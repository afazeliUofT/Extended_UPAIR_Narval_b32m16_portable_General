#!/usr/bin/env bash
# Runtime sanity probe for the UPAIR feature/PRB/d-model patch.
# Run from the repository root, preferably on a short interactive GPU allocation.
set -euo pipefail

ROOT="$(pwd)"
if [[ ! -d "${ROOT}/src/upair5g" || ! -d "${ROOT}/scripts" || ! -d "${ROOT}/configs" ]]; then
  echo "[PROBE] Please run this from the UPAIR repository root." >&2
  exit 1
fi

source "${ROOT}/upair_portable_env.sh"
upair_activate

python - <<'PY'
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import tensorflow as tf

from upair5g.config import load_config, get_cfg, set_cfg
from upair5g.builders import (
    build_pusch_transmitter,
    build_channel,
    build_ls_estimator,
    get_resource_grid,
    extract_pilot_mask_per_stream,
)
from upair5g.estimator import UPAIRChannelEstimator
from upair5g.training import _make_batch
from upair5g.utils import broadcast_like_err

ROOT = Path.cwd()
failures: list[str] = []


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    failures.append(msg)


def check(cond: bool, msg: str) -> None:
    ok(msg) if cond else fail(msg)

cfg = load_config("configs/twc_comprehensive_mu32_base.yaml")
check(int(get_cfg(cfg, "pusch.n_size_grid")) == 8, "pusch.n_size_grid is 8")
check(int(get_cfg(cfg, "pusch.n_size_bwp")) == 8, "pusch.n_size_bwp is 8")
check(str(get_cfg(cfg, "model.pilot_mask_mode", "")) == "per_stream", "model.pilot_mask_mode is per_stream")
check(str(get_cfg(cfg, "model.error_feature_mode", "")) == "per_user", "model.error_feature_mode is per_user")
check([float(x) for x in get_cfg(cfg, "training.val_user_count_weights")] == [1.0, 3.0, 6.0, 10.0], "validation user-count weights match training weights")
check(int(get_cfg(cfg, "model.d_model")) == 256, "base model.d_model is 256")

# Load Optuna/common variants without importing TensorFlow-heavy comprehensive runner.
spec = importlib.util.spec_from_file_location("optuna_1dmrs_common", ROOT / "scripts" / "optuna_1dmrs_common.py")
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
expected_variants = {
    "main_d256_b4_r2": 256,
    "shallow_d256_b2_r2": 256,
    "deep_d256_b6_r2": 256,
    "narrow_d192_b4_r2": 192,
    "wide_d320_b4_r2": 320,
    "wide_deep_d320_b6_r2": 320,
    "mlpwide_d256_b4_r4": 256,
}
check(set(mod.VARIANTS) == set(expected_variants), "Optuna variant names are the new d192/d256/d320 names")
for name, expected_d in expected_variants.items():
    actual_d = int(mod.VARIANTS[name]["model.d_model"])
    check(actual_d == expected_d, f"{name} has d_model={expected_d}")

# Static expected feature-channel accounting after patch:
# h_LS real/imag: 2*16*4 = 128
# y real/imag:    2*16   = 32
# LS err per user:         4
# N0:                      1
# pilot mask per user:     4
expected_feature_channels = 169
for name, d in expected_variants.items():
    check(d > expected_feature_channels, f"{name}: d_model={d} > feature_channels={expected_feature_channels}")


def summarize_case(additional_position: int, num_users: int) -> None:
    local_cfg = copy.deepcopy(cfg)
    set_cfg(local_cfg, "multiuser.dmrs.length", 1)
    set_cfg(local_cfg, "multiuser.dmrs.additional_position", additional_position)
    set_cfg(local_cfg, "multiuser.fixed_num_users", num_users)

    tx, _ = build_pusch_transmitter(local_cfg, num_users=num_users)
    channel = build_channel(local_cfg, tx)
    ls = build_ls_estimator(tx, local_cfg)
    rg = get_resource_grid(tx)
    mask_stream = extract_pilot_mask_per_stream(rg)
    estimator = UPAIRChannelEstimator(ls_estimator=ls, resource_grid=rg, cfg=local_cfg)

    batch = _make_batch(
        tx=tx,
        channel=channel,
        cfg=local_cfg,
        batch_size=1,
        training=False,
        fixed_ebno_db=0.0,
    )
    h_ls, err_ls = estimator._call_ls(batch["y"], batch["no"], ls)
    feat, _, _, _ = estimator._build_features(batch["y"], h_ls, err_ls, batch["no"], pilot_mask=mask_stream)
    h_hat, err_hat, h_ls2, err_ls2 = estimator.estimate_with_ls(
        batch["y"],
        batch["no"],
        training=False,
        ls_estimator=ls,
        pilot_mask=mask_stream,
    )

    print("\n" + "=" * 88)
    print(f"case additional_position={additional_position} num_users={num_users}")
    print(f"feature_shape={tuple(feat.shape)} input_channels={estimator.input_channels}")
    print(f"pilot_stream_shape={tuple(mask_stream.shape)} pilot_stream_sums={tf.reduce_sum(mask_stream, axis=[0,1]).numpy()}")

    check(tuple(feat.shape[1:3]) == (14, 96), f"case addpos={additional_position}, U={num_users}: grid is 14 x 96")
    check(int(feat.shape[-1]) == expected_feature_channels, f"case addpos={additional_position}, U={num_users}: feature channels are {expected_feature_channels}")
    check(int(estimator.input_channels) == expected_feature_channels, f"case addpos={additional_position}, U={num_users}: estimator.input_channels matches feature channels")

    # Feature layout after patch: [h_ri:128 | y_ri:32 | err_user:4 | no:1 | pilot_user:4]
    feat_np = feat.numpy()
    err_user = feat_np[..., 160:164]
    no_map = feat_np[..., 164]
    pilot_user = feat_np[..., 165:169]

    active_err_sum = np.sum(np.abs(err_user[..., :num_users]))
    inactive_err_sum = np.sum(np.abs(err_user[..., num_users:]))
    active_pilot_sum = np.sum(np.abs(pilot_user[..., :num_users]))
    inactive_pilot_sum = np.sum(np.abs(pilot_user[..., num_users:]))
    check(active_err_sum > 0.0, f"case addpos={additional_position}, U={num_users}: active per-user error features are nonzero")
    check(inactive_err_sum == 0.0, f"case addpos={additional_position}, U={num_users}: inactive per-user error features are zero-padded")
    check(active_pilot_sum > 0.0, f"case addpos={additional_position}, U={num_users}: active per-user pilot features are nonzero")
    check(inactive_pilot_sum == 0.0, f"case addpos={additional_position}, U={num_users}: inactive per-user pilot features are zero-padded")
    check(len(np.unique(np.round(no_map.reshape(-1), 10))) == 1, f"case addpos={additional_position}, U={num_users}: N0 feature remains a broadcast scalar")

    h_diff = tf.reduce_max(tf.abs(h_hat - h_ls)).numpy()
    err_anchor = broadcast_like_err(err_ls, h_ls)
    err_diff = tf.reduce_max(tf.abs(err_hat - err_anchor)).numpy()
    print(f"max|h_hat_init - h_ls|={float(h_diff):.3e}")
    print(f"max|err_hat_init - err_ls_broadcast|={float(err_diff):.3e}")
    check(float(h_diff) < 1e-7, f"case addpos={additional_position}, U={num_users}: channel head starts exactly from LS")
    check(float(err_diff) < 1e-7, f"case addpos={additional_position}, U={num_users}: error head starts exactly from LS variance")


for addpos in (0, 1):
    for users in (1, 2, 3, 4):
        summarize_case(addpos, users)

# Check pyproject no longer references README.md.
pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
check('readme = "README.md"' not in pyproject_text, "pyproject.toml does not reference missing README.md")

# Check clean probe no longer uses compileall.
probe_text = (ROOT / "upair_probe_clean_start.sh").read_text(encoding="utf-8")
check("compileall" not in probe_text, "clean-start probe no longer calls compileall")

if failures:
    print("\n[PROBE] FAILED")
    raise SystemExit(1)
print("\n[PROBE] PASSED feature/PRB/d-model patch sanity checks")
PY
