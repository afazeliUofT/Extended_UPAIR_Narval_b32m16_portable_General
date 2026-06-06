#!/usr/bin/env bash
# Runtime probe for the exact Stage-A warm-up path that failed during Optuna.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/upair_portable_env.sh"
upair_activate

python - <<'PY'
from __future__ import annotations
import tensorflow as tf

from upair5g.config import load_config, set_cfg
from upair5g.training import _build_system_for_num_users, _make_batch
from upair5g.estimator import UPAIRChannelEstimator
from upair5g.utils import btfnu_to_tensor7

cfg = load_config("configs/twc_comprehensive_mu32_base.yaml")
set_cfg(cfg, "multiuser.enabled", True)
set_cfg(cfg, "multiuser.max_num_users", 4)
set_cfg(cfg, "multiuser.fixed_num_users", 4)
set_cfg(cfg, "multiuser.dmrs.length", 1)
set_cfg(cfg, "multiuser.dmrs.additional_position", 0)

system = _build_system_for_num_users(cfg, 4)
estimator = UPAIRChannelEstimator(
    ls_estimator=system["ls_estimator"],
    resource_grid=system["resource_grid"],
    cfg=cfg,
)

batch = _make_batch(
    tx=system["tx"],
    channel=system["channel"],
    cfg=cfg,
    batch_size=2,   # exact warm-up size used by train_model when train batch >=2
    training=False,
    fixed_ebno_db=0.0,
)

h_ls, err_ls = estimator._call_ls(batch["y"], batch["no"], system["ls_estimator"])
feat, h_ls_btfnu, err_btfnu, _ = estimator._build_features(
    batch["y"], h_ls, err_ls, batch["no"], pilot_mask=system["pilot_mask"]
)

print("[PROBE] y shape       :", batch["y"].shape)
print("[PROBE] h_ls shape    :", h_ls.shape)
print("[PROBE] err_ls shape  :", err_ls.shape)
print("[PROBE] feat shape    :", feat.shape)
print("[PROBE] input_channels:", estimator.input_channels)

assert int(feat.shape[0]) == 2, f"feature batch dimension should be 2, got {feat.shape[0]}"
assert int(feat.shape[1]) == 14, f"time dimension should be 14, got {feat.shape[1]}"
assert int(feat.shape[2]) == 96, f"frequency dimension should be 96 for 8 RB, got {feat.shape[2]}"
assert int(feat.shape[3]) == 169, f"feature channels should be 169, got {feat.shape[3]}"
assert int(estimator.input_channels) == int(feat.shape[3]), "input_channels mismatch"

h_hat, err_hat, h_ls2, err_ls2 = estimator.estimate_with_ls(
    batch["y"],
    batch["no"],
    training=False,
    ls_estimator=system["ls_estimator"],
    pilot_mask=system["pilot_mask"],
)

_, _, err_btfnu2, _ = estimator._build_features(
    batch["y"], h_ls2, err_ls2, batch["no"], pilot_mask=system["pilot_mask"]
)
err_anchor_tensor7 = tf.cast(btfnu_to_tensor7(err_btfnu2), tf.float32)

max_h = float(tf.reduce_max(tf.abs(h_hat - h_ls2)).numpy())
max_err = float(tf.reduce_max(tf.abs(err_hat - err_anchor_tensor7)).numpy())

print(f"[PROBE] max |h_hat_init - h_ls| = {max_h:.3e}")
print(f"[PROBE] max |err_hat_init - err_ls_broadcast| = {max_err:.3e}")

assert max_h < 1e-7, "channel head no longer starts exactly from LS"
assert max_err < 1e-7, "error head no longer starts exactly from LS variance"

print("[PROBE] PASSED Stage-A batch-size-2 feature concat/warm-up test.")
PY
