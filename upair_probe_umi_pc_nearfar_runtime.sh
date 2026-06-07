#!/usr/bin/env bash
# Runtime probe for UMi pathloss/shadowing + fractional power-control mean re-referencing.
# Run on a GPU node if possible.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/upair_portable_env.sh"
upair_activate

python - <<'PY'
from __future__ import annotations
import numpy as np
import tensorflow as tf

from upair5g.config import load_config, set_cfg, get_cfg
from upair5g.training import _build_system_for_num_users, _make_batch
from upair5g.estimator import UPAIRChannelEstimator
from upair5g.utils import btfnu_to_tensor7

def run_alpha(alpha: float, batch_size: int = 4):
    cfg = load_config("configs/twc_comprehensive_mu32_base.yaml")
    set_cfg(cfg, "near_far.alpha_eval", float(alpha))
    set_cfg(cfg, "near_far.alpha_train_min", float(alpha))
    set_cfg(cfg, "near_far.alpha_train_max", float(alpha))
    set_cfg(cfg, "system.batch_size_train", batch_size)
    set_cfg(cfg, "system.batch_size_eval", batch_size)

    system = _build_system_for_num_users(cfg, 4)
    channel = system["channel"]
    print("\n" + "="*100)
    print(f"[ALPHA] {alpha}")
    print("[PROBE] channel class:", channel.__class__.__name__)
    print("[PROBE] UMi pathloss/shadow:", get_cfg(cfg, "channel.umi.enable_pathloss"), get_cfg(cfg, "channel.umi.enable_shadow_fading"))
    print("[PROBE] normalize_channel:", get_cfg(cfg, "channel.normalize_channel"))

    batch = _make_batch(system["tx"], system["channel"], cfg, batch_size=batch_size, training=False, fixed_ebno_db=0.0)
    stats = channel.last_near_far_stats
    if not stats:
        raise SystemExit("[FAIL] channel did not expose last_near_far_stats")

    raw = stats["raw_power"].numpy()
    post = stats["post_power"].numpy()
    raw_spread = stats["raw_spread_db"].numpy()
    post_spread = stats["post_spread_db"].numpy()
    alpha_used = stats["alpha"].numpy().reshape(-1)
    post_mean = stats["post_power_mean"].numpy()

    print("[PROBE] y shape:", batch["y"].shape)
    print("[PROBE] h shape:", batch["h"].shape)
    print("[PROBE] alpha_used:", alpha_used)
    print("[PROBE] raw_power first row:", raw[0])
    print("[PROBE] post_power first row:", post[0])
    print("[PROBE] raw_spread_db:", raw_spread)
    print("[PROBE] post_spread_db:", post_spread)
    print("[PROBE] post_power_mean:", post_mean)

    assert tuple(batch["y"].shape) == (batch_size, 1, 16, 14, 96)
    assert int(batch["h"].shape[3]) == 4
    assert np.allclose(post_mean, 1.0, rtol=1e-5, atol=1e-5)

    expected_ratio = max(0.0, 1.0 - float(alpha))
    if np.max(raw_spread) > 1e-3:
        ratio = post_spread / np.maximum(raw_spread, 1e-9)
        print("[PROBE] spread ratio post/raw:", ratio, "expected:", expected_ratio)
        assert np.allclose(ratio, expected_ratio, rtol=2e-3, atol=2e-3)

    if abs(alpha - 1.0) < 1e-12:
        assert np.max(post_spread) < 1e-3

    return cfg, system, batch

# Test limiting and realistic cases. Training will not use alpha close to zero,
# but alpha=0 is useful as a mathematical stress probe.
cfg, system, batch = run_alpha(1.0)
run_alpha(0.8)
run_alpha(0.7)
run_alpha(0.0)

# Feature/estimator path under realistic alpha_eval=0.8
cfg = load_config("configs/twc_comprehensive_mu32_base.yaml")
set_cfg(cfg, "near_far.alpha_eval", 0.8)
system = _build_system_for_num_users(cfg, 4)
estimator = UPAIRChannelEstimator(
    ls_estimator=system["ls_estimator"],
    resource_grid=system["resource_grid"],
    cfg=cfg,
    pilot_mask=system["pilot_mask"],
)
batch = _make_batch(system["tx"], system["channel"], cfg, batch_size=2, training=False, fixed_ebno_db=0.0)
h_ls, err_ls = estimator._call_ls(batch["y"], batch["no"], system["ls_estimator"])
feat, _, _, _ = estimator._build_features(batch["y"], h_ls, err_ls, batch["no"], pilot_mask=system["pilot_mask"])
print("\n[PROBE] feature shape:", feat.shape)
assert tuple(feat.shape) == (2, 14, 96, 169)

h_hat, err_hat, h_ls2, err_ls2 = estimator.estimate_with_ls(batch["y"], batch["no"], training=False, ls_estimator=system["ls_estimator"], pilot_mask=system["pilot_mask"])
_, _, err_btfnu2, _ = estimator._build_features(batch["y"], h_ls2, err_ls2, batch["no"], pilot_mask=system["pilot_mask"])
err_anchor = tf.cast(btfnu_to_tensor7(err_btfnu2), tf.float32)
max_h = float(tf.reduce_max(tf.abs(h_hat - h_ls2)).numpy())
max_e = float(tf.reduce_max(tf.abs(err_hat - err_anchor)).numpy())
print(f"[PROBE] max |h_hat_init - h_ls| = {max_h:.3e}")
print(f"[PROBE] max |err_hat_init - err_ls_broadcast| = {max_e:.3e}")
assert max_h < 1e-7
assert max_e < 1e-7

print("\n[PROBE] PASSED UMi fractional-power-control near-far runtime probe.")
PY
