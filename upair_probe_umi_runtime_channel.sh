#!/usr/bin/env bash
# Runtime probe for randomized UMi channel path.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
source "${ROOT}/upair_portable_env.sh"
upair_activate
python - <<'PY'
from __future__ import annotations
import tensorflow as tf
from upair5g.config import load_config, get_cfg
from upair5g.training import _build_system_for_num_users, _make_batch
from upair5g.estimator import UPAIRChannelEstimator
from upair5g.utils import btfnu_to_tensor7
cfg = load_config("configs/twc_comprehensive_mu32_base.yaml")
print("[PROBE] channel.family:", get_cfg(cfg, "channel.family", None))
print("[PROBE] channel.model:", get_cfg(cfg, "channel.model", None))
print("[PROBE] channel.umi:", get_cfg(cfg, "channel.umi", {}))
assert str(get_cfg(cfg, "channel.family", "")).lower() == "umi"
assert str(get_cfg(cfg, "channel.model", "")).lower() == "umi"
for u in [1,2,3,4]:
    print("\n" + "="*100)
    print(f"[PROBE] Building UMi system for U={u}")
    system = _build_system_for_num_users(cfg, u)
    channel = system["channel"]
    print("[PROBE] channel class:", channel.__class__.__name__)
    assert "TopologyRefreshingOFDMChannel" in channel.__class__.__name__, channel.__class__
    pilot_sums = tf.reduce_sum(system["pilot_mask"], axis=[0,1]).numpy()
    print("[PROBE] true DMRS mask shape/sums:", system["pilot_mask"].shape, pilot_sums)
    assert list(pilot_sums[:u]) == [32.0] * u
    batch = _make_batch(system["tx"], system["channel"], cfg, batch_size=2, training=True, fixed_ebno_db=0.0)
    print("[PROBE] y shape:", batch["y"].shape)
    print("[PROBE] h shape:", batch["h"].shape)
    print("[PROBE] no:", float(tf.reshape(batch["no"], [-1])[0].numpy()))
    assert int(batch["y"].shape[0]) == 2
    assert int(batch["y"].shape[2]) == int(cfg["channel"]["num_rx_ant"])
    assert int(batch["y"].shape[-2]) == 14
    assert int(batch["y"].shape[-1]) == 96
    assert int(batch["h"].shape[0]) == 2
    assert int(batch["h"].shape[3]) == u
system = _build_system_for_num_users(cfg, 4)
estimator = UPAIRChannelEstimator(ls_estimator=system["ls_estimator"], resource_grid=system["resource_grid"], cfg=cfg, pilot_mask=system["pilot_mask"])
batch = _make_batch(system["tx"], system["channel"], cfg, batch_size=2, training=False, fixed_ebno_db=0.0)
h_ls, err_ls = estimator._call_ls(batch["y"], batch["no"], system["ls_estimator"])
feat, _, _, _ = estimator._build_features(batch["y"], h_ls, err_ls, batch["no"], pilot_mask=system["pilot_mask"])
print("\n[PROBE] feature shape:", feat.shape)
assert tuple(feat.shape) == (2,14,96,169)
h_hat, err_hat, h_ls2, err_ls2 = estimator.estimate_with_ls(batch["y"], batch["no"], training=False, ls_estimator=system["ls_estimator"], pilot_mask=system["pilot_mask"])
_, _, err_btfnu2, _ = estimator._build_features(batch["y"], h_ls2, err_ls2, batch["no"], pilot_mask=system["pilot_mask"])
err_anchor = tf.cast(btfnu_to_tensor7(err_btfnu2), tf.float32)
max_h = float(tf.reduce_max(tf.abs(h_hat - h_ls2)).numpy())
max_e = float(tf.reduce_max(tf.abs(err_hat - err_anchor)).numpy())
print(f"[PROBE] max |h_hat_init - h_ls| = {max_h:.3e}")
print(f"[PROBE] max |err_hat_init - err_ls_broadcast| = {max_e:.3e}")
assert max_h < 1e-7
assert max_e < 1e-7
print("\n[PROBE] PASSED randomized UMi runtime channel/feature probe.")
PY
