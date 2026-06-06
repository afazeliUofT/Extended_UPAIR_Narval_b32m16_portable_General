#!/usr/bin/env bash
# Verify that the UPAIR estimator now receives TRUE nonzero DMRS RE masks per port.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/upair_portable_env.sh"
upair_activate

python - <<'PY'
from __future__ import annotations

import numpy as np
import tensorflow as tf

from upair5g.config import load_config, set_cfg
from upair5g.builders import (
    build_pusch_transmitter,
    get_resource_grid,
    extract_pilot_mask_per_stream,
    extract_true_dmrs_mask_per_stream,
)
from upair5g.training import _build_system_for_num_users, _make_batch
from upair5g.estimator import UPAIRChannelEstimator
from upair5g.utils import btfnu_to_tensor7


def get_mods(mask_tf):
    mask = np.asarray(tf.convert_to_tensor(mask_tf).numpy())
    out = {}
    for u in range(mask.shape[-1]):
        nz = np.argwhere(mask[..., u] > 0)
        by_t = {}
        for t, k in nz:
            by_t.setdefault(int(t), []).append(int(k))
        out[u] = {t: sorted(set(k % 12 for k in ks)) for t, ks in by_t.items()}
    return out


def assert_equal(label, got, expected):
    if got != expected:
        raise SystemExit(f"[FAIL] {label}: got {got}, expected {expected}")
    print(f"[OK] {label}: {got}")


def check_case(addpos: int):
    cfg = load_config("configs/twc_comprehensive_mu32_base.yaml")
    set_cfg(cfg, "pusch.n_size_grid", 8)
    set_cfg(cfg, "pusch.n_size_bwp", 8)
    set_cfg(cfg, "multiuser.enabled", True)
    set_cfg(cfg, "multiuser.max_num_users", 4)
    set_cfg(cfg, "multiuser.fixed_num_users", 4)
    set_cfg(cfg, "multiuser.dmrs.config_type", 2)
    set_cfg(cfg, "multiuser.dmrs.length", 1)
    set_cfg(cfg, "multiuser.dmrs.additional_position", addpos)
    set_cfg(cfg, "multiuser.dmrs.type_a_position", 2)
    set_cfg(cfg, "multiuser.dmrs.num_cdm_groups_without_data", 3)
    set_cfg(cfg, "multiuser.dmrs.port_sets", [[0], [1], [2], [3]])

    tx, _ = build_pusch_transmitter(cfg, num_users=4)
    rg = get_resource_grid(tx)
    broad = extract_pilot_mask_per_stream(rg)
    true = extract_true_dmrs_mask_per_stream(tx, rg)

    print("\n" + "=" * 90)
    print(f"addpos={addpos}")
    print(f"broad_mask_shape={broad.shape}, sums={tf.reduce_sum(broad, axis=[0,1]).numpy()}")
    print(f"true_mask_shape ={true.shape}, sums={tf.reduce_sum(true, axis=[0,1]).numpy()}")

    if addpos == 0:
        expected_sums = [32.0, 32.0, 32.0, 32.0]
        expected_symbols = [2]
    else:
        expected_sums = [64.0, 64.0, 64.0, 64.0]
        expected_symbols = [2, 11]

    sums = [float(x) for x in tf.reduce_sum(true, axis=[0,1]).numpy().tolist()]
    assert_equal(f"true mask sums addpos={addpos}", sums, expected_sums)

    mods = get_mods(true)
    for u in [0, 1]:
        assert_equal(f"user {u+1} symbols addpos={addpos}", sorted(mods[u]), expected_symbols)
        for t in expected_symbols:
            assert_equal(f"user {u+1} k mod 12 at symbol {t}", mods[u][t], [0, 1, 6, 7])
    for u in [2, 3]:
        assert_equal(f"user {u+1} symbols addpos={addpos}", sorted(mods[u]), expected_symbols)
        for t in expected_symbols:
            assert_equal(f"user {u+1} k mod 12 at symbol {t}", mods[u][t], [2, 3, 8, 9])


def check_estimator_feature_path():
    cfg = load_config("configs/twc_comprehensive_mu32_base.yaml")
    set_cfg(cfg, "multiuser.enabled", True)
    set_cfg(cfg, "multiuser.max_num_users", 4)
    set_cfg(cfg, "multiuser.fixed_num_users", 4)
    set_cfg(cfg, "multiuser.dmrs.additional_position", 0)

    system = _build_system_for_num_users(cfg, 4)
    print("\n" + "=" * 90)
    print("training system pilot mask sums:", tf.reduce_sum(system["pilot_mask"], axis=[0,1]).numpy())

    estimator = UPAIRChannelEstimator(
        ls_estimator=system["ls_estimator"],
        resource_grid=system["resource_grid"],
        cfg=cfg,
        pilot_mask=system["pilot_mask"],
    )
    batch = _make_batch(system["tx"], system["channel"], cfg, batch_size=2, training=False, fixed_ebno_db=0.0)
    h_ls, err_ls = estimator._call_ls(batch["y"], batch["no"], system["ls_estimator"])
    feat, _, _, _ = estimator._build_features(
        batch["y"], h_ls, err_ls, batch["no"], pilot_mask=system["pilot_mask"]
    )
    print("feature shape:", feat.shape)
    assert int(feat.shape[0]) == 2
    assert int(feat.shape[1]) == 14
    assert int(feat.shape[2]) == 96
    assert int(feat.shape[3]) == 169
    pilot_features = feat[..., -4:]
    sums = tf.reduce_sum(pilot_features, axis=[0,1,2]).numpy().tolist()
    assert_equal("estimator feature pilot sums for batch=2", [float(x) for x in sums], [64.0, 64.0, 64.0, 64.0])

    h_hat, err_hat, h_ls2, err_ls2 = estimator.estimate_with_ls(
        batch["y"], batch["no"], training=False, ls_estimator=system["ls_estimator"], pilot_mask=system["pilot_mask"]
    )
    _, _, err_btfnu2, _ = estimator._build_features(batch["y"], h_ls2, err_ls2, batch["no"], pilot_mask=system["pilot_mask"])
    err_anchor = tf.cast(btfnu_to_tensor7(err_btfnu2), tf.float32)
    max_h = float(tf.reduce_max(tf.abs(h_hat - h_ls2)).numpy())
    max_e = float(tf.reduce_max(tf.abs(err_hat - err_anchor)).numpy())
    print(f"max |h_hat_init - h_ls| = {max_h:.3e}")
    print(f"max |err_hat_init - err_ls_broadcast| = {max_e:.3e}")
    assert max_h < 1e-7
    assert max_e < 1e-7


check_case(0)
check_case(1)
check_estimator_feature_path()
print("\n[PROBE] PASSED true-DMRS-mask patch probe.")
PY
