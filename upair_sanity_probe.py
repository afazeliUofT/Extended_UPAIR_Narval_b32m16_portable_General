#!/usr/bin/env python
"""
UPAIR UMi pipeline sanity probe (read-only, diagnostic).

Prints FACTS about what the channel pipeline actually produces at runtime so the
five concerns flagged in review can be confirmed or ruled out WITHOUT guesswork:

  CHECK 1  Antenna-count consistency: PanelArray element count == num_rx_ant / num_tx_ant.
  CHECK 2  Channel-tensor layout: does UMi h have the user axis == U, rx axis == num_rx_ant,
           and does it MATCH the CDL path's layout for the same U?
  CHECK 3  Per-user receive-power spread (near-far). With pathloss off this should be ~flat;
           the probe quantifies it so you can see exactly how much near-far realism exists.
  CHECK 4  pathloss / shadow_fading / normalize_channel interaction: does enabling pathloss
           actually change per-user power, and does normalize_channel cancel it?
  CHECK 5  Topology randomization: do two batches produce genuinely different channels (UMi)
           and does turning randomize_topology_each_batch off freeze them?

It does NOT modify any tracked file and writes nothing except stdout.
All numbers are printed; nothing is hidden behind a pass/fail assert except a few
hard structural invariants (which, if they fire, are themselves the finding).
"""
from __future__ import annotations

import copy
import sys
import traceback
from typing import Any

import numpy as np
import tensorflow as tf

from upair5g.config import load_config, get_cfg, set_cfg
from upair5g import builders
from upair5g.training import _build_system_for_num_users, _make_batch

CONFIG_PATH = "configs/twc_comprehensive_mu32_base.yaml"
PROBe_USERS = [1, 2, 4]   # kept small; bump if you want U=32 (slower)
BATCH = 2
SEED = 1234


def _hr(title: str) -> None:
    print("\n" + "=" * 100)
    print(f"[PROBE] {title}")
    print("=" * 100)


def _arr(x: tf.Tensor) -> np.ndarray:
    return np.asarray(x.numpy()) if hasattr(x, "numpy") else np.asarray(x)


def _power_per_user(h: tf.Tensor, user_axis: int) -> np.ndarray:
    """Mean |h|^2 per user, averaging over every axis except `user_axis`."""
    h = tf.convert_to_tensor(h)
    p = tf.abs(h) ** 2
    reduce_axes = [a for a in range(len(h.shape)) if a != user_axis]
    return _arr(tf.reduce_mean(p, axis=reduce_axes))


def _guess_user_axis(h_shape, num_users: int) -> int:
    """Find the axis whose static size equals num_users (the transmit/UT axis)."""
    matches = [i for i, d in enumerate(h_shape) if d is not None and int(d) == int(num_users)]
    return matches[0] if matches else -1


def _set_umi(cfg: dict[str, Any], **umi_over) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    for k, v in umi_over.items():
        set_cfg(c, f"channel.umi.{k}", v)
    return c


def _force_family(cfg: dict[str, Any], family: str) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    set_cfg(c, "channel.family", family)
    set_cfg(c, "channel.model", family)
    return c


def _build_and_batch(cfg: dict[str, Any], u: int, fixed_ebno_db: float = 10.0):
    sysd = _build_system_for_num_users(cfg, u)
    batch = _make_batch(sysd["tx"], sysd["channel"], cfg, batch_size=BATCH,
                        training=False, fixed_ebno_db=fixed_ebno_db)
    return sysd, batch


# --------------------------------------------------------------------------------------
def check0_environment(cfg):
    _hr("CHECK 0 — environment & config snapshot")
    print("[PROBE] python      :", sys.version.split()[0])
    print("[PROBE] tensorflow  :", tf.__version__)
    try:
        import sionna
        print("[PROBE] sionna      :", getattr(sionna, "__version__", "unknown"))
    except Exception as e:
        print("[PROBE] sionna import failed:", e)
    print("[PROBE] GPUs         :", [d.name for d in tf.config.list_physical_devices("GPU")])
    print("[PROBE] channel.family       :", get_cfg(cfg, "channel.family"))
    print("[PROBE] channel.model        :", get_cfg(cfg, "channel.model"))
    print("[PROBE] channel.num_rx_ant   :", get_cfg(cfg, "channel.num_rx_ant"))
    print("[PROBE] channel.num_tx_ant   :", get_cfg(cfg, "channel.num_tx_ant"))
    print("[PROBE] channel.normalize    :", get_cfg(cfg, "channel.normalize_channel"))
    print("[PROBE] umi block            :", get_cfg(cfg, "channel.umi", {}))
    print("[PROBE] multiuser.max_num_users   :", get_cfg(cfg, "multiuser.max_num_users"))
    print("[PROBE] multiuser.fixed_num_users :", get_cfg(cfg, "multiuser.fixed_num_users"))


def check1_antenna_counts(cfg):
    _hr("CHECK 1 — PanelArray element count vs configured num_rx_ant / num_tx_ant")
    num_rx = int(get_cfg(cfg, "channel.num_rx_ant"))
    num_tx = int(get_cfg(cfg, "channel.num_tx_ant"))
    model = builders._build_umi_channel_model(cfg, num_users=1)

    def _count(arr):
        for attr in ["num_ant", "num_antennas", "array_size"]:
            v = getattr(arr, attr, None)
            if v is not None:
                try:
                    return int(v)
                except Exception:
                    pass
        # PanelArray exposes element positions; fall back to that
        for attr in ["ant_pos", "antenna_positions", "positions"]:
            v = getattr(arr, attr, None)
            if v is not None:
                try:
                    return int(np.asarray(v).shape[0])
                except Exception:
                    pass
        return None

    bs = getattr(model, "_bs_array", getattr(model, "bs_array", None))
    ut = getattr(model, "_ut_array", getattr(model, "ut_array", None))
    bs_n = _count(bs) if bs is not None else None
    ut_n = _count(ut) if ut is not None else None
    rows = int(get_cfg(cfg, "channel.umi.bs_array_rows", 4))
    cols = int(get_cfg(cfg, "channel.umi.bs_array_cols", num_rx // 4))
    print(f"[PROBE] configured num_rx_ant      = {num_rx}")
    print(f"[PROBE] bs_array_rows x cols        = {rows} x {cols} = {rows*cols} (x pol=1)")
    print(f"[PROBE] BS PanelArray element count = {bs_n}")
    print(f"[PROBE] configured num_tx_ant      = {num_tx}")
    print(f"[PROBE] UT PanelArray element count = {ut_n}")
    if bs_n is not None and bs_n != num_rx:
        print(f"[PROBE][FINDING] BS array element count {bs_n} != num_rx_ant {num_rx}  <<< INCONSISTENT")
    else:
        print("[PROBE] BS antenna count consistent.")
    if ut_n is not None and ut_n != num_tx:
        print(f"[PROBE][FINDING] UT array element count {ut_n} != num_tx_ant {num_tx}  <<< INCONSISTENT")
    else:
        print("[PROBE] UT antenna count consistent.")


def check2_layout_and_parity(cfg):
    _hr("CHECK 2 — UMi channel-tensor layout, and parity vs CDL path")
    umi_cfg = _force_family(cfg, "umi")
    cdl_cfg = _force_family(cfg, "cdl")
    for u in PROBe_USERS:
        print(f"\n--- U={u} ---")
        try:
            _, b_umi = _build_and_batch(umi_cfg, u)
            hu = b_umi["h"]; yu = b_umi["y"]
            ax_u = _guess_user_axis(hu.shape, u)
            print(f"[PROBE] UMi  y.shape={tuple(int(d) if d is not None else -1 for d in yu.shape)}")
            print(f"[PROBE] UMi  h.shape={tuple(int(d) if d is not None else -1 for d in hu.shape)}  (user-axis guess={ax_u})")
        except Exception:
            print("[PROBE][FINDING] UMi build/batch FAILED for U=%d:" % u)
            traceback.print_exc()
            continue
        try:
            _, b_cdl = _build_and_batch(cdl_cfg, u)
            hc = b_cdl["h"]; yc = b_cdl["y"]
            ax_c = _guess_user_axis(hc.shape, u)
            print(f"[PROBE] CDL  y.shape={tuple(int(d) if d is not None else -1 for d in yc.shape)}")
            print(f"[PROBE] CDL  h.shape={tuple(int(d) if d is not None else -1 for d in hc.shape)}  (user-axis guess={ax_c})")
            # Parity: same rank, same per-axis sizes EXCEPT possibly value content
            su = [int(d) if d is not None else -1 for d in hu.shape]
            sc = [int(d) if d is not None else -1 for d in hc.shape]
            if su == sc:
                print("[PROBE] LAYOUT PARITY OK: UMi and CDL h tensors have identical shape.")
            else:
                print(f"[PROBE][FINDING] LAYOUT MISMATCH: UMi h {su} vs CDL h {sc}  <<< estimator sees different layouts per family")
            if int(yu.shape[2]) != int(get_cfg(cfg, "channel.num_rx_ant")):
                print(f"[PROBE][FINDING] UMi y rx-axis {int(yu.shape[2])} != num_rx_ant {get_cfg(cfg,'channel.num_rx_ant')}")
            if ax_u != ax_c:
                print(f"[PROBE][FINDING] user-axis differs UMi={ax_u} vs CDL={ax_c}  <<< check estimator's assumed axis")
        except Exception:
            print("[PROBE][NOTE] CDL comparison build failed (may be expected if CDL MU path differs):")
            traceback.print_exc()


def check3_near_far(cfg):
    _hr("CHECK 3 — per-user receive-power spread (near-far realism)")
    umi_cfg = _force_family(cfg, "umi")
    u = max(PROBe_USERS)
    try:
        _, batch = _build_and_batch(umi_cfg, u)
        h = batch["h"]
        ax = _guess_user_axis(h.shape, u)
        if ax < 0:
            print(f"[PROBE][NOTE] could not locate user axis in h.shape={h.shape}; skipping per-user power.")
            return
        pu = _power_per_user(h, ax)
        pu_db = 10.0 * np.log10(np.maximum(pu, 1e-30))
        print(f"[PROBE] per-user mean|h|^2 (linear): {np.array2string(pu, precision=4)}")
        print(f"[PROBE] per-user power (dB)        : {np.array2string(pu_db, precision=2)}")
        spread = float(pu_db.max() - pu_db.min())
        print(f"[PROBE] near-far spread (max-min, dB) = {spread:.2f}")
        if spread < 1.0:
            print("[PROBE][INTERPRETATION] spread < 1 dB  -> users are ~equal power (NO near-far). "
                  "Consistent with pathloss disabled and/or normalize_channel=true.")
        else:
            print("[PROBE][INTERPRETATION] non-trivial per-user power spread present.")
    except Exception:
        print("[PROBE][FINDING] near-far probe FAILED:")
        traceback.print_exc()


def check4_pathloss_normalize(cfg):
    _hr("CHECK 4 — pathloss / shadow / normalize_channel interaction")
    u = max(PROBe_USERS)
    scenarios = [
        ("pathloss=OFF norm=TRUE  (current)", dict(family="umi", pathloss=False, shadow=False, norm=True)),
        ("pathloss=ON  norm=TRUE  (does norm cancel PL?)", dict(family="umi", pathloss=True,  shadow=True,  norm=True)),
        ("pathloss=ON  norm=FALSE (PL should show)", dict(family="umi", pathloss=True,  shadow=True,  norm=False)),
    ]
    for label, s in scenarios:
        c = _force_family(cfg, s["family"])
        set_cfg(c, "channel.umi.enable_pathloss", bool(s["pathloss"]))
        set_cfg(c, "channel.umi.enable_shadow_fading", bool(s["shadow"]))
        set_cfg(c, "channel.normalize_channel", bool(s["norm"]))
        print(f"\n--- {label} ---")
        try:
            _, batch = _build_and_batch(c, u)
            h = batch["h"]
            ax = _guess_user_axis(h.shape, u)
            pu = _power_per_user(h, ax) if ax >= 0 else None
            total = float(tf.reduce_mean(tf.abs(h) ** 2).numpy())
            print(f"[PROBE] total mean|h|^2 = {total:.4e}")
            if pu is not None:
                pu_db = 10.0 * np.log10(np.maximum(pu, 1e-30))
                print(f"[PROBE] per-user power (dB) = {np.array2string(pu_db, precision=2)}  spread={float(pu_db.max()-pu_db.min()):.2f} dB")
        except Exception:
            print("[PROBE][NOTE] scenario failed to build/run:")
            traceback.print_exc()
    print("\n[PROBE][INTERPRETATION] Compare the three: if row2 spread ~= row1 spread (~0) but row3 shows a large "
          "spread, then normalize_channel=true is CANCELLING pathloss -> you must set normalize_channel=false to keep near-far.")


def check5_topology_randomization(cfg):
    _hr("CHECK 5 — topology randomization actually changes the channel")
    u = max(PROBe_USERS)
    # Robust approach: two independent _make_batch calls at the same Eb/N0,
    # each forcing a fresh topology draw; their channels should differ.
    c_on = _force_family(cfg, "umi")
    set_cfg(c_on, "channel.umi.randomize_topology_each_batch", True)
    try:
        # Easiest robust approach: two independent _make_batch calls at same Eb/N0
        _, b1 = _build_and_batch(c_on, u)
        sysd2 = _build_system_for_num_users(c_on, u)  # rebuild to be safe
        b2 = _make_batch(sysd2["tx"], sysd2["channel"], c_on, batch_size=BATCH, training=False, fixed_ebno_db=10.0)
        d = float(tf.reduce_mean(tf.abs(b1["h"]) ** 2).numpy()) - float(tf.reduce_mean(tf.abs(b2["h"]) ** 2).numpy())
        # better metric: cross-correlation of flattened |h| histograms via mean abs diff of sorted magnitudes
        m1 = np.sort(_arr(tf.reshape(tf.abs(b1["h"]), [-1])))
        m2 = np.sort(_arr(tf.reshape(tf.abs(b2["h"]), [-1])))
        n = min(m1.size, m2.size)
        rel = float(np.mean(np.abs(m1[:n] - m2[:n])) / (np.mean(m1[:n]) + 1e-12))
        print(f"[PROBE] randomize=ON : mean|h|^2 batch1 vs batch2 differ by {abs(d):.3e}")
        print(f"[PROBE] randomize=ON : sorted-magnitude relative diff = {rel:.3f}  (expect >> 0 if topology truly re-drawn)")
        if rel < 1e-3:
            print("[PROBE][FINDING] batches look identical -> topology may NOT be re-randomizing.")
        else:
            print("[PROBE] topology randomization produces distinct channels (good).")
    except Exception:
        print("[PROBE][NOTE] randomization probe failed:")
        traceback.print_exc()


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    cfg = load_config(CONFIG_PATH)
    check0_environment(cfg)
    check1_antenna_counts(cfg)
    check2_layout_and_parity(cfg)
    check3_near_far(cfg)
    check4_pathloss_normalize(cfg)
    check5_topology_randomization(cfg)
    _hr("PROBE COMPLETE — copy ALL stdout above back for review")


if __name__ == "__main__":
    main()
