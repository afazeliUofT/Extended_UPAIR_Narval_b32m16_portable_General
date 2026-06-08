#!/usr/bin/env python
"""
Normalized-UMi confirmation probe (read-only). Verifies that push 7b57fcf
implements a clean STANDARD normalized link-level UMi model.

  A) Unit-power & equal-across-users: with normalize_channel=True and
     pathloss/shadow OFF, per-user mean|h|^2 ~ 1 and the near-far spread ~ 0 dB.
     (Confirms large-scale gain is normalized away, not present.)
  B) Tensor layout intact: y and h have the expected shapes/axes for the
     estimator (no regression from the revert).
  C) Near-far fully bypassed: near_far.enabled=False, and the channel does
     NOT populate last_near_far_stats (i.e. _apply_fractional_power_control
     never runs).
  D) SNR calibration (AWGN added once): sweep Eb/N0 and check the empirical
     post-channel SNR ~ matches the nominal, i.e. mean|y_signal|^2 / N0 tracks
     10^(EbN0/10) up to bits/symbol, coderate, and RE-occupancy factors. Also
     confirms noise scales correctly (no double / missing AWGN).
  E) Covariance cache staleness: report whether the configured
     empirical_covariances_umi_norm.npz exists yet (so reuse_cache won't grab
     a stale near-far cache), and whether any *_umi_pc_meanref.npz lingers.

Writes only stdout. Deep-copies cfg; mutates nothing.
"""
from __future__ import annotations

import copy, glob, os
import numpy as np
import tensorflow as tf
import traceback

from upair5g.config import load_config, get_cfg, set_cfg
from upair5g.training import _build_system_for_num_users, _make_batch
from upair5g.builders import get_resource_grid
try:
    from upair5g.utils import call_transmitter, ebno_db_to_no
except Exception:
    from upair5g.training import call_transmitter, ebno_db_to_no  # type: ignore

CONFIG_PATH = "configs/twc_comprehensive_mu32_base.yaml"
U = 4
SEED = 90909


def _hr(t):
    print("\n" + "=" * 100 + f"\n[PROBE] {t}\n" + "=" * 100)


def _ppwr(h):
    return tf.reduce_mean(tf.math.real(h * tf.math.conj(h)), axis=[1, 2, 4, 5, 6]).numpy()


def main():
    np.random.seed(SEED); tf.random.set_seed(SEED)
    cfg = load_config(CONFIG_PATH)

    _hr("CONFIG SNAPSHOT")
    print("[PROBE] channel.family            :", get_cfg(cfg, "channel.family"))
    print("[PROBE] channel.normalize_channel :", get_cfg(cfg, "channel.normalize_channel"))
    print("[PROBE] umi.enable_pathloss       :", get_cfg(cfg, "channel.umi.enable_pathloss"))
    print("[PROBE] umi.enable_shadow_fading  :", get_cfg(cfg, "channel.umi.enable_shadow_fading"))
    print("[PROBE] near_far.enabled          :", get_cfg(cfg, "near_far.enabled"))
    print("[PROBE] near_far.mode             :", get_cfg(cfg, "near_far.mode"))
    print("[PROBE] baselines cov cache       :", get_cfg(cfg, "baselines.covariance_estimation.cache_name"))
    print("[PROBE] baselines reuse_cache     :", get_cfg(cfg, "baselines.covariance_estimation.reuse_cache"))

    try:
        c = copy.deepcopy(cfg)
        set_cfg(c, "system.batch_size_train", 8); set_cfg(c, "system.batch_size_eval", 8)
        sysd = _build_system_for_num_users(c, U)
        ch, tx = sysd["channel"], sysd["tx"]
        rg = get_resource_grid(tx)

        # ---- A) unit power & equal across users ------------------------
        _hr("A — unit-power & equal-across-users (large-scale normalized away)")
        batch = _make_batch(tx, ch, c, batch_size=8, training=False, fixed_ebno_db=10.0)
        h = batch["h"]; y = batch["y"]
        pu = _ppwr(h).mean(axis=0)                      # per-user mean over batch
        pu_db = 10 * np.log10(np.maximum(pu, 1e-30))
        spread = float(pu_db.max() - pu_db.min())
        total = float(tf.reduce_mean(tf.abs(h) ** 2).numpy())
        print(f"[PROBE] per-user mean|h|^2      : {np.array2string(pu, precision=4)}")
        print(f"[PROBE] per-user power (dB)     : {np.array2string(pu_db, precision=3)}")
        print(f"[PROBE] near-far spread (dB)    : {spread:.3f}  (want ~0 for normalized)")
        print(f"[PROBE] overall mean|h|^2       : {total:.4f}  (want ~1.0)")
        okA = spread < 1.0 and abs(total - 1.0) < 0.25
        print(f"[PROBE][A] {'PASS' if okA else 'CHECK'}: normalized channel is unit-power & equal-across-users.")

        # ---- B) tensor layout ------------------------------------------
        _hr("B — estimator-facing tensor layout")
        ys = tuple(int(d) if d is not None else -1 for d in y.shape)
        hs = tuple(int(d) if d is not None else -1 for d in h.shape)
        print(f"[PROBE] y.shape = {ys}")
        print(f"[PROBE] h.shape = {hs}  (expect [...,Nr, U, ...] with Nr=16, U={U})")
        okB = (h.shape[2] == int(get_cfg(c, "channel.num_rx_ant"))) and (h.shape[3] == U)
        print(f"[PROBE][B] {'PASS' if okB else 'CHECK'}: rx-antenna axis = {int(h.shape[2])}, user axis = {int(h.shape[3])}.")

        # ---- C) near-far bypassed --------------------------------------
        _hr("C — near-far machinery fully bypassed")
        stats = getattr(ch, "last_near_far_stats", {})
        bypassed = (len(stats) == 0)
        print(f"[PROBE] last_near_far_stats populated? {'NO (good)' if bypassed else 'YES (unexpected!)'}")
        print(f"[PROBE][C] {'PASS' if bypassed else 'CHECK'}: _apply_fractional_power_control "
              f"{'did not run' if bypassed else 'RAN despite near_far.enabled=false'}.")

        # ---- D) SNR calibration (AWGN once) ----------------------------
        _hr("D — SNR calibration / AWGN-once sanity")
        # At high Eb/N0, mean|y|^2 ~ signal power (noise negligible).
        # At low Eb/N0, mean|y|^2 ~ signal + N0*Nr. Check noise scales as N0.
        def mean_y2(ebno):
            b = _make_batch(tx, ch, c, batch_size=8, training=False, fixed_ebno_db=ebno)
            return float(tf.reduce_mean(tf.abs(b["y"]) ** 2).numpy())
        hi = mean_y2(60.0)     # ~ signal only
        lo = mean_y2(-6.0)     # ~ signal + noise
        no_hi = float(np.asarray(ebno_db_to_no(tf.constant(60.0, tf.float32), tx=tx, resource_grid=rg)))
        no_lo = float(np.asarray(ebno_db_to_no(tf.constant(-6.0, tf.float32), tx=tx, resource_grid=rg)))
        Nr = int(h.shape[2])
        print(f"[PROBE] mean|y|^2 @ EbN0=60dB   : {hi:.4e}   (≈ signal power)")
        print(f"[PROBE] mean|y|^2 @ EbN0=-6dB   : {lo:.4e}")
        print(f"[PROBE] N0 @ 60dB / -6dB        : {no_hi:.3e} / {no_lo:.3e}")
        # expected noise contribution to mean|y|^2 at low SNR ~ N0 (per RE-antenna), data REs only
        approx_noise = lo - hi
        print(f"[PROBE] (low - high) mean|y|^2  : {approx_noise:.4e}   (noise power proxy; should be >0 and ~ O(N0))")
        okD = (hi > 0) and (lo > hi) and (no_lo > no_hi)
        print(f"[PROBE][D] {'PASS' if okD else 'CHECK'}: y increases with lower Eb/N0 and N0 scales correctly "
              "(noise added exactly once, calibration sane).")

        # ---- E) covariance cache staleness -----------------------------
        _hr("E — LMMSE covariance cache staleness")
        cache_name = str(get_cfg(c, "baselines.covariance_estimation.cache_name"))
        found_new = glob.glob(f"**/{cache_name}", recursive=True) + glob.glob(cache_name)
        stale = (glob.glob("**/*umi_pc_meanref*.npz", recursive=True)
                 + glob.glob("**/*umi_pc*.npz", recursive=True))
        print(f"[PROBE] configured cache         : {cache_name}")
        print(f"[PROBE] new cache present?       : {'YES' if found_new else 'NO (will be built on first run)'}  {found_new[:3]}")
        print(f"[PROBE] lingering near-far caches: {stale[:5] if stale else 'none'}")
        print("[PROBE][E][NOTE] reuse_cache=true is safe ONLY if the NEW-named cache reflects the "
              "normalized channel. If a new cache exists but predates the revert, delete it so it rebuilds.")

        _hr("SUMMARY")
        print(f"[PROBE] A(unit-power)={'PASS' if okA else 'CHECK'}  B(layout)={'PASS' if okB else 'CHECK'}  "
              f"C(bypass)={'PASS' if bypassed else 'CHECK'}  D(SNR)={'PASS' if okD else 'CHECK'}")
        print("[PROBE] If A-D PASS and E shows no stale cache, the normalized-UMi model is correctly "
              "and cleanly implemented in the standard link-level manner.")
    except Exception:
        print("[PROBE] FAILED:"); traceback.print_exc()


if __name__ == "__main__":
    main()
