#!/usr/bin/env python
"""
UPAIR near-far follow-up probe (read-only). Settles the two open points
from the previous run:

  CHECK D2  Fast-fading contamination of P_u (Issue 1), DONE CORRECTLY.
            The previous CHECK D was confounded: it rebuilt the system each
            draw, so the topology (user locations) re-randomized and the CV
            it reported was geometry spread, not fast fading.
            Here we build ONE system, fix the topology
            (randomize_topology_each_batch=False, batch_size=1 => a single
            geometry), and call the SAME channel object many times so only
            fast fading varies. CV across calls is then pure fast-fading
            contamination of the grid-averaged P_u.
              CV < ~0.05 (<0.2 dB) -> P_u ~= beta_u, Issue 1 negligible.
              CV ~ 0.1-0.3 (~0.5-1.3 dB) -> Issue 1 real, sized here.

  CHECK E   eps-floor / realized-power diagnostic. The previous CHECK C
            showed the returned h_star averaged 0.926 (-0.33 dB) instead of
            1.0, while the internal stat reported mean p_star = 1.000000.
            This compares, per (batch,user), the EMPIRICAL mean|h_star|^2 of
            the RETURNED channel against the INTERNAL post_power (p_star),
            and counts how many user-instances sit near the epsilon floor.
            If the deficit lives on the users whose raw P_u approaches eps,
            the cause is the eps=1e-12 clamp; otherwise it is elsewhere.

Writes only stdout. Reuses one system object where required; never mutates
tracked files or checkpoints.
"""
from __future__ import annotations

import copy
import numpy as np
import tensorflow as tf
import traceback

from upair5g.config import load_config, get_cfg, set_cfg
from upair5g.training import _build_system_for_num_users, _make_batch

CONFIG_PATH = "configs/twc_comprehensive_mu32_base.yaml"
U = 4
SEED = 770077


def _hr(t):
    print("\n" + "=" * 100 + f"\n[PROBE] {t}\n" + "=" * 100)


def _np(x):
    return np.asarray(x.numpy()) if hasattr(x, "numpy") else np.asarray(x)


def _umi(cfg):
    c = copy.deepcopy(cfg)
    set_cfg(c, "channel.family", "umi")
    set_cfg(c, "channel.model", "umi")
    set_cfg(c, "near_far.enabled", True)
    return c


# ------------------------------------------------------------------ D2
def checkD2(cfg):
    _hr("CHECK D2 — fast-fading contamination of P_u  (FIXED geometry)")
    c = _umi(cfg)
    set_cfg(c, "near_far.alpha_eval", 0.0)                 # raw powers exposed
    set_cfg(c, "channel.umi.randomize_topology_each_batch", False)  # freeze geometry
    try:
        # Build ONE system; batch_size=1 => a single fixed drop/geometry.
        sysd = _build_system_for_num_users(c, U)
        tx, ch = sysd["tx"], sysd["channel"]
        raws = []
        for i in range(12):
            _ = _make_batch(tx, ch, c, batch_size=1, training=False, fixed_ebno_db=10.0)
            st = getattr(ch, "last_near_far_stats", {})
            if not st:
                print("[PROBE] no last_near_far_stats; abort D2"); return
            raws.append(_np(st["raw_power"])[0])           # [U] for the single drop
        R = np.stack(raws, axis=0)                          # [calls, U]
        mean_u = R.mean(axis=0)
        cv = R.std(axis=0) / np.maximum(mean_u, 1e-30)
        cv_db = 10.0 * np.log10(1.0 + cv)
        # sanity: geometry should be ~fixed => mean ratios between users stable
        print(f"[PROBE] fixed-geometry per-user mean raw power: {np.array2string(mean_u, precision=3)}")
        print(f"[PROBE] per-user CV across fast-fading draws  : {np.array2string(cv, precision=4)}")
        print(f"[PROBE] approx fluctuation (dB)               : {np.array2string(cv_db, precision=3)}")
        print(f"[PROBE] max CV                                : {float(cv.max()):.4f}")
        m = float(cv.max())
        verdict = ("NEGLIGIBLE (P_u ~= beta_u)" if m < 0.05 else
                   "MODEST (sub-dB to ~1 dB)" if m < 0.35 else
                   "LARGE (fast fading dominates P_u)")
        print(f"[PROBE][VERDICT] Issue 1 fast-fading contamination: {verdict}")
        print("[PROBE][NOTE] If user-to-user mean ratios here are NOT stable, the geometry "
              "did not actually freeze (report this).")
    except Exception:
        print("[PROBE] CHECK D2 FAILED:"); traceback.print_exc()


# ------------------------------------------------------------------ E
def checkE(cfg):
    _hr("CHECK E — eps-floor / realized-power deficit diagnostic")
    c = _umi(cfg)
    set_cfg(c, "near_far.alpha_eval", 0.8)
    eps = float(get_cfg(c, "near_far.epsilon", 1e-12))
    try:
        sysd = _build_system_for_num_users(c, U)
        tx, ch = sysd["tx"], sysd["channel"]
        batch = _make_batch(tx, ch, c, batch_size=16, training=False, fixed_ebno_db=60.0)
        h = batch["h"]
        st = getattr(ch, "last_near_far_stats", {})
        p_star = _np(st["post_power"])         # [B,U] intended
        raw = _np(st["raw_power"])             # [B,U] grid-averaged raw P_u
        # empirical realized power of returned h_star, per (B,U)
        abs2 = tf.math.real(h * tf.math.conj(h))
        emp = _np(tf.reduce_mean(abs2, axis=[1, 2, 4, 5, 6]))   # [B,U]
        ratio = emp / np.maximum(p_star, 1e-30)                  # should be ~1
        print(f"[PROBE] internal mean_u p_star (per batch, want 1): "
              f"{np.array2string(p_star.mean(axis=1)[:6], precision=5)} ...")
        print(f"[PROBE] realized  mean_u |h*|^2 (per batch)       : "
              f"{np.array2string(emp.mean(axis=1)[:6], precision=5)} ...")
        print(f"[PROBE] overall mean realized/intended ratio      : {float(ratio.mean()):.5f}")
        print(f"[PROBE] min/median/max ratio                      : "
              f"{float(ratio.min()):.4f} / {float(np.median(ratio)):.4f} / {float(ratio.max()):.4f}")
        near_floor = int(np.sum(raw < 10 * eps))
        very_near = int(np.sum(raw < eps))
        print(f"[PROBE] raw P_u below 10*eps (=1e-11) : {near_floor} of {raw.size} user-instances")
        print(f"[PROBE] raw P_u below eps   (=1e-12)  : {very_near} of {raw.size} user-instances")
        # correlation: is the deficit concentrated on low-raw-power users?
        lo = ratio[raw < np.median(raw)].mean()
        hi = ratio[raw >= np.median(raw)].mean()
        print(f"[PROBE] mean ratio for LOW-raw-power half : {float(lo):.4f}")
        print(f"[PROBE] mean ratio for HIGH-raw-power half: {float(hi):.4f}")
        print("[PROBE][INTERPRETATION] If overall ratio < 1 AND it is the LOW-raw-power half "
              "that is depressed (and some instances sit near eps), the deficit is the eps=1e-12 "
              "clamp biasing deeply-faded users; fix by lowering eps (e.g. 1e-20). If ratio ~1 "
              "everywhere, the earlier 0.926 was a measurement artifact and there is no bug.")
    except Exception:
        print("[PROBE] CHECK E FAILED:"); traceback.print_exc()


def main():
    np.random.seed(SEED); tf.random.set_seed(SEED)
    cfg = load_config(CONFIG_PATH)
    checkD2(cfg)
    checkE(cfg)
    _hr("FOLLOW-UP PROBE COMPLETE — send back ALL stdout")


if __name__ == "__main__":
    main()
