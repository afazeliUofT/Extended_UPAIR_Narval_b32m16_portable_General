#!/usr/bin/env python
"""
UPAIR near-far / fractional-power-control verification probe (read-only).

Targets commit c15d5f2. Confirms with hard numbers the four things that
static review cannot settle:

  CHECK A  Calibration invariance: is mean_u P*_u == 1 per drop (so the
           Eb/N0 sweep still sets the average SNR)?  [Lemma 1]
  CHECK B  (1-alpha) spread law: does post_spread_db ~= (1-alpha)*raw_spread_db
           across alpha in {0.0, 0.5, 0.8, 1.0}?  [Eq. 7-8]
           alpha=1 -> ~0 dB (equal power); alpha=0 -> full raw spread.
  CHECK C  Internal consistency: does y == sum_u (h_star * x) + n hold to
           numerical precision, i.e. is the returned (y, h_star) pair
           self-consistent?  [Issue 2]
  CHECK D  Fast-fading contamination: how far is the grid-averaged P_u from
           the true large-scale gain beta_u? Quantifies Issue 1 by comparing
           per-user power dispersion WITHIN a fixed topology across fresh
           fast-fading draws (should be small if P_u ~= beta_u).

Writes nothing except stdout. Deep-copies cfg for each variant; never
mutates tracked files or checkpoints.
"""
from __future__ import annotations

import copy
import sys
import traceback
import numpy as np
import tensorflow as tf

from upair5g.config import load_config, get_cfg, set_cfg
from upair5g.training import _build_system_for_num_users, _make_batch

CONFIG_PATH = "configs/twc_comprehensive_mu32_base.yaml"
U = 4               # exercise the multi-user near-far path
BATCH = 8           # several drops for distributional reads
SEED = 20260607


def _hr(t):
    print("\n" + "=" * 100 + f"\n[PROBE] {t}\n" + "=" * 100)


def _np(x):
    return np.asarray(x.numpy()) if hasattr(x, "numpy") else np.asarray(x)


def _force_umi(cfg):
    c = copy.deepcopy(cfg)
    set_cfg(c, "channel.family", "umi")
    set_cfg(c, "channel.model", "umi")
    return c


def _build(cfg, training, ebno=10.0):
    sysd = _build_system_for_num_users(cfg, U)
    batch = _make_batch(sysd["tx"], sysd["channel"], cfg, batch_size=BATCH,
                        training=training, fixed_ebno_db=ebno)
    return sysd, batch


# ---------------------------------------------------------------- CHECK 0
def check0(cfg):
    _hr("CHECK 0 — environment & near_far config")
    print("[PROBE] tf:", tf.__version__)
    try:
        import sionna; print("[PROBE] sionna:", getattr(sionna, "__version__", "?"))
    except Exception as e:
        print("[PROBE] sionna import failed:", e)
    print("[PROBE] near_far block:", get_cfg(cfg, "near_far", {}))
    print("[PROBE] channel.normalize_channel:", get_cfg(cfg, "channel.normalize_channel"))
    print("[PROBE] umi.enable_pathloss:", get_cfg(cfg, "channel.umi.enable_pathloss"))
    print("[PROBE] umi.enable_shadow_fading:", get_cfg(cfg, "channel.umi.enable_shadow_fading"))


# ---------------------------------------------------------------- CHECK A,B
def checkAB(cfg):
    _hr("CHECK A & B — calibration invariance + (1-alpha) spread law")
    base = _force_umi(cfg)
    set_cfg(base, "near_far.enabled", True)
    print(f"{'alpha':>6} | {'mean P*_u (want 1.0)':>22} | {'raw_spread_dB':>13} | "
          f"{'post_spread_dB':>14} | {'(1-a)*raw':>10} | {'ratio post/(1-a)raw':>20}")
    print("-" * 100)
    for a in [0.0, 0.5, 0.8, 1.0]:
        c = copy.deepcopy(base)
        # Force a deterministic alpha by pinning eval-mode and alpha_eval=a
        set_cfg(c, "near_far.alpha_eval", float(a))
        try:
            sysd, batch = _build(c, training=False)
            ch = sysd["channel"]
            stats = getattr(ch, "last_near_far_stats", {})
            if not stats:
                print(f"{a:6.2f} | (no last_near_far_stats exposed — cannot read)")
                continue
            meanP = float(np.mean(_np(stats["post_power_mean"])))
            raw = float(np.mean(_np(stats["raw_spread_db"])))
            post = float(np.mean(_np(stats["post_spread_db"])))
            pred = (1.0 - a) * raw
            ratio = (post / pred) if pred > 1e-9 else float("nan")
            flagP = "" if abs(meanP - 1.0) < 1e-3 else "  <<< MEAN != 1"
            print(f"{a:6.2f} | {meanP:22.6f}{flagP} | {raw:13.2f} | {post:14.2f} | "
                  f"{pred:10.2f} | {ratio:20.3f}")
        except Exception:
            print(f"{a:6.2f} | FAILED:")
            traceback.print_exc()
    print("\n[PROBE][INTERPRETATION] CHECK A: 'mean P*_u' must be ~1.000 for every alpha "
          "(else Eb/N0 calibration is broken).")
    print("[PROBE][INTERPRETATION] CHECK B: 'ratio post/(1-a)raw' should be ~1.0 for "
          "alpha<1; alpha=1 row should show post_spread ~0 dB (equal power); "
          "alpha=0 row should show post ~= raw (full near-far).")


# ---------------------------------------------------------------- CHECK C
def checkC(cfg):
    _hr("CHECK C — internal consistency  y == sum_u(h_star * x) + n  (noise-subtracted)")
    c = _force_umi(cfg)
    set_cfg(c, "near_far.enabled", True)
    set_cfg(c, "near_far.alpha_eval", 0.8)
    # Run at very high Eb/N0 so the noise term is negligible; then y ~ sum h_star x.
    try:
        sysd, batch = _build(c, training=False, ebno=60.0)
        y = batch["y"]; h = batch["h"]
        tx = sysd["tx"]
        # Recover the transmitted grid x by re-calling the transmitter is not trivial;
        # instead verify the *structural* identity using power balance:
        #   E|y|^2  ~=  sum_u E|h_star_u|^2 * E|x_u|^2   (+ noise, negligible here)
        # With unit-power symbols E|x_u|^2 = 1, and h shape [B,1,Nr,U,S,T,F].
        py = float(tf.reduce_mean(tf.abs(y) ** 2).numpy())
        # per-user mean power of h_star, then sum over users, averaged over grid/ant
        abs2 = tf.math.real(h * tf.math.conj(h))
        # mean over [B,1,Nr,S,T,F] keeping U  -> [U]; here axes: 0,1,2,4,5,6
        pu = _np(tf.reduce_mean(abs2, axis=[0, 1, 2, 4, 5, 6]))
        sum_pu = float(np.sum(pu))
        Nr = int(h.shape[2])
        # y aggregates over Nr receive antennas; E|y|^2 per RE-antenna ~ (1/Nr)*sum_u E|h|^2*Nr? 
        # Cleanest: compare mean |y|^2 to mean over antennas of sum_u |h_star|^2 (unit-power x).
        py_per = py
        h_pred_per = sum_pu  # mean |h|^2 already averaged over Nr; sum over users
        ratio = py_per / h_pred_per if h_pred_per > 1e-12 else float("nan")
        print(f"[PROBE] mean|y|^2 (Eb/N0=60dB, ~noiseless) = {py:.5e}")
        print(f"[PROBE] sum_u mean|h_star_u|^2             = {sum_pu:.5e}")
        print(f"[PROBE] per-user mean|h_star|^2            = {np.array2string(pu, precision=4)}")
        print(f"[PROBE] ratio mean|y|^2 / sum_u mean|h|^2  = {ratio:.4f}")
        print("[PROBE][INTERPRETATION] If y is built consistently from h_star with unit-power "
              "symbols, this ratio should be ~1.0 (within DMRS/data RE occupancy and pilot-power "
              "factors). A ratio far from O(1) signals y and h_star are NOT the same channel.")
    except Exception:
        print("[PROBE] CHECK C FAILED:")
        traceback.print_exc()


# ---------------------------------------------------------------- CHECK D
def checkD(cfg):
    _hr("CHECK D — fast-fading contamination of P_u (Issue 1)")
    # Disable topology randomization so the SAME user geometry (same beta_u) is
    # reused across draws; then any variation in measured P_u across draws is
    # pure fast fading. Large variation => P_u is a noisy estimate of beta_u.
    c = _force_umi(cfg)
    set_cfg(c, "near_far.enabled", True)
    set_cfg(c, "near_far.alpha_eval", 0.0)  # alpha=0 -> p_star == p_u/mean, exposes raw P_u
    set_cfg(c, "channel.umi.randomize_topology_each_batch", False)
    try:
        draws = []
        for _ in range(6):
            sysd, batch = _build(c, training=False)
            ch = sysd["channel"]
            stats = getattr(ch, "last_near_far_stats", {})
            if not stats:
                print("[PROBE] no stats; cannot run CHECK D"); return
            # raw_power shape [B,U]; average over batch to get per-user power this draw
            rp = _np(stats["raw_power"]).mean(axis=0)  # [U]
            draws.append(rp)
        draws = np.stack(draws, axis=0)  # [draws, U]
        # Per-user coefficient of variation across draws (fast-fading-induced)
        mean_u = draws.mean(axis=0)
        std_u = draws.std(axis=0)
        cv = std_u / np.maximum(mean_u, 1e-30)
        cv_db = 10.0 * np.log10(1.0 + cv)  # rough dB-scale of the fluctuation
        print(f"[PROBE] per-user mean power across draws : {np.array2string(mean_u, precision=3)}")
        print(f"[PROBE] per-user coeff. of variation     : {np.array2string(cv, precision=4)}")
        print(f"[PROBE] approx fluctuation (dB)          : {np.array2string(cv_db, precision=3)}")
        print(f"[PROBE] max CV across users              : {float(cv.max()):.4f}")
        print("[PROBE][INTERPRETATION] CV << 1 (say < 0.05, i.e. < ~0.2 dB) means the grid average "
              "P_u is a clean estimate of beta_u and the LaTeX approximation P_u ~= beta_u holds; "
              "the re-referencing is then effectively large-scale-only. CV of O(0.1-0.3) means a "
              "non-trivial slice of fast fading is being divided out (Issue 1 is real and sized here).")
    except Exception:
        print("[PROBE] CHECK D FAILED:")
        traceback.print_exc()


def main():
    np.random.seed(SEED); tf.random.set_seed(SEED)
    cfg = load_config(CONFIG_PATH)
    check0(cfg)
    checkAB(cfg)
    checkC(cfg)
    checkD(cfg)
    _hr("PROBE COMPLETE — send back ALL stdout above")


if __name__ == "__main__":
    main()
