#!/usr/bin/env python
"""
UMiPC post-ca5ca0f verification (read-only).

The repo's power_integrity probe checks realized==target (which, after the
eps->1e-20 fix, holds essentially by construction). This probe answers the
TWO questions that one does NOT:

  CHECK 1  Did the eps fix actually remove the weak-user corruption AT THE
           NEW eps, and is aggregate calibration intact?
           - realized/intended ratio across alpha (want ~1, min not << 1)
           - mean_u P*_u == 1 per drop (Lemma)
           - count of pre-clamp dips below the NEW eps (want 0)

  CHECK 2  OVER-EQUALIZATION (Issue 2b): does the delivered h_star still
           carry physical fast-fading power fluctuation, or has dividing by
           instantaneous P_u flattened it?
           Method: freeze topology; draw fast fading N times; for a FIXED
           alpha, measure the std (dB) of the *delivered* per-user
           |h_star|^2 across draws.
             ~0 dB  -> fast fading power REMOVED (over-equalized; the
                       receiver never sees users fade in/out in power).
                       Real fractional PC compensates slow gain only, so
                       physical fluctuation should remain -> this is the
                       Issue 2b modeling gap, CONFIRMED.
             >~1 dB -> delivered channel retains fluctuation (gap absent).
           For contrast we also report the std of the RAW channel power
           (what a physically-correct slow-gain PC would preserve).

Writes only stdout. Deep-copies cfg; never mutates tracked files.
"""
from __future__ import annotations

import copy
import numpy as np
import tensorflow as tf
import traceback

from upair5g.config import load_config, get_cfg, set_cfg
from upair5g.training import _build_system_for_num_users
from upair5g.builders import get_resource_grid
try:
    from upair5g.utils import call_transmitter, ebno_db_to_no
except Exception:
    from upair5g.training import call_transmitter, ebno_db_to_no  # type: ignore

CONFIG_PATH = "configs/twc_comprehensive_mu32_base.yaml"
U = 4


def _hr(t):
    print("\n" + "=" * 100 + f"\n[PROBE] {t}\n" + "=" * 100)


def _ppwr(h):
    return tf.reduce_mean(tf.math.real(h * tf.math.conj(h)), axis=[1, 2, 4, 5, 6]).numpy()


def _umi(cfg):
    c = copy.deepcopy(cfg)
    set_cfg(c, "channel.family", "umi"); set_cfg(c, "channel.model", "umi")
    set_cfg(c, "near_far.enabled", True)
    return c


# --------------------------------------------------------------- CHECK 1
def check1(cfg):
    _hr("CHECK 1 — eps fix at runtime + aggregate calibration")
    eps = float(get_cfg(cfg, "near_far.epsilon", 1e-12))
    print(f"[PROBE] near_far.epsilon = {eps:.1e}  (expect 1e-20 after fix)")
    c = _umi(cfg)
    set_cfg(c, "system.batch_size_train", 8); set_cfg(c, "system.batch_size_eval", 8)
    try:
        sysd = _build_system_for_num_users(c, U)
        ch, tx = sysd["channel"], sysd["tx"]
        rg = get_resource_grid(tx)
        for a in [1.0, 0.8, 0.0]:
            set_cfg(c, "near_far.alpha_eval", float(a)); ch.cfg = c
            ch.set_training_mode(False)
            ratios, meanU, dips = [], [], 0
            for _ in range(15):
                x, _ = call_transmitter(tx, 8)
                no = ebno_db_to_no(tf.constant(0.0, tf.float32), tx=tx, resource_grid=rg)
                ch._set_topology(8)
                _, h_raw = ch._call_clean_ofdm(x)
                p_true = _ppwr(h_raw).astype(np.float64)
                _, h_star = ch._apply_fractional_power_control(h_raw, x, no)
                st = ch.last_near_far_stats
                p_tgt = np.asarray(st["post_power"].numpy(), np.float64)
                p_real = _ppwr(h_star).astype(np.float64)
                ratios.append((p_real / np.maximum(p_tgt, 1e-300)).reshape(-1))
                meanU.append(p_tgt.mean(axis=1).reshape(-1))
                dips += int((p_true < eps).sum())
            r = np.concatenate(ratios); mU = np.concatenate(meanU)
            print(f"  alpha={a:4.2f} | realized/target min/mean = {r.min():.4f}/{r.mean():.4f} "
                  f"| mean_u P* (want 1) = {mU.mean():.5f} | pre-clamp dips<eps = {dips}")
        print("\n[PROBE][INTERPRETATION] PASS if, for every alpha: realized/target min ~>=0.99, "
              "mean ~1.000, mean_u P* ~1.000, and dips==0. That confirms the eps clamp no longer "
              "corrupts weak users and aggregate SNR calibration holds.")
    except Exception:
        print("[PROBE] CHECK 1 FAILED:"); traceback.print_exc()


# --------------------------------------------------------------- CHECK 2
def check2(cfg):
    _hr("CHECK 2 — over-equalization of fast fading in delivered h_star (Issue 2b)")
    c = _umi(cfg)
    set_cfg(c, "system.batch_size_train", 1); set_cfg(c, "system.batch_size_eval", 1)
    set_cfg(c, "channel.umi.randomize_topology_each_batch", False)  # freeze geometry
    alpha = 0.8
    set_cfg(c, "near_far.alpha_eval", alpha)
    draws = 24
    try:
        sysd = _build_system_for_num_users(c, U)
        ch, tx = sysd["channel"], sysd["tx"]
        rg = get_resource_grid(tx)
        ch.set_training_mode(False)
        x, _ = call_transmitter(tx, 1)
        ch._set_topology(1)            # fix ONE geometry for all draws
        raw_p, deliv_p = [], []
        for _ in range(draws):
            _, h_raw = ch._call_clean_ofdm(x)               # fresh fast fading, same geometry
            no = ebno_db_to_no(tf.constant(0.0, tf.float32), tx=tx, resource_grid=rg)
            _, h_star = ch._apply_fractional_power_control(h_raw, x, no)
            raw_p.append(_ppwr(h_raw)[0])                   # [U]
            deliv_p.append(_ppwr(h_star)[0])                # [U]
        raw_p = np.asarray(raw_p, np.float64); deliv_p = np.asarray(deliv_p, np.float64)
        raw_db = 10 * np.log10(np.maximum(raw_p, 1e-300))
        dlv_db = 10 * np.log10(np.maximum(deliv_p, 1e-300))
        raw_std = raw_db.std(axis=0)      # physical fast-fading fluctuation (what slow-gain PC keeps)
        dlv_std = dlv_db.std(axis=0)      # fluctuation in DELIVERED channel
        print(f"[PROBE] frozen geometry, {draws} fast-fading draws, alpha={alpha}")
        print(f"[PROBE] RAW per-user power std (dB)       : {np.array2string(raw_std, precision=3)}")
        print(f"[PROBE] DELIVERED h_star power std (dB)   : {np.array2string(dlv_std, precision=3)}")
        print(f"[PROBE] max RAW std={raw_std.max():.3f} dB   max DELIVERED std={dlv_std.max():.3f} dB")
        if dlv_std.max() < 0.2 and raw_std.max() > 1.0:
            print("[PROBE][VERDICT] OVER-EQUALIZATION CONFIRMED: the raw channel fluctuates "
                  f"({raw_std.max():.2f} dB) but the delivered h_star is flat ({dlv_std.max():.2f} dB). "
                  "Dividing by instantaneous P_u removed the physical fast-fading power variation. "
                  "A slow-gain-based PC would preserve it. This is Issue 2b.")
        elif dlv_std.max() >= 1.0:
            print("[PROBE][VERDICT] Delivered channel retains fast-fading power variation; "
                  "Issue 2b is NOT present in the delivered channel.")
        else:
            print("[PROBE][VERDICT] Partial: inspect the per-user numbers above.")
        print("[PROBE][NOTE] Issue 2a (P_u != beta_u, the ~1-2.6 dB contamination of the PC TARGET) "
              "is measured by upair_probe_umi_pc_fast_fading_contamination.sh; this CHECK 2 is about "
              "whether the DELIVERED channel is over-flattened.")
    except Exception:
        print("[PROBE] CHECK 2 FAILED:"); traceback.print_exc()


def main():
    np.random.seed(13); tf.random.set_seed(13)
    cfg = load_config(CONFIG_PATH)
    check1(cfg)
    check2(cfg)
    _hr("PROBE COMPLETE — send back ALL stdout")


if __name__ == "__main__":
    main()
