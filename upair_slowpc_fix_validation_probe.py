#!/usr/bin/env python
"""
Validate the corrected near-far fix design (read-only prototype).

CONTEXT. The slow_gain_formula probe ruled out deriving beta_u from Sionna's
basic_pathloss+sf (7.7 dB residual: antenna directivity + O2I are not in
those tensors). The right slow gain is therefore the REALIZED power averaged
over fast fading at fixed geometry -- which captures PL+shadow+antenna+O2I
(everything real RSRP sees) while removing fast fading.

This probe prototypes that fix WITHOUT modifying the channel, and checks:

  A) beta_hat from N-draw averaging is STABLE (fast fading removed):
     std of independent N-averages should be << the single-slot 3-6 dB.
  B) SLOW-PC delivered channel (scale by sqrt(pstar_slow/beta_hat), a slow
     per-user constant) RETAINS full fast-fading fluctuation, unlike the
     current per-slot normalization which compresses it to (1-alpha).
  C) Calibration holds IN EXPECTATION: mean over users of the slot-mean
     delivered power ~ 1 (so the Eb/N0 sweep still sets the average SNR),
     while per-slot power fluctuates physically.
  D) Cost: N extra channel draws per topology to estimate beta_hat.

If A/B/C pass, the fix is sound and ready to implement in
_apply_fractional_power_control (estimate beta via batch-replicated topology
in one vectorized call, scale by slow gain, re-reference by slow mean).

Writes only stdout. Reuses one fixed-topology channel object; mutates nothing.
"""
from __future__ import annotations

import copy
import numpy as np
import tensorflow as tf
import traceback

from upair5g.config import load_config, get_cfg, set_cfg
from upair5g.training import _build_system_for_num_users
try:
    from upair5g.utils import call_transmitter
except Exception:
    from upair5g.training import call_transmitter  # type: ignore

CONFIG_PATH = "configs/twc_comprehensive_mu32_base.yaml"
U = 4
ALPHA = 0.8
N_BETA = 32       # draws to estimate beta_hat
M_TEST = 24       # test slots
SEED = 4242


def _hr(t):
    print("\n" + "=" * 100 + f"\n[PROBE] {t}\n" + "=" * 100)


def _ppwr(h):
    return tf.reduce_mean(tf.math.real(h * tf.math.conj(h)), axis=[1, 2, 4, 5, 6]).numpy()


def db(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, np.float64), 1e-300))


def main():
    np.random.seed(SEED); tf.random.set_seed(SEED)
    cfg = load_config(CONFIG_PATH)
    set_cfg(cfg, "channel.family", "umi"); set_cfg(cfg, "channel.model", "umi")
    set_cfg(cfg, "near_far.enabled", True)
    set_cfg(cfg, "near_far.alpha_eval", ALPHA)
    set_cfg(cfg, "system.batch_size_train", 1); set_cfg(cfg, "system.batch_size_eval", 1)
    set_cfg(cfg, "channel.umi.randomize_topology_each_batch", False)  # freeze geometry

    try:
        sysd = _build_system_for_num_users(cfg, U)
        ch, tx = sysd["channel"], sysd["tx"]
        if not hasattr(ch, "_call_clean_ofdm"):
            raise SystemExit("[FAIL] channel lacks _call_clean_ofdm")
        ch.set_training_mode(False)
        x, _ = call_transmitter(tx, 1)
        ch._set_topology(1)            # ONE fixed geometry for the whole probe
        rho = 1.0 - ALPHA

        # ---- A) beta_hat stability -------------------------------------
        _hr("A — beta_hat from N-draw averaging vs single-slot P_u")
        def estimate_beta(n):
            acc = np.zeros(U, np.float64)
            for _ in range(n):
                _, h = ch._call_clean_ofdm(x)
                acc += _ppwr(h)[0].astype(np.float64)
            return acc / n
        beta_runs = np.stack([estimate_beta(N_BETA) for _ in range(4)], axis=0)  # [4,U]
        beta_hat = beta_runs.mean(axis=0)
        beta_stab = db(beta_runs).std(axis=0)   # stability of the N-average
        # single-slot error vs beta_hat
        singles = np.stack([_ppwr(h)[0] for h in
                            (ch._call_clean_ofdm(x)[1] for _ in range(M_TEST))], axis=0)  # [M,U]
        single_err = (db(singles) - db(beta_hat)[None, :]).std(axis=0)
        print(f"[PROBE] beta_hat (dB)              : {np.array2string(db(beta_hat), precision=2)}")
        print(f"[PROBE] N={N_BETA} estimate stability std (dB): {np.array2string(beta_stab, precision=3)}  (want << single-slot)")
        print(f"[PROBE] single-slot P_u error vs beta_hat std (dB): {np.array2string(single_err, precision=3)}  (the 3-6 dB contamination)")
        ok_A = beta_stab.max() < 0.5
        print(f"[PROBE][A] {'PASS' if ok_A else 'CHECK'}: N-averaging {'removes' if ok_A else 'may not fully remove'} fast fading "
              f"(stability {beta_stab.max():.2f} dB vs single-slot {single_err.max():.2f} dB)")

        # ---- B) slow-PC vs current per-slot, delivered fluctuation -----
        _hr("B — delivered fast-fading fluctuation: SLOW-PC fix vs CURRENT per-slot")
        pstar_slow = beta_hat ** rho
        pstar_slow = pstar_slow / pstar_slow.mean()          # slow target, fixed across slots
        raw_p, slow_deliv, cur_deliv = [], [], []
        for _ in range(M_TEST):
            _, h = ch._call_clean_ofdm(x)
            p = _ppwr(h)[0].astype(np.float64)               # [U]
            raw_p.append(p)
            slow_deliv.append(p * pstar_slow / beta_hat)     # SLOW scale: retains fast fading
            cur_deliv.append(p ** rho / np.mean(p ** rho))   # CURRENT per-slot normalize
        raw_p = np.asarray(raw_p); slow_deliv = np.asarray(slow_deliv); cur_deliv = np.asarray(cur_deliv)
        raw_std = db(raw_p).std(axis=0)
        slow_std = db(slow_deliv).std(axis=0)
        cur_std = db(cur_deliv).std(axis=0)
        print(f"[PROBE] RAW fast-fading std (dB)        : {np.array2string(raw_std, precision=3)}")
        print(f"[PROBE] SLOW-PC delivered std (dB)      : {np.array2string(slow_std, precision=3)}  (want ~= RAW)")
        print(f"[PROBE] CURRENT per-slot delivered (dB) : {np.array2string(cur_std, precision=3)}  (~ (1-alpha)*RAW = {1-ALPHA:.1f}x)")
        ok_B = np.allclose(slow_std, raw_std, atol=0.3) and (cur_std.max() < 0.6 * raw_std.max())
        print(f"[PROBE][B] {'PASS' if ok_B else 'CHECK'}: SLOW-PC preserves physical fast-fading dynamics; "
              f"CURRENT suppresses them to ~{(cur_std/np.maximum(raw_std,1e-9)).mean():.2f}x.")

        # ---- C) calibration in expectation -----------------------------
        _hr("C — calibration: mean over users of slot-mean delivered power")
        slow_mean_users = slow_deliv.mean(axis=1)            # per slot
        cur_mean_users = cur_deliv.mean(axis=1)
        print(f"[PROBE] SLOW-PC: mean_u per slot -> overall mean = {slow_mean_users.mean():.4f} "
              f"(want ~1; fluctuates per slot by design, std={slow_mean_users.std():.3f})")
        print(f"[PROBE] CURRENT: mean_u per slot -> overall mean = {cur_mean_users.mean():.4f} "
              f"(=1 exactly per slot, std={cur_mean_users.std():.3e})")
        ok_C = abs(slow_mean_users.mean() - 1.0) < 0.1
        print(f"[PROBE][C] {'PASS' if ok_C else 'CHECK'}: SLOW-PC calibrates in expectation "
              "(Eb/N0 sets AVERAGE SNR; per-slot SNR fluctuates physically).")

        _hr("D — cost & summary")
        print(f"[PROBE] cost: {N_BETA} extra channel draws per topology to estimate beta_hat "
              "(efficient via batch-replicated topology in ONE vectorized call).")
        print(f"[PROBE] OVERALL: A={'PASS' if ok_A else 'CHECK'}  B={'PASS' if ok_B else 'CHECK'}  C={'PASS' if ok_C else 'CHECK'}")
        print("[PROBE] If all PASS, the slow-gain-via-multi-draw fix is validated and ready to implement.")
    except Exception:
        print("[PROBE] FAILED:"); traceback.print_exc()


if __name__ == "__main__":
    main()
