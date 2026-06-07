#!/usr/bin/env bash
# Quantify whether near_far.epsilon/p_safe clamping corrupts realized post-PC channel powers.
# Run after UMiPC patch. Fails if realized channel powers do not match intended p_star.
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
from upair5g.training import _build_system_for_num_users
from upair5g.builders import get_resource_grid
try:
    from upair5g.utils import call_transmitter, ebno_db_to_no
except Exception:
    from upair5g.training import call_transmitter, ebno_db_to_no  # type: ignore

def mean_user_power(h):
    return tf.reduce_mean(tf.math.real(h * tf.math.conj(h)), axis=[1,2,4,5,6]).numpy()

cfg = load_config("configs/twc_comprehensive_mu32_base.yaml")
if not bool(get_cfg(cfg, "near_far.enabled", False)):
    raise SystemExit("[FAIL] near_far.enabled is false; this probe is for UMiPC only.")

eps = float(get_cfg(cfg, "near_far.epsilon", 1e-12))
print(f"[PROBE] near_far.epsilon={eps:.3e}")
print(f"[PROBE] channel.normalize_channel={get_cfg(cfg, 'channel.normalize_channel', None)}")
print(f"[PROBE] pathloss/shadow={get_cfg(cfg, 'channel.umi.enable_pathloss', None)} {get_cfg(cfg, 'channel.umi.enable_shadow_fading', None)}")

set_cfg(cfg, "system.batch_size_train", 8)
set_cfg(cfg, "system.batch_size_eval", 8)
system = _build_system_for_num_users(cfg, 4)
channel = system["channel"]
if not hasattr(channel, "_call_clean_ofdm") or not hasattr(channel, "_apply_fractional_power_control"):
    raise SystemExit("[FAIL] channel does not expose UMiPC internals; apply UMiPC patch first.")

tx = system["tx"]
rg = get_resource_grid(tx)
B = 8
num_drops = 20
alphas = [1.0, 0.8, 0.7, 0.0]

global_bad = False

for alpha in alphas:
    set_cfg(cfg, "near_far.alpha_eval", float(alpha))
    set_cfg(cfg, "near_far.alpha_train_min", float(alpha))
    set_cfg(cfg, "near_far.alpha_train_max", float(alpha))
    channel.cfg = cfg
    channel.set_training_mode(False)

    ratios = []
    p_true_all = []
    p_safe_all = []
    p_target_all = []
    p_real_all = []

    for _ in range(num_drops):
        x, _ = call_transmitter(tx, B)
        no = ebno_db_to_no(tf.constant(0.0, tf.float32), tx=tx, resource_grid=rg)
        channel._set_topology(B)
        _, h_raw = channel._call_clean_ofdm(x)
        p_true = mean_user_power(h_raw)

        _, h_star = channel._apply_fractional_power_control(h_raw, x, no)
        stats = channel.last_near_far_stats
        p_safe = np.asarray(stats["raw_power"].numpy(), dtype=np.float64)
        p_target = np.asarray(stats["post_power"].numpy(), dtype=np.float64)
        p_real = mean_user_power(h_star).astype(np.float64)

        ratio = p_real / np.maximum(p_target, 1e-300)
        ratios.append(ratio.reshape(-1))
        p_true_all.append(p_true.reshape(-1))
        p_safe_all.append(p_safe.reshape(-1))
        p_target_all.append(p_target.reshape(-1))
        p_real_all.append(p_real.reshape(-1))

    ratios = np.concatenate(ratios)
    p_true_all = np.concatenate(p_true_all)
    p_safe_all = np.concatenate(p_safe_all)
    p_target_all = np.concatenate(p_target_all)
    p_real_all = np.concatenate(p_real_all)

    clamp_pre = p_true_all < eps
    safe_eq = np.isclose(p_safe_all, eps, rtol=0.0, atol=max(eps*1e-6, 1e-30))
    low_ratio = ratios < 0.99

    print("\n" + "="*100)
    print(f"[ALPHA] {alpha}")
    print(f"[INFO] samples={ratios.size} preclamp_below_eps={clamp_pre.sum()} ({clamp_pre.mean()*100:.2f}%) p_safe_eq_eps={safe_eq.sum()}")
    print(f"[INFO] p_true min/p01/median/max = {np.min(p_true_all):.3e} / {np.quantile(p_true_all,0.01):.3e} / {np.median(p_true_all):.3e} / {np.max(p_true_all):.3e}")
    print(f"[INFO] realized/target ratio min/p01/mean/median = {np.min(ratios):.6f} / {np.quantile(ratios,0.01):.6f} / {np.mean(ratios):.6f} / {np.median(ratios):.6f}")
    if low_ratio.any():
        idx = np.argsort(ratios)[:8]
        print("[DETAIL] worst ratios:")
        for i in idx:
            print(f"  ratio={ratios[i]:.6e} p_true={p_true_all[i]:.3e} p_safe={p_safe_all[i]:.3e} p_target={p_target_all[i]:.3e} p_real={p_real_all[i]:.3e}")
    if low_ratio.mean() > 0.001 or np.min(ratios) < 0.99:
        print("[FAIL] Realized post-PC powers do not match intended p_star. Epsilon clamp is corrupting weak users.")
        global_bad = True
    else:
        print("[OK] Realized post-PC powers match intended p_star for this alpha.")

if global_bad:
    raise SystemExit("[PROBE] FAILED UMiPC power-integrity probe")
print("\n[PROBE] PASSED UMiPC power-integrity probe")
PY
