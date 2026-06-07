#!/usr/bin/env bash
# Quantify fast-fading contamination of P_u by freezing UMi topology and re-drawing fast fading.
# This does not necessarily fail the pipeline; it measures whether P_u is a good slow-gain proxy.
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
    from upair5g.utils import call_transmitter
except Exception:
    from upair5g.training import call_transmitter  # type: ignore

def mean_user_power(h):
    return tf.reduce_mean(tf.math.real(h * tf.math.conj(h)), axis=[1,2,4,5,6]).numpy()

cfg = load_config("configs/twc_comprehensive_mu32_base.yaml")
if not bool(get_cfg(cfg, "near_far.enabled", False)):
    raise SystemExit("[FAIL] near_far.enabled is false; this probe is for UMiPC only.")

set_cfg(cfg, "system.batch_size_train", 1)
set_cfg(cfg, "system.batch_size_eval", 1)
system = _build_system_for_num_users(cfg, 4)
channel = system["channel"]
if not hasattr(channel, "_set_topology") or not hasattr(channel, "_call_clean_ofdm"):
    raise SystemExit("[FAIL] channel does not expose UMi internals; apply UMiPC patch first.")

tx = system["tx"]
B = 1
draws = int(float(get_cfg(cfg, "near_far.fast_fading_probe_draws", 16)))
draws = max(draws, 16)
alpha = 0.8

print(f"[PROBE] fixed-topology fast-fading draws={draws}, alpha_for_pstar={alpha}")
print(f"[PROBE] pathloss/shadow={get_cfg(cfg, 'channel.umi.enable_pathloss', None)} {get_cfg(cfg, 'channel.umi.enable_shadow_fading', None)}")

x, _ = call_transmitter(tx, B)
channel._set_topology(B)

p = []
for k in range(draws):
    # Re-use the same topology, but call the stochastic channel again.
    _, h_raw = channel._call_clean_ofdm(x)
    p.append(mean_user_power(h_raw)[0])
p = np.asarray(p, dtype=np.float64)  # [draws,U]
p_db = 10*np.log10(np.maximum(p, 1e-300))

mean_p = np.mean(p, axis=0)
std_p = np.std(p, axis=0)
cv = std_p / np.maximum(mean_p, 1e-300)
std_db = np.std(p_db, axis=0)
range_db = np.max(p_db, axis=0) - np.min(p_db, axis=0)

# How much alpha=0.8 PC target would move solely because P_u changes across fast fading.
rho = 1.0 - alpha
logp = np.log(np.maximum(p, 1e-300))
logeff = rho * logp
logmean = np.log(np.mean(np.exp(logeff), axis=1, keepdims=True))
pstar = np.exp(logeff - logmean)
pstar_db = 10*np.log10(np.maximum(pstar, 1e-300))
pstar_std_db = np.std(pstar_db, axis=0)

print("\n" + "="*100)
print("[RESULT] Per-user variation with topology fixed")
for u in range(p.shape[1]):
    print(
        f"user={u+1} meanP={mean_p[u]:.3e} CV={cv[u]:.3f} "
        f"std_dB={std_db[u]:.3f} range_dB={range_db[u]:.3f} "
        f"alpha0p8_pstar_std_dB={pstar_std_db[u]:.3f}"
    )

print("\n[SUMMARY]")
print(f"max CV={np.max(cv):.3f}")
print(f"max raw-power std_dB={np.max(std_db):.3f}")
print(f"max alpha0.8 pstar std_dB={np.max(pstar_std_db):.3f}")

if np.max(std_db) > 1.0:
    print("[WARN] P_u contains >1 dB fast-fading variation under frozen topology. This is a modeling limitation if P_u is intended to approximate only large-scale gain.")
else:
    print("[OK] Frozen-topology P_u variation is below 1 dB in this draw.")

print("[PROBE] DONE fast-fading contamination probe")
PY
