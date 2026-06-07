#!/usr/bin/env bash
# Probe which Sionna internal attributes reproduce the slow large-scale gain used by UMi.
# This avoids guessing whether LSP.sf is a linear gain, inverse loss, dB term, amplitude factor, etc.
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
try:
    from upair5g.utils import call_transmitter
except Exception:
    from upair5g.training import call_transmitter  # type: ignore

def mean_user_power(h):
    return tf.reduce_mean(tf.math.real(h * tf.math.conj(h)), axis=[1,2,4,5,6]).numpy()

def db(x):
    return 10.0*np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1e-300))

cfg = load_config("configs/twc_comprehensive_mu32_base.yaml")
set_cfg(cfg, "system.batch_size_train", 8)
set_cfg(cfg, "system.batch_size_eval", 8)

system = _build_system_for_num_users(cfg, 4)
channel = system["channel"]
if not hasattr(channel, "_set_topology") or not hasattr(channel, "_call_clean_ofdm"):
    raise SystemExit("[FAIL] UMiPC/UMi topology wrapper internals not found.")

B = 8
U = 4
draws = 64

x, _ = call_transmitter(system["tx"], B)
channel._set_topology(B)

cm = channel.channel_model
sc = cm._scenario
lsp = cm._lsp

pl_db = np.asarray(sc.basic_pathloss.numpy(), dtype=np.float64).reshape(B, U)
sf = np.asarray(lsp.sf.numpy(), dtype=np.float64).reshape(B, U)

powers = []
for _ in range(draws):
    _, h_raw = channel._call_clean_ofdm(x)
    powers.append(mean_user_power(h_raw))
powers = np.asarray(powers, dtype=np.float64)  # [draws,B,U]
p_mean = np.mean(powers, axis=0)
p_std_db = np.std(db(powers), axis=0)

cand = {}
base_gain = 10.0**(-pl_db/10.0)
cand["10^(-PL/10)"] = base_gain
cand["10^(-PL/10) * sf"] = base_gain * sf
cand["10^(-PL/10) / sf"] = base_gain / np.maximum(sf, 1e-300)
cand["10^(-PL/10) * sf^2"] = base_gain * sf**2
cand["10^(-PL/10) / sf^2"] = base_gain / np.maximum(sf, 1e-300)**2
# Some codebases store shadow fading in dB even if values look odd; include these candidates.
cand["10^(-(PL+sf)/10)"] = 10.0**(-(pl_db + sf)/10.0)
cand["10^(-(PL-sf)/10)"] = 10.0**(-(pl_db - sf)/10.0)

print("[PROBE] Sionna slow-gain formula probe")
print(f"[INFO] B={B}, U={U}, fast-fading draws={draws}")
print("[INFO] pathloss/shadow:", get_cfg(cfg, "channel.umi.enable_pathloss"), get_cfg(cfg, "channel.umi.enable_shadow_fading"))
print("[INFO] pl_db first row:", pl_db[0])
print("[INFO] sf first row:", sf[0])
print("[INFO] p_mean first row:", p_mean[0])
print("[INFO] fast-fading p_dB std first row:", p_std_db[0])

rows = []
for name, g in cand.items():
    ratio_db = db(p_mean / np.maximum(g, 1e-300))
    rows.append((float(np.std(ratio_db)), float(np.mean(ratio_db)), name, ratio_db))
rows.sort(key=lambda x: x[0])

print("\n" + "="*100)
print("[CANDIDATE FORMULA FIT]")
for std, mean, name, ratio_db in rows:
    print(f"{name:24s} ratio_dB_std={std:8.3f} ratio_dB_mean={mean:9.3f} ratio_dB_min/max={np.min(ratio_db):9.3f}/{np.max(ratio_db):9.3f}")

best = rows[0]
print("\n" + "="*100)
print(f"[BEST] {best[2]} with ratio_dB_std={best[0]:.3f}, mean_bias={best[1]:.3f} dB")
if best[0] > 2.0:
    print("[WARN] Even the best candidate has >2 dB cross-user/drop ratio variation. Need deeper inspection before patching.")
else:
    print("[OK] A stable slow-gain candidate was found. This can be used for slow-gain-based fractional power control after patching.")

# Show how much frozen-topology P* would vary under realized-power vs slow-gain candidate.
alpha = 0.8
rho = 1.0 - alpha

def pstar_from_power(p):
    lp = np.log(np.maximum(p, 1e-300))
    le = rho*lp
    lm = np.log(np.mean(np.exp(le), axis=-1, keepdims=True))
    return np.exp(le-lm)

pstar_realized = np.asarray([pstar_from_power(powers[k]) for k in range(draws)])
pstar_slow = pstar_from_power(best[3]*0 + p_mean)  # placeholder not used

# Use the selected gain itself for slow-gain pstar, fixed over fast fading.
g_best = cand[best[2]]
pstar_slow_fixed = pstar_from_power(g_best)
slow_stack = np.repeat(pstar_slow_fixed[None, ...], draws, axis=0)

realized_std = np.std(db(pstar_realized), axis=0)
slow_std = np.std(db(slow_stack), axis=0)
print("\n[PC TARGET JITTER AT alpha=0.8]")
print("realized-Pu pstar std_dB max/mean:", float(np.max(realized_std)), float(np.mean(realized_std)))
print("slow-gain  pstar std_dB max/mean:", float(np.max(slow_std)), float(np.mean(slow_std)))
print("[PROBE] DONE slow-gain formula probe")
PY
