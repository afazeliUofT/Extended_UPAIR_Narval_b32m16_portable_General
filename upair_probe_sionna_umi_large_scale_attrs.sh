#!/usr/bin/env bash
# Inspect Sionna UMi channel object for large-scale/pathloss/shadow/LSP tensors.
# Use this to determine whether we can implement power control from true slow gains instead of realized slot power P_u.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
source "${ROOT}/upair_portable_env.sh"
upair_activate

python - <<'PY'
from __future__ import annotations
import numpy as np
import tensorflow as tf
from collections import deque

from upair5g.config import load_config, set_cfg, get_cfg
from upair5g.training import _build_system_for_num_users

cfg = load_config("configs/twc_comprehensive_mu32_base.yaml")
set_cfg(cfg, "system.batch_size_train", 2)
set_cfg(cfg, "system.batch_size_eval", 2)
system = _build_system_for_num_users(cfg, 4)
channel = system["channel"]
if not hasattr(channel, "_set_topology"):
    raise SystemExit("[FAIL] UMi topology wrapper not found.")

channel._set_topology(2)
cm = channel.channel_model

keywords = ["path", "loss", "shadow", "gain", "lsp", "sf", "distance", "dist", "los", "pl", "scenario", "topology"]

def summarize_value(v):
    try:
        if isinstance(v, (tf.Tensor, tf.Variable)):
            arr = v.numpy()
        elif isinstance(v, np.ndarray):
            arr = v
        else:
            return None
        arr = np.asarray(arr)
        if arr.size == 0:
            return f"array shape={arr.shape} dtype={arr.dtype} empty"
        if np.iscomplexobj(arr):
            mag = np.abs(arr)
            return f"array shape={arr.shape} dtype={arr.dtype} |.| min/mean/max={mag.min():.3e}/{mag.mean():.3e}/{mag.max():.3e}"
        arrf = arr.astype(np.float64, copy=False) if np.issubdtype(arr.dtype, np.number) or arr.dtype == bool else arr
        if np.issubdtype(arr.dtype, np.number) or arr.dtype == bool:
            flat = arrf.reshape(-1)
            preview = flat[:min(8, flat.size)]
            return f"array shape={arr.shape} dtype={arr.dtype} min/mean/max={flat.min():.3e}/{flat.mean():.3e}/{flat.max():.3e} first={preview}"
        return f"array shape={arr.shape} dtype={arr.dtype}"
    except Exception as e:
        return f"<could not summarize: {type(e).__name__}: {e}>"

seen = set()
queue = deque([(cm, "channel_model", 0)])
print("[PROBE] Inspecting channel_model:", type(cm))
print("[PROBE] channel.umi:", get_cfg(cfg, "channel.umi", {}))

hits = []
while queue:
    obj, name, depth = queue.popleft()
    oid = id(obj)
    if oid in seen or depth > 2:
        continue
    seen.add(oid)

    try:
        attrs = sorted(set(list(getattr(obj, "__dict__", {}).keys()) + [a for a in dir(obj) if not a.startswith("__")]))
    except Exception:
        continue

    for attr in attrs:
        low = attr.lower()
        relevant = any(k in low for k in keywords)
        if not relevant and depth >= 1:
            continue
        try:
            val = getattr(obj, attr)
        except Exception as e:
            if relevant:
                hits.append((f"{name}.{attr}", f"<getattr failed: {type(e).__name__}: {e}>"))
            continue
        if callable(val):
            continue
        summary = summarize_value(val)
        if relevant or summary is not None:
            hits.append((f"{name}.{attr}", summary if summary is not None else f"type={type(val)}"))
        if depth < 2 and hasattr(val, "__dict__") and not isinstance(val, (str, bytes)):
            if any(k in low for k in ["scenario", "topology", "lsp", "path", "loss", "shadow"]):
                queue.append((val, f"{name}.{attr}", depth+1))

print("\n" + "="*100)
print("[CANDIDATE ATTRIBUTES]")
for k, v in hits:
    print(f"{k}: {v}")

print("\n[PROBE] DONE. Look for tensors with shapes like [batch, num_bs, num_ut] or [batch, num_ut] and names related to pathloss/shadow/sf/distance/los.")
PY
