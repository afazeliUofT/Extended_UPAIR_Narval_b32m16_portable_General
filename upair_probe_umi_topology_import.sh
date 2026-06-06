#!/usr/bin/env bash
# Probe that the General repo resolves the UMi topology helper from the correct Sionna module.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/upair_portable_env.sh"
upair_activate

python - <<'PY'
from __future__ import annotations
from pathlib import Path
import inspect
import importlib

from upair5g.compat import resolve_attr

src = Path("src/upair5g/builders.py").read_text()
needle = 'resolve_attr(["sionna.phy.channel", "sionna.phy.channel.tr38901"], "gen_single_sector_topology")'
bad = 'resolve_attr(["sionna.phy.channel.tr38901", "sionna.channel.tr38901"], "gen_single_sector_topology")'

if needle not in src:
    raise SystemExit("[FAIL] builders.py does not use sionna.phy.channel for gen_single_sector_topology")
if bad in src:
    raise SystemExit("[FAIL] builders.py still contains old bad lookup path")
print("[OK] builders.py uses sionna.phy.channel for gen_single_sector_topology")

mod = importlib.import_module("sionna.phy.channel")
assert hasattr(mod, "gen_single_sector_topology")
fn = resolve_attr(["sionna.phy.channel", "sionna.phy.channel.tr38901"], "gen_single_sector_topology")
print("[OK] resolved function:", fn)
print("[OK] signature:", inspect.signature(fn))

# Generate a tiny topology on CPU; this should not need a GPU.
topology = fn(batch_size=1, num_ut=1, scenario="umi", min_ut_velocity=8.33, max_ut_velocity=16.67)
print("[OK] topology type:", type(topology), "length:", len(topology))
if not isinstance(topology, (tuple, list)) or len(topology) < 6:
    raise SystemExit("[FAIL] unexpected topology object")

from sionna.phy.channel.tr38901 import UMi
print("[OK] UMi.set_topology signature:", inspect.signature(UMi.set_topology))
print("[PROBE] PASSED UMi topology import/source probe.")
PY
