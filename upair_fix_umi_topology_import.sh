#!/usr/bin/env bash
# Fix UMi topology generator import path for Sionna 1.2.1.
# Your installed Sionna exposes gen_single_sector_topology under sionna.phy.channel,
# not under sionna.phy.channel.tr38901.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if [[ "${ROOT}" != *"Extended_UPAIR_Narval_b32m16_portable_General"* && "${UPAIR_ALLOW_NON_GENERAL_PATCH:-0}" != "1" ]]; then
  echo "[FIX] Refusing to run outside the General repo copy." >&2
  echo "[FIX] ROOT=${ROOT}" >&2
  exit 1
fi

[[ -f src/upair5g/builders.py ]] || { echo "[FIX] builders.py not found; run from repo root." >&2; exit 1; }

python - <<'PY'
from pathlib import Path

p = Path("src/upair5g/builders.py")
s = p.read_text()

old = 'resolve_attr(["sionna.phy.channel.tr38901", "sionna.channel.tr38901"], "gen_single_sector_topology")'
new = 'resolve_attr(["sionna.phy.channel", "sionna.phy.channel.tr38901"], "gen_single_sector_topology")'

if old in s:
    s = s.replace(old, new)
    p.write_text(s)
    print("[FIX] Replaced UMi topology helper lookup with sionna.phy.channel first.")
elif new in s:
    print("[FIX] UMi topology helper lookup is already correct.")
else:
    raise SystemExit("[FIX] Could not find expected gen_single_sector_topology lookup in builders.py")

# Check the replacement is specifically in _set_topology.
text = p.read_text()
if 'resolve_attr(["sionna.phy.channel", "sionna.phy.channel.tr38901"], "gen_single_sector_topology")' not in text:
    raise SystemExit("[FIX] Corrected lookup not found after patch.")
if 'resolve_attr(["sionna.phy.channel.tr38901", "sionna.channel.tr38901"], "gen_single_sector_topology")' in text:
    raise SystemExit("[FIX] Old bad lookup still present.")
PY

echo "[FIX] Done. Run:"
echo "  bash upair_probe_umi_topology_import.sh"
echo "  bash upair_probe_umi_runtime_channel.sh"
