#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/upair_portable_env.sh"
upair_activate

python - <<'PY'
from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys

print("=" * 100)
print("[SIONNA API PROBE]")
try:
    import sionna
    print("sionna module:", sionna)
    print("sionna version:", getattr(sionna, "__version__", "<no __version__>"))
except Exception as e:
    print("[FAIL] could not import sionna:", repr(e))
    raise

candidate_modules = [
    "sionna.phy.channel.tr38901",
    "sionna.phy.channel.tr38901.utils",
    "sionna.phy.channel.tr38901.system_level_scenario",
    "sionna.phy.channel.tr38901.umi",
    "sionna.phy.channel.tr38901.antenna",
    "sionna.phy.channel",
    "sionna.channel.tr38901",
    "sionna.channel.tr38901.utils",
]

print("\n" + "=" * 100)
print("[DIRECT MODULE CHECKS]")
imported = {}
for name in candidate_modules:
    try:
        mod = importlib.import_module(name)
        imported[name] = mod
        print(f"[OK] import {name}")
        names = dir(mod)
        interesting = [
            x for x in names
            if any(k in x.lower() for k in ["umi", "uma", "rma", "topology", "sector", "panel", "antenna", "scenario"])
        ]
        print("     interesting names:", interesting[:80])
        for attr in ["UMi", "UMa", "RMa", "PanelArray", "Antenna", "gen_single_sector_topology"]:
            if hasattr(mod, attr):
                obj = getattr(mod, attr)
                print(f"     HAS {attr}: {obj}")
                try:
                    print(f"       signature: {inspect.signature(obj)}")
                except Exception as e:
                    print(f"       signature unavailable: {e!r}")
    except Exception as e:
        print(f"[NO] import {name}: {type(e).__name__}: {e}")

print("\n" + "=" * 100)
print("[WALK sionna.phy.channel.tr38901 SUBMODULES]")
try:
    base = importlib.import_module("sionna.phy.channel.tr38901")
    if hasattr(base, "__path__"):
        for m in pkgutil.walk_packages(base.__path__, base.__name__ + "."):
            name = m.name
            try:
                mod = importlib.import_module(name)
            except Exception as e:
                print(f"[SKIP] {name}: {type(e).__name__}: {e}")
                continue
            names = dir(mod)
            hits = [
                x for x in names
                if any(k in x.lower() for k in ["topology", "sector", "scenario", "umi", "uma", "rma", "panel"])
            ]
            if hits:
                print(f"[SUBMODULE] {name}")
                print("  hits:", hits[:120])
                for attr in hits:
                    if attr == "gen_single_sector_topology" or "topology" in attr.lower():
                        obj = getattr(mod, attr)
                        print(f"  candidate {attr}: {obj}")
                        try:
                            print(f"    signature: {inspect.signature(obj)}")
                        except Exception as e:
                            print(f"    signature unavailable: {e!r}")
    else:
        print("sionna.phy.channel.tr38901 has no __path__; cannot walk.")
except Exception as e:
    print("[FAIL] walking tr38901 failed:", repr(e))

print("\n" + "=" * 100)
print("[UMi CLASS DETAILS]")
UMi = None
for name, mod in imported.items():
    if hasattr(mod, "UMi"):
        UMi = getattr(mod, "UMi")
        print("UMi found in:", name)
        break

if UMi is not None:
    try:
        print("UMi signature:", inspect.signature(UMi))
    except Exception as e:
        print("UMi signature unavailable:", repr(e))
    print("UMi methods/attrs containing topology/scenario:")
    for x in dir(UMi):
        if any(k in x.lower() for k in ["topology", "scenario", "lsp", "pathloss", "shadow"]):
            print(" ", x)
    if hasattr(UMi, "set_topology"):
        try:
            print("UMi.set_topology signature:", inspect.signature(UMi.set_topology))
        except Exception as e:
            print("UMi.set_topology signature unavailable:", repr(e))
else:
    print("[FAIL] UMi class not found in direct candidate modules.")

print("\n[PROBE] DONE")
PY
