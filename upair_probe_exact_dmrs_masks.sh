#!/usr/bin/env bash
# Probe exact Sionna PUSCH DMRS occupied REs per user/port and compare them
# with the current pilot mask feature fed to UPAIR.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/upair_portable_env.sh"
upair_activate

python - <<'PY'
from __future__ import annotations

import numpy as np
import tensorflow as tf

from upair5g.config import load_config, set_cfg
from upair5g.builders import (
    build_pusch_transmitter,
    get_resource_grid,
    extract_pilot_mask_per_stream,
    extract_pilot_mask,
)

def as_np(x):
    return np.asarray(tf.convert_to_tensor(x).numpy())

def summarize_binary_mask(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        # Try to reduce any singleton layer/stream axis for printing.
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
    if arr.ndim != 2:
        print(f"{name}: shape={arr.shape}; cannot summarize as [T,F]")
        return
    nz = np.argwhere(np.abs(arr) > 0)
    print(f"{name}: shape={arr.shape}, nonzero_count={len(nz)}")
    if len(nz) == 0:
        return
    by_symbol = {}
    for t, k in nz:
        by_symbol.setdefault(int(t), []).append(int(k))
    for t in sorted(by_symbol):
        sc = sorted(by_symbol[t])
        mods = sorted(set(k % 12 for k in sc))
        counts_by_mod = {m: sum(1 for k in sc if k % 12 == m) for m in mods}
        print(f"  symbol {t}: count={len(sc)}, k_mod_12={mods}, counts_by_mod={counts_by_mod}")
        print(f"    first 48 absolute k: {sc[:48]}")

def get_attr_chain(obj, names):
    for name in names:
        if hasattr(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                pass
    return None

def candidate_dmrs_grids(pusch_cfg):
    candidates = []
    # Common direct attributes
    for name in ["dmrs_grid", "_dmrs_grid"]:
        value = get_attr_chain(pusch_cfg, [name])
        if value is not None:
            candidates.append((f"pusch_config.{name}", value))
    dmrs = getattr(pusch_cfg, "dmrs", None)
    if dmrs is not None:
        for name in ["dmrs_grid", "_dmrs_grid", "pilot_grid", "_pilot_grid"]:
            value = get_attr_chain(dmrs, [name])
            if value is not None:
                candidates.append((f"pusch_config.dmrs.{name}", value))
    return candidates

def main(additional_position: int = 0, num_users: int = 4) -> None:
    cfg = load_config("configs/twc_comprehensive_mu32_base.yaml")
    set_cfg(cfg, "pusch.n_size_grid", 8)
    set_cfg(cfg, "pusch.n_size_bwp", 8)
    set_cfg(cfg, "multiuser.enabled", True)
    set_cfg(cfg, "multiuser.max_num_users", 4)
    set_cfg(cfg, "multiuser.fixed_num_users", num_users)
    set_cfg(cfg, "multiuser.dmrs.config_type", 2)
    set_cfg(cfg, "multiuser.dmrs.length", 1)
    set_cfg(cfg, "multiuser.dmrs.additional_position", additional_position)
    set_cfg(cfg, "multiuser.dmrs.type_a_position", 2)
    set_cfg(cfg, "multiuser.dmrs.num_cdm_groups_without_data", 3)
    set_cfg(cfg, "multiuser.dmrs.port_sets", [[0], [1], [2], [3]])

    tx, _ = build_pusch_transmitter(cfg, num_users=num_users)
    rg = get_resource_grid(tx)
    pcs = list(getattr(tx, "_upair_pusch_configs", []))

    print("=" * 100)
    print(f"additional_position={additional_position}, num_users={num_users}, RB=8")
    print("Expected if your diagram is the actual nonzero DMRS pattern:")
    print("  ports 0/1 -> k mod 12 in [0,1,6,7], count per DMRS symbol = 4*8 = 32")
    print("  ports 2/3 -> k mod 12 in [2,3,8,9], count per DMRS symbol = 4*8 = 32")
    print("  addpos=1 should double counts if there are two DMRS symbols.")
    print()

    feature_mask = as_np(extract_pilot_mask_per_stream(rg))  # [T,F,S] from current code path
    collapsed_mask = as_np(extract_pilot_mask(rg))
    print(f"CURRENT feature-mask source extract_pilot_mask_per_stream(rg): shape={feature_mask.shape}")
    for s in range(feature_mask.shape[-1]):
        summarize_binary_mask(f"current_feature_mask_stream_{s}", feature_mask[..., s])
    summarize_binary_mask("current_collapsed_mask", collapsed_mask)
    print()

    if not pcs:
        print("No tx._upair_pusch_configs found; cannot inspect per-user PUSCHConfig objects.")
        return

    for u, pc in enumerate(pcs, start=1):
        port_set = None
        dmrs = getattr(pc, "dmrs", None)
        if dmrs is not None:
            for attr in ["dmrs_port_set", "_dmrs_port_set", "port_set", "_port_set"]:
                if hasattr(dmrs, attr):
                    try:
                        port_set = getattr(dmrs, attr)
                        break
                    except Exception:
                        pass
        print("-" * 100)
        print(f"User {u}: inferred port_set={port_set}")
        candidates = candidate_dmrs_grids(pc)
        if not candidates:
            print("  No dmrs_grid-like attribute found on this PUSCHConfig. Available hints:")
            print("  pusch_config attrs containing 'dmrs' or 'grid':",
                  [a for a in dir(pc) if 'dmrs' in a.lower() or 'grid' in a.lower()][:80])
            if dmrs is not None:
                print("  pusch_config.dmrs attrs containing 'grid' or 'pilot':",
                      [a for a in dir(dmrs) if 'grid' in a.lower() or 'pilot' in a.lower()][:80])
            continue

        for name, grid in candidates:
            arr = as_np(grid)
            print(f"  candidate {name}: raw_shape={arr.shape}, dtype={arr.dtype}")
            # Try to squeeze to [T,F] or choose first layer if needed.
            arr_sq = np.squeeze(arr)
            # If shape is [layers,T,F], summarize each layer.
            if arr_sq.ndim == 3:
                for layer in range(arr_sq.shape[0]):
                    summarize_binary_mask(f"  {name}[layer={layer}]", arr_sq[layer])
            else:
                summarize_binary_mask(f"  {name}", arr_sq)

if __name__ == "__main__":
    main(additional_position=0, num_users=4)
    main(additional_position=1, num_users=4)
PY
