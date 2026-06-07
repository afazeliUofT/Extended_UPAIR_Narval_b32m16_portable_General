#!/usr/bin/env bash
# Minimal safety fix for UMiPC epsilon clamp: lower epsilon and preserve current UMiPC mode.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

python - <<'PY'
from pathlib import Path
import yaml

p = Path("configs/twc_comprehensive_mu32_base.yaml")
cfg = yaml.safe_load(p.read_text())
nf = cfg.setdefault("near_far", {})
old = nf.get("epsilon", None)
nf["epsilon"] = 1.0e-20
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"[FIX] near_far.epsilon: {old} -> {nf['epsilon']}")
PY

echo "[FIX] Run: bash upair_probe_umi_pc_power_integrity.sh"
