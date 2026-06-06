#!/usr/bin/env bash
# Clean the General UPAIR repo for a fresh UMi workflow.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
if [[ "${ROOT}" != *"Extended_UPAIR_Narval_b32m16_portable_General"* && "${UPAIR_ALLOW_NON_GENERAL_CLEAN:-0}" != "1" ]]; then
  echo "[CLEAN] Refusing to run outside the General repo copy." >&2
  echo "[CLEAN] ROOT=${ROOT}" >&2
  exit 1
fi
echo "[CLEAN] ROOT=${ROOT}"
rm -rf optuna logs TWC_plots_comprehensive outputs plots metrics checkpoints artifacts patch_backups _smoke_* __pycache__
find . -type d \( -name '__pycache__' -o -name '*.egg-info' -o -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' -o -name '.ipynb_checkpoints' \) -prune -exec rm -rf {} +
find . -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.out' -o -name '*.err' -o -name '*.log' -o -name '*.db' -o -name '*.sqlite' -o -name '*.sqlite3' -o -name '*.weights.h5' -o -name '*.data-*' \) -delete
mkdir -p optuna logs/optuna logs/submit logs/train_eval logs/smoke
python - <<'PY'
from pathlib import Path
p = Path(".gitignore")
s = p.read_text() if p.exists() else ""
items = ["__pycache__/", "*.py[cod]", "*.pyo", "*.egg-info/", "optuna/", "logs/", "TWC_plots_comprehensive/", "outputs/", "plots/", "metrics/", "checkpoints/", "artifacts/", "patch_backups/", "_smoke_*/", "*.out", "*.err", "*.log", "*.db", "*.sqlite", "*.sqlite3", "*.weights.h5", "*.data-*", ".cache/", ".venv/", ".venv*/", ".vevn_upair_potable/"]
lines = s.splitlines()
for item in items:
    if item not in lines:
        lines.append(item)
p.write_text("\n".join(lines).rstrip() + "\n")
print("[CLEAN] .gitignore updated with runtime-output ignore rules.")
PY
echo "[CLEAN] Completed. Runtime directories recreated empty:"
find optuna logs -maxdepth 2 -type d | sort
echo "[CLEAN] Git status summary:"
git status --short || true
