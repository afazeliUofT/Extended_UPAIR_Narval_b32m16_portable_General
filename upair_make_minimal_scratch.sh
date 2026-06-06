#!/usr/bin/env bash
# Destructive cleanup for a minimal from-scratch UPAIR package.
# It removes old Optuna/training/evaluation evidence and leaves only code/configs/wrappers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if [[ "${1:-}" != "--yes" && "${UPAIR_CLEAN_CONFIRM:-}" != "YES" ]]; then
  cat >&2 <<MSG
This will permanently remove generated/evidence folders and unnecessary logs/readmes from:
  ${ROOT}

It removes, among others:
  TWC_plots_comprehensive/ optuna/ manifest/ portable_tools/ logs/ old README*.md old *.log

Run exactly:
  bash $0 --yes
MSG
  exit 2
fi

required=(src scripts configs pyproject.toml)
for item in "${required[@]}"; do
  [[ -e "${ROOT}/${item}" ]] || { echo "[CLEAN] Missing required item before cleanup: ${item}" >&2; exit 1; }
done

echo "[CLEAN] Removing old generated/evidence directories..."
rm -rf \
  "${ROOT}/TWC_plots_comprehensive" \
  "${ROOT}/optuna" \
  "${ROOT}/logs" \
  "${ROOT}/manifest" \
  "${ROOT}/portable_tools" \
  "${ROOT}/.pytest_cache" \
  "${ROOT}/.mypy_cache" \
  "${ROOT}/.ruff_cache" \
  "${ROOT}/htmlcov" \
  "${ROOT}/wandb" \
  "${ROOT}/mlruns" \
  "${ROOT}/outputs" \
  "${ROOT}/checkpoints" \
  "${ROOT}/plots" \
  "${ROOT}/metrics" \
  "${ROOT}/artifacts" \
  "${ROOT}/__pycache__"

# Remove generated files/logs and old documentation snapshots at the repository root.
echo "[CLEAN] Removing old root logs/readmes/helper snapshots..."
rm -f \
  "${ROOT}"/PORTABLE_README.md \
  "${ROOT}"/README_CLEAN_OPTUNA.md \
  "${ROOT}"/README_CLEAN_OPTUNA_MEMORY_FIX.md \
  "${ROOT}"/README_NARVAL_B32_MICRO16.md \
  "${ROOT}"/README_NARVAL_OPTUNA.md \
  "${ROOT}"/manifest_summary_after_cleanup_*.log \
  "${ROOT}"/make_dmrs_grid_*.log \
  "${ROOT}"/rerender_dmrs*.log \
  "${ROOT}"/slurm-*.out \
  "${ROOT}"/*.out \
  "${ROOT}"/*.err \
  "${ROOT}"/*.log \
  "${ROOT}"/core.* \
  "${ROOT}"/delete_old_unclean_portable_backups.sh \
  "${ROOT}"/make_dmrs_grid_4users_12sc.sh \
  "${ROOT}"/rerender_dmrs_grid_4users_2x2_large_cells.sh

# Remove Python caches anywhere.
find "${ROOT}" -type d \( -name '__pycache__' -o -name '.ipynb_checkpoints' \) -prune -exec rm -rf {} +
find "${ROOT}" -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.tmp' -o -name '*.bak' -o -name '*~' \) -delete

# Create only empty runtime roots needed by wrappers.
mkdir -p "${ROOT}/optuna" "${ROOT}/logs/optuna" "${ROOT}/logs/train_eval"

echo "[CLEAN] Done. Current top-level size:"
du -sh "${ROOT}"
echo "[CLEAN] Top-level contents:"
find "${ROOT}" -maxdepth 1 -mindepth 1 -printf '  %f\n' | sort
