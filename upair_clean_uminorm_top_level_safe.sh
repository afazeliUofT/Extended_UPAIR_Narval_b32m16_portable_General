#!/usr/bin/env bash
# Clean redundant top-level shell scripts and transient runtime artifacts.
# This script intentionally does NOT modify .gitignore.
# Default is dry-run. Apply with:
#   UPAIR_CLEAN_MODE=apply bash upair_clean_uminorm_top_level_safe.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

MODE="${UPAIR_CLEAN_MODE:-dryrun}"
echo "[CLEAN] MODE=${MODE}"
echo "[CLEAN] ROOT=${ROOT}"

KEEP_SH=(
  upair_portable_env.sh
  upair_submit_lib.sh
  upair_make_minimal_scratch.sh
  upair_submit_stageA_all.sh
  upair_submit_stageB_all.sh
  upair_submit_8variant_pipeline.sh
  upair_variant_pipeline_worker.sh
  upair_probe_uminorm_pipeline_ready.sh
  upair_probe_uminorm_batch_eval_policy.sh
  upair_probe_normalized_umi_ready.sh
  upair_probe_normalized_umi_runtime.sh
  upair_clean_uminorm_top_level_safe.sh
)

is_keep() {
  local f="$1"
  for k in "${KEEP_SH[@]}"; do
    [[ "$f" == "$k" ]] && return 0
  done
  return 1
}

echo
echo "================================================================================"
echo "[CLEAN] Top-level .sh files"
mapfile -t all_sh < <(find . -maxdepth 1 -type f -name '*.sh' -printf '%f\n' | sort)
for f in "${all_sh[@]}"; do
  if is_keep "$f"; then
    echo "[KEEP]   $f"
  else
    echo "[REMOVE] $f"
  fi
done

echo
echo "================================================================================"
echo "[CLEAN] Transient folders/files to remove locally"
TRANSIENT=(
  _isolated_eval_chunks
  _isolated_eval_chunks_smoke
  _smoke_umi_norm_runtime
  _smoke_umi_pc_runtime
  _smoke_true_dmrs_runtime
  _smoke_*
  _stress_*
  logs/eval_iso
  logs/train_eval
  logs/pipeline
)
for x in "${TRANSIENT[@]}"; do
  compgen -G "$x" >/dev/null || continue
  for y in $x; do
    [[ -e "$y" ]] && echo "[REMOVE] $y"
  done
done

echo
echo "[CLEAN] Generated folders to untrack from git only, not delete from disk:"
echo "[UNTRACK] TWC_plots_comprehensive/  logs/  optuna/  _isolated_eval_chunks*/"

if [[ "$MODE" != "apply" ]]; then
  echo
  echo "[CLEAN] Dry run only. To apply:"
  echo "  UPAIR_CLEAN_MODE=apply bash upair_clean_uminorm_top_level_safe.sh"
  exit 0
fi

echo
echo "[CLEAN] Applying cleanup..."

for f in "${all_sh[@]}"; do
  if ! is_keep "$f"; then
    rm -f "$f"
  fi
done

rm -rf _isolated_eval_chunks _isolated_eval_chunks_smoke _smoke_* _stress_* logs/eval_iso logs/train_eval logs/pipeline
mkdir -p logs/submit logs/pipeline

find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

git rm -r --cached --ignore-unmatch TWC_plots_comprehensive logs optuna _isolated_eval_chunks _isolated_eval_chunks_smoke >/dev/null 2>&1 || true

echo
echo "[CLEAN] Final top-level .sh files:"
find . -maxdepth 1 -type f -name '*.sh' -printf '%f\n' | sort
echo
echo "[CLEAN] Done. Review with:"
echo "  git status --short"
