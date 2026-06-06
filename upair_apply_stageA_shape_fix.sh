#!/usr/bin/env bash
# Fix Stage-A Optuna warm-up shape failure caused by per-user LS-error features
# using raw err_ls with batch dimension 1 instead of broadcasted err_bc/err_btfnu.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

[[ -f src/upair5g/estimator.py ]] || { echo "[PATCH] Run from repo root; src/upair5g/estimator.py not found." >&2; exit 1; }

python - <<'PY'
from pathlib import Path

p = Path("src/upair5g/estimator.py")
s = p.read_text()

old_multi = '''                # Sionna's err_var usually has one receive-error dimension.  Average that
                # dimension but keep the user/stream dimension explicit: [B,T,F,Umax].
                err_raw_btfnu = tensor7_to_btfnu(err_ls)
                err_user_map = tf.reduce_mean(err_raw_btfnu, axis=-2)
                err_map = self._pad_feature_dim(err_user_map, self.max_num_users)
'''
new_multi = '''                # Sionna's err_var can have batch dimension 1 even when y/h have
                # batch dimension B>1. Use the broadcasted error tensor err_btfnu,
                # not raw err_ls, so every feature has the same [B,T,F,*] axes.
                err_user_map = tf.reduce_mean(err_btfnu, axis=-2)
                err_map = self._pad_feature_dim(err_user_map, self.max_num_users)
'''

old_single = '''            if per_user_error:
                err_raw_btfnu = tensor7_to_btfnu(err_ls)
                err_map = tf.reduce_mean(err_raw_btfnu, axis=-2)
'''
new_single = '''            if per_user_error:
                # Use broadcasted err_btfnu for the same reason as in the
                # multi-user branch: raw err_ls may carry batch dimension 1.
                err_map = tf.reduce_mean(err_btfnu, axis=-2)
'''

if old_multi not in s:
    raise SystemExit("[PATCH] Could not find multi-user raw err_ls block in estimator.py")
if old_single not in s:
    raise SystemExit("[PATCH] Could not find single-user raw err_ls block in estimator.py")

s = s.replace(old_multi, new_multi)
s = s.replace(old_single, new_single)
p.write_text(s)
print("[PATCH] estimator.py now builds per-user LS-error features from broadcasted err_btfnu.")
PY

# Restore generated-output ignores. These were intentionally removed to push logs,
# but they should not remain unignored for normal runs.
python - <<'PY'
from pathlib import Path
p = Path(".gitignore")
s = p.read_text() if p.exists() else ""
needed = [
    "optuna/",
    "checkpoints/",
    "artifacts/",
    "TWC_plots_comprehensive/",
    "logs/",
    "outputs/",
    "plots/",
    "metrics/",
    "*.out",
    "*.err",
    "*.log",
    "patch_backups/",
]
lines = s.splitlines()
for item in needed:
    if item not in lines:
        lines.append(item)
p.write_text("\n".join(lines).rstrip() + "\n")
print("[PATCH] .gitignore contains runtime-output ignore rules.")
PY

# Remove failed Stage-A evidence for the smart prefix unless explicitly kept.
if [[ "${UPAIR_KEEP_FAILED_STAGEA:-0}" != "1" ]]; then
  prefix="${UPAIR_OPTUNA_STAGEA_PREFIX:-clean_b32_prb8_d256_40k_smart_u34610_1dmrs_stageA}"
  echo "[PATCH] Removing failed Stage-A artifacts for prefix: ${prefix}"
  rm -f "optuna/${prefix}"_*.db
  rm -rf "optuna/runs_1dmrs/${prefix}"_*
  rm -f "logs/optuna/${prefix}"_*.out "logs/optuna/${prefix}"_*.err
  rm -f logs/submit/stageA_*.sbatch
fi

echo "[PATCH] Done. Run: bash upair_probe_stageA_batch_shape_fix.sh"
