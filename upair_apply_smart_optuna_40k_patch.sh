#!/usr/bin/env bash
# Patch portable UPAIR repo for 40k final training and a bounded/effective two-stage Optuna plan.
# Run from /home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable
set -euo pipefail

ROOT="$(pwd)"
if [[ ! -d "${ROOT}/src" || ! -d "${ROOT}/scripts" || ! -d "${ROOT}/configs" ]]; then
  echo "[PATCH] Please run this from the UPAIR repository root." >&2
  exit 1
fi

python - <<'PY'
from __future__ import annotations
from pathlib import Path
import re

root = Path.cwd()

def read(path: str) -> str:
    return (root / path).read_text(encoding="utf-8")

def write(path: str, text: str) -> None:
    (root / path).write_text(text, encoding="utf-8")

def replace_once(text: str, old: str, new: str, path: str) -> str:
    if old not in text:
        raise SystemExit(f"[PATCH] Could not find expected text in {path}: {old!r}")
    return text.replace(old, new, 1)

# ---------------------------------------------------------------------
# 1) Final training config: 40k steps; keep PRB=8 and patched feature modes.
# ---------------------------------------------------------------------
cfg_path = "configs/twc_comprehensive_mu32_base.yaml"
cfg = read(cfg_path)
cfg = re.sub(r"(?m)^(\s*)steps:\s*20000\s*$", r"\1steps: 40000", cfg, count=1)
cfg = re.sub(r"(?m)^(\s*)learning_rate_decay_steps:\s*20000\s*$", r"\1learning_rate_decay_steps: 40000", cfg, count=1)
# Reduce validation overhead for 40k final training while keeping enough validation points.
cfg = re.sub(r"(?m)^(\s*)eval_every:\s*1000\s*$", r"\1eval_every: 2000", cfg, count=1)
# Align periodic checkpointing with validation/logging cadence for the longer run.
cfg = re.sub(r"(?m)^(\s*)checkpoint_every:\s*500\s*$", r"\1checkpoint_every: 1000", cfg, count=1)
write(cfg_path, cfg)

# ---------------------------------------------------------------------
# 2) Optuna common defaults: compact Stage A, promotion-style Stage B, 40k Stage C fallback.
# ---------------------------------------------------------------------
common_path = "scripts/optuna_1dmrs_common.py"
common = read(common_path)
old = '''STAGE_DEFAULTS: dict[str, dict[str, int]] = {
    "A": {"steps": 6000, "target_total_trials": 30, "source_top_k": 0},
    "B": {"steps": 10000, "target_total_trials": 8, "source_top_k": 8},
    "C": {"steps": 20000, "target_total_trials": 3, "source_top_k": 3},
}
'''
new = '''STAGE_DEFAULTS: dict[str, dict[str, int]] = {
    # Smart bounded schedule for the PRB8/d256 package:
    # Stage A explores broadly but cheaply; Stage B re-runs the best candidates longer.
    "A": {"steps": 4000, "target_total_trials": 20, "source_top_k": 0},
    "B": {"steps": 12000, "target_total_trials": 6, "source_top_k": 6},
    "C": {"steps": 40000, "target_total_trials": 3, "source_top_k": 3},
}
'''
common = replace_once(common, old, new, common_path)
common = common.replace('parser.add_argument("--tpe-startup-trials", type=int, default=12)', 'parser.add_argument("--tpe-startup-trials", type=int, default=8)')
common = common.replace('parser.add_argument("--pruner-startup-trials", type=int, default=10)', 'parser.add_argument("--pruner-startup-trials", type=int, default=6)')
common = common.replace('parser.add_argument("--pruner-min-trials", type=int, default=5)', 'parser.add_argument("--pruner-min-trials", type=int, default=4)')
write(common_path, common)

# ---------------------------------------------------------------------
# 3) Stage-A wrapper defaults and extra smart controls.
# ---------------------------------------------------------------------
stage_a_path = "upair_submit_stageA_all.sh"
s = read(stage_a_path)
s = s.replace('PREFIX="${UPAIR_OPTUNA_STAGEA_PREFIX:-clean_b32_prb8_d256_u34610_1dmrs_stageA}"',
              'PREFIX="${UPAIR_OPTUNA_STAGEA_PREFIX:-clean_b32_prb8_d256_40k_smart_u34610_1dmrs_stageA}"')
s = s.replace('TRIALS="${UPAIR_OPTUNA_STAGEA_TRIALS:-30}"', 'TRIALS="${UPAIR_OPTUNA_STAGEA_TRIALS:-20}"')
s = s.replace('STEPS="${UPAIR_OPTUNA_STAGEA_STEPS:-6000}"', 'STEPS="${UPAIR_OPTUNA_STAGEA_STEPS:-4000}"')
s = s.replace('MAX_ATTEMPTS="${UPAIR_OPTUNA_STAGEA_MAX_ATTEMPTS:-${TRIALS}}"', 'MAX_ATTEMPTS="${UPAIR_OPTUNA_STAGEA_MAX_ATTEMPTS:-$((TRIALS + 10))}"')
s = s.replace('TIME_LIMIT="${UPAIR_TIME_STAGE_A:-12:00:00}"', 'TIME_LIMIT="${UPAIR_TIME_STAGE_A:-30:00:00}"')
insert_after = '''VAL_MB="${UPAIR_VAL_MICROBATCH:-16}"
'''
insert_vars = '''VAL_MB="${UPAIR_VAL_MICROBATCH:-16}"
TPE_STARTUP="${UPAIR_OPTUNA_TPE_STARTUP_TRIALS:-8}"
PRUNER_STARTUP="${UPAIR_OPTUNA_PRUNER_STARTUP_TRIALS:-6}"
PRUNER_MIN_TRIALS="${UPAIR_OPTUNA_PRUNER_MIN_TRIALS:-4}"
OBJECTIVE_RECENT_K="${UPAIR_OPTUNA_OBJECTIVE_RECENT_K:-2}"
OBJECTIVE_MIN_STEP="${UPAIR_OPTUNA_OBJECTIVE_MIN_STEP:-1000}"
'''
s = replace_once(s, insert_after, insert_vars, stage_a_path)
old_tail = '''  --validation-batch-size "${VAL_B}" \\
  --validation-microbatch-size "${VAL_MB}" \\
  --seed "${SEED}"
'''
new_tail = '''  --validation-batch-size "${VAL_B}" \\
  --validation-microbatch-size "${VAL_MB}" \\
  --seed "${SEED}" \\
  --tpe-startup-trials "${TPE_STARTUP}" \\
  --pruner-startup-trials "${PRUNER_STARTUP}" \\
  --pruner-min-trials "${PRUNER_MIN_TRIALS}" \\
  --objective-recent-k "${OBJECTIVE_RECENT_K}" \\
  --objective-min-step "${OBJECTIVE_MIN_STEP}"
'''
s = replace_once(s, old_tail, new_tail, stage_a_path)
write(stage_a_path, s)

# ---------------------------------------------------------------------
# 4) Stage-B wrapper defaults and extra smart controls.
# ---------------------------------------------------------------------
stage_b_path = "upair_submit_stageB_all.sh"
s = read(stage_b_path)
s = s.replace('A_PREFIX="${UPAIR_OPTUNA_STAGEA_PREFIX:-clean_b32_prb8_d256_u34610_1dmrs_stageA}"',
              'A_PREFIX="${UPAIR_OPTUNA_STAGEA_PREFIX:-clean_b32_prb8_d256_40k_smart_u34610_1dmrs_stageA}"')
s = s.replace('B_PREFIX="${UPAIR_OPTUNA_STAGEB_PREFIX:-clean_b32_prb8_d256_u34610_1dmrs_stageB}"',
              'B_PREFIX="${UPAIR_OPTUNA_STAGEB_PREFIX:-clean_b32_prb8_d256_40k_smart_u34610_1dmrs_stageB}"')
s = s.replace('TRIALS="${UPAIR_OPTUNA_STAGEB_TRIALS:-8}"', 'TRIALS="${UPAIR_OPTUNA_STAGEB_TRIALS:-6}"')
s = s.replace('STEPS="${UPAIR_OPTUNA_STAGEB_STEPS:-10000}"', 'STEPS="${UPAIR_OPTUNA_STAGEB_STEPS:-12000}"')
s = s.replace('SOURCE_TOP_K="${UPAIR_OPTUNA_STAGEB_SOURCE_TOP_K:-8}"', 'SOURCE_TOP_K="${UPAIR_OPTUNA_STAGEB_SOURCE_TOP_K:-6}"')
s = s.replace('MAX_ATTEMPTS="${UPAIR_OPTUNA_STAGEB_MAX_ATTEMPTS:-${TRIALS}}"', 'MAX_ATTEMPTS="${UPAIR_OPTUNA_STAGEB_MAX_ATTEMPTS:-$((TRIALS + 4))}"')
s = s.replace('TIME_LIMIT="${UPAIR_TIME_STAGE_B:-12:00:00}"', 'TIME_LIMIT="${UPAIR_TIME_STAGE_B:-30:00:00}"')
insert_after = '''VAL_MB="${UPAIR_VAL_MICROBATCH:-16}"
'''
insert_vars = '''VAL_MB="${UPAIR_VAL_MICROBATCH:-16}"
TPE_STARTUP="${UPAIR_OPTUNA_STAGEB_TPE_STARTUP_TRIALS:-4}"
PRUNER_STARTUP="${UPAIR_OPTUNA_STAGEB_PRUNER_STARTUP_TRIALS:-4}"
PRUNER_MIN_TRIALS="${UPAIR_OPTUNA_STAGEB_PRUNER_MIN_TRIALS:-3}"
OBJECTIVE_RECENT_K="${UPAIR_OPTUNA_STAGEB_OBJECTIVE_RECENT_K:-3}"
OBJECTIVE_MIN_STEP="${UPAIR_OPTUNA_STAGEB_OBJECTIVE_MIN_STEP:-2000}"
'''
s = replace_once(s, insert_after, insert_vars, stage_b_path)
old_tail = '''  --validation-batch-size "${VAL_B}" \\
  --validation-microbatch-size "${VAL_MB}" \\
  --seed "${SEED}"
'''
new_tail = '''  --validation-batch-size "${VAL_B}" \\
  --validation-microbatch-size "${VAL_MB}" \\
  --seed "${SEED}" \\
  --tpe-startup-trials "${TPE_STARTUP}" \\
  --pruner-startup-trials "${PRUNER_STARTUP}" \\
  --pruner-min-trials "${PRUNER_MIN_TRIALS}" \\
  --objective-recent-k "${OBJECTIVE_RECENT_K}" \\
  --objective-min-step "${OBJECTIVE_MIN_STEP}"
'''
s = replace_once(s, old_tail, new_tail, stage_b_path)
write(stage_b_path, s)

# ---------------------------------------------------------------------
# 5) Final train/eval and Stage-B probe use the new Stage-B prefix and longer walltime.
# ---------------------------------------------------------------------
train_eval_path = "upair_submit_train_eval_all.sh"
s = read(train_eval_path)
s = s.replace('B_PREFIX="${UPAIR_OPTUNA_STAGEB_PREFIX:-clean_b32_prb8_d256_u34610_1dmrs_stageB}"',
              'B_PREFIX="${UPAIR_OPTUNA_STAGEB_PREFIX:-clean_b32_prb8_d256_40k_smart_u34610_1dmrs_stageB}"')
s = s.replace('TIME_LIMIT="${UPAIR_TIME_TRAIN_EVAL:-12:00:00}"', 'TIME_LIMIT="${UPAIR_TIME_TRAIN_EVAL:-30:00:00}"')
write(train_eval_path, s)

probe_b_path = "upair_probe_after_stageB.sh"
s = read(probe_b_path)
s = s.replace('B_PREFIX="${UPAIR_OPTUNA_STAGEB_PREFIX:-clean_b32_prb8_d256_u34610_1dmrs_stageB}"',
              'B_PREFIX="${UPAIR_OPTUNA_STAGEB_PREFIX:-clean_b32_prb8_d256_40k_smart_u34610_1dmrs_stageB}"')
s = s.replace('A_PREFIX="${UPAIR_OPTUNA_STAGEA_PREFIX:-clean_b32_prb8_d256_u34610_1dmrs_stageA}"',
              'A_PREFIX="${UPAIR_OPTUNA_STAGEA_PREFIX:-clean_b32_prb8_d256_40k_smart_u34610_1dmrs_stageA}"')
write(probe_b_path, s)

# ---------------------------------------------------------------------
# 6) Restore ignore rules removed in baca3d9 and ignore/remove patch_backups.
# ---------------------------------------------------------------------
gitignore_path = ".gitignore"
gi = read(gitignore_path) if (root / gitignore_path).exists() else ""
needed = [
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
if not gi.endswith("\n"):
    gi += "\n"
missing = [x for x in needed if x not in gi.splitlines()]
if missing:
    gi += "\n# UPAIR generated runtime outputs and patch scratch\n" + "\n".join(missing) + "\n"
write(gitignore_path, gi)

print("[PATCH] Updated config, Optuna defaults/wrappers, train/eval prefix, Stage-B probe, and .gitignore.")
PY

chmod +x \
  upair_submit_stageA_all.sh \
  upair_submit_stageB_all.sh \
  upair_submit_train_eval_all.sh \
  upair_probe_after_stageB.sh \
  upair_probe_clean_start.sh 2>/dev/null || true

if [[ "${UPAIR_KEEP_PATCH_BACKUPS:-0}" != "1" && -d patch_backups ]]; then
  echo "[PATCH] Removing local patch_backups/ because it only contains pre-patch snapshots. Set UPAIR_KEEP_PATCH_BACKUPS=1 to keep it."
  rm -rf patch_backups
fi

echo "[PATCH] Done. Run: bash upair_probe_smart_optuna_40k.sh"
