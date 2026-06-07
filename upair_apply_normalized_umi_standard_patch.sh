#!/usr/bin/env bash
# Revert General repo to a standard normalized UMi link-level model.
# Keeps true-DMRS/per-user features, 8-RB grid, 8 architecture variants including prompt-MLP ablation,
# and permanent smart Optuna defaults. Removes stale runtime evidence.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if [[ "${ROOT}" != *"Extended_UPAIR_Narval_b32m16_portable_General"* && "${UPAIR_ALLOW_NON_GENERAL_PATCH:-0}" != "1" ]]; then
  echo "[NORM-UMI-PATCH] Refusing to patch outside the General repo copy." >&2
  echo "[NORM-UMI-PATCH] ROOT=${ROOT}" >&2
  exit 1
fi

[[ -f configs/twc_comprehensive_mu32_base.yaml && -f src/upair5g/builders.py && -f scripts/optuna_1dmrs_common.py ]] || {
  echo "[NORM-UMI-PATCH] Run from repo root." >&2
  exit 1
}

python - <<'PY'
from __future__ import annotations
from pathlib import Path
import re
import yaml

A_PREFIX = "clean_b32_prb8_d256_40k_smart_trueDMRS_UMiNorm_u34610_1dmrs_stageA"
B_PREFIX = "clean_b32_prb8_d256_40k_smart_trueDMRS_UMiNorm_u34610_1dmrs_stageB"

# ------------------------------------------------------------------
# 1) Config: standard normalized UMi.
# ------------------------------------------------------------------
cfg_path = Path("configs/twc_comprehensive_mu32_base.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

cfg.setdefault("experiment", {})["name"] = "rx16_prb8_umi_norm_trueDMRS_1dmrs_u3"

ch = cfg.setdefault("channel", {})
ch["family"] = "umi"
ch["model"] = "umi"
ch.setdefault("cdl_model", "C")
ch["normalize_channel"] = True
ch.setdefault("num_rx_ant", 16)
ch.setdefault("num_tx_ant", 1)
ch.setdefault("min_speed_mps", 8.33)
ch.setdefault("max_speed_mps", 16.67)

umi = ch.setdefault("umi", {})
umi.update({
    "scenario": "umi",
    "o2i_model": "low",
    "enable_pathloss": False,
    "enable_shadow_fading": False,
    "always_generate_lsp": False,
    "randomize_topology_each_batch": True,
    "antenna_pattern_bs": "38.901",
    "antenna_pattern_ut": "omni",
    "bs_array_rows": 4,
    "bs_array_cols": 4,
    "ut_array_rows": 1,
    "ut_array_cols": 1,
    "min_bs_ut_dist": None,
    "isd": None,
    "bs_height": None,
    "min_ut_height": None,
    "max_ut_height": None,
    "indoor_probability": None,
    "min_ut_velocity_mps": float(ch.get("min_speed_mps", 8.33)),
    "max_ut_velocity_mps": float(ch.get("max_speed_mps", 16.67)),
})

# Keep near_far section but disable it explicitly so old UMiPC code paths cannot fire.
nf = cfg.setdefault("near_far", {})
nf.update({
    "enabled": False,
    "mode": "disabled",
    "alpha_train_min": 1.0,
    "alpha_train_max": 1.0,
    "alpha_eval": 1.0,
    "alpha_sampling": "fixed",
    "epsilon": 1.0e-20,
    "log_stats": False,
    "headroom_db": None,
})

# Preserve current architecture setup.
cfg.setdefault("model", {})["prompt_mlp_ratio"] = 1.0
cfg["model"]["pilot_mask_mode"] = "per_stream"
cfg["model"]["error_feature_mode"] = "per_user"
cfg["model"]["error_variance_update"] = "multiplicative"

# Keep covariance cache distinct from CDL-C and UMiPC.
cfg.setdefault("baselines", {}).setdefault("covariance_estimation", {})["cache_name"] = "empirical_covariances_umi_norm.npz"

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("[NORM-UMI-PATCH] Config set to normalized UMi: pathloss/shadow off, normalize_channel=true, near_far disabled.")

# ------------------------------------------------------------------
# 2) Ensure topology helper path is compatible with Sionna 1.2.1.
# ------------------------------------------------------------------
bp = Path("src/upair5g/builders.py")
b = bp.read_text()
b = b.replace(
    'resolve_attr(["sionna.phy.channel.tr38901", "sionna.channel.tr38901"], "gen_single_sector_topology")',
    'resolve_attr(["sionna.phy.channel", "sionna.phy.channel.tr38901"], "gen_single_sector_topology")',
)
b = b.replace(
    'resolve_attr(\n            ["sionna.phy.channel.tr38901", "sionna.channel.tr38901"],\n            "gen_single_sector_topology",\n        )',
    'resolve_attr(\n            ["sionna.phy.channel", "sionna.phy.channel.tr38901"],\n            "gen_single_sector_topology",\n        )',
)
bp.write_text(b)
print("[NORM-UMI-PATCH] builders.py topology helper lookup set to sionna.phy.channel first.")

# ------------------------------------------------------------------
# 3) Replace UMiPC/UMi old prefixes with UMiNorm prefixes in wrappers/probes.
# ------------------------------------------------------------------
prefix_repls = {
    "clean_b32_prb8_d256_40k_smart_trueDMRS_UMiPC_u34610_1dmrs_stageA": A_PREFIX,
    "clean_b32_prb8_d256_40k_smart_trueDMRS_UMiPC_u34610_1dmrs_stageB": B_PREFIX,
    "clean_b32_prb8_d256_40k_smart_trueDMRS_UMi_u34610_1dmrs_stageA": A_PREFIX,
    "clean_b32_prb8_d256_40k_smart_trueDMRS_UMi_u34610_1dmrs_stageB": B_PREFIX,
    "clean_b32_prb8_d256_40k_smart_u34610_1dmrs_stageA": A_PREFIX,
    "clean_b32_prb8_d256_40k_smart_u34610_1dmrs_stageB": B_PREFIX,
}
for path in [
    "upair_submit_stageA_all.sh",
    "upair_submit_stageB_all.sh",
    "upair_submit_train_eval_all.sh",
    "upair_probe_after_stageB.sh",
    "upair_probe_after_train_eval.sh",
    "upair_probe_clean_start.sh",
    "upair_probe_smart_optuna_40k.sh",
    "upair_probe_general_umi_ready.sh",
]:
    p = Path(path)
    if not p.exists():
        continue
    text = p.read_text()
    for old, new in prefix_repls.items():
        text = text.replace(old, new)
    p.write_text(text)

# Permanent smart Optuna defaults remain.
def replace_line(text: str, prefix: str, newline: str) -> str:
    pattern = re.compile(rf'^{re.escape(prefix)}.*$', re.M)
    if pattern.search(text):
        return pattern.sub(newline, text, count=1)
    return text

p = Path("upair_submit_stageA_all.sh")
if p.exists():
    s = p.read_text()
    s = replace_line(s, 'TRIALS="${UPAIR_OPTUNA_STAGEA_TRIALS:-', 'TRIALS="${UPAIR_OPTUNA_STAGEA_TRIALS:-24}"')
    s = replace_line(s, 'STEPS="${UPAIR_OPTUNA_STAGEA_STEPS:-', 'STEPS="${UPAIR_OPTUNA_STAGEA_STEPS:-4000}"')
    s = replace_line(s, 'MAX_ATTEMPTS="${UPAIR_OPTUNA_STAGEA_MAX_ATTEMPTS:-', 'MAX_ATTEMPTS="${UPAIR_OPTUNA_STAGEA_MAX_ATTEMPTS:-36}"')
    s = replace_line(s, 'TIME_LIMIT="${UPAIR_TIME_STAGE_A:-', 'TIME_LIMIT="${UPAIR_TIME_STAGE_A:-30:00:00}"')
    p.write_text(s)

p = Path("upair_submit_stageB_all.sh")
if p.exists():
    s = p.read_text()
    s = replace_line(s, 'TRIALS="${UPAIR_OPTUNA_STAGEB_TRIALS:-', 'TRIALS="${UPAIR_OPTUNA_STAGEB_TRIALS:-10}"')
    s = replace_line(s, 'STEPS="${UPAIR_OPTUNA_STAGEB_STEPS:-', 'STEPS="${UPAIR_OPTUNA_STAGEB_STEPS:-12000}"')
    s = replace_line(s, 'SOURCE_TOP_K="${UPAIR_OPTUNA_STAGEB_SOURCE_TOP_K:-', 'SOURCE_TOP_K="${UPAIR_OPTUNA_STAGEB_SOURCE_TOP_K:-8}"')
    s = replace_line(s, 'MAX_ATTEMPTS="${UPAIR_OPTUNA_STAGEB_MAX_ATTEMPTS:-', 'MAX_ATTEMPTS="${UPAIR_OPTUNA_STAGEB_MAX_ATTEMPTS:-16}"')
    s = replace_line(s, 'TIME_LIMIT="${UPAIR_TIME_STAGE_B:-', 'TIME_LIMIT="${UPAIR_TIME_STAGE_B:-30:00:00}"')
    p.write_text(s)

# ------------------------------------------------------------------
# 4) Clean stale runtime evidence from UMiPC/smoke/Optuna.
# ------------------------------------------------------------------
import shutil
for folder in ["optuna", "logs", "TWC_plots_comprehensive", "_smoke_umi_runtime", "_smoke_umi_pc_runtime", "_smoke_true_dmrs_runtime"]:
    path = Path(folder)
    if path.exists():
        shutil.rmtree(path)
Path("optuna").mkdir(exist_ok=True)
Path("logs/optuna").mkdir(parents=True, exist_ok=True)
Path("logs/submit").mkdir(parents=True, exist_ok=True)
Path("logs/train_eval").mkdir(parents=True, exist_ok=True)
Path("logs/smoke").mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 5) Update gitignore.
# ------------------------------------------------------------------
gp = Path(".gitignore")
gs = gp.read_text() if gp.exists() else ""
items = [
    "optuna/", "logs/", "TWC_plots_comprehensive/", "_smoke_*/", "patch_backups/",
    "*.out", "*.err", "*.log", "*.db", "*.sqlite", "*.sqlite3",
    "__pycache__/", "*.py[cod]", "*.weights.h5", "*.data-*",
]
lines = gs.splitlines()
for item in items:
    if item not in lines:
        lines.append(item)
gp.write_text("\n".join(lines).rstrip() + "\n")

print("[NORM-UMI-PATCH] Prefixes set to UMiNorm and runtime evidence removed.")
PY

echo "[NORM-UMI-PATCH] Done. Next:"
echo "  bash upair_probe_normalized_umi_ready.sh"
echo "  bash upair_probe_normalized_umi_runtime.sh"
