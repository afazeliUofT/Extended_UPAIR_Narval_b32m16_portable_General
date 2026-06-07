#!/usr/bin/env bash
# Static/minimal probe for the standard normalized UMi link-level model.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/upair_portable_env.sh"
upair_ensure_venv

PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from __future__ import annotations
from pathlib import Path
import importlib.util
import sys
import yaml

fail = False
def ok(msg): print("[OK]", msg)
def bad(msg):
    global fail
    print("[FAIL]", msg, file=sys.stderr)
    fail = True

cfg = yaml.safe_load(Path("configs/twc_comprehensive_mu32_base.yaml").read_text())
ch = cfg.get("channel", {})
umi = ch.get("umi", {})
nf = cfg.get("near_far", {})

if str(ch.get("family", "")).lower() == "umi" and str(ch.get("model", "")).lower() == "umi":
    ok("channel.family/model are UMi")
else:
    bad(f"channel family/model wrong: {ch.get('family')} {ch.get('model')}")

if ch.get("normalize_channel") is True:
    ok("channel.normalize_channel is true")
else:
    bad(f"channel.normalize_channel should be true, got {ch.get('normalize_channel')}")

if umi.get("enable_pathloss") is False and umi.get("enable_shadow_fading") is False:
    ok("UMi pathloss and shadow fading are disabled")
else:
    bad(f"pathloss/shadow should be false/false, got {umi.get('enable_pathloss')} {umi.get('enable_shadow_fading')}")

if nf.get("enabled") is False:
    ok("near_far is explicitly disabled")
else:
    bad(f"near_far.enabled should be false, got {nf.get('enabled')}")

if cfg["pusch"]["n_size_grid"] == 8 and cfg["pusch"]["n_size_bwp"] == 8:
    ok("8-RB grid preserved")
else:
    bad("8-RB grid not preserved")

if cfg["model"]["pilot_mask_mode"] == "per_stream" and cfg["model"]["error_feature_mode"] == "per_user":
    ok("true-DMRS/per-user feature settings preserved")
else:
    bad("feature settings changed unexpectedly")

if cfg["baselines"]["covariance_estimation"]["cache_name"] == "empirical_covariances_umi_norm.npz":
    ok("normalized UMi covariance cache name is distinct")
else:
    bad("covariance cache name is not empirical_covariances_umi_norm.npz")

builders = Path("src/upair5g/builders.py").read_text()
for token in ["TopologyRefreshingOFDMChannel", "sionna.phy.channel", "gen_single_sector_topology"]:
    if token in builders:
        ok(f"builders.py contains {token}")
    else:
        bad(f"builders.py missing {token}")

if '["sionna.phy.channel.tr38901", "sionna.channel.tr38901"], "gen_single_sector_topology"' in builders:
    bad("builders.py still contains old bad one-line topology lookup")
else:
    ok("old bad topology lookup is absent")

# Optuna variants and prompt MLP ablation.
spec = importlib.util.spec_from_file_location("optuna_common_probe", "scripts/optuna_1dmrs_common.py")
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
variants = dict(mod.VARIANTS)

if "promptmlp_d256_b4_r2_pr2" in variants:
    ok("prompt-MLP-ratio ablation is still present")
else:
    bad("prompt-MLP-ratio ablation missing")

if mod.STAGE_DEFAULTS["A"]["target_total_trials"] == 24 and mod.STAGE_DEFAULTS["A"]["steps"] == 4000:
    ok("Stage A common defaults remain 24 trials x 4000 steps")
else:
    bad(f"unexpected Stage A defaults: {mod.STAGE_DEFAULTS['A']}")

if mod.STAGE_DEFAULTS["B"]["target_total_trials"] == 10 and mod.STAGE_DEFAULTS["B"]["source_top_k"] == 8 and mod.STAGE_DEFAULTS["B"]["steps"] == 12000:
    ok("Stage B common defaults remain 10 trials, top-8, 12000 steps")
else:
    bad(f"unexpected Stage B defaults: {mod.STAGE_DEFAULTS['B']}")

for wrapper in ["upair_submit_stageA_all.sh", "upair_submit_stageB_all.sh", "upair_submit_train_eval_all.sh"]:
    text = Path(wrapper).read_text()
    if "trueDMRS_UMiNorm" in text:
        ok(f"{wrapper} uses UMiNorm prefix")
    else:
        bad(f"{wrapper} does not use UMiNorm prefix")
    if "trueDMRS_UMiPC" in text:
        bad(f"{wrapper} still contains UMiPC prefix")

# No old Optuna DBs should exist after patch.
dbs = list(Path("optuna").glob("*.db")) if Path("optuna").exists() else []
if not dbs:
    ok("no old Optuna DBs remain")
else:
    bad("old Optuna DBs remain: " + ", ".join(str(p) for p in dbs[:5]))

# Syntax compile.
for root in ["src", "scripts"]:
    for path in Path(root).rglob("*.py"):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            bad(f"syntax error in {path}: {exc}")

if fail:
    raise SystemExit("[PROBE] FAILED normalized UMi static probe")
print("[PROBE] PASSED normalized UMi static/minimal probe")
PY
