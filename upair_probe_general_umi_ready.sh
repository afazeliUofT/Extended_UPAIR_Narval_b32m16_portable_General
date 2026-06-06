#!/usr/bin/env bash
# Static/minimal probe for the General UMi repo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
[[ "${ROOT}" == *"Extended_UPAIR_Narval_b32m16_portable_General"* ]] || { echo "[FAIL] not in General repo copy: ${ROOT}" >&2; exit 1; }
python - <<'PY'
from pathlib import Path
import yaml, sys
fail = False
def ok(msg): print("[OK]", msg)
def bad(msg):
    global fail
    print("[FAIL]", msg, file=sys.stderr)
    fail = True
cfg = yaml.safe_load(Path("configs/twc_comprehensive_mu32_base.yaml").read_text())
ch = cfg.get("channel", {})
if str(ch.get("family","")).lower() == "umi" and str(ch.get("model","")).lower() == "umi": ok("config channel.family/model are UMi")
else: bad(f"config channel not UMi: family={ch.get('family')} model={ch.get('model')}")
umi = ch.get("umi", {})
for key in ["scenario","o2i_model","randomize_topology_each_batch","bs_array_rows","bs_array_cols"]:
    ok(f"channel.umi.{key} exists: {umi[key]!r}") if key in umi else bad(f"missing channel.umi.{key}")
ok("8-RB grid preserved") if cfg["pusch"]["n_size_grid"] == 8 and cfg["pusch"]["n_size_bwp"] == 8 else bad("8-RB grid not preserved")
ok("true-DMRS/per-user feature settings preserved") if cfg["model"]["pilot_mask_mode"] == "per_stream" and cfg["model"]["error_feature_mode"] == "per_user" else bad("feature settings changed unexpectedly")
builders = Path("src/upair5g/builders.py").read_text()
for token in ["TopologyRefreshingOFDMChannel", "extract_true_dmrs_mask_per_stream", "gen_single_sector_topology", 'family in {"umi"']:
    ok(f"builders.py contains {token}") if token in builders else bad(f"builders.py missing {token}")
for folder in ["TWC_plots_comprehensive", "_smoke_true_dmrs_runtime"]:
    ok(f"generated folder absent: {folder}") if not Path(folder).exists() else bad(f"generated folder still exists: {folder}")
old_dbs = list(Path(".").glob("optuna/*stageA*.db")) + list(Path(".").glob("optuna/*stageB*.db"))
ok("no old Optuna DBs present") if not old_dbs else bad("Optuna DBs still exist after UMi cleanup: " + ", ".join(str(p) for p in old_dbs[:5]))
stagea = Path("upair_submit_stageA_all.sh").read_text()
stageb = Path("upair_submit_stageB_all.sh").read_text()
train = Path("upair_submit_train_eval_all.sh").read_text()
ok("Stage A/B/train-eval wrapper defaults use UMi-specific prefixes") if "trueDMRS_UMi" in stagea and "trueDMRS_UMi" in stageb and "trueDMRS_UMi" in train else bad("some wrapper defaults do not contain trueDMRS_UMi")
raise SystemExit(1 if fail else 0)
PY
source "${ROOT}/upair_portable_env.sh"
upair_ensure_venv
python - <<'PY'
from pathlib import Path
failed = False
for root in ["src", "scripts"]:
    for path in Path(root).rglob("*.py"):
        try: compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            print(f"[SYNTAX-FAIL] {path}: {exc}")
            failed = True
raise SystemExit(1 if failed else 0)
PY
echo "[OK] Python syntax compile passed without writing bytecode"
echo "[PROBE] PASSED General UMi static/minimal probe"
