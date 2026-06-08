#!/usr/bin/env bash
# Apply CDL-C-grade memory-safe, resumable, process-isolated training/eval pipeline
# to the normalized-UMi General repo.
#
# IMPORTANT: This script intentionally does NOT edit .gitignore.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

[[ -f configs/twc_comprehensive_mu32_base.yaml && -d src/upair5g && -f scripts/run_comprehensive_mu32_ablation.py ]] || {
  echo "[UMINORM-MASTER-PATCH] Run from repo root." >&2
  exit 1
}

python - <<'PY'
from pathlib import Path
import yaml

cfg_path = Path("configs/twc_comprehensive_mu32_base.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

# Keep the normalized UMi model explicit and protected.
ch = cfg.setdefault("channel", {})
ch["family"] = "umi"
ch["model"] = "umi"
ch["normalize_channel"] = True
umi = ch.setdefault("umi", {})
umi["enable_pathloss"] = False
umi["enable_shadow_fading"] = False
umi["randomize_topology_each_batch"] = True

nf = cfg.setdefault("near_far", {})
nf["enabled"] = False
nf["mode"] = "disabled"

# Preserve Optuna-compatible training/validation batch policy.
sys = cfg.setdefault("system", {})
sys["batch_size_train"] = 32
sys["batch_size_eval"] = 32

tr = cfg.setdefault("training", {})
tr["steps"] = 40000
tr["resume"] = True
tr["checkpoint_every"] = 1000
tr["eval_every"] = 2000
tr["val_microbatch_size"] = 16
tr["val_memory_cleanup_every_microbatch"] = True
tr["val_memory_cleanup_every_batch"] = True
tr["memory_cleanup_after_validation"] = True
tr["memory_cleanup_every_steps"] = 100

# Final evaluation policy: BLER-only, no long-lived compiled receiver graph,
# process-isolated chunks will control total Monte Carlo depth.
ev = cfg.setdefault("evaluation", {})
ev["logical_batch_size"] = 64
ev["receiver_microbatch_size"] = 8
ev["stream_eval_microbatches"] = True
ev["compiled_receiver_error_counts"] = False
ev["receiver_call_jit_compile"] = False
ev["num_batches_per_point"] = 512
ev["min_num_batches_per_point"] = 16
ev["max_num_batches_per_point"] = 2000
ev["target_block_errors_per_receiver"] = 100
ev["reliable_min_block_errors"] = 100
ev["reliable_min_bit_errors"] = 1000
ev["save_example_batch"] = False
ev["nmse_receivers"] = []
ev["memory_cleanup_every_batches"] = 1
ev["memory_cleanup_every_microbatch"] = True
ev["per_receiver_stopping"] = True
ev["target_bler_floor"] = 1.0e-5
ev["log_latency"] = True
ev["log_gpu_memory"] = True

# Keep UMi-normalized covariance cache distinct.
cfg.setdefault("baselines", {}).setdefault("covariance_estimation", {})["cache_name"] = "empirical_covariances_umi_norm.npz"

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("[UMINORM-MASTER-PATCH] Config updated: normalized UMi + memory-safe BLER eval policy.")
PY

python - <<'PY'
from pathlib import Path

p = Path("scripts/run_comprehensive_mu32_ablation.py")
s = p.read_text()

old = 'set_cfg(cfg, "evaluation.save_example_batch", variant_name == "main_d256_b4_r2" and num_users == 4)'
new = 'set_cfg(cfg, "evaluation.save_example_batch", bool(get_cfg(cfg, "evaluation.save_example_batch", False)))'
if old in s:
    s = s.replace(old, new)
    p.write_text(s)
    print("[UMINORM-MASTER-PATCH] Patched _eval_cfg to preserve evaluation.save_example_batch=False.")
elif new in s:
    print("[UMINORM-MASTER-PATCH] _eval_cfg already preserves evaluation.save_example_batch from config.")
else:
    print("[UMINORM-MASTER-PATCH][WARN] Did not find save_example_batch line to patch; probe will check.")
PY

mkdir -p scripts logs/submit logs/pipeline

cat > scripts/run_isolated_eval_chunk.py <<'PY'
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from upair5g.config import get_cfg, load_config, set_cfg  # noqa: E402
from upair5g.evaluation import evaluate_model  # noqa: E402
from scripts.run_comprehensive_mu32_ablation import (  # noqa: E402
    _apply_optuna_best_1dmrs,
    _eval_cfg,
    _variant_cfg,
)


def _safe_tag(x: Any) -> str:
    s = str(x)
    return s.replace("-", "m").replace("+", "p").replace(".", "p").replace(",", "_")


def _read_single_row(curves_path: Path, receiver: str, ebno: float, num_users: int) -> dict[str, Any]:
    df = pd.read_csv(curves_path)
    if "receiver" in df.columns:
        df = df[df["receiver"].astype(str) == str(receiver)]
    if "ebno_db" in df.columns:
        df = df[df["ebno_db"].astype(float) == float(ebno)]
    if "num_users" in df.columns:
        df = df[df["num_users"].astype(int) == int(num_users)]
    if len(df) != 1:
        raise RuntimeError(
            f"Expected one curve row for receiver={receiver}, ebno={ebno}, num_users={num_users}, "
            f"got {len(df)} rows from {curves_path}"
        )
    return df.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one normalized-UMi isolated evaluation chunk.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_base.yaml"))
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dmrs-case", default="1dmrs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-users", type=int, required=True)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--ebno-db", type=float, required=True)
    parser.add_argument("--chunk-idx", type=int, required=True)
    parser.add_argument("--chunk-batches", type=int, default=20)
    parser.add_argument("--receiver-microbatch-size", type=int, default=8)
    parser.add_argument("--stageb-prefix", default="clean_b32_prb8_d256_40k_smart_trueDMRS_UMiNorm_u34610_1dmrs_stageB")
    parser.add_argument("--optuna-dir", default=str(PROJECT_ROOT / "optuna"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "_isolated_eval_chunks"))
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    # Must be set before TensorFlow initializes in this process.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_CPP_VMODULE", "bfc_allocator=0")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    cfg = load_config(args.config)
    train_cfg = _variant_cfg(cfg, args.variant, args.dmrs_case, args.seed)
    _apply_optuna_best_1dmrs(
        train_cfg,
        args.variant,
        args.dmrs_case,
        storage_dir=args.optuna_dir,
        study_prefix=args.stageb_prefix,
        require_external=True,
    )

    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        PROJECT_ROOT
        / "TWC_plots_comprehensive"
        / "runs_rx16"
        / f"seed{args.seed}"
        / args.dmrs_case
        / args.variant
        / "checkpoints"
        / str(get_cfg(train_cfg, "training.checkpoint_name", "best.weights.h5"))
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for isolated eval: {checkpoint}")

    cfg_eval = _eval_cfg(train_cfg, args.variant, args.dmrs_case, args.num_users)

    # Distinct deterministic evaluation seed per chunk to avoid replaying the same Monte Carlo drops.
    base_eval_seed = int(get_cfg(cfg_eval, "system.evaluation_seed", args.seed + 1000))
    ebno_offset = int(round((float(args.ebno_db) + 100.0) * 1000.0))
    chunk_seed = base_eval_seed + 100003 * int(args.chunk_idx) + 17 * ebno_offset + 1009 * int(args.num_users)
    set_cfg(cfg_eval, "system.evaluation_seed", int(chunk_seed))
    set_cfg(cfg_eval, "system.seed", int(chunk_seed))

    tag = (
        f"{args.variant}_u{args.num_users}_{args.receiver}_"
        f"ebno{_safe_tag(args.ebno_db)}_chunk{args.chunk_idx:04d}_"
        f"m{args.receiver_microbatch_size}_b{args.chunk_batches}"
    )
    out_root = Path(args.output_root)
    set_cfg(cfg_eval, "experiment.output_root", str(out_root))
    set_cfg(cfg_eval, "experiment.name", tag)

    set_cfg(cfg_eval, "system.ebno_db_eval", [float(args.ebno_db)])
    set_cfg(cfg_eval, "baselines.enabled_receivers", [str(args.receiver)])

    # BLER-only, memory-safe isolated chunk settings.
    set_cfg(cfg_eval, "evaluation.nmse_receivers", [])
    set_cfg(cfg_eval, "evaluation.save_example_batch", False)
    set_cfg(cfg_eval, "evaluation.compiled_receiver_error_counts", False)
    set_cfg(cfg_eval, "evaluation.receiver_call_jit_compile", False)
    set_cfg(cfg_eval, "evaluation.receiver_microbatch_size", int(args.receiver_microbatch_size))
    set_cfg(cfg_eval, "evaluation.stream_eval_microbatches", True)
    set_cfg(cfg_eval, "evaluation.memory_cleanup_every_batches", 1)
    set_cfg(cfg_eval, "evaluation.memory_cleanup_every_microbatch", True)
    set_cfg(cfg_eval, "evaluation.min_num_batches_per_point", int(args.chunk_batches))
    set_cfg(cfg_eval, "evaluation.max_num_batches_per_point", int(args.chunk_batches))
    set_cfg(cfg_eval, "evaluation.target_block_errors_per_receiver", 0)
    set_cfg(cfg_eval, "evaluation.per_receiver_stopping", False)
    set_cfg(cfg_eval, "evaluation.force", True)
    set_cfg(cfg_eval, "evaluation.progress_every_batches", max(1, min(10, int(args.chunk_batches))))
    set_cfg(cfg_eval, "baselines.covariance_estimation.reuse_cache", True)

    print("[ISO-CHUNK] variant:", args.variant)
    print("[ISO-CHUNK] receiver:", args.receiver)
    print("[ISO-CHUNK] num_users:", args.num_users)
    print("[ISO-CHUNK] ebno_db:", args.ebno_db)
    print("[ISO-CHUNK] chunk_idx:", args.chunk_idx)
    print("[ISO-CHUNK] chunk_batches:", args.chunk_batches)
    print("[ISO-CHUNK] receiver_microbatch_size:", args.receiver_microbatch_size)
    print("[ISO-CHUNK] chunk_seed:", chunk_seed)
    print("[ISO-CHUNK] checkpoint:", checkpoint)
    print("[ISO-CHUNK] TF_GPU_ALLOCATOR:", os.environ.get("TF_GPU_ALLOCATOR"))
    print("[ISO-CHUNK] output tag:", tag)

    result = evaluate_model(cfg_eval, checkpoint_path=str(checkpoint), num_users=int(args.num_users))
    curves_path = Path(result["curves_path"])
    row = _read_single_row(curves_path, args.receiver, float(args.ebno_db), int(args.num_users))
    row.update(
        {
            "variant": args.variant,
            "dmrs_case": args.dmrs_case,
            "seed": int(args.seed),
            "training_seed": int(args.seed),
            "evaluation_seed": int(chunk_seed),
            "receiver": args.receiver,
            "num_users": int(args.num_users),
            "ebno_db": float(args.ebno_db),
            "chunk_idx": int(args.chunk_idx),
            "chunk_batches_requested": int(args.chunk_batches),
            "receiver_microbatch_size": int(args.receiver_microbatch_size),
            "checkpoint_path": str(checkpoint),
            "chunk_output_dir": str(result["output_dir"]),
            "chunk_curves_path": str(curves_path),
        }
    )

    out_dir = Path(result["output_dir"])
    pd.DataFrame([row]).to_csv(out_dir / "chunk_result.csv", index=False)
    with open(out_dir / "chunk_result.json", "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, sort_keys=True)
    print("[ISO-CHUNK] wrote:", out_dir / "chunk_result.csv")
    print("[ISO-CHUNK] DONE")


if __name__ == "__main__":
    main()
PY

cat > scripts/merge_isolated_eval_chunks.py <<'PY'
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUM_COLS = [
    "bit_errors",
    "num_bits",
    "block_errors",
    "num_blocks",
    "num_batches_run",
    "point_elapsed_s",
    "data_elapsed_s",
    "receiver_elapsed_s",
]


def _load_rows(input_root: Path) -> pd.DataFrame:
    paths = sorted(input_root.rglob("chunk_result.csv"))
    if not paths:
        raise FileNotFoundError(f"No chunk_result.csv files found under {input_root}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["chunk_result_path"] = str(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _merge_group(group: pd.DataFrame) -> dict[str, Any]:
    first = group.iloc[0].to_dict()
    out: dict[str, Any] = {
        "receiver": str(first["receiver"]),
        "variant": str(first.get("variant", "")),
        "dmrs_case": str(first.get("dmrs_case", "")),
        "seed": int(first.get("seed", 7)),
        "training_seed": int(first.get("training_seed", first.get("seed", 7))),
        "num_users": int(first["num_users"]),
        "ebno_db": float(first["ebno_db"]),
        "num_chunks": int(len(group)),
        "chunk_indices": ",".join(str(int(x)) for x in sorted(group["chunk_idx"].astype(int).tolist())),
    }
    for col in SUM_COLS:
        if col in group.columns:
            out[col] = float(group[col].sum())
    for col in ["bit_errors", "num_bits", "block_errors", "num_blocks", "num_batches_run"]:
        if col in out:
            out[col] = int(round(float(out[col])))

    bit_errors = int(out.get("bit_errors", 0))
    num_bits = int(out.get("num_bits", 0))
    block_errors = int(out.get("block_errors", 0))
    num_blocks = int(out.get("num_blocks", 0))
    rx_time = float(out.get("receiver_elapsed_s", 0.0))
    batches = int(out.get("num_batches_run", 0))

    out["ber"] = float(bit_errors / num_bits) if num_bits else np.nan
    out["bler"] = float(block_errors / num_blocks) if num_blocks else np.nan
    out["nmse"] = np.nan
    out["mc_stop_reason"] = "isolated_chunk_merge"
    out["target_bler_floor"] = np.nan
    out["bler_zero_error_upper_bound"] = float(3.0 / num_blocks) if num_blocks and block_errors == 0 else np.nan
    out["ber_zero_error_upper_bound"] = float(3.0 / num_bits) if num_bits and bit_errors == 0 else np.nan
    out["receiver_ms_per_batch"] = float(1000.0 * rx_time / max(batches, 1))
    out["receiver_ms_per_frame"] = float(1000.0 * rx_time / max(num_blocks, 1))
    out["reliable_ber"] = bool(bit_errors >= 1000)
    out["reliable_bler"] = bool(block_errors >= 100)
    for col in ["gpu_mem_current_gib", "gpu_mem_peak_gib", "gpu_mem", "peak"]:
        if col in group.columns:
            try:
                out[col] = float(group[col].max())
            except Exception:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge process-isolated evaluation chunks.")
    ap.add_argument("--input-root", default="_isolated_eval_chunks")
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--summary-json", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--receiver", default=None)
    ap.add_argument("--num-users", type=int, default=None)
    ap.add_argument("--ebno-db", type=float, default=None)
    args = ap.parse_args()

    input_root = Path(args.input_root)
    df = _load_rows(input_root)
    if args.variant:
        df = df[df["variant"].astype(str) == args.variant]
    if args.receiver:
        df = df[df["receiver"].astype(str) == args.receiver]
    if args.num_users is not None:
        df = df[df["num_users"].astype(int) == int(args.num_users)]
    if args.ebno_db is not None:
        df = df[np.isclose(df["ebno_db"].astype(float), float(args.ebno_db))]
    if df.empty:
        raise RuntimeError("No chunk rows remain after filters.")

    group_cols = ["variant", "dmrs_case", "seed", "receiver", "num_users", "ebno_db"]
    rows = [_merge_group(g) for _, g in df.groupby(group_cols, dropna=False)]
    out_df = pd.DataFrame(rows).sort_values(["variant", "num_users", "ebno_db", "receiver"])

    out_csv = Path(args.output_csv) if args.output_csv else input_root / "merged_curves.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    summary = {
        "input_root": str(input_root),
        "output_csv": str(out_csv),
        "num_input_chunks": int(len(df)),
        "num_merged_rows": int(len(out_df)),
        "rows": rows,
    }
    summary_json = Path(args.summary_json) if args.summary_json else out_csv.with_suffix(".summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("[ISO-MERGE] chunks:", len(df))
    print("[ISO-MERGE] rows:", len(out_df))
    print("[ISO-MERGE] wrote:", out_csv)
    print("[ISO-MERGE] wrote:", summary_json)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
PY

cat > scripts/isolated_eval_status.py <<'PY'
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_float_equal(a: Any, b: float) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except Exception:
        return False


def _rows(root: Path, variant: str, receiver: str, num_users: int, ebno_db: float) -> pd.DataFrame:
    frames = []
    for p in sorted(root.rglob("chunk_result.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        row = df.iloc[0]
        if str(row.get("variant", "")) != variant:
            continue
        if str(row.get("receiver", "")) != receiver:
            continue
        try:
            if int(row.get("num_users")) != int(num_users):
                continue
        except Exception:
            continue
        if not _safe_float_equal(row.get("ebno_db"), ebno_db):
            continue
        df["chunk_result_path"] = str(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _parse_int_set(df: pd.DataFrame) -> set[int]:
    out: set[int] = set()
    if "chunk_idx" not in df.columns:
        return out
    for x in df["chunk_idx"].tolist():
        try:
            out.add(int(x))
        except Exception:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="_isolated_eval_chunks")
    ap.add_argument("--config", default="configs/twc_comprehensive_mu32_base.yaml")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--num-users", type=int, required=True)
    ap.add_argument("--ebno-db", type=float, required=True)
    ap.add_argument("--chunk-batches", type=int, default=20)
    ap.add_argument("--target-block-errors", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--min-batches", type=int, default=None)
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()

    cfg = _load_config(Path(args.config))
    ev = cfg.get("evaluation", {})
    target = int(args.target_block_errors if args.target_block_errors is not None else ev.get("target_block_errors_per_receiver", 100))
    max_batches = int(args.max_batches if args.max_batches is not None else ev.get("max_num_batches_per_point", 2000))
    min_batches = int(args.min_batches if args.min_batches is not None else ev.get("min_num_batches_per_point", 16))
    chunk_batches = max(1, int(args.chunk_batches))
    max_chunks = (max_batches + chunk_batches - 1) // chunk_batches

    df = _rows(Path(args.input_root), args.variant, args.receiver, args.num_users, args.ebno_db)
    if df.empty:
        chunks_done: set[int] = set()
        bit_errors = num_bits = block_errors = num_blocks = num_batches = 0
    else:
        chunks_done = _parse_int_set(df)
        bit_errors = int(df.get("bit_errors", pd.Series(dtype=float)).fillna(0).sum())
        num_bits = int(df.get("num_bits", pd.Series(dtype=float)).fillna(0).sum())
        block_errors = int(df.get("block_errors", pd.Series(dtype=float)).fillna(0).sum())
        num_blocks = int(df.get("num_blocks", pd.Series(dtype=float)).fillna(0).sum())
        num_batches = int(df.get("num_batches_run", pd.Series(dtype=float)).fillna(0).sum())

    target_met = (num_batches >= min_batches) and (block_errors >= target)
    max_met = num_batches >= max_batches or len(chunks_done) >= max_chunks
    done = bool(target_met or max_met)

    next_chunk = None
    if not done:
        for k in range(max_chunks):
            if k not in chunks_done:
                next_chunk = k
                break
        if next_chunk is None:
            done = True
            max_met = True

    status = {
        "done": done,
        "reason": "target_block_errors" if target_met else ("max_batches" if max_met else "continue"),
        "next_chunk": -1 if next_chunk is None else int(next_chunk),
        "chunks_done": sorted(chunks_done),
        "num_chunks_done": len(chunks_done),
        "bit_errors": bit_errors,
        "num_bits": num_bits,
        "block_errors": block_errors,
        "num_blocks": num_blocks,
        "num_batches": num_batches,
        "target_block_errors": target,
        "min_batches": min_batches,
        "max_batches": max_batches,
        "chunk_batches": chunk_batches,
        "max_chunks": max_chunks,
        "ber": float(bit_errors / num_bits) if num_bits else None,
        "bler": float(block_errors / num_blocks) if num_blocks else None,
    }

    if args.shell:
        for k, v in status.items():
            if isinstance(v, bool):
                v = 1 if v else 0
            elif isinstance(v, list):
                v = ",".join(str(x) for x in v)
            print(f"{k.upper()}={v}")
    else:
        print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
PY

cat > upair_variant_pipeline_worker.sh <<'BASH'
#!/usr/bin/env bash
# Worker for exactly one normalized-UMi architecture variant.
# Resubmittable: resumes training and resumes process-isolated eval chunks.
set -euo pipefail

VARIANT="${1:?Usage: bash upair_variant_pipeline_worker.sh <variant>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/upair_portable_env.sh"
upair_activate

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

CONFIG="${UPAIR_CONFIG:-${ROOT}/configs/twc_comprehensive_mu32_base.yaml}"
DMRS_CASE="${UPAIR_DMRS_CASE:-1dmrs}"
SEED="${UPAIR_SEED:-7}"
STAGEB_PREFIX="${UPAIR_OPTUNA_STAGEB_PREFIX:-clean_b32_prb8_d256_40k_smart_trueDMRS_UMiNorm_u34610_1dmrs_stageB}"
OUT_ROOT="${UPAIR_EVAL_CHUNK_ROOT:-${ROOT}/_isolated_eval_chunks}"

RECEIVERS_RAW="${UPAIR_PIPELINE_RECEIVERS:-${UPAIR_EVAL_RECEIVERS:-baseline_ls_lmmse,baseline_ls_2dlmmse_lmmse,upair5g_lmmse,perfect_csi_lmmse}}"
USERS_RAW="${UPAIR_PIPELINE_USERS:-${UPAIR_EVAL_USERS:-1,2,3,4}}"
EBNOS_RAW="${UPAIR_PIPELINE_EBNOS:-${UPAIR_EVAL_EBNOS:--4,-3,-2,-1,0,1,2,3,4}}"
CHUNK_BATCHES="${UPAIR_PIPELINE_CHUNK_BATCHES:-${UPAIR_EVAL_CHUNK_BATCHES:-20}}"
MICRO="${UPAIR_PIPELINE_MICRO:-${UPAIR_EVAL_MICRO:-8}}"
TARGET_BLOCK_ERRORS="${UPAIR_PIPELINE_TARGET_BLOCK_ERRORS:-}"
MAX_BATCHES="${UPAIR_PIPELINE_MAX_BATCHES:-}"
MIN_BATCHES="${UPAIR_PIPELINE_MIN_BATCHES:-}"

mkdir -p "${OUT_ROOT}" logs/pipeline

split_csv() {
  local raw="${1//,/ }"
  # shellcheck disable=SC2206
  local arr=( ${raw} )
  printf '%s\n' "${arr[@]}"
}

training_complete() {
  python - "$VARIANT" <<'PY'
import json, sys
from pathlib import Path
v = sys.argv[1]
p = Path(f"TWC_plots_comprehensive/runs_rx16/seed7/1dmrs/{v}/metrics/train_state.json")
if not p.exists():
    sys.exit(1)
try:
    d = json.loads(p.read_text())
except Exception:
    sys.exit(1)
sys.exit(0 if bool(d.get("training_complete", False)) else 1)
PY
}

echo "================================================================================"
echo "[PIPELINE] variant=${VARIANT}"
echo "[PIPELINE] root=${ROOT}"
echo "[PIPELINE] stageB_prefix=${STAGEB_PREFIX}"
echo "[PIPELINE] receivers=${RECEIVERS_RAW}"
echo "[PIPELINE] users=${USERS_RAW}"
echo "[PIPELINE] ebnos=${EBNOS_RAW}"
echo "[PIPELINE] chunk_batches=${CHUNK_BATCHES} micro=${MICRO}"
echo "================================================================================"

best_json="${ROOT}/optuna/${STAGEB_PREFIX}_${VARIANT}_best_params.json"
best_db="${ROOT}/optuna/${STAGEB_PREFIX}_${VARIANT}.db"
if [[ ! -s "${best_json}" && ! -s "${best_db}" ]]; then
  echo "[PIPELINE] Missing Stage-B best for ${VARIANT}. Run Stage A/B first." >&2
  echo "  ${best_json}" >&2
  echo "  ${best_db}" >&2
  exit 2
fi

if training_complete; then
  echo "[PIPELINE] training already complete for ${VARIANT}; skipping training."
else
  echo "[PIPELINE] training missing/incomplete for ${VARIANT}; running training-only resume."
  export UPAIR_COMPREHENSIVE_SKIP_FINAL_EVAL=1
  python -u "${ROOT}/scripts/run_comprehensive_mu32_ablation.py" \
    --config "${CONFIG}" \
    --variants "${VARIANT}" \
    --dmrs-cases "${DMRS_CASE}" \
    --seeds "${SEED}" \
    --eval-users "${USERS_RAW}" \
    --use-optuna-best-1dmrs \
    --optuna-best-storage-dir "${ROOT}/optuna" \
    --optuna-best-study-prefix "${STAGEB_PREFIX}" \
    --require-optuna-best \
    --no-global-summary

  if ! training_complete; then
    echo "[PIPELINE] Training is still incomplete for ${VARIANT}; resubmit this same pipeline later." >&2
    exit 20
  fi
fi

echo "[PIPELINE] starting/resuming isolated evaluation for ${VARIANT}."

status_args_base=(--input-root "${OUT_ROOT}" --config "${CONFIG}" --variant "${VARIANT}" --chunk-batches "${CHUNK_BATCHES}")
if [[ -n "${TARGET_BLOCK_ERRORS}" ]]; then
  status_args_base+=(--target-block-errors "${TARGET_BLOCK_ERRORS}")
fi
if [[ -n "${MAX_BATCHES}" ]]; then
  status_args_base+=(--max-batches "${MAX_BATCHES}")
fi
if [[ -n "${MIN_BATCHES}" ]]; then
  status_args_base+=(--min-batches "${MIN_BATCHES}")
fi

while IFS= read -r receiver; do
  [[ -n "${receiver}" ]] || continue
  while IFS= read -r users; do
    [[ -n "${users}" ]] || continue
    while IFS= read -r ebno; do
      [[ -n "${ebno}" ]] || continue

      echo
      echo "--------------------------------------------------------------------------------"
      echo "[PIPELINE] eval point variant=${VARIANT} receiver=${receiver} U=${users} Eb/N0=${ebno}"
      echo "--------------------------------------------------------------------------------"

      while true; do
        status_file="$(mktemp)"
        python "${ROOT}/scripts/isolated_eval_status.py" \
          "${status_args_base[@]}" \
          --receiver "${receiver}" \
          --num-users "${users}" \
          --ebno-db "${ebno}" \
          --shell > "${status_file}"
        # shellcheck disable=SC1090
        source "${status_file}"
        rm -f "${status_file}"

        echo "[PIPELINE] status done=${DONE} reason=${REASON} chunks=${NUM_CHUNKS_DONE} batches=${NUM_BATCHES} block_errors=${BLOCK_ERRORS}/${TARGET_BLOCK_ERRORS} next_chunk=${NEXT_CHUNK}"

        if [[ "${DONE}" == "1" ]]; then
          break
        fi

        python -u "${ROOT}/scripts/run_isolated_eval_chunk.py" \
          --config "${CONFIG}" \
          --variant "${VARIANT}" \
          --dmrs-case "${DMRS_CASE}" \
          --seed "${SEED}" \
          --num-users "${users}" \
          --receiver "${receiver}" \
          --ebno-db "${ebno}" \
          --chunk-idx "${NEXT_CHUNK}" \
          --chunk-batches "${CHUNK_BATCHES}" \
          --receiver-microbatch-size "${MICRO}" \
          --stageb-prefix "${STAGEB_PREFIX}" \
          --optuna-dir "${ROOT}/optuna" \
          --output-root "${OUT_ROOT}"
      done

      safe_ebno="${ebno//-/m}"
      safe_ebno="${safe_ebno//./p}"
      merged_csv="${OUT_ROOT}/merged_${VARIANT}_u${users}_${receiver}_e${safe_ebno}.csv"
      python "${ROOT}/scripts/merge_isolated_eval_chunks.py" \
        --input-root "${OUT_ROOT}" \
        --output-csv "${merged_csv}" \
        --variant "${VARIANT}" \
        --receiver "${receiver}" \
        --num-users "${users}" \
        --ebno-db "${ebno}"

    done < <(split_csv "${EBNOS_RAW}")
  done < <(split_csv "${USERS_RAW}")
done < <(split_csv "${RECEIVERS_RAW}")

echo "[PIPELINE] COMPLETE variant=${VARIANT}"
BASH
chmod +x upair_variant_pipeline_worker.sh

cat > upair_submit_8variant_pipeline.sh <<'BASH'
#!/usr/bin/env bash
# Submit one normalized-UMi job per architecture variant. Each job requests exactly one GPU
# and runs evaluation chunks sequentially inside that one job/GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/upair_submit_lib.sh"
upair_ensure_venv

mkdir -p "${UPAIR_REPO_ROOT}/logs/pipeline" "${UPAIR_REPO_ROOT}/logs/submit"

TIME_LIMIT="${UPAIR_TIME_PIPELINE:-30:00:00}"
export UPAIR_GRES="${UPAIR_GRES:-gpu:h100:1}"
export UPAIR_MEM="${UPAIR_MEM:-32G}"
export UPAIR_CPUS="${UPAIR_CPUS:-8}"

echo "[PIPELINE-SUBMIT] ROOT=${UPAIR_REPO_ROOT}"
echo "[PIPELINE-SUBMIT] VENV=${UPAIR_VENV_PATH}"
echo "[PIPELINE-SUBMIT] GRES=${UPAIR_GRES} TIME=${TIME_LIMIT}"
echo "[PIPELINE-SUBMIT] variants:"
upair_variants | sed 's/^/  - /'

while IFS= read -r variant; do
  [[ -n "${variant}" ]] || continue

  job="uNormP-$(upair_first_n_chars "${variant}" 12)"
  log="${UPAIR_REPO_ROOT}/logs/pipeline/pipeline_${variant}_%j.out"
  jobfile="${UPAIR_REPO_ROOT}/logs/submit/pipeline_${variant}.sbatch"
  upair_write_sbatch_header "${jobfile}" "${job}" "${TIME_LIMIT}" "${log}"

  cat >> "${jobfile}" <<SBATCH
set -euo pipefail
cd "${UPAIR_REPO_ROOT}"
source "${UPAIR_REPO_ROOT}/upair_portable_env.sh"
upair_activate

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL="\${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="\${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="\${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="\${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

export UPAIR_CONFIG="${UPAIR_CONFIG:-}"
export UPAIR_DMRS_CASE="${UPAIR_DMRS_CASE:-1dmrs}"
export UPAIR_SEED="${UPAIR_SEED:-7}"
export UPAIR_OPTUNA_STAGEB_PREFIX="${UPAIR_OPTUNA_STAGEB_PREFIX:-clean_b32_prb8_d256_40k_smart_trueDMRS_UMiNorm_u34610_1dmrs_stageB}"
export UPAIR_PIPELINE_RECEIVERS="${UPAIR_PIPELINE_RECEIVERS:-}"
export UPAIR_PIPELINE_USERS="${UPAIR_PIPELINE_USERS:-}"
export UPAIR_PIPELINE_EBNOS="${UPAIR_PIPELINE_EBNOS:-}"
export UPAIR_PIPELINE_CHUNK_BATCHES="${UPAIR_PIPELINE_CHUNK_BATCHES:-20}"
export UPAIR_PIPELINE_MICRO="${UPAIR_PIPELINE_MICRO:-8}"
export UPAIR_PIPELINE_TARGET_BLOCK_ERRORS="${UPAIR_PIPELINE_TARGET_BLOCK_ERRORS:-}"
export UPAIR_PIPELINE_MAX_BATCHES="${UPAIR_PIPELINE_MAX_BATCHES:-}"
export UPAIR_PIPELINE_MIN_BATCHES="${UPAIR_PIPELINE_MIN_BATCHES:-}"

bash "${UPAIR_REPO_ROOT}/upair_variant_pipeline_worker.sh" "${variant}"
SBATCH

  echo "[PIPELINE-SUBMIT] submitting ${variant}"
  upair_submit_job_script "${jobfile}"
done < <(upair_variants)
BASH
chmod +x upair_submit_8variant_pipeline.sh

cat > upair_probe_uminorm_pipeline_ready.sh <<'BASH'
#!/usr/bin/env bash
# Read-only sanity probe. Does NOT modify .gitignore.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

fail=0
ok(){ echo "[OK] $*"; }
bad(){ echo "[FAIL] $*" >&2; fail=1; }
warn(){ echo "[WARN] $*" >&2; }

for f in \
  configs/twc_comprehensive_mu32_base.yaml \
  src/upair5g/builders.py \
  scripts/run_comprehensive_mu32_ablation.py \
  scripts/run_isolated_eval_chunk.py \
  scripts/merge_isolated_eval_chunks.py \
  scripts/isolated_eval_status.py \
  upair_portable_env.sh \
  upair_submit_lib.sh \
  upair_submit_stageA_all.sh \
  upair_submit_stageB_all.sh \
  upair_variant_pipeline_worker.sh \
  upair_submit_8variant_pipeline.sh
do
  [[ -e "$f" ]] && ok "$f exists" || bad "$f missing"
done

python - <<'PY'
from pathlib import Path
import ast, json, sys, yaml

fail = False
def ok(msg): print("[OK]", msg)
def bad(msg):
    global fail
    print("[FAIL]", msg, file=sys.stderr)
    fail = True
def warn(msg): print("[WARN]", msg, file=sys.stderr)

cfg = yaml.safe_load(Path("configs/twc_comprehensive_mu32_base.yaml").read_text())
def get(path, default=None):
    node = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node

checks = [
    ("channel.family", "umi"),
    ("channel.model", "umi"),
    ("channel.normalize_channel", True),
    ("channel.umi.enable_pathloss", False),
    ("channel.umi.enable_shadow_fading", False),
    ("near_far.enabled", False),
    ("near_far.mode", "disabled"),
    ("pusch.n_size_grid", 8),
    ("pusch.n_size_bwp", 8),
    ("model.pilot_mask_mode", "per_stream"),
    ("model.error_feature_mode", "per_user"),
    ("system.batch_size_train", 32),
    ("system.batch_size_eval", 32),
    ("training.val_microbatch_size", 16),
    ("evaluation.logical_batch_size", 64),
    ("evaluation.receiver_microbatch_size", 8),
    ("evaluation.compiled_receiver_error_counts", False),
    ("evaluation.receiver_call_jit_compile", False),
    ("evaluation.stream_eval_microbatches", True),
    ("evaluation.nmse_receivers", []),
    ("evaluation.save_example_batch", False),
    ("evaluation.max_num_batches_per_point", 2000),
    ("evaluation.target_block_errors_per_receiver", 100),
    ("baselines.covariance_estimation.cache_name", "empirical_covariances_umi_norm.npz"),
]
for path, want in checks:
    got = get(path)
    if got == want:
        ok(f"{path} = {got!r}")
    else:
        bad(f"{path}: expected {want!r}, got {got!r}")

for py in ["scripts/run_isolated_eval_chunk.py", "scripts/merge_isolated_eval_chunks.py", "scripts/isolated_eval_status.py"]:
    ast.parse(Path(py).read_text())
    ok(f"syntax parse: {py}")

rc = Path("scripts/run_comprehensive_mu32_ablation.py").read_text()
if 'evaluation.save_example_batch", bool(get_cfg(cfg, "evaluation.save_example_batch", False))' in rc:
    ok("_eval_cfg preserves save_example_batch=False from config")
else:
    bad("_eval_cfg may still force save_example_batch=True for main/u4")

if "UPAIR_COMPREHENSIVE_SKIP_FINAL_EVAL=1" in Path("upair_variant_pipeline_worker.sh").read_text():
    ok("variant worker uses training-only mode before isolated eval")
else:
    bad("variant worker does not force training-only mode")

if "clean_b32_prb8_d256_40k_smart_trueDMRS_UMiNorm_u34610_1dmrs_stageB" in Path("upair_submit_8variant_pipeline.sh").read_text():
    ok("master pipeline uses UMiNorm Stage-B prefix")
else:
    bad("master pipeline Stage-B prefix is not UMiNorm")

# Check variants from optuna_common and submit_lib.
ns = {"__file__": str(Path("scripts/optuna_1dmrs_common.py").resolve())}
exec(compile(Path("scripts/optuna_1dmrs_common.py").read_text(), "scripts/optuna_1dmrs_common.py", "exec"), ns)
variants = list(ns["VARIANTS"].keys())
print("[INFO] optuna variants:", ",".join(variants))
if len(variants) == 8 and "promptmlp_d256_b4_r2_pr2" in variants:
    ok("8 variants including prompt-MLP ablation are present")
else:
    bad("expected 8 variants including prompt-MLP ablation")

# Stage-B best is optional before Optuna has run. Make it fatal only if requested.
require_stageb = str(__import__("os").environ.get("UPAIR_REQUIRE_STAGEB", "0")).lower() in {"1","true","yes"}
missing = []
for v in variants:
    json_path = Path(f"optuna/clean_b32_prb8_d256_40k_smart_trueDMRS_UMiNorm_u34610_1dmrs_stageB_{v}_best_params.json")
    db_path = Path(f"optuna/clean_b32_prb8_d256_40k_smart_trueDMRS_UMiNorm_u34610_1dmrs_stageB_{v}.db")
    if json_path.exists() or db_path.exists():
        ok(f"Stage-B best exists for {v}")
    else:
        missing.append(v)
        warn(f"Stage-B best missing for {v}; normal before Stage A/B has completed")
if missing and require_stageb:
    bad("Stage-B best is required but missing for: " + ",".join(missing))

if fail:
    raise SystemExit("[PROBE] FAILED normalized-UMi pipeline readiness probe")
print("[PROBE] PASSED normalized-UMi pipeline readiness probe")
PY

source "${ROOT}/upair_portable_env.sh"
upair_ensure_venv

[[ "$fail" == "0" ]] || exit 1
BASH
chmod +x upair_probe_uminorm_pipeline_ready.sh

cat > upair_probe_uminorm_batch_eval_policy.sh <<'BASH'
#!/usr/bin/env bash
# Read-only batch/reliability sanity probe. Does NOT modify .gitignore.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

python - <<'PY'
from pathlib import Path
import yaml, math

cfg = yaml.safe_load(Path("configs/twc_comprehensive_mu32_base.yaml").read_text())
sys = cfg["system"]; tr = cfg["training"]; ev = cfg["evaluation"]

train_b = int(sys["batch_size_train"])
val_b = int(sys["batch_size_eval"])
val_mb = int(tr["val_microbatch_size"])
eval_logical_b = int(ev.get("logical_batch_size", val_b))
eval_mb = int(ev["receiver_microbatch_size"])
target_be = int(ev["target_block_errors_per_receiver"])
max_batches = int(ev["max_num_batches_per_point"])
min_batches = int(ev["min_num_batches_per_point"])

print("="*90)
print("[BATCH POLICY]")
print(f"training batch size                  = {train_b}")
print(f"Optuna/final validation batch size    = {val_b}")
print(f"validation microbatch size            = {val_mb}")
print(f"final BLER logical batch size         = {eval_logical_b}")
print(f"receiver microbatch size              = {eval_mb}")
print(f"min/max batches per point             = {min_batches}/{max_batches}")
print(f"target block errors per receiver      = {target_be}")

if train_b == 32 and val_b == 32 and val_mb == 16:
    print("[OK] Training/validation batch policy matches Optuna defaults: train=32, val=32, val_micro=16.")
else:
    print("[WARN] Training/validation batch policy differs from Optuna defaults.")

if eval_logical_b >= eval_mb and eval_logical_b % eval_mb == 0:
    print("[OK] Evaluation logical batch is compatible with receiver microbatching.")
else:
    print("[WARN] Evaluation logical batch is not an integer multiple of receiver microbatch.")

print()
print("="*90)
print("[BLER RELIABILITY CALCULATIONS]")
for b in sorted({32, 64, eval_logical_b}):
    for mb in [1500, 2000, 8000, 16000]:
        frames = b * mb
        exp_at_1e4 = frames * 1e-4
        ub_zero = 3.0 / frames
        print(
            f"batch={b:>3d}, max_batches={mb:>5d}: "
            f"frames={frames:>8d}, expected errors at 1e-4={exp_at_1e4:>6.1f}, "
            f"zero-error UB≈{ub_zero:.2e}"
        )

print()
for target in [50, 100]:
    frames_needed = int(math.ceil(target / 1e-4))
    print(f"To get {target} block errors at BLER=1e-4:")
    print(f"  frames needed = {frames_needed}")
    print(f"  batches at batch=32 = {math.ceil(frames_needed/32)}")
    print(f"  batches at batch=64 = {math.ceil(frames_needed/64)}")
PY
BASH
chmod +x upair_probe_uminorm_batch_eval_policy.sh

cat > upair_clean_uminorm_top_level_safe.sh <<'BASH'
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
BASH
chmod +x upair_clean_uminorm_top_level_safe.sh

echo "[UMINORM-MASTER-PATCH] Done. Run:"
echo "  bash upair_probe_uminorm_pipeline_ready.sh"
echo "  bash upair_probe_uminorm_batch_eval_policy.sh"
echo "  bash upair_clean_uminorm_top_level_safe.sh   # dry-run only"
