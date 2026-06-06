#!/usr/bin/env bash
# Apply UPAIR feature-tensor, d-model, validation, PRB, and hygiene updates.
# Run from the repository root: /home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable
set -euo pipefail

ROOT="$(pwd)"
if [[ ! -d "${ROOT}/src/upair5g" || ! -d "${ROOT}/scripts" || ! -d "${ROOT}/configs" ]]; then
  echo "[PATCH] Please run this from the UPAIR repository root." >&2
  exit 1
fi

stamp="$(date +%Y%m%d_%H%M%S)"
backup_dir="${ROOT}/patch_backups/feature_prb8_dmodel_${stamp}"
mkdir -p "${backup_dir}"

files_to_backup=(
  src/upair5g/estimator.py
  src/upair5g/training.py
  configs/twc_comprehensive_mu32_base.yaml
  scripts/optuna_1dmrs_common.py
  scripts/run_comprehensive_mu32_ablation.py
  upair_submit_lib.sh
  upair_submit_stageA_all.sh
  upair_submit_stageB_all.sh
  upair_submit_train_eval_all.sh
  upair_probe_clean_start.sh
  upair_probe_after_stageB.sh
  upair_probe_after_train_eval.sh
  upair_portable_env.sh
  pyproject.toml
  .gitignore
)

for f in "${files_to_backup[@]}"; do
  if [[ -e "${f}" ]]; then
    mkdir -p "${backup_dir}/$(dirname "${f}")"
    cp -a "${f}" "${backup_dir}/${f}"
  fi
done

echo "[PATCH] Backed up edited files under: ${backup_dir}"

python - <<'PATCHPY'
from __future__ import annotations
from pathlib import Path
import re

ROOT = Path.cwd()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_required(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"[PATCH-FAIL] Could not find block for {label}")
    return text.replace(old, new)


def replace_once_or_keep(text: str, old: str, new: str) -> str:
    return text.replace(old, new) if old in text else text

# -------------------------------------------------------------------------
# 1) Estimator: per-stream pilot mask, per-user LS-error features,
#    multiplicative LS-exact error-variance head.
# -------------------------------------------------------------------------
est_path = "src/upair5g/estimator.py"
s = read(est_path)
s = replace_required(
    s,
    "from .builders import extract_pilot_mask\n",
    "from .builders import extract_pilot_mask_per_stream\n",
    "estimator import extract_pilot_mask_per_stream",
)

s = replace_required(
    s,
    """        self.use_noise_feature = bool(cfg["model"]["use_noise_feature"])
        self.use_pilot_mask_feature = bool(cfg["model"]["use_pilot_mask_feature"])
        self.residual_scale = float(cfg["model"]["residual_scale"])
        self.eps = 1e-6

        extra_channels = 1 + int(self.use_noise_feature) + int(self.use_pilot_mask_feature)
        if self.max_num_users > 1:
            input_channels = 2 * self.num_rx_ant * self.max_num_users + 2 * self.num_rx_ant + extra_channels
        else:
            input_channels = 4 * self.num_rx_ant + extra_channels
""",
    """        model_cfg = cfg.get("model", {})
        self.use_noise_feature = bool(model_cfg.get("use_noise_feature", True))
        self.use_pilot_mask_feature = bool(model_cfg.get("use_pilot_mask_feature", True))
        self.pilot_mask_mode = str(model_cfg.get("pilot_mask_mode", "per_stream")).lower()
        self.error_feature_mode = str(model_cfg.get("error_feature_mode", "per_user")).lower()
        self.residual_scale = float(model_cfg.get("residual_scale", 0.35))
        self.eps = 1e-6

        per_stream_mask = self.pilot_mask_mode in {"per_stream", "per_user", "stream", "user"}
        per_user_error = self.error_feature_mode in {"per_user", "per_stream", "user", "stream"}
        self.pilot_mask_channels = self.max_num_users if (self.use_pilot_mask_feature and per_stream_mask) else int(self.use_pilot_mask_feature)
        self.error_feature_channels = self.max_num_users if per_user_error else 1

        extra_channels = self.error_feature_channels + int(self.use_noise_feature) + self.pilot_mask_channels
        if self.max_num_users > 1:
            input_channels = 2 * self.num_rx_ant * self.max_num_users + 2 * self.num_rx_ant + extra_channels
        else:
            input_channels = 4 * self.num_rx_ant + extra_channels
""",
    "estimator feature-channel accounting",
)

s = replace_required(
    s,
    """        self.err_head = tf.keras.layers.Conv2D(
            self.num_rx_ant * self.max_num_users,
            kernel_size=1,
            padding="same",
            kernel_initializer="zeros",
            bias_initializer=tf.keras.initializers.Constant(-4.0),
            name="err_head",
        )

        self.pilot_mask = tf.cast(extract_pilot_mask(resource_grid), tf.float32)
""",
    """        self.err_head = tf.keras.layers.Conv2D(
            self.num_rx_ant * self.max_num_users,
            kernel_size=1,
            padding="same",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="err_head",
        )

        self.pilot_mask = tf.cast(extract_pilot_mask_per_stream(resource_grid), tf.float32)
""",
    "LS-exact error head and per-stream stored pilot mask",
)

old_pilot_func = """    def _pilot_mask_for_batch(self, pilot_mask: tf.Tensor | None, batch: tf.Tensor, time: tf.Tensor, freq: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(tf.convert_to_tensor(pilot_mask if pilot_mask is not None else self.pilot_mask), tf.float32)
        if mask.shape.rank == 2:
            mask = mask[..., tf.newaxis]
        if mask.shape.rank != 3:
            raise ValueError(f"Expected pilot mask rank 2 or 3, got {mask.shape.rank}.")
        mask = tf.reduce_max(mask, axis=-1, keepdims=True)
        return tf.broadcast_to(mask[tf.newaxis, ...], [batch, time, freq, 1])
"""
new_pilot_func = """    def _pad_feature_dim(self, x: tf.Tensor, target_channels: int) -> tf.Tensor:
        x = tf.convert_to_tensor(x)
        if x.shape.rank != 4:
            raise ValueError(f"Expected rank-4 feature map [B,T,F,C], got rank {x.shape.rank}.")
        target_channels = int(target_channels)
        pad_channels = tf.maximum(target_channels - tf.shape(x)[-1], 0)
        paddings = tf.stack(
            [
                tf.constant([0, 0], dtype=tf.int32),
                tf.constant([0, 0], dtype=tf.int32),
                tf.constant([0, 0], dtype=tf.int32),
                tf.stack([tf.constant(0, dtype=tf.int32), tf.cast(pad_channels, tf.int32)]),
            ]
        )
        return tf.pad(x, paddings)[..., :target_channels]

    def _pad_mask_streams(self, mask: tf.Tensor, target_streams: int) -> tf.Tensor:
        mask = tf.convert_to_tensor(mask)
        if mask.shape.rank != 3:
            raise ValueError(f"Expected rank-3 pilot mask [T,F,S], got rank {mask.shape.rank}.")
        target_streams = int(target_streams)
        pad_streams = tf.maximum(target_streams - tf.shape(mask)[-1], 0)
        paddings = tf.stack(
            [
                tf.constant([0, 0], dtype=tf.int32),
                tf.constant([0, 0], dtype=tf.int32),
                tf.stack([tf.constant(0, dtype=tf.int32), tf.cast(pad_streams, tf.int32)]),
            ]
        )
        return tf.pad(mask, paddings)[..., :target_streams]

    def _pilot_mask_for_batch(
        self,
        pilot_mask: tf.Tensor | None,
        batch: tf.Tensor,
        time: tf.Tensor,
        freq: tf.Tensor,
        *,
        collapse: bool = False,
    ) -> tf.Tensor:
        mask = tf.cast(tf.convert_to_tensor(pilot_mask if pilot_mask is not None else self.pilot_mask), tf.float32)
        if mask.shape.rank == 2:
            mask = mask[..., tf.newaxis]
        if mask.shape.rank != 3:
            raise ValueError(f"Expected pilot mask rank 2 or 3, got {mask.shape.rank}.")

        per_stream = self.pilot_mask_mode in {"per_stream", "per_user", "stream", "user"}
        if collapse or not per_stream:
            mask = tf.reduce_max(mask, axis=-1, keepdims=True)
            channels = 1
        else:
            mask = self._pad_mask_streams(mask, self.max_num_users)
            channels = self.max_num_users
        return tf.broadcast_to(mask[tf.newaxis, ...], [batch, time, freq, channels])
"""
s = replace_required(s, old_pilot_func, new_pilot_func, "pilot mask function")

old_feature_block = """        err_bc = broadcast_like_err(err_ls, h_ls)
        if self.max_num_users > 1:
            h_ls_btfnu = tensor7_to_btfnu(h_ls)
            err_btfnu = tensor7_to_btfnu(err_bc)
            h_feat = pad_user_dim(h_ls_btfnu, self.max_num_users)
            err_feat = pad_user_dim(err_btfnu, self.max_num_users)
            b = tf.shape(y_btfnc)[0]
            t = tf.shape(y_btfnc)[1]
            f = tf.shape(y_btfnc)[2]
            h_ri = complex_to_ri_channels(tf.reshape(h_feat, [b, t, f, self.num_rx_ant * self.max_num_users]))
            err_map = tf.reduce_mean(err_feat, axis=[-2, -1], keepdims=False)[..., tf.newaxis]
        else:
            h_ls_btfnu = tensor7_to_btfnu(h_ls)
            err_btfnu = tensor7_to_btfnu(err_bc)
            h_ls_btfnc = tf.squeeze(h_ls_btfnu, axis=-1)
            err_btfnc = tf.squeeze(err_btfnu, axis=-1)
            h_ri = complex_to_ri_channels(h_ls_btfnc)
            err_map = tf.reduce_mean(err_btfnc, axis=-1, keepdims=True)
"""
new_feature_block = """        err_bc = broadcast_like_err(err_ls, h_ls)
        per_user_error = self.error_feature_mode in {"per_user", "per_stream", "user", "stream"}
        if self.max_num_users > 1:
            h_ls_btfnu = tensor7_to_btfnu(h_ls)
            err_btfnu = tensor7_to_btfnu(err_bc)
            h_feat = pad_user_dim(h_ls_btfnu, self.max_num_users)
            b = tf.shape(y_btfnc)[0]
            t = tf.shape(y_btfnc)[1]
            f = tf.shape(y_btfnc)[2]
            h_ri = complex_to_ri_channels(tf.reshape(h_feat, [b, t, f, self.num_rx_ant * self.max_num_users]))
            if per_user_error:
                # Sionna's err_var usually has one receive-error dimension.  Average that
                # dimension but keep the user/stream dimension explicit: [B,T,F,Umax].
                err_raw_btfnu = tensor7_to_btfnu(err_ls)
                err_user_map = tf.reduce_mean(err_raw_btfnu, axis=-2)
                err_map = self._pad_feature_dim(err_user_map, self.max_num_users)
            else:
                err_feat = pad_user_dim(err_btfnu, self.max_num_users)
                err_map = tf.reduce_mean(err_feat, axis=[-2, -1], keepdims=False)[..., tf.newaxis]
        else:
            h_ls_btfnu = tensor7_to_btfnu(h_ls)
            err_btfnu = tensor7_to_btfnu(err_bc)
            h_ls_btfnc = tf.squeeze(h_ls_btfnu, axis=-1)
            h_ri = complex_to_ri_channels(h_ls_btfnc)
            if per_user_error:
                err_raw_btfnu = tensor7_to_btfnu(err_ls)
                err_map = tf.reduce_mean(err_raw_btfnu, axis=-2)
            else:
                err_btfnc = tf.squeeze(err_btfnu, axis=-1)
                err_map = tf.reduce_mean(err_btfnc, axis=-1, keepdims=True)
"""
s = replace_required(s, old_feature_block, new_feature_block, "per-user LS-error feature block")

s = replace_required(
    s,
    """        mask = self._pilot_mask_for_batch(pilot_mask, b, t, f)
""",
    """        mask = self._pilot_mask_for_batch(pilot_mask, b, t, f, collapse=True)
""",
    "collapsed prompt mask",
)

s = replace_required(
    s,
    """        err_hat_btfnu = err_anchor + tf.nn.softplus(err_delta) + self.eps
""",
    """        # Multiplicative positive correction.  With the zero-initialized err_head,
        # this starts exactly from the LS error variance instead of adding
        # softplus(-4) to it.
        err_scale = tf.exp(tf.clip_by_value(err_delta, -6.0, 6.0))
        err_hat_btfnu = tf.maximum(err_anchor * err_scale, tf.cast(self.eps, err_anchor.dtype))
""",
    "multiplicative LS-exact error variance update",
)
write(est_path, s)

# -------------------------------------------------------------------------
# 2) Training systems pass per-stream masks.
# -------------------------------------------------------------------------
train_path = "src/upair5g/training.py"
s = read(train_path)
s = replace_required(
    s,
    "from .builders import build_channel, build_ls_estimator, build_pusch_transmitter, extract_pilot_mask, get_resource_grid, max_num_users, multiuser_enabled\n",
    "from .builders import build_channel, build_ls_estimator, build_pusch_transmitter, extract_pilot_mask_per_stream, get_resource_grid, max_num_users, multiuser_enabled\n",
    "training import extract_pilot_mask_per_stream",
)
s = replace_required(
    s,
    '        "pilot_mask": extract_pilot_mask(resource_grid),\n',
    '        "pilot_mask": extract_pilot_mask_per_stream(resource_grid),\n',
    "training per-stream pilot mask",
)
write(train_path, s)

# -------------------------------------------------------------------------
# 3) Base config: PRB=8, main d=256, per-stream/per-user features,
#    validation distribution aligned with training.
# -------------------------------------------------------------------------
cfg_path = "configs/twc_comprehensive_mu32_base.yaml"
s = read(cfg_path)
s = replace_once_or_keep(s, "name: rx16_prb12_1dmrs_u3", "name: rx16_prb8_1dmrs_u3")
s = re.sub(r"(?m)^(\s*)n_size_grid:\s*12\s*$", r"\1n_size_grid: 8", s)
s = re.sub(r"(?m)^(\s*)n_size_bwp:\s*12\s*$", r"\1n_size_bwp: 8", s)
s = re.sub(r"(?m)^(\s*)d_model:\s*96\s*$", r"\1d_model: 256", s, count=1)
if "pilot_mask_mode:" not in s:
    s = s.replace(
        "  use_pilot_mask_feature: true\n",
        "  use_pilot_mask_feature: true\n  pilot_mask_mode: per_stream\n  error_feature_mode: per_user\n  error_variance_update: multiplicative\n",
    )
s = replace_required(
    s,
    """  val_user_count_weights:
  - 1.0
  - 2.0
  - 3.0
  - 4.0
""",
    """  val_user_count_weights:
  - 1.0
  - 3.0
  - 6.0
  - 10.0
""",
    "validation user-count weights in base config",
)
write(cfg_path, s)

# -------------------------------------------------------------------------
# 4) Optuna/common and comprehensive variants: rename and increase d.
# -------------------------------------------------------------------------
variant_name_replacements = {
    "main_d96_b4_r2": "main_d256_b4_r2",
    "shallow_d96_b2_r2": "shallow_d256_b2_r2",
    "deep_d96_b6_r2": "deep_d256_b6_r2",
    "narrow_d64_b4_r2": "narrow_d192_b4_r2",
    "wide_d128_b4_r2": "wide_d320_b4_r2",
    "wide_deep_d128_b6_r2": "wide_deep_d320_b6_r2",
    "mlpwide_d96_b4_r4": "mlpwide_d256_b4_r4",
}
text_files_for_names = [
    "scripts/optuna_1dmrs_common.py",
    "scripts/run_comprehensive_mu32_ablation.py",
    "upair_submit_lib.sh",
    "upair_submit_stageA_all.sh",
    "upair_submit_stageB_all.sh",
    "upair_submit_train_eval_all.sh",
    "upair_probe_after_stageB.sh",
    "upair_probe_after_train_eval.sh",
]
for path in text_files_for_names:
    p = ROOT / path
    if not p.exists():
        continue
    s = p.read_text(encoding="utf-8")
    for old, new in variant_name_replacements.items():
        s = s.replace(old, new)
    if path in {"scripts/optuna_1dmrs_common.py", "scripts/run_comprehensive_mu32_ablation.py"}:
        s = s.replace('"model.d_model": 96', '"model.d_model": 256')
        s = s.replace('"model.d_model": 64', '"model.d_model": 192')
        s = s.replace('"model.d_model": 128', '"model.d_model": 320')
        s = s.replace('"label": "d=96,', '"label": "d=256,')
        s = s.replace('"label": "d=64,', '"label": "d=192,')
        s = s.replace('"label": "d=128,', '"label": "d=320,')
        s = s.replace('default=[1.0, 2.0, 3.0, 4.0])', 'default=[1.0, 3.0, 6.0, 10.0])')
    p.write_text(s, encoding="utf-8")

# Update default study prefixes to avoid accidental reuse of old incompatible Optuna evidence.
prefix_replacements = {
    "clean_b32_iso_u34610_1dmrs_stageA": "clean_b32_prb8_d256_u34610_1dmrs_stageA",
    "clean_b32_iso_u34610_1dmrs_stageB": "clean_b32_prb8_d256_u34610_1dmrs_stageB",
    "clean_b32_iso_u34610_1dmrs_stageC": "clean_b32_prb8_d256_u34610_1dmrs_stageC",
}
for path in [
    "scripts/optuna_1dmrs_common.py",
    "scripts/run_comprehensive_mu32_ablation.py",
    "upair_submit_stageA_all.sh",
    "upair_submit_stageB_all.sh",
    "upair_submit_train_eval_all.sh",
    "upair_probe_clean_start.sh",
    "upair_probe_after_stageB.sh",
    "upair_probe_after_train_eval.sh",
]:
    p = ROOT / path
    if not p.exists():
        continue
    s = p.read_text(encoding="utf-8")
    for old, new in prefix_replacements.items():
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")

# -------------------------------------------------------------------------
# 5) Prevent __pycache__ generation where possible and make the clean probe
#    perform no-side-effect syntax compilation.
# -------------------------------------------------------------------------
env_path = "upair_portable_env.sh"
s = read(env_path)
if "PYTHONDONTWRITEBYTECODE" not in s:
    s = s.replace(
        'export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"\n',
        'export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"\nexport PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"\n',
    )
write(env_path, s)

probe_path = "upair_probe_clean_start.sh"
s = read(probe_path)
old = """upair_ensure_venv
python -m compileall -q src scripts
check_ok "Python syntax compile passed for src/ and scripts/"
"""
new = """upair_ensure_venv
python - <<'PY'
from __future__ import annotations
from pathlib import Path

failed = False
for root in ("src", "scripts"):
    for path in Path(root).rglob("*.py"):
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            print(f"[SYNTAX-FAIL] {path}: {exc}")
            failed = True
raise SystemExit(1 if failed else 0)
PY
check_ok "Python syntax compile passed for src/ and scripts/ without writing __pycache__"
"""
s = replace_required(s, old, new, "clean probe compileall replacement")
write(probe_path, s)

# -------------------------------------------------------------------------
# 6) pyproject: do not reference missing README.md.
# -------------------------------------------------------------------------
pyproject = "pyproject.toml"
s = read(pyproject)
s = s.replace('readme = "README.md"\n', '')
write(pyproject, s)

# -------------------------------------------------------------------------
# 7) .gitignore: add remaining runtime outputs/logs.
# -------------------------------------------------------------------------
gitignore = ROOT / ".gitignore"
s = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
for line in [
    "TWC_plots_comprehensive/",
    "logs/",
    "outputs/",
    "plots/",
    "metrics/",
    "*.out",
    "*.err",
    "*.log",
]:
    if line not in s.splitlines():
        s += ("\n" if s and not s.endswith("\n") else "") + line + "\n"
gitignore.write_text(s, encoding="utf-8")
PATCHPY

# Remove generated bytecode immediately after patching.
find "${ROOT}" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "${ROOT}" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

chmod +x \
  upair_portable_env.sh \
  upair_make_minimal_scratch.sh \
  upair_submit_stageA_all.sh \
  upair_submit_stageB_all.sh \
  upair_submit_train_eval_all.sh \
  upair_probe_clean_start.sh \
  upair_probe_after_stageB.sh \
  upair_probe_after_train_eval.sh 2>/dev/null || true

echo "[PATCH] Applied feature/PRB/d-model updates."
echo "[PATCH] Important new defaults: PRB=8, feature_channels=169, d={192,256,320}, Stage prefix=clean_b32_prb8_d256_u34610_1dmrs_stageA/B."
echo "[PATCH] Run: bash upair_probe_clean_start.sh"
echo "[PATCH] Then run the runtime sanity probe on a GPU node."
