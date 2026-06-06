#!/usr/bin/env bash
# Patch UPAIR to feed TRUE nonzero DMRS RE masks per user/port into the neural
# estimator, instead of Sionna resource-grid no-data masks for full DMRS symbols.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

[[ -f src/upair5g/builders.py && -f src/upair5g/estimator.py && -f src/upair5g/training.py && -f src/upair5g/evaluation.py ]] || {
  echo "[PATCH] Run this from the repository root." >&2
  exit 1
}

python - <<'PY'
from pathlib import Path

# ----------------------------
# 1) builders.py: add true DMRS-mask extractor
# ----------------------------
p = Path("src/upair5g/builders.py")
s = p.read_text()

anchor = '''def extract_pilot_mask(resource_grid: Any) -> tf.Tensor:
    mask = extract_pilot_mask_per_stream(resource_grid)
    return tf.reduce_max(mask, axis=-1, keepdims=True)
'''

new = r'''def extract_pilot_mask(resource_grid: Any) -> tf.Tensor:
    mask = extract_pilot_mask_per_stream(resource_grid)
    return tf.reduce_max(mask, axis=-1, keepdims=True)


def _grid_mask_to_tf_time_freq(
    grid: Any,
    *,
    target_time: int,
    target_freq: int,
    context: str,
) -> tf.Tensor:
    # Convert a Sionna DMRS grid into a binary [T,F] nonzero-RE mask.
    # In Sionna 1.2.1, PUSCHConfig.dmrs_grid is commonly [num_layers, F, T].
    # Some objects may expose [num_layers, T, F]. We infer orientation from the
    # resource-grid shape and reduce all leading singleton/layer axes.
    mask = tf.cast(tf.not_equal(tf.abs(tf.convert_to_tensor(grid)), 0), tf.float32)
    if mask.shape.rank is None or mask.shape.rank < 2:
        raise ValueError(f"{context}: expected DMRS grid rank >=2, got {mask.shape.rank}.")
    if mask.shape.rank > 2:
        reduce_axes = list(range(mask.shape.rank - 2))
        mask = tf.reduce_max(mask, axis=reduce_axes)
    static = mask.shape.as_list()
    if len(static) != 2:
        raise ValueError(f"{context}: expected reduced DMRS grid rank 2, got shape {mask.shape}.")
    a, b = static
    if a == target_time and b == target_freq:
        return tf.cast(mask, tf.float32)
    if a == target_freq and b == target_time:
        return tf.cast(tf.transpose(mask, [1, 0]), tf.float32)

    shape = tf.shape(mask)
    is_ft = tf.logical_and(tf.equal(shape[0], target_freq), tf.equal(shape[1], target_time))
    is_tf = tf.logical_and(tf.equal(shape[0], target_time), tf.equal(shape[1], target_freq))
    def as_ft() -> tf.Tensor:
        return tf.transpose(mask, [1, 0])
    def as_tf() -> tf.Tensor:
        return mask
    out = tf.cond(is_ft, as_ft, as_tf)
    with tf.control_dependencies([
        tf.debugging.assert_equal(
            tf.logical_or(is_ft, is_tf),
            True,
            message=f"{context}: DMRS grid shape does not match resource-grid [T,F]=[{target_time},{target_freq}].",
        )
    ]):
        return tf.cast(tf.identity(out), tf.float32)


def _dmrs_grid_from_pusch_config(pusch_config: Any) -> Any | None:
    value = first_present_attr(pusch_config, ["dmrs_grid", "_dmrs_grid"], None)
    if value is not None:
        return value
    dmrs = getattr(pusch_config, "dmrs", None)
    if dmrs is not None:
        value = first_present_attr(dmrs, ["dmrs_grid", "_dmrs_grid", "pilot_grid", "_pilot_grid"], None)
        if value is not None:
            return value
    return None


def extract_true_dmrs_mask_per_stream(tx: Any, resource_grid: Any | None = None) -> tf.Tensor:
    # Return TRUE nonzero-DMRS RE masks as [T,F,U] for active users/streams.
    # This differs from resource_grid.pilot_pattern.mask, which can mark all
    # no-data REs in a DMRS OFDM symbol when num_cdm_groups_without_data reserves
    # additional subcarriers.
    if resource_grid is None:
        resource_grid = get_resource_grid(tx)

    fallback = extract_pilot_mask_per_stream(resource_grid)
    target_time = int(fallback.shape[0] or tf.shape(fallback)[0].numpy())
    target_freq = int(fallback.shape[1] or tf.shape(fallback)[1].numpy())

    pusch_configs = list(first_present_attr(tx, ["_upair_pusch_configs"], []) or [])
    if not pusch_configs:
        return fallback

    masks: list[tf.Tensor] = []
    for idx, pusch_config in enumerate(pusch_configs):
        grid = _dmrs_grid_from_pusch_config(pusch_config)
        if grid is None:
            stream_count = int(fallback.shape[-1] or tf.shape(fallback)[-1].numpy())
            stream_idx = min(idx, max(stream_count - 1, 0))
            masks.append(tf.cast(fallback[..., stream_idx], tf.float32))
            continue
        masks.append(
            _grid_mask_to_tf_time_freq(
                grid,
                target_time=target_time,
                target_freq=target_freq,
                context=f"PUSCHConfig[{idx}].dmrs_grid",
            )
        )

    return tf.stack(masks, axis=-1)
'''

if anchor not in s:
    raise SystemExit("[PATCH] Could not locate extract_pilot_mask() anchor in builders.py.")
if "def extract_true_dmrs_mask_per_stream" not in s:
    s = s.replace(anchor, new)
else:
    print("[PATCH] builders.py already has extract_true_dmrs_mask_per_stream(); leaving existing function.")
p.write_text(s)
print("[PATCH] builders.py: true DMRS-mask extractor available.")

# ----------------------------
# 2) estimator.py: accept optional pilot_mask argument in constructor
# ----------------------------
p = Path("src/upair5g/estimator.py")
s = p.read_text()

old_sig = '''        resource_grid: Any,
        cfg: dict[str, Any],
        **kwargs: Any,
'''
new_sig = '''        resource_grid: Any,
        cfg: dict[str, Any],
        pilot_mask: tf.Tensor | None = None,
        **kwargs: Any,
'''
if "pilot_mask: tf.Tensor | None = None" not in s:
    if old_sig not in s:
        raise SystemExit("[PATCH] Could not locate estimator __init__ signature block.")
    s = s.replace(old_sig, new_sig, 1)

old_line = '''        self.pilot_mask = tf.cast(extract_pilot_mask_per_stream(resource_grid), tf.float32)
'''
new_line = '''        self.pilot_mask = tf.cast(
            tf.convert_to_tensor(pilot_mask) if pilot_mask is not None else extract_pilot_mask_per_stream(resource_grid),
            tf.float32,
        )
'''
if old_line in s:
    s = s.replace(old_line, new_line, 1)
elif "tf.convert_to_tensor(pilot_mask) if pilot_mask is not None" in s:
    pass
else:
    raise SystemExit("[PATCH] Could not locate estimator self.pilot_mask assignment.")
p.write_text(s)
print("[PATCH] estimator.py: constructor supports exact pilot_mask override.")

# ----------------------------
# 3) training.py: use true DMRS mask in systems and estimator constructor
# ----------------------------
p = Path("src/upair5g/training.py")
s = p.read_text()

s = s.replace(
    "from .builders import build_channel, build_ls_estimator, build_pusch_transmitter, extract_pilot_mask_per_stream, get_resource_grid, max_num_users, multiuser_enabled",
    "from .builders import build_channel, build_ls_estimator, build_pusch_transmitter, extract_true_dmrs_mask_per_stream, get_resource_grid, max_num_users, multiuser_enabled",
)
s = s.replace(
    '''        "pilot_mask": extract_pilot_mask_per_stream(resource_grid),
''',
    '''        "pilot_mask": extract_true_dmrs_mask_per_stream(tx, resource_grid),
''',
)
s = s.replace(
    '''    estimator = UPAIRChannelEstimator(
        ls_estimator=reference_system["ls_estimator"],
        resource_grid=reference_system["resource_grid"],
        cfg=cfg,
    )
''',
    '''    estimator = UPAIRChannelEstimator(
        ls_estimator=reference_system["ls_estimator"],
        resource_grid=reference_system["resource_grid"],
        cfg=cfg,
        pilot_mask=reference_system["pilot_mask"],
    )
''',
)
p.write_text(s)
print("[PATCH] training.py: systems use true DMRS masks.")

# ----------------------------
# 4) evaluation.py: use true DMRS mask for estimator self-mask
# ----------------------------
p = Path("src/upair5g/evaluation.py")
s = p.read_text()

s = s.replace(
    "from .builders import build_channel, build_ls_estimator, build_pusch_transmitter, build_receiver, get_resource_grid, max_num_users, multiuser_enabled",
    "from .builders import build_channel, build_ls_estimator, build_pusch_transmitter, build_receiver, extract_true_dmrs_mask_per_stream, get_resource_grid, max_num_users, multiuser_enabled",
)
s = s.replace(
    '''    ls_estimator = build_ls_estimator(tx, cfg, interpolation_type="lin")
    estimator = UPAIRChannelEstimator(ls_estimator=ls_estimator, resource_grid=get_resource_grid(tx), cfg=cfg)
''',
    '''    resource_grid = get_resource_grid(tx)
    pilot_mask = extract_true_dmrs_mask_per_stream(tx, resource_grid)
    ls_estimator = build_ls_estimator(tx, cfg, interpolation_type="lin")
    estimator = UPAIRChannelEstimator(ls_estimator=ls_estimator, resource_grid=resource_grid, cfg=cfg, pilot_mask=pilot_mask)
''',
)
p.write_text(s)
print("[PATCH] evaluation.py: proposed receiver estimator uses true DMRS self-mask.")

# ----------------------------
# 5) Update .gitignore runtime-output rules
# ----------------------------
p = Path(".gitignore")
s = p.read_text() if p.exists() else ""
items = [
    "optuna/",
    "logs/",
    "TWC_plots_comprehensive/",
    "outputs/",
    "plots/",
    "metrics/",
    "checkpoints/",
    "artifacts/",
    "*.out",
    "*.err",
    "*.log",
    "patch_backups/",
]
lines = s.splitlines()
for item in items:
    if item not in lines:
        lines.append(item)
p.write_text("\n".join(lines).rstrip() + "\n")
print("[PATCH] .gitignore: runtime-output rules present.")
PY

echo "[PATCH] Done. Run: bash upair_probe_true_dmrs_mask_patch.sh"
