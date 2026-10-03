# Shared TrainingSpec (schema version 1)

`fitcheck.training_spec` is the single workload contract for the planned harness,
calculator, CLI and setup generator. It uses Python 3.10+ and only the standard
library. There is no training framework import, network access or model download.

From the repository root:

```python
from pathlib import Path
from fitcheck import TrainingSpec

spec = TrainingSpec.from_json(Path("examples/training-spec.json").read_text())
print(spec.to_json())                 # canonical JSON, including resolved defaults
print(spec.fingerprint())             # SHA-256 hex digest of that JSON
for warning in spec.coverage_warnings():
    print(warning)
assert TrainingSpec.from_dict(spec.to_dict()) == spec
```

Run the offline tests with `python3 -S -m unittest discover -s tests -v`.
`-S` disables site packages; no dependency installation is needed.

## Fields and defaults

| Field | Meaning / default |
|---|---|
| `schema_version` | Integer `1`; omission means v1, not “latest”. Other versions are rejected. |
| `model_id`, `model_revision` | Required nonempty strings. Use a resolved immutable revision for reproducible runs. |
| `training_method` | Required: `full`, `lora` or `qlora`. Distillation is outside the revised MVP. |
| `sequence_length`, `micro_batch_size` | Required positive integers; tokens per sequence and sequences **per device**, respectively. |
| `gradient_accumulation_steps` | Positive integer, default `1`. Effective global batch = micro-batch × accumulation × GPU count. |
| `max_steps`, `num_epochs` | Exactly one required: positive integer **optimizer update** count, or positive finite epoch count (fractional allowed). No implicit precedence. |
| `precision` | `bf16` (MVP default), `fp16` or `fp32`; training compute precision, separate from weight quantization. |
| `quantization` | `none` by default; QLoRA requires explicit `nf4` or `fp4` (4-bit base weights). Other methods require `none`. |
| `double_quantization` | Boolean, default `false`; requires quantized weights. |
| `optimizer` | `adamw` (MVP default), `adam` or `sgd`. Other optimizer implementations are not represented in v1. |
| `gradient_checkpointing`, `packing` | Booleans, both default `true`, as pinned in GOAL.md. Packing means fixed-length packed batches. |
| `gpu_count` | Integer 1–4, default `1`. One node only. |
| `distributed_strategy` | `none` (default) for one GPU; explicit `fsdp` for 2–4 GPUs. DDP/ZeRO and single-GPU FSDP are outside this MVP contract. |
| `lora` | Required for LoRA/QLoRA, forbidden for full FT. See below. |
| `fsdp` | Required for FSDP, otherwise forbidden. See below. |

`LoRAConfig` requires positive integer `rank`, positive finite `alpha`, and a
nonempty `target_modules` list/tuple of names (or `["all-linear"]` alone).
`dropout` defaults to `0.0` and must be in `[0, 1)`. No rank/alpha/target default is
inferred: RESEARCH.md proposes rank 16–64 and all linear layers for the assistant,
but does not fix them for the benchmark grid. V1 models ordinary LoRA scaling,
without trainable bias, DoRA or rank-stabilized variants.

`FSDPConfig` requires explicit `sharding_strategy` (`full_shard`, `shard_grad_op`
or `no_shard`), `wrap_layer_classes` (nonempty names), and boolean
`use_orig_params`; boolean `cpu_offload` defaults to `false`. These are FSDP1-style
settings with transformer-layer class wrapping. LoRA with this wrapping requires
`use_orig_params=true`: [PyTorch documents the frozen/trainable parameter
restriction](https://docs.pytorch.org/docs/stable/fsdp.html). Specialized wrapping
that separates adapter parameters and FSDP2 settings are not represented. Runtime
adapters must validate layer names and supported library options before execution.
CPU offload with gradient accumulation also produces an advisory: the runtime
must use `no_sync()` during accumulation micro-steps, per the same PyTorch docs.
**Multi-GPU QLoRA is rejected**, consistent with GOAL.md's MVP boundary.

Direct Python construction uses `LoRAConfig` / `FSDPConfig` instances; JSON and
`from_dict` use nested objects. Instances are immutable. Invalid values,
incompatible combinations, unknown fields, duplicate JSON keys and nonfinite
numbers raise `ValueError`; missing required constructor arguments raise Python's
`TypeError` (converted to `ValueError` by `from_dict` / `from_json`). Numeric strings
and booleans in numeric fields are rejected.

## Coverage and unresolved choices

Validation is distinct from calibration. `coverage_warnings()` always reports that
no measured calibration dataset exists yet. It also flags deviations from the
pinned defaults and planned grid: single GPU Qwen3 0.6B/1.7B/4B/8B, seq 512/2048,
micro-batch 1/4, with full FT only at 0.6B/1.7B; four GPUs at 4B/8B, seq 2048,
micro-batch 1, LoRA/full FT. Two/three GPUs, other models, disabled packing or
checkpointing, and other listed precision/optimizer options are accepted with
advisories. A valid object does not establish that a workload fits in memory.

The goals do not pin revisions, accumulation, adapter settings, FSDP details or
the QLoRA quantization variant. Those remain uncalibrated choices. Defaults of
accumulation 1, adapter dropout 0, no double quantization and no CPU offload are
v1 contract choices, **not measured grid settings**. The example's rank 16, alpha
32 and 50 optimizer steps are illustrative. Its revision `main` is mutable;
replace it with the downloaded snapshot's commit before recording a benchmark.

Arbitrary model identifiers are allowed but flagged: without fetching metadata we
cannot verify the MVP's ≤8B parameter limit, context length, revision existence,
target modules or wrapping classes. Runtime consumers must check these. Epochs
require dataset size/accounting before a calculator can derive steps; the spec
does not profile datasets. Hardware identity, library pins, dataset identity,
learning-rate schedules, checkpoint I/O, and benchmark warm-up/measurement windows
belong to future run metadata/execution policies, not implicit training defaults.

## Serialization and intended integration

The v1 canonical format is `json.dumps(..., sort_keys=True, separators=(",", ":"),
ensure_ascii=True, allow_nan=False)` over all fields, including nulls and resolved
defaults. Numeric real fields normalize integers to floats and negative zero to
zero. Module/class lists are unordered sets of unique names, serialized in sorted
order. SHA-256 hashes the UTF-8 bytes. Key order, whitespace and omitted defaults
do not affect the fingerprint; material field changes do. This is a defined v1
Python JSON format, not a claim of RFC 8785 compliance. Consumers must use this
implementation rather than hashing arbitrary input JSON.

The fingerprint identifies configuration, not model contents or a complete run.
A mutable revision can point to changed weights without changing the fingerprint.
Hardware, software and dataset metadata must accompany it for reproducibility.
Changing defaults or semantics requires an explicit schema-version/migration
decision; unsupported versions are never silently upgraded.

No consumer code exists on the agreed base branch. Intended integration:

- **Harness:** accept `TrainingSpec`, translate it into trainer settings, and log
  canonical JSON/fingerprint beside hardware, versions, git revision and metrics.
- **Calculator:** accept the same object, expose coverage warnings, then match
  measured configuration and environment before claiming calibrated coverage.
- **CLI/assistant:** validate through `from_dict`/`from_json`, ask for missing
  required choices, and display advisories. Do not build a second workload schema.
- **Setup generator:** render this same spec into training/Slurm inputs, validate
  runtime compatibility, and retain the fingerprint in generated run metadata.
