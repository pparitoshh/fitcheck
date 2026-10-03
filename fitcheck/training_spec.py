"""Validated workload settings, independent of any training framework."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from typing import Literal


def _integer(name: str, value: object, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _boolean(name: str, value: object) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")


def _text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")


def _choice(name: str, value: object, choices: tuple[str, ...]) -> None:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {choices}")


def _real(name: str, value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return 0.0 if result == 0 else result


def _names(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty list or tuple of names")
    for item in value:
        _text(name, item)
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicate names")
    return tuple(sorted(value))


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    """Explicit adapter choices; GOAL.md does not pin rank or scaling."""

    rank: int
    alpha: float
    target_modules: tuple[str, ...]
    dropout: float = 0.0

    def __post_init__(self) -> None:
        _integer("lora.rank", self.rank)
        alpha = _real("lora.alpha", self.alpha)
        dropout = _real("lora.dropout", self.dropout)
        if alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("lora.alpha must be positive and dropout must be in [0, 1)")
        targets = _names("lora.target_modules", self.target_modules)
        if "all-linear" in targets and len(targets) != 1:
            raise ValueError("all-linear must be the only lora.target_modules entry")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "dropout", dropout)
        object.__setattr__(self, "target_modules", targets)


@dataclass(frozen=True, slots=True)
class FSDPConfig:
    """Single-node FSDP choices; wrapping must be specified for the model."""

    sharding_strategy: Literal["full_shard", "shard_grad_op", "no_shard"]
    wrap_layer_classes: tuple[str, ...]
    use_orig_params: bool
    cpu_offload: bool = False

    def __post_init__(self) -> None:
        _choice("fsdp.sharding_strategy", self.sharding_strategy,
                ("full_shard", "shard_grad_op", "no_shard"))
        _boolean("fsdp.use_orig_params", self.use_orig_params)
        _boolean("fsdp.cpu_offload", self.cpu_offload)
        object.__setattr__(self, "wrap_layer_classes",
                           _names("fsdp.wrap_layer_classes", self.wrap_layer_classes))


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    """The version-1 workload contract shared by all fitcheck consumers.

    Construct directly or use from_dict/from_json at input boundaries. Invalid
    inputs raise ValueError. Consult coverage_warnings before reporting estimates.
    """

    model_id: str
    model_revision: str
    training_method: Literal["full", "lora", "qlora"]
    sequence_length: int
    micro_batch_size: int
    max_steps: int | None = None
    num_epochs: float | None = None
    gradient_accumulation_steps: int = 1
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    quantization: Literal["none", "nf4", "fp4"] = "none"
    double_quantization: bool = False
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    gradient_checkpointing: bool = True
    packing: bool = True
    gpu_count: int = 1
    distributed_strategy: Literal["none", "fsdp"] = "none"
    lora: LoRAConfig | None = None
    fsdp: FSDPConfig | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported schema_version; expected integer 1")
        _text("model_id", self.model_id)
        _text("model_revision", self.model_revision)
        _choice("training_method", self.training_method, ("full", "lora", "qlora"))
        for name in ("sequence_length", "micro_batch_size", "gradient_accumulation_steps",
                     "gpu_count"):
            _integer(name, getattr(self, name))
        if self.gpu_count > 4:
            raise ValueError("the MVP supports one node with 1–4 GPUs")
        if (self.max_steps is None) == (self.num_epochs is None):
            raise ValueError("specify exactly one of max_steps or num_epochs")
        if self.max_steps is not None:
            _integer("max_steps", self.max_steps)
        if self.num_epochs is not None:
            epochs = _real("num_epochs", self.num_epochs)
            if epochs <= 0:
                raise ValueError("num_epochs must be positive")
            object.__setattr__(self, "num_epochs", epochs)
        _choice("precision", self.precision, ("bf16", "fp16", "fp32"))
        _choice("quantization", self.quantization, ("none", "nf4", "fp4"))
        _choice("optimizer", self.optimizer, ("adamw", "adam", "sgd"))
        _choice("distributed_strategy", self.distributed_strategy, ("none", "fsdp"))
        for name in ("gradient_checkpointing", "packing", "double_quantization"):
            _boolean(name, getattr(self, name))
        if self.lora is not None and not isinstance(self.lora, LoRAConfig):
            raise ValueError("lora must be a LoRAConfig; use from_dict for JSON objects")
        if self.fsdp is not None and not isinstance(self.fsdp, FSDPConfig):
            raise ValueError("fsdp must be an FSDPConfig; use from_dict for JSON objects")
        if (self.training_method == "full") != (self.lora is None):
            raise ValueError("LoRA/QLoRA require lora settings; full FT forbids them")
        if (self.training_method == "qlora") != (self.quantization != "none"):
            raise ValueError("QLoRA requires nf4/fp4; full FT and LoRA require no quantization")
        if self.double_quantization and self.quantization == "none":
            raise ValueError("double_quantization requires quantization")
        if self.training_method == "qlora" and self.gpu_count != 1:
            raise ValueError("multi-GPU QLoRA is unsupported in the MVP")
        if (self.gpu_count > 1) != (self.distributed_strategy == "fsdp"):
            raise ValueError("use none for 1 GPU and fsdp for 2–4 GPUs")
        if (self.distributed_strategy == "fsdp") != (self.fsdp is not None):
            raise ValueError("fsdp settings are required exactly when strategy is fsdp")
        if self.fsdp and self.lora and not self.fsdp.use_orig_params:
            raise ValueError("MVP FSDP LoRA requires use_orig_params=true for mixed frozen/trainable parameters")

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingSpec":
        if not isinstance(data, dict):
            raise ValueError("TrainingSpec must be a JSON object")
        data = data.copy()
        for name, config_type in (("lora", LoRAConfig), ("fsdp", FSDPConfig)):
            if data.get(name) is not None:
                data[name] = _from_dict(config_type, data[name])
        return _from_dict(cls, data)

    @classmethod
    def from_json(cls, text: str) -> "TrainingSpec":
        return cls.from_dict(json.loads(text, object_pairs_hook=_unique_object,
                                        parse_constant=_invalid_constant))

    def to_dict(self) -> dict:
        """Return an independent JSON-compatible object with all defaults resolved."""
        return json.loads(self.to_json())

    def to_json(self) -> str:
        """Canonical v1 JSON: sorted keys, compact separators, ASCII, no NaN."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def coverage_warnings(self) -> tuple[str, ...]:
        """Advisories, never a claim of measured/calibrated coverage.

        GOAL.md defines a planned grid, but no measurements ship yet. Future
        calculators must additionally match the measured hardware/software stack.
        """
        warnings = ["No measured calibration data is available; planned-grid membership is not calibrated coverage."]
        for name, pinned in (("precision", "bf16"), ("optimizer", "adamw"),
                             ("gradient_checkpointing", True), ("packing", True)):
            if getattr(self, name) != pinned:
                warnings.append(f"{name} differs from the MVP pinned setting {pinned!r}.")
        small_models = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")
        large_models = ("Qwen/Qwen3-4B", "Qwen/Qwen3-8B")
        if self.model_id not in small_models + large_models:
            warnings.append("model_id is outside the planned Qwen3 grid; parameter count and context limit are unverified.")
        if self.gpu_count == 1:
            if self.sequence_length not in (512, 2048):
                warnings.append("sequence_length is outside the planned 512/2048 grid.")
            if self.micro_batch_size not in (1, 4):
                warnings.append("micro_batch_size is outside the planned 1/4 grid.")
            if self.training_method == "full" and self.model_id not in small_models:
                warnings.append("single-GPU full FT is only planned for Qwen3-0.6B/1.7B.")
        elif not (self.gpu_count == 4 and self.model_id in large_models
                  and self.sequence_length == 2048 and self.micro_batch_size == 1):
            warnings.append("outside the planned FSDP grid: 4 GPUs, Qwen3-4B/8B, seq 2048, micro-batch 1.")
        if self.fsdp and self.fsdp.cpu_offload and self.gradient_accumulation_steps > 1:
            warnings.append("FSDP CPU offload with accumulation requires runtime no_sync() on accumulation micro-steps.")
        warnings.append("Model revision, accumulation, duration, quantization and LoRA/FSDP details are not pinned by the calibration plan.")
        return tuple(warnings)


def _from_dict(config_type, data):
    if not isinstance(data, dict):
        raise ValueError(f"{config_type.__name__} must be a JSON object")
    unknown = data.keys() - {field.name for field in fields(config_type)}
    if unknown:
        raise ValueError(f"unknown {config_type.__name__} fields: {sorted(unknown, key=str)}")
    try:
        return config_type(**data)
    except TypeError as exc:
        raise ValueError(f"invalid {config_type.__name__}: {exc}") from exc


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")
