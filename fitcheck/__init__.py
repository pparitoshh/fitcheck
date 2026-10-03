"""CPU-only shared configuration for fitcheck."""

from .training_spec import FSDPConfig, LoRAConfig, TrainingSpec

__all__ = ["FSDPConfig", "LoRAConfig", "TrainingSpec"]
