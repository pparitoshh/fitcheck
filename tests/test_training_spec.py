"""Offline contract tests; run with python3 -S -m unittest discover -s tests -v."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import unittest

from fitcheck import FSDPConfig, LoRAConfig, TrainingSpec


def full_spec(**changes):
    values = dict(model_id="Qwen/Qwen3-0.6B", model_revision="test-revision",
                  training_method="full", sequence_length=512, micro_batch_size=1,
                  max_steps=50)
    return TrainingSpec(**(values | changes))


def adapter(**changes):
    return LoRAConfig(**(dict(rank=16, alpha=32, target_modules=("all-linear",)) | changes))


def fsdp(**changes):
    return FSDPConfig(**(dict(sharding_strategy="full_shard",
                             wrap_layer_classes=("Qwen3DecoderLayer",),
                             use_orig_params=True) | changes))


class TrainingSpecTests(unittest.TestCase):
    def test_mvp_defaults(self):
        spec = full_spec()
        self.assertEqual((spec.optimizer, spec.precision, spec.gradient_checkpointing,
                          spec.packing, spec.gpu_count, spec.gradient_accumulation_steps),
                         ("adamw", "bf16", True, True, 1, 1))
        self.assertEqual(spec.schema_version, 1)
        self.assertEqual(spec.quantization, "none")

    def test_valid_methods_durations_and_distributed_configs(self):
        specs = [full_spec(), full_spec(max_steps=None, num_epochs=1.5),
                 full_spec(training_method="lora", lora=adapter()),
                 full_spec(training_method="qlora", lora=adapter(), quantization="nf4"),
                 full_spec(training_method="qlora", lora=adapter(), quantization="fp4",
                           double_quantization=True)]
        for gpu_count in (2, 3, 4):
            specs.extend([
                full_spec(gpu_count=gpu_count, distributed_strategy="fsdp", fsdp=fsdp()),
                full_spec(training_method="lora", lora=adapter(), gpu_count=gpu_count,
                          distributed_strategy="fsdp", fsdp=fsdp())])
        for spec in specs:
            with self.subTest(spec=spec):
                self.assertEqual(TrainingSpec.from_json(spec.to_json()), spec)
                self.assertEqual(TrainingSpec.from_dict(spec.to_dict()), spec)

    def test_invalid_types_and_ranges(self):
        cases = {"model_id": ["", " spaced ", 12], "model_revision": ["", None],
                 "sequence_length": [0, -1, True, 512.0, "512"],
                 "micro_batch_size": [0, False, 1.5],
                 "gradient_accumulation_steps": [0, True, "2"],
                 "max_steps": [0, -1, True, 50.0], "gpu_count": [0, 5, True, 1.0],
                 "precision": ["int8", None], "optimizer": ["unknown"],
                 "packing": [1, "true"], "gradient_checkpointing": [0, None],
                 "double_quantization": [1], "training_method": ["distillation", "unknown"],
                 "quantization": ["int8"], "distributed_strategy": ["ddp", "zero"]}
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    full_spec(**{field: value})

    def test_invalid_duration(self):
        for changes in ({"max_steps": None}, {"num_epochs": 1},
                        *({"max_steps": None, "num_epochs": value}
                          for value in (0, -1, True, "1", float("nan"), float("inf")))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                full_spec(**changes)

    def test_incompatible_combinations(self):
        cases = [dict(lora=adapter()), dict(training_method="lora"),
                 dict(training_method="qlora", lora=adapter()),
                 dict(quantization="nf4"), dict(double_quantization=True),
                 dict(training_method="lora", lora=adapter(), quantization="nf4"),
                 dict(training_method="qlora", lora=adapter(), quantization="nf4",
                      gpu_count=4, distributed_strategy="fsdp", fsdp=fsdp()),
                 dict(gpu_count=2), dict(distributed_strategy="fsdp", fsdp=fsdp()),
                 dict(gpu_count=4, distributed_strategy="fsdp"), dict(fsdp=fsdp()),
                 dict(training_method="lora", lora=adapter(), gpu_count=4,
                      distributed_strategy="fsdp", fsdp=fsdp(use_orig_params=False)),
                 dict(lora={}), dict(fsdp={})]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                full_spec(**changes)

    def test_adapter_and_fsdp_validation(self):
        for changes in (dict(rank=0), dict(rank=True), dict(alpha=0), dict(alpha=True),
                        dict(alpha=float("inf")), dict(dropout=-0.1), dict(dropout=1),
                        dict(dropout=float("nan")), dict(target_modules=[]),
                        dict(target_modules="all-linear"), dict(target_modules=[""]),
                        dict(target_modules=["q_proj", "q_proj"]),
                        dict(target_modules=["all-linear", "q_proj"])):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                adapter(**changes)
        for changes in (dict(sharding_strategy="hybrid_shard"),
                        dict(wrap_layer_classes=[]), dict(wrap_layer_classes=[1]),
                        dict(use_orig_params="true"), dict(cpu_offload=1)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                fsdp(**changes)

    def test_json_input_is_strict(self):
        spec = full_spec().to_dict()
        for data in ([], None, {}, spec | {"typo": 1}, spec | {"max_steps": "50"},
                     spec | {"training_method": "lora", "lora": {}},
                     spec | {"lora": []}, spec | {"fsdp": "auto"},
                     spec | {"fsdp": {"typo": 1}},
                     spec | {"lora": {"rank": 16, "alpha": 32,
                                       "target_modules": ["all-linear"], "typo": 1}}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                TrainingSpec.from_json(json.dumps(data))
        for text in ('{', '{"schema_version":1,"schema_version":1}',
                     '{"lora":{"rank":16,"rank":32}}', '{"num_epochs":NaN}',
                     '{"num_epochs":Infinity}', '{"num_epochs":-Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                TrainingSpec.from_json(text)

    def test_schema_version(self):
        for version in (0, 2, "1", 1.0, True, None):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "schema_version"):
                    full_spec(schema_version=version)
                with self.assertRaisesRegex(ValueError, "schema_version"):
                    TrainingSpec.from_dict(full_spec().to_dict() | {"schema_version": version})

    def test_fingerprint_canonicalization(self):
        spec = full_spec(training_method="lora", lora=adapter())
        data = spec.to_dict()
        data["lora"]["alpha"] = 32.0
        data["lora"] = dict(reversed(list(data["lora"].items())))
        reordered = json.dumps(dict(reversed(list(data.items()))), indent=4)
        self.assertEqual(spec.fingerprint(), TrainingSpec.from_json(reordered).fingerprint())
        minimal = dict(model_id=spec.model_id, model_revision=spec.model_revision,
                       training_method="lora", sequence_length=512, micro_batch_size=1,
                       max_steps=50, lora=dict(rank=16, alpha=32, target_modules=["all-linear"]))
        self.assertEqual(spec.fingerprint(), TrainingSpec.from_dict(minimal).fingerprint())
        self.assertEqual(spec.fingerprint(), hashlib.sha256(spec.to_json().encode()).hexdigest())
        self.assertIn('"schema_version":1', spec.to_json())
        self.assertEqual(len(spec.fingerprint()), 64)
        a = replace(spec, lora=adapter(target_modules=["v_proj", "q_proj"], dropout=-0.0))
        b = replace(spec, lora=adapter(target_modules=("q_proj", "v_proj"), dropout=0))
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertEqual(full_spec(max_steps=None, num_epochs=2).fingerprint(),
                         full_spec(max_steps=None, num_epochs=2.0).fingerprint())

    def test_every_material_field_changes_fingerprint(self):
        base = full_spec()
        changes = [dict(model_id="Qwen/Qwen3-1.7B"), dict(model_revision="another-revision"),
                   dict(sequence_length=2048), dict(micro_batch_size=4), dict(max_steps=51),
                   dict(max_steps=None, num_epochs=2), dict(gradient_accumulation_steps=2),
                   dict(precision="fp16"), dict(optimizer="sgd"), dict(packing=False),
                   dict(gradient_checkpointing=False),
                   dict(training_method="lora", lora=adapter()),
                   dict(gpu_count=4, distributed_strategy="fsdp", fsdp=fsdp())]
        for change in changes:
            with self.subTest(change=change):
                self.assertNotEqual(base.fingerprint(), replace(base, **change).fingerprint())
        lora_spec = full_spec(training_method="qlora", lora=adapter(), quantization="nf4")
        for change in (dict(rank=32), dict(alpha=64), dict(dropout=0.1),
                       dict(target_modules=("q_proj",))):
            self.assertNotEqual(lora_spec.fingerprint(),
                                replace(lora_spec, lora=adapter(**change)).fingerprint())
        for change in (dict(quantization="fp4"), dict(double_quantization=True),
                       dict(training_method="lora", quantization="none")):
            self.assertNotEqual(lora_spec.fingerprint(), replace(lora_spec, **change).fingerprint())
        distributed = full_spec(gpu_count=4, distributed_strategy="fsdp", fsdp=fsdp())
        for change in (dict(sharding_strategy="shard_grad_op"), dict(cpu_offload=True),
                       dict(use_orig_params=False), dict(wrap_layer_classes=("OtherLayer",))):
            self.assertNotEqual(distributed.fingerprint(),
                                replace(distributed, fsdp=fsdp(**change)).fingerprint())

    def test_coverage_advisories(self):
        self.assertIn("No measured calibration", full_spec().coverage_warnings()[0])
        changes = dict(precision="fp32", optimizer="adam", gradient_checkpointing=False,
                       packing=False, sequence_length=4096, micro_batch_size=2,
                       model_id="custom/model")
        warnings = " ".join(full_spec(**changes).coverage_warnings())
        for field in changes:
            self.assertIn(field, warnings)
        self.assertIn("single-GPU full FT", warnings)
        two_gpu = full_spec(gpu_count=2, distributed_strategy="fsdp", fsdp=fsdp())
        self.assertTrue(any("outside the planned FSDP grid" in w for w in two_gpu.coverage_warnings()))
        planned = replace(two_gpu, gpu_count=4, model_id="Qwen/Qwen3-4B", sequence_length=2048)
        self.assertFalse(any("outside the planned FSDP grid" in w for w in planned.coverage_warnings()))
        offloaded = replace(planned, fsdp=fsdp(cpu_offload=True), gradient_accumulation_steps=2)
        self.assertTrue(any("no_sync()" in w for w in offloaded.coverage_warnings()))

    def test_immutable_and_detached_from_input(self):
        targets = ["q_proj"]
        spec = full_spec(training_method="lora", lora=adapter(target_modules=targets))
        fingerprint = spec.fingerprint()
        targets.append("v_proj")
        data = spec.to_dict()
        data["lora"]["target_modules"].append("k_proj")
        self.assertEqual(spec.fingerprint(), fingerprint)
        with self.assertRaises(FrozenInstanceError):
            spec.max_steps = 10
        with self.assertRaises(FrozenInstanceError):
            spec.lora.rank = 8

    def test_example(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "training-spec.json"
        spec = TrainingSpec.from_json(path.read_text())
        self.assertEqual(spec.training_method, "lora")
        self.assertEqual(spec, TrainingSpec.from_json(spec.to_json()))
        # Freeze the v1 wire format, including defaults, across processes/releases.
        self.assertEqual(spec.fingerprint(),
                         "8604860afc70b08ebaf3f49f4c36adc1adc39e94cb9e4a13e9f3d857b8094112")


if __name__ == "__main__":
    unittest.main()
