# fitcheck

**Will your training run fit?** Measured GPU, memory, time and cost estimates for fine-tuning SLMs/LLMs on HPC clusters, plus a ready-to-submit uv + Slurm setup.

You pick the model and the method (full FT, LoRA, QLoRA). fitcheck tells you what it needs, from real runs on Leonardo (CINECA) A100 64 GB nodes, with the estimate's error shown, and writes the job script for you.

> Status: planning. Built during the European AI Hackathon (Oct 2026) by Team Alembic.

See [GOAL.md](GOAL.md) for the plan, MVP scope and timeline.

## Shared workload configuration

The CPU-only `TrainingSpec` validates workload settings and provides canonical
JSON and a SHA-256 configuration fingerprint. See the [usage and defaults](docs/training-spec.md)
and [example configuration](examples/training-spec.json). Requires Python 3.10+;
no third-party dependencies. Run offline tests from the repository root:

```sh
python3 -S -m unittest discover -s tests -v
```

## License

MIT
