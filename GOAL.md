# fitcheck — Goals & Expected Outcomes

*European AI Hackathon · Oct 6–29, 2026 · Team Alembic (5 people) · Leonardo (CINECA)*
*Self-contained project: every goal, piece of code and dataset below is built inside this repo. Nothing depends on another project.*

---

## 1. Mission

**Help people who are new to HPC train SLMs/LLMs on a cluster without wasting compute.**

The user picks the model and the training method. We tell them, from **measured runs on real hardware**, what it will need and cost (GPUs, memory, time, GPU-hours, €), and we generate a uv + Slurm setup that works the first time. A small local assistant (Qwen3-0.6B) turns plain-language requests into these tool calls; the same tools also work from Claude/ChatGPT.

## 2. The problem

- Newcomers get cluster time (EuroHPC, university clusters) but have no HPC background. They over-request or under-request resources, hit OOM after waiting in the queue, and lose runs to the 24h wall-time limit.
- GPU waste is well documented: in one Princeton study, 52% of users had at least one job with an idle GPU, and users routinely over-request memory and CPUs (see §14).
- Existing VRAM/cost calculators (ftune, web calculators) and general chatbots **estimate from formulas**. None of them reports how far its estimates are from real runs.
- Cluster know-how lives in mentors' heads and emails. On Leonardo: no internet on compute nodes (download to shared storage first; outbound only via a reverse proxy through the login nodes), use uv/pixi instead of modules, run in `$WORK`/`$SCRATCH`, checkpoint because of the 24h limit.

## 3. Who it is for

| Persona | Situation | What they need from us |
|---|---|---|
| **HPC newcomer with an allocation** (primary) | Researcher, student or startup with EuroHPC / Leonardo time, no Slurm experience | "Will my job fit, how long, how many GPU-hours, and what script do I submit?" |
| **University Slurm cluster user** | Same problem on a different cluster | The same, with that cluster's rules |
| **Cloud GPU renter** | Pays per hour on RunPod/Lambda/AWS | The same estimates in € per run |

## 4. Design decisions

| Decision | Choice | Why |
|---|---|---|
| Method choice | **The user decides** (full FT / LoRA / QLoRA / distillation). We do **not** recommend methods. | Method advice goes stale quickly; measured costs stay true on that hardware. |
| Alternatives | Show **measured** costs of other methods/settings next to the user's choice, as facts | "QLoRA: 14 GB, 2.9h vs your LoRA: 38 GB, 2.1h" helps without giving advice |
| Scope | Models **up to 8B, one node (1–4× A100 64 GB)** | Covers most newcomer use cases and fits the shared reservation |
| Core differentiator | **Calculator calibrated on measured Leonardo runs**, reporting its own error | Claude/ChatGPT and public calculators don't have this |
| Cost units | **GPU-hours / node-hours** (allocations) and **€** (cloud, from an editable price table) | What allocation holders and cloud renters each count in |
| Interface | **CLI + thin MCP server** (usable from Claude/ChatGPT) **and the local Qwen3-0.6B assistant** | The value lives in the tools; the 0.6B model is the free, offline front end |
| Numbers | The model **never writes numbers**. It picks the tool and arguments; numbers are filled into a template from the tool output. | Small models make up numbers |
| Environment | uv-based, no modules | Mentor's advice for Leonardo; same as our local setup |
| Team reality | **MVP must be doable by 2 people working part-time.** Everything else is bolt-on. | That's how hackathons go |

## 5. Scope tiers

### MVP — must ship, sized for 2 part-time people

| # | Piece | Minimum version |
|---|---|---|
| M1 | **Benchmark harness** | One script + Slurm job array: warm-up, ~50 steps, logs peak memory, tokens/s, step time, OOM. Runs unattended. |
| M2 | **Small grid** | **~40–60 configs**: Qwen3 0.6B / 1.7B / 4B / 8B × LoRA / QLoRA (+ full FT up to 1.7B) × seq 512 / 2048 × 1 GPU, plus a few 4-GPU runs. ≈ 10 GPU-hours. |
| M3 | **Calculator** | Formula baseline + correction fitted on the grid. Outputs memory, OOM yes/no, time, GPU-hours, € with an error band. Tested on held-out configs and against one public calculator. |
| M4 | **Setup generator** | Templates for uv env + `sbatch` (checkpoint/resume, `$WORK`/`$SCRATCH`, `HF_HUB_OFFLINE=1`, pre-download step). Checked with `bash -n` + `sbatch --test-only`. |
| M5 | **CLI** | `plan` (estimates + measured alternatives) and `setup` (writes the files). |
| M6 | **0.6B assistant** | Qwen3-0.6B fine-tuned with a **new, simple SFT script written in this repo** (LoRA or full FT, using fitcheck's own setup generator) on teacher-generated request → tool-call pairs. Asks a follow-up question when info is missing. |
| M7 | **Case study** | Plan the 0.6B assistant's own training run with the tool, run it on Leonardo, report predicted vs. actual. |

### Should-have — bolt-ons for whoever has time (each independent of the others)

- **S1:** MCP server wrapping the CLI (small task, high value).
- **S2:** Bigger grid: 100–150 configs, seq 4096, 2-GPU runs, gradient-checkpointing settings, repeat runs for noise.
- **S3:** 5-minute smoke-test job generated alongside each script.
- **S4:** Leonardo knowledge base + "plan your training run on Leonardo" guide.
- **S5:** "Did it work?" step: loss sanity checks + small eval after the run.
- **S6:** Second cluster profile (a university Slurm cluster).

### Stretch

- Multi-node (2 nodes), models > 8B, live GPU-idle warnings from job statistics.

## 6. Goals (definition of done = MVP)

- **G1 — Measured grid:** ≥ 40 configurations measured on Leonardo, published as an open dataset.
- **G2 — Calibrated calculator:** on held-out configs it predicts OOM correctly and gives memory/time with stated error, **lower than at least one popular public calculator**.
- **G3 — Working setup:** generated scripts pass `bash -n` and `sbatch --test-only`; the case-study run (M7) uses one and finishes.
- **G4 — Local assistant:** fine-tuned 0.6B makes correct tool calls more often than base 0.6B on a held-out request set; runs on a laptop.
- **G5 — Case study:** predicted vs. actual memory and time for one full-length run, reported honestly.
- **G6 — Open release:** public MIT repo, grid dataset, README with a 5-minute quickstart.

## 7. Success metrics (provisional; fixed after the first grid results)

| Metric | Target |
|---|---|
| OOM prediction accuracy on held-out configs | ≥ 90% |
| Peak memory error (median absolute %) | ≤ 10% |
| Time error (median absolute %) | ≤ 20% |
| Error vs. a public calculator on the same configs | lower |
| Tool-call exact match (tool + arguments) on 100–200 held-out requests | beats base 0.6B; Claude reported as reference |
| Assistant on laptop (Q4_K_M GGUF) | < 4 GB RAM |

**Integrity rules:** split held-out configs before fitting; report errors per dimension (say where the calculator is bad); report negative results.

## 8. Compute plan and constraints

- **Accounts:** active Oct 6 (Day 0), new password around Oct 15, **expire Nov 2**. Consent form due **Oct 3, 23:59 CEST** for every member.
- **Async days:** shared project, jobs may queue → the MVP grid is short job-array jobs.
- **Hackathon days (Day 1 = Oct 13, Days 2–4):** **one shared node reservation for all teams.** Request due **Oct 8 EoD**; changes by Oct 15.
- **Budget:** MVP grid ≈ 10 GPU-hours, 0.6B training < 1 GPU-hour. S2 adds ≈ 20–30 GPU-hours.
- **Proposed request:** Day 1: 1 GPU (smoke tests, first grid jobs). Days 2–4: 1–2 GPUs.
- **Teacher** for assistant training data: external API through the reverse proxy.
- **Ask the mentor:** how many GPUs can our team use at once in the shared reservation?

## 9. Timeline with cut lines

| When | Milestone | Cut line if behind |
|---|---|---|
| **Oct 3** | All 5 members submit the consent form | — |
| **Oct 4–5** | Harness runs locally on 0.6B (LoRA/QLoRA); formula baseline; pick the public calculator to compare against | — |
| **Oct 6 (Day 0)** | Cluster intro; weights to `$WORK`; uv env on Leonardo; first grid job | — |
| **Oct 8** | Send the GPU request | — |
| **Oct 6–12** | MVP grid running (async); setup templates; teacher generates request → tool-call data | Drop 8B full-precision configs |
| **Oct 13–16 (Days 1–4)** | Finish MVP grid; fit + test calculator; CLI; train 0.6B (= case study run) | **Freeze grid at what's measured by Oct 16** |
| **Oct 17–26** | Evaluations; bolt-ons (S1–S6) by whoever has time | Ship MVP only |
| **Oct 27–29** | Release repo + dataset; demo | — |
| **Nov 2** | Accounts expire. All data copied off Leonardo before this. | — |

## 10. Team roles (to confirm after pitching to the team)

| Role | Owns | Commitment |
|---|---|---|
| **Core A** (Paritosh) | Harness, grid, calculator (M1–M3), integration | MVP owner |
| **Core B** | Setup generator, CLI, 0.6B assistant + case study (M4–M7) | MVP owner |
| **Contributors** (×3) | Pick any S-item; each is self-contained and nothing in the MVP waits on it | Part-time OK |

## 11. Non-goals

- Recommending a training method.
- Predicting model quality or exact data requirements (only cited rough guidance, labelled "not measured").
- Models > 8B or multi-node in the MVP.
- Supporting every cluster (Leonardo first).
- A GUI.

## 12. Built from scratch here (no reuse from other projects)

Earlier experiments in other repos are not dependencies. Anything they covered is **redone in this repo** and listed here:

| Item | What is redone | Where it lands |
|---|---|---|
| Training script for the assistant | New minimal SFT script (HF `transformers` + `peft`/`trl`), runnable locally and on Leonardo | M6 |
| Request → tool-call dataset | New teacher-generated data and held-out test set, made for the `plan`/`setup` tool schema | M6 |
| Evaluation of tool calls | New exact-match scorer (tool + arguments) | M6 / G4 |
| Benchmark and calculator code | All new | M1–M3 |

If a piece from an older project is copied in, it is rewritten to fit this repo and tested here; it is not imported.

## 13. Demo story

"I want to LoRA-fine-tune Qwen3-4B on 20k examples on Leonardo."
→ assistant asks for the sequence length → calls `plan` → "1× A100, ~22 GB, ~1.8 h, 1.8 GPU-hours, ~€4 on cloud; QLoRA alternative: 11 GB, 2.4 h" (with error bands) → calls `setup` → ready-to-submit `sbatch` + uv env.
Then: "We planned our own assistant's training this way. Predicted X, actual Y."

## 14. Evidence and references

- Princeton Research Computing on GPU utilisation: https://researchcomputing.princeton.edu/document/6891
- NERSC Perlmutter resource utilisation study: https://hgpu.org/?p=27716
- LANL HPC support ticket analysis: https://arxiv.org/abs/2010.04321
- Existing calculators: ftune (https://pypi.org/project/ftuneai/), https://inventivehq.com/tools/fine-tuning-vram-calculator
- Related agents: Hugging Face hf-llm-trainer skill (https://huggingface.co/blog/hf-skills-training), SlurmWise (https://github.com/armanbishal/slurmwise)
- CINECA storage areas: https://docs.hpc.cineca.it/hpc/hpc_data_storage.html
