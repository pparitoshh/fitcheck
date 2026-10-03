# fitcheck — Goals & Expected Outcomes

*European AI Hackathon · Oct 6–29, 2026 · Team Alembic (5 people) · Leonardo (CINECA)*

---

## 1. Mission

**Help people who are new to HPC train SLMs/LLMs on a cluster without wasting compute.**

The user picks the model and the training method. From **measured runs on real hardware**, we tell them what it will need and cost (GPUs, memory, time, GPU-hours, €). Every number comes with an **error band measured on held-out runs**, and we generate a uv + Slurm setup that works the first time. We support the whole climb: start small, see real numbers, scale up with confidence. A small local assistant (Qwen3-0.6B) turns plain-language requests into tool calls; the same tools also work from Claude/ChatGPT.

## 2. The problem

- Newcomers get cluster time (EuroHPC, university clusters) but have no HPC background. They over- or under-request resources, hit OOM after waiting in the queue, and lose runs to the 24h wall-time limit.
- GPU waste is well documented: in one Princeton study, 52% of users had at least one job with an idle GPU, and users routinely over-request memory and CPUs (see §17).
- Existing VRAM/cost calculators (ftune, web calculators) and general chatbots **estimate from formulas**. None reports how far its estimates are from real runs, so none can say how much headroom to request.
- Cluster know-how lives in mentors' heads and emails. On Leonardo: no internet on compute nodes (download to shared storage first; outbound only via a reverse proxy through the login nodes), use uv instead of modules, run in `$WORK`/`$SCRATCH`, checkpoint because of the 24h limit.

## 3. Who it is for

| Persona | Situation | What they need from us |
|---|---|---|
| **HPC newcomer with an allocation** (primary) | Researcher, student or startup with EuroHPC / Leonardo time, no Slurm experience | "Will my job fit, how long, how many GPU-hours, and what script do I submit?" |
| **University Slurm cluster user** | Same problem on a different cluster | The same, with that cluster's rules |
| **Cloud GPU renter** | Pays per hour on RunPod/Lambda/AWS | The same estimates in € per run (**transferred from Leonardo, not measured** on their GPU; labelled as such) |

## 4. Design decisions

| Decision | Choice | Why |
|---|---|---|
| Method choice | **The user decides** (full FT / LoRA / QLoRA). We do **not** recommend methods. | Method advice goes stale quickly; measured costs stay true on that hardware. |
| Alternatives | Show **measured** costs of other methods/settings next to the user's choice, as facts | "QLoRA: 14 GB, 2.9h vs your LoRA: 38 GB, 2.1h" helps without giving advice |
| Scope | Models **up to 8B, one node (1–4× A100 64 GB)** | Covers most newcomer use cases and fits the shared reservation |
| Multi-GPU | **FSDP only** in the MVP | DDP, FSDP and ZeRO use memory very differently; modelling one keeps the grid honest. Others are a bolt-on. |
| Pinned settings | Gradient checkpointing **on**, AdamW, bf16, fixed-length packed batches | Each free setting multiplies the grid. The CLI warns when the user's config differs from the pinned settings. |
| Core differentiator | **Calculator calibrated on measured Leonardo runs**, reporting its own held-out error | Claude/ChatGPT and public calculators don't have this |
| Cost units | **GPU-hours / node-hours** (allocations) and **€** (cloud, from an editable price table) | What allocation holders and cloud renters each count in |
| Interface | **CLI + thin MCP server** (usable from Claude/ChatGPT) **and the local Qwen3-0.6B assistant** | The value lives in the tools; the 0.6B model is the free, offline front end |
| Numbers | The model **never writes numbers**. It picks the tool and arguments; numbers are filled into a template from the tool output. | Small models make up numbers |
| Environment | uv-based, no modules | Mentor's advice for Leonardo; same as our local setup |

## 5. What makes this different (the wedge)

The baseline formula (weights + gradients + optimizer states + activations) is **not** novel; ftune and the web calculators already do it. Our contribution is not the algorithm. It is:

1. **A measured dataset**: real runs on real Leonardo hardware, published openly.
2. **A measured error band** on every number, taken from held-out runs. We say "24 GB, request 27 GB: 9 out of 10 held-out runs stayed under that", where everyone else gives a bare number and stops.

The band is what makes the number *actionable*: it tells the user how much headroom to request so they don't OOM. The algorithm is deliberately simple *because the data is the contribution*, which is a sturdier position than depending on a clever model.

**Positioning vs. scaling laws:** scaling laws (Kaplan, Chinchilla) predict model *quality* from size/data/compute. fitcheck predicts *cost and fit*: **"scaling laws tell you what to train; fitcheck tells you what it costs to train it."** We borrow the *shape* of their method (measure a few points, fit, extrapolate) but for resources, not loss. Predicting quality stays a non-goal.

## 6. How the estimate works (method)

1. **Analytical formula** as the baseline: parameters × bytes per parameter for weights, gradients and optimizer states, plus activations scaled by micro-batch and sequence length. This gets most of the way.
2. **Residual correction:** fit the **ratio** `log(measured / formula)`, not the difference. The overhead the formula misses (CUDA context, allocator fragmentation, kernel workspaces) grows with model size, so a ratio fits better than a fixed offset. Keep it simple: **linear regression on log features** (log params, log seq, log micro-batch, method as a category). Gradient boosting is ruled out: with ~50 points trees overfit and **cannot extrapolate** past the largest measured model, which would break the 8B prediction.
3. **Error band from held-out residuals (an empirical prediction interval):**
   - **Leave-one-model-size-out cross-validation**: fit on three sizes, predict the fourth, repeat for all four. Every config gets one held-out residual (~50 residuals, not ~10).
   - Report **extrapolation** (holding out 0.6B or 8B) and **interpolation** (holding out 1.7B or 4B) separately. Extrapolation is the case that matters to users scaling up, and it will have the wider band.
   - Memory gets a **one-sided upper bound** (90th percentile of held-out error): under-predicting causes an OOM, over-predicting only wastes a little. Time gets a two-sided band.
   - With ~50 residuals the 90th percentile is rough. We say so, and quote the max held-out error next to it.
   - The shipped calculator is then refit on all data; the band comes from the cross-validation.
4. **Noise floor:** 3 configs × 3 repeats measure run-to-run jitter. A band narrower than the jitter is not believable; we clamp it to that minimum.
5. **OOM** is a threshold rule: predicted upper-bound memory vs. usable card memory. It is not a separate model.

**On sample size:** the ~50 configs aren't 50 repeats of one experiment, so t-distribution logic doesn't apply. They are 50 *different* points pinning down a curve with few parameters. What matters is **spread** across the dimensions (model size, seq length, micro-batch, method), roughly ≥10 points per fitted coefficient, not the raw count. A well-spread 15 beats a lazy 40.

**Targets:** peak memory (GB) and step time (s) are the two continuous targets. GPU-hours, € and total time derive from them (`steps × step time × (1 + checkpoint overhead) + startup`). OOM derives from memory. Queue wait is not predicted.

## 7. Scope tiers

### MVP — must ship

| # | Piece | Minimum version |
|---|---|---|
| M1 | **Benchmark harness** | One script + Slurm job array: warm-up, ~50 steps, logs peak memory (`torch.cuda.max_memory_reserved` **and** the `nvidia-smi` peak, since OOM happens against the latter), median step time, tokens/s, OOM. Fixed-length packed batches so memory isn't noisy. Logs library versions (torch, CUDA, transformers, peft, bitsandbytes, flash-attn) and the git commit with every run. Runs unattended. |
| M2 | **Small grid** | **~50 configs** (see the table below), spread across all dimensions and including configs that are expected to sit **near or over** 64 GB, so the OOM rule is tested. ≈ 10 GPU-hours. 4-GPU runs go in the async window (Oct 6–12), not the shared-reservation days. |
| M3 | **Calculator** | Formula + residual correction (§6). Outputs memory, OOM yes/no, time, GPU-hours, €, **each with its error band**. Compared fairly against one public calculator: same batch, gradient checkpointing, optimizer and precision. |
| M4 | **Setup generator** | Templates for uv env + `sbatch` (checkpoint/resume, `$WORK`/`$SCRATCH`, `HF_HUB_OFFLINE=1`, pre-download step, `--time` from the upper time band). Runs longer than 24h are split into chained jobs. Checked with `bash -n` + `sbatch --test-only`. |
| M5 | **CLI** | `plan` (estimates + measured alternatives) and `setup` (writes the files). |
| M6 | **0.6B assistant** | Qwen3-0.6B fine-tuned with a **simple SFT script** (LoRA; full FT as fallback; see RESEARCH.md §1) on teacher-generated request → tool-call pairs. Asks a follow-up when required info is missing, and is **scored on whether it asks when it should**, not only on exact match. |
| M7 | **Case study** | Plan the assistant's own training run with the tool, run it on Leonardo, report predicted vs. actual. If budget allows, add one 4B/8B LoRA run so the case study isn't only the easiest config to predict. |

**MVP grid (M2):**

| Block | Dimensions | Configs |
|---|---|---|
| LoRA / QLoRA, 1 GPU | 4 sizes (0.6B/1.7B/4B/8B) × 2 methods × seq 512/2048 × micro-batch 1/4 | 32 |
| Full FT, 1 GPU | 0.6B/1.7B × seq 512/2048 × micro-batch 1/4 | 8 |
| FSDP, 4 GPUs | 4B/8B × LoRA/full FT, seq 2048, micro-batch 1 (some are expected to OOM) | ~4–6 |
| Repeats (noise floor) | 3 configs × 2 extra runs | 6 |
| | **Total** | **~50** |

Cut line if behind: drop micro-batch 4 for the 0.6B rows first, then the 8B full-FT FSDP row.

**Where each piece runs:**

| # | Where it runs | GPU? | LLM? |
|---|---|---|---|
| M1–M2 | Leonardo compute nodes (Slurm job array) | Yes, ≈ 10 GPU-hours | Fine-tunes Qwen3 models for ~50 steps **only to measure** memory and time; no model is asked to generate anything |
| M3 | Anywhere (laptop, login node, CI). Fitted once on the grid; the coefficients ship with the package | No | No: formula + linear regression, millisecond predictions |
| M4 | Anywhere. `bash -n` locally; `sbatch --test-only` on a Leonardo login node | No | No: fills templates with the user's config and M3's estimates |
| M5 | User's laptop or a Leonardo login node | No | No: `plan` calls M3, `setup` calls M4 |
| M6 | Training on Leonardo (< 1 GPU-hour); inference on a laptop (Q4_K_M GGUF) | Training only | Yes: Qwen3-0.6B is an **optional** front end that turns plain language into `plan`/`setup` calls. The teacher (external API, via the reverse proxy) is used once to generate its training data |
| M7 | Leonardo compute nodes | Yes | Trains M6's model |

The core product (M3–M5) is lightweight and works offline, including on login nodes. Users can skip the assistant and use the CLI directly, or call the same tools from Claude/ChatGPT via the MCP server (S1).

### Should-have — bolt-ons for whoever has time (each independent of the others)

- **S1:** MCP server wrapping the CLI (small task, high value).
- **S2:** Bigger grid: 100–150 configs, seq 4096, 2-GPU runs, gradient checkpointing off, more repeats.
- **S3:** 5-minute smoke-test job generated next to each script. It prints measured vs. predicted memory and step time, so the user's first small run checks the estimate before they scale up.
- **S4:** Leonardo knowledge base + "plan your training run on Leonardo" guide.
- **S5:** "Did it work?" step: loss sanity checks + small eval after the run.
- **S6:** Second cluster profile (a university Slurm cluster), calibrated with the same harness.
- **S7:** `fitcheck record`: append the user's own smoke-test and full runs to a local calibration file, so their estimates get tighter on their setup.

### Stretch

- Multi-node (2 nodes), models > 8B, live GPU-idle warnings from job statistics.

## 8. Goals (definition of done = MVP)

- **G1 — Measured grid:** ≥ 40 configurations measured on Leonardo, published as an open dataset with library versions per run.
- **G2 — Calibrated calculator:** meets the §9 targets on held-out configs, with held-out error **lower than at least one popular public calculator**.
- **G3 — Working setup:** generated scripts pass `bash -n` and `sbatch --test-only`; the case-study run (M7) uses one and finishes.
- **G4 — Local assistant:** fine-tuned 0.6B makes correct tool calls more often than base 0.6B on a held-out request set, asks follow-ups when info is missing, and runs on a laptop.
- **G5 — Case study:** predicted vs. actual memory and time for one full-length run, reported honestly, including whether the actual values fell inside the band.
- **G6 — Open release:** public MIT repo, grid dataset, README with a 5-minute quickstart.

## 9. Success metrics (provisional; fixed after the first grid results)

| Metric | Target |
|---|---|
| **Missed OOMs** on held-out configs (predicted "fits", actually OOM) | 0 |
| False OOM alarms on held-out configs (predicted OOM, actually fits) | ≤ 10% |
| Peak memory error, median absolute % (interpolation / extrapolation reported separately) | ≤ 10% |
| Step time error, median absolute % | ≤ 20% |
| Band coverage: held-out runs inside the stated 90% band | 80–95% (checks that the band is honest, not just wide) |
| Error vs. a public calculator on the same configs | lower |
| Tool-call exact match (tool + arguments) on 100–200 held-out requests | beats base 0.6B; Claude reported as reference |
| Follow-up behaviour: asks on incomplete requests / doesn't ask on complete ones | ≥ 80% / ≥ 90% |
| Assistant on laptop (Q4_K_M GGUF) | < 4 GB RAM; FP16 → Q4 drop in exact match reported |

**Integrity rules:** fix the held-out split before fitting; report errors per dimension (say where the calculator is bad); report negative results; never tune the method on the held-out size.

## 10. Timeline with cut lines

| When | Milestone | Cut line if behind |
|---|---|---|
| **Oct 3** | All 5 members submit the consent form | — |
| **Oct 4–5** | Harness runs locally on 0.6B (LoRA/QLoRA); formula baseline written; public calculator chosen; grid table and held-out protocol frozen | — |
| **Oct 6 (Day 0)** | Cluster intro; weights to `$WORK`; uv env on Leonardo; first grid job | — |
| **Oct 8** | Send the GPU request for the shared reservation | — |
| **Oct 6–12** | MVP grid running (async), including all 4-GPU configs; setup templates; teacher generates request → tool-call data | Drop the 8B full-FT FSDP configs |
| **Oct 13–16 (Days 1–4)** | Finish grid; fit correction; cross-validation → error bands; CLI; train 0.6B (= case-study run) | **Freeze the grid at what's measured by Oct 16** |
| **Oct 17–26** | Evaluations, metrics; bolt-ons (S1–S7) by whoever has time | Ship MVP only |
| **Oct 27–29** | Release repo + dataset; demo | — |
| **by Nov 1** | All data copied off Leonardo (accounts expire Nov 2) | — |

## 11. Compute plan and constraints

- **Accounts:** active Oct 6 (Day 0), new password around Oct 15, **expire Nov 2**. Consent form due **Oct 3, 23:59 CEST** for every member.
- **Async days:** shared project, jobs may queue, so the MVP grid is short job-array jobs.
- **Hackathon days (Day 1 = Oct 13, Days 2–4):** **one shared node reservation for all teams.** Request due **Oct 8 EoD**; changes by Oct 15.
- **Budget:** MVP grid ≈ 10 GPU-hours, 0.6B training < 1 GPU-hour. S2 adds ≈ 20–30 GPU-hours.
- **Proposed request:** Day 1: 1 GPU (smoke tests, remaining grid jobs). Days 2–4: 1–2 GPUs.
- **Teacher** for assistant training data: external API through the reverse proxy.
- **Ask the mentor:** how many GPUs can our team use at once in the shared reservation?

## 12. Team roles (to confirm after pitching to the team)

| Role | Owns | Commitment |
|---|---|---|
| **Core A** (Paritosh) | Harness, grid, calculator (M1–M3), integration | MVP owner |
| **Core B** | Setup generator, CLI, 0.6B assistant + case study (M4–M7) | MVP owner |
| **Contributors** (×3) | Pick any S-item; each is self-contained and nothing in the MVP waits on it | Part-time OK |

## 13. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Queue delays in the async window | Grid unfinished by Oct 13 | Short job-array jobs; submit Day 0; grid frozen Oct 16 |
| QLoRA + FSDP is fragile (4-bit storage dtype, wrapping policy) | 4-GPU QLoRA rows fail | MVP 4-GPU block uses LoRA/full FT only; QLoRA multi-GPU is S2 |
| Library version changes shift memory | Old measurements stop matching | Pin versions in `uv.lock`; log them per run; the band applies to that pinned stack |
| Extrapolation to 8B is poor | G2 missed for the size users care most about | Report it per size as a negative result; never hide it in an average |
| Base 0.6B tool calls are unreliable | G4 weak | Schema validation + one retry in the CLI; full-FT fallback (RESEARCH.md §1) |
| Not enough near-OOM configs | OOM metric meaningless (everything clearly fits) | Grid deliberately includes configs at the 64 GB edge |

## 14. Non-goals

- Recommending a training method.
- Predicting model quality or data requirements (scaling-law territory; only cited rough guidance, labelled "not measured").
- Predicting queue wait time.
- Models > 8B or multi-node in the MVP.
- Supporting every cluster (Leonardo first; others via the same harness).
- A GUI.

## 15. Components to build

| Item | What we build | Where it lands |
|---|---|---|
| Training script for the assistant | Minimal SFT script (HF `transformers` + `peft`/`trl`), runnable locally and on Leonardo | M6 |
| Request → tool-call dataset | Teacher-generated data and held-out test set, made for the `plan`/`setup` tool schema, in the Hermes template | M6 |
| Evaluation of tool calls | Exact-match scorer (tool + arguments) and follow-up scorer | M6 / G4 |
| Benchmark and calculator code | Harness, grid runner, formula baseline, residual fit | M1–M3 |

## 16. Demo story

"I want to LoRA-fine-tune Qwen3-4B on 20k examples on Leonardo."
→ assistant asks for the sequence length → calls `plan` → "1× A100, ~22 GB (request 25 GB), ~1.8 h (1.6–2.1 h), 1.8 GPU-hours, ~€4 on cloud (transferred estimate); measured QLoRA alternative: 11 GB, 2.4 h" → calls `setup` → ready-to-submit `sbatch` + uv env + 5-minute smoke test.
Then: "We planned our own assistant's training this way. Predicted X, actual Y, inside the band."

## 17. Evidence and references

- Princeton Research Computing on GPU utilisation: https://researchcomputing.princeton.edu/document/6891
- NERSC Perlmutter resource utilisation study: https://hgpu.org/?p=27716
- LANL HPC support ticket analysis: https://arxiv.org/abs/2010.04321
- Existing calculators: ftune (https://pypi.org/project/ftuneai/), https://inventivehq.com/tools/fine-tuning-vram-calculator
- Related agents: Hugging Face hf-llm-trainer skill (https://huggingface.co/blog/hf-skills-training), SlurmWise (https://github.com/armanbishal/slurmwise)
- CINECA storage areas: https://docs.hpc.cineca.it/hpc/hpc_data_storage.html
- Qwen3-0.6B tool-calling findings: [RESEARCH.md](RESEARCH.md)
