# fitcheck — Research Notes

Findings that back decisions in [GOAL.md](GOAL.md). Checked 2026-10-03. Items marked *unverified* come from search snippets only; confirm before citing.

---

## 1. Qwen3-0.6B for tool calling (M6 / G4)

Question: does Qwen3-0.6B support tool calling and MCP well enough to be the local assistant?

### From official Qwen documentation

| Topic | Finding |
|---|---|
| Template | Hermes-style tool template: schemas in the prompt, model answers with `<tool_call>{"name": ..., "arguments": ...}</tool_call>`. |
| Template caveat | Avoid stop-word templates (ReAct) for reasoning models. The model may emit stop words inside its thinking and break tool calls. |
| Serving | vLLM has a built-in parser: `--enable-auto-tool-choice --tool-call-parser hermes`. Qwen-Agent is the "canonical implementation". |
| Reliability | Docs: "not guaranteed that the model generation will always follow the protocol". Fine-tuning on domain data may be needed. |
| MCP | Not in the function-calling doc. The 0.6B model card says Qwen-Agent accepts MCP config files. MCP is a client-side layer; the model only needs to emit valid tool calls. |
| Model card claim | "Excels in tool calling" (marketing; not a measurement for 0.6B). |
| Sampling | Thinking mode: temp 0.6, top-p 0.95, top-k 20, min-p 0; **no greedy** (repetition loops). Non-thinking: temp 0.7, top-p 0.8, top-k 20, min-p 0. |
| Mode switch | `enable_thinking=False` or `/no_think`. |

### Third-party numbers (*unverified*, search snippets only)

- Base Qwen3-0.6B: ~62% on BFCL v3 non-live in FP16; lower when quantized.
- STAR (0.6B, distillation + RL): 51.7% on BFCL v3, 53.0% on ACEBench. Not comparable to the line above (different settings).
- Community 0.6B tool fine-tunes: ~66% (exact-args) to ~92% (argument parsing) on narrow, self-made tests. Not comparable to each other.

### Decisions that follow

1. Fine-tune with **LoRA** (rank 16–64, all linear layers; full FT on 0.6B is the fallback if G4 is missed).
2. Generate training data in the **Hermes template** so the model works unchanged with llama.cpp, Ollama, vLLM and MCP clients.
3. Run in **non-thinking mode** for tool calls.
4. CLI **validates tool calls against the JSON schema and retries once**.
5. Evaluate the **Q4_K_M GGUF as well as FP16** and report the quantization drop.
6. Keep the model to 2–3 tools and never let it write numbers (see GOAL.md §4).

### Open questions

- Exact-match score of base 0.6B on our own held-out set (baseline for G4).
- Quantization loss on our tool set.
- Whether 0.6B asks the follow-up question reliably or needs extra training examples for it.

---

## 2. Sources

- Qwen function calling docs: https://qwen.readthedocs.io/en/latest/framework/function_call.html
- Qwen3-0.6B model card: https://huggingface.co/Qwen/Qwen3-0.6B
- Small Reasoning Models are Instruction Followers in Function Calling: https://arxiv.org/pdf/2608.22472
- STAR: Super-Tiny Function Calling Models: https://arxiv.org/pdf/2602.03022
- iromu/Qwen3-0.6B-tools: https://featherless.ai/models/iromu/Qwen3-0.6B-tools
- AL1 model B: https://featherless.ai/models/karthik-2905/AL1-model-B
