# Real pretrained LLM cache study

The original short-context protocol uses **HuggingFaceTB/SmolLM2-135M-Instruct**, checkpoint `12fd25f77366fa6b3b4b768ec3050bf629380bac`. Actual pretrained weights, original Hugging Face model forwards, 30 layers, 9 query heads, 3 KV heads, head dimension64. All comparisons use float32. Weight files are downloaded and SHA256 checked; none are included in this package.

| Set | Notebook | Purpose |
|---|---|---|
| A | Colab_Set_A.ipynb | Ordinary greedy decode; verify applicability and inference baselines. Zero C++ rewrite matches. |
| B | Colab_Set_B.ipynb | Two-candidate lookahead; test concat commit factoring. 60 C++ rewrite matches. |

Set B is a custom two-candidate, one-step lookahead driver using the pretrained model. It is not Hugging Face beam search or a speculative acceptance/rejection implementation. Set A's zero-match result is intentional; we do not manufacture an optimized timing for identical IR.

## Run in Colab

1. Open `Colab_Set_A.ipynb` or `Colab_Set_B.ipynb` from this repository. The final repaired long-context evaluation is in `../dram`; these notebooks reproduce the earlier protocol.
2. In Colab choose **File → Upload notebook**, then upload your `.ipynb`.
3. Choose **Runtime → Change runtime type → T4 GPU**. Prefer the same GPU type on both accounts; within-account comparisons remain primary.
4. Run notebook cells in order. The first code cell clones the repository and restores prepared IR. The environment cell retains Colab's CUDA-enabled Torch and installs pinned IREE/Turbine/Transformers versions in a virtualenv. It avoids the previous `ensurepip` failure.
5. Mount Google Drive when prompted, then run the last main cell.
6. Leave `QUICK=False` for the full study. For an initial pipeline check, use `QUICK=True`; rerun with False for final measurements. Keep `SKIP_TORCH_COMPILE=True`; project performance tables compare IREE variants.
7. Retain the complete `real_llm_set_*.tar.gz` archives. Setup failures are recorded in `setup.log`.

The full run uses contexts128/512/2048, two prompt templates, 31 trials × 3 repetitions, five warmups, and16-token continuations. Prompt templates are repeated/truncated to control context length, not sampled from a representative corpus. Total duration depends on Colab, context lengths and compilation; it is separate from measured step latency. Independent runs may use separate GPUs; within-run comparisons remain primary.

A heartbeat prints every minute. Partial archives are saved to Drive every five minutes; a final archive is saved and a browser download attempted even after an ordinary error or interruption. Browser permissions or abrupt runtime termination can prevent the download/final callback. Drive contains the latest completed backup. This does not keep Colab alive after a disconnect.

## What is measured

- Whole-model one-token decode/selection latency, with real prompt-derived KV caches resident on device. No random KV tensors are used for measurements.
- Strong IREE baseline and optimized IREE where the pass actually matches. Torch is the correctness reference; historical raw records retain auxiliary backend measurements. The default summary tables contain IREE only.
- Randomized backend order for every trial, repeated measurements, raw samples and per-repetition medians. Inputs are uploaded before timing and verified unchanged afterward. Every returned cache/logit/winner is checked before and after timing.
- `resident_outputs` measures synchronous invocation with outputs left on device. IREE's synchronous main ABI has an explicit fence await; Torch CUDA is synchronized. `logits_to_host` additionally downloads logits and the winner scalar. No full KV download is timed.
- Prefill, checkpoint download, parameter creation and compilation are excluded from step latency. Their costs are not hidden inside execution measurements; IREE compilation and Torch first-call costs are recorded separately.
- Short host-driven autoregressive continuations retain device KV outputs. Only logits, scalar winner and small token/mask/position data cross the host boundary. Their token agreement and throughput are diagnostic: single continuations, not repeated publication-quality generation measurements.

Correctness uses a fixed `atol=5e-4, rtol=3e-3` against the float32 HF eager-attention reference; integer outputs must agree exactly. GPU numerical failures stop timing and preserve partial logs. Token agreement is reported separately; no failed sample is silently removed. Torch compile failures are logged, marked unavailable and never silently replaced by eager timings.

No allocation, GPU DRAM-traffic, fp16, large-model, or general serving speedup is claimed by this preparation. Logical cache factoring, physical allocation elimination, reduced memory traffic and runtime improvements require separate evidence. CPU low-trial results included here only validate the runner; they are not presentable performance estimates.

## Compiler optimization

Natural HF DynamicCache appends new K/V rows via `torch.cat`; it does not match the original insert-slice/read-forwarding prototype. The real C++ extension recognizes:

```
select(c, concat(history, new_a), concat(history, new_b))
  -> concat(history, select(c, new_a, new_b))
```

The selected rows are small. The common history is unchanged. Attention users of both candidate caches remain unchanged, so this transformation does not promise to eliminate every candidate allocation. It applies to a specific branch-commit pattern, not all GQA graphs. Ordinary decode has no branch selection and no matches.

The C++ pass checks scalar i1 condition, all-parallel identity-indexed selection, equal append types/shapes, common history SSA value and concat axis, static nonempty append shape and no encoded/lowered tensor. Unsupported patterns are left intact. Both original insert-slice and new concat forms are preserved in source. Pass anchoring accepts IREE util.func bodies through an unregistered-op generic IR bridge; both before/after are verified again by the pinned IREE compiler.

Both variants use O3, `--iree-opt-generalize-matmul`, and `--iree-dispatch-creation-enable-aggressive-reshape-movement`. CUDA also uses these shared workarounds for an IREE NVPTX wide vocabulary reduction crash:

```
--iree-dispatch-creation-enable-split-reduction
--iree-dispatch-creation-split-argmax-threshold=1024
--iree-dispatch-creation-split-argmax-tile-size=256
```

The original HF model attention implementation is `eager` for export. Dynamic export requires a small range-proved simplification of exporter guard `sym_min` nodes unsupported by Turbine; it changes only symbolic guard arithmetic. A failed range proof raises an error. It does not hand-rewrite tensor/cache operations. `text_only.py` disables inherited optional audio/vision packages for pinned Transformers text-model imports.

## Local reproduction

Prepared IR is included; Colab does not build MLIR or re-export the model. For full source regeneration on Ubuntu24.04 x86_64:

```
python -m pip install -r requirements-colab.txt cmake ninja
# Install a compatible Torch separately: CPU for local checks, CUDA for GPU runs.
python scripts/download_model.py
python compiler/scripts/fetch_mlir.py
bash compiler/scripts/build.sh
python scripts/prepare.py --set A
python scripts/prepare.py --set B
python tests/test_concat_factor.py
python tests/test_concat_numerical.py
python benchmark.py --set A --device cpu --contexts 8 32 --prompts 1 --trials 2 --repetitions 1 --warmup 1 --steps 2 --output local_checks/final_A
python benchmark.py --set B --device cpu --contexts 8 32 --prompts 1 --trials 2 --repetitions 1 --warmup 1 --steps 2 --output local_checks/final_B
python tests/validate_pretrained.py
python tests/trace_fullgraph.py
python tests/validate_sdpa.py
```

The SDK fetch verifies pinned MLIR/LLVM19.1.1 Debian package hashes and does not change the system installation. Tested export environment: Torch2.11.0+cpu, Python3.12, Transformers4.57.1, Turbine3.9.0, IREE compiler/runtime3.13.0 (compiler commit89ece87b89852100c0a4ac3d515e14842d090967). Exact environment and commands accompany result archives. Colab reuses its installed CUDA Torch; a different version is recorded and must pass numerical checks.

For manual GPU execution:

```
python scripts/run_colab_set.py --set A --skip-torch-compile
# In a separate GPU session:
python scripts/run_colab_set.py --set B --skip-torch-compile
```

## Recorded outputs

Result archives contain: `environment.json`, `correctness.json`, `latency.json`, `compilation.json`, `torch_compilation.json`, `generation*.json`, `summary.json`, `table.csv`, compiler commands/logs, PTX/codegen and any `torch_compile_error.log`. Evaluation uses within-run baseline/optimized ratios, numerical eligibility and token agreement. Do not compare absolute times across different GPU sessions as an ablation.
