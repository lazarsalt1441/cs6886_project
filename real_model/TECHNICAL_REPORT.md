# Real-model cache commit factoring: preparation results

This package tests a concrete compiler transformation on a pinned pretrained LLM. It does not claim a new publication result or a GPU speedup before the accompanying Colab runs.

The model is SmolLM2-135M-Instruct (revision12fd25f77366fa6b3b4b768ec3050bf629380bac), with30 Llama decoder layers,9 query heads and3 KV heads. We use actual checkpoint weights and real prompt-derived caches, in float32. The original Hugging Face model forward and DynamicCache update are retained. The cache updates lower to `tensor.concat`; they do not match the original insert-slice/read-forwarding prototype.

## Two workloads

**A: ordinary decode.** Feed one next token and the existing cache to the model, return next-token logits and its updated cache. The compiler's before/after IR is identical: zero matches. We benchmark natural Torch eager, Torch SDPA, optional Torch compile and strong IREE. There is no meaningful optimized-IREE ablation for an identical program.

**B: two-candidate lookahead.** Take the prefix's two most likely next tokens. Evaluate each using the same pretrained model/history. Score each path using its first-token log probability plus its best next-token log probability; commit only the better candidate's cache. This custom one-step lookahead is a valid inference driver, but is neither standard HF beam search nor a speculative acceptance algorithm.

The genuine C++ MLIR extension factors the committed selection:

```
select(c, concat(history, a), concat(history, b))
  => concat(history, select(c, a, b))
```

It rewrites60 cache outputs (K and V in30 layers). The appended one-token rows are selected before appending the common history. Both candidate attention computations and their cache users remain intact. The pass does not imply all candidate caches can disappear.

## Validation completed locally

- Rebuilt the complete standalone C++ optimizer with pinned MLIR/LLVM19.1.1. Actual pass execution, not a manual before/after model substitution.
- Eight structural/FileCheck cases pass: positive static and dynamic histories, multirow append, idempotence, rejection of different histories, per-element condition and unprofitable static patch.
- Eight compiled bitwise cases pass for static/dynamic histories, both winners, NaNs, infinities and signed zero.
- Full pretrained CPU integration passes at contexts8/32 for both programs. Full-model extended validation passes42 case/variant combinations at contexts32/128/512 and two prompts, including forced left/right winners. All61/62 outputs per invocation are checked (logits,60 KV tensors, optional winner). Largest observed absolute error against HF float32 is1.1014938354492188e-4, below fixed atol5e-4/rtol3e-3.
- Short autoregressive continuations preserve token agreement in the local checks. The latest runner additionally passes Torch SDPA comparisons and verifies cache input immutability after timing. Natural SDPA/eager comparisons pass for both workloads at contexts8/128.
- Both full-model wrappers capture with Torch Dynamo fullgraph using backend=eager. This checks graph capture, not Torch Inductor CUDA performance or support.
- All three IREE variants compile for T4 sm_75: A baseline, B baseline, B optimized. GPU execution, accuracy and latency remain untested locally.
- Colab-style fresh parameter archive generation is checked separately: no MLIR re-export or compiler build is needed on the user's machine.

CPU latency files have deliberately tiny trial counts and were produced during concurrent preparation. They validate runner behavior and must not be used as publishable speedup results.

## Toolchain and compilation finding

IREE3.13.0 compiler commit89ece87b89852100c0a4ac3d515e14842d090967; Turbine3.9.0; Transformers4.57.1; Torch2.11.0 CPU for export/checks. The prepared IR uses HF eager attention. Torch SDPA is retained as a natural reference baseline rather than assuming eager attention is the fastest backend.

The standalone C++ tool uses MLIR19; IREE uses LLVM24. The bridge prints generic operations, preserves unknown IREE operations, runs the C++ passes, then reparses both variants with IREE. Numerical execution checks this bridge on the actual model.

Dynamic export initially failed on an exporter symbolic-stride guard. Deferred assertions expose unsupported `sym_min` nodes; `shape_legalize.py` proves their operands ordered over the export's integer ranges, substitutes the provably smaller operand and removes tautological assertions. This changes guard arithmetic only. It does not replace the model's cache math.

Without reduction splitting, compiling the full lookahead vocabulary reduction triggered an LLVM NVPTX SmallVector capacity error in the earlier preparation. Both baseline and optimized CUDA builds use the same explicit reduction-splitting and argmax thresholds, recorded in README and compiler command JSON. The recovered package recompiles all three variants successfully with those settings; it does not claim to fix the underlying upstream bug.

## What the evidence does and does not establish

Logical commit selection is factored in60 actual full-model graph locations. CPU semantics and source-level applicability are supported by recorded tests. This is stronger than a standalone hand-written attention-only graph.

Physical allocation elimination is not established. Candidate attention users remain. Actual DRAM reduction is not established: codegen can already fuse/CSE portions of the original graph. Offline PTX has33 distinct kernels for baseline B and34 for optimized B; these are compiled kernel definitions, not dynamic launch counts or memory-traffic measurements. An extra kernel can offset a smaller commit operation.

Runtime improvement on a GPU remains an experiment. The notebook keeps KV resident, separates compilation from execution, randomizes backend order, logs numerical errors and tests an eager, SDPA and compiled Torch comparison where available. It can honestly return a regression or no improvement. Different accounts must not be compared as a baseline/optimized ablation.

Next evidence needed: both result archives; within-run baseline/optimized repeated medians; correctness and token agreement; profile cache-commit dispatches and measured GPU traffic where profiling is permitted; then test a representative real branching decoder and more architectures/dtypes. Ordinary decoding has zero current matches, so success in the lookahead driver would not establish a general greedy-decoding speedup.
