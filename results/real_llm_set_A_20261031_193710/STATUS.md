# Status

The two-set Colab package is ready for GPU execution. No new real-model GPU speedup is claimed.

- M0: pinned actual pretrained SmolLM2-135M-Instruct checkpoint, tokenizer and SHA256 manifest; reproducible Torch/IREE/MLIR environment.
- M1: actual full-model IR investigation. HF DynamicCache uses concat. A ordinary decode has zero matches and identical baseline/optimized hashes. B custom two-candidate lookahead has60 concat commit matches.
- M2: real standalone C++ pass rebuilt with MLIR19.1.1; eight structural/FileCheck and eight compiled bitwise cases pass. Original pass patterns retained; unsupported patterns safely reject.
- M3: actual checkpoint CPU integration, dynamic cache continuations and input immutability pass. Extended42 numerical case/variant checks at contexts32/128/512, two prompts and both winners. Largest observed absolute error1.1014938354492188e-4. Torch SDPA reference, fullgraph Dynamo capture and fresh parameter archives checked separately.
- M4 preparation: all three IREE variants compile for sm_75 with identical strong options and documented reduction-splitting workarounds. PTX saved. Local CPU timings are low-trial protocol diagnostics, not performance claims. GPU execution, Torch Inductor GPU and hardware counters await Colab.
- M5: extension source, prepared IR, full scripts, two notebooks, technical report, backup/download handling and reproduction instructions included.

Resolved failures: a workspace restart lost unpersisted scratch files; source was reconstructed, validations rerun and downloadable checkpoints saved. Dynamic exporter guard/Turbine sym_min issue solved with range-proved guard-only simplification. CUDA wide-reduction compiler crash avoided by shared upstream split-reduction settings, not by changing model semantics. Temporary IRPA files excluded from artifacts.

Remaining: run SetA and SetB on separate Colab accounts, return archives, inspect correctness/regressions, and only then assess measured benefit and further research scope. All-GQA applicability, fp16, real serving throughput and novelty are unproven.
