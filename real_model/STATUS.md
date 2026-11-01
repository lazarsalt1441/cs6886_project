# Status

The two-set export and validation pipeline is implemented. This file records the preparation-stage checks; final GPU results and the long-context repair are documented in the repository root README.

- M0: pinned actual pretrained SmolLM2-135M-Instruct checkpoint, tokenizer and SHA256 manifest; reproducible Torch/IREE/MLIR environment.
- M1: actual full-model IR investigation. HF DynamicCache uses concat. A ordinary decode has zero matches and identical baseline/optimized hashes. B custom two-candidate lookahead has60 concat commit matches.
- M2: real standalone C++ pass rebuilt with MLIR19.1.1; eight structural/FileCheck and eight compiled bitwise cases pass. Original pass patterns retained; unsupported patterns safely reject.
- M3: actual checkpoint CPU integration, dynamic cache continuations and input immutability pass. Extended42 numerical case/variant checks at contexts32/128/512, two prompts and both winners. Largest observed absolute error1.1014938354492188e-4. Torch SDPA reference, fullgraph Dynamo capture and fresh parameter archives checked separately.
- M4 preparation: all three IREE variants compile for sm_75 with identical strong options and documented reduction-splitting workarounds. PTX saved. Local CPU timings are low-trial protocol diagnostics, not performance claims. Final GPU execution and counter datasets are under `../results/dram`; these preparation-stage compilation checks are not themselves GPU timing evidence.
- M5: extension source, prepared IR, full scripts, two notebooks, technical report, backup/download handling and reproduction instructions included.

Resolved compiler/export issues: Dynamic exporter guard/Turbine sym_min issue solved with range-proved guard-only simplification. CUDA wide-reduction compiler crash avoided by shared upstream split-reduction settings, not by changing model semantics. Temporary IRPA files excluded from artifacts.


Scope: ordinary decode has no commit rewrite. Universal GQA applicability, lower-precision serving throughput and global novelty are not established.
