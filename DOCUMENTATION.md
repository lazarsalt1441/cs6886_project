# Documentation guide

## Current reproduction and evaluation

* [Submission report](report/CS6886_Report_Concise.pdf) and [LaTeX](report/CS6886_Report_Concise.tex): two main pages plus references, matching the course report limit.
* [Expanded report](report/CS6886_Report.pdf) and [LaTeX source](report/CS6886_Report.tex): five pages including references, with a source/IR explanation of the two GQA flags and examples of every transformation. It is supplementary to the submission report.
* [README.md](README.md): project scope, final IREE comparisons, installation,
  C++ build, CPU checks, Colab execution and raw-result reanalysis.
* [STATUS.md](STATUS.md): completed implementation/evaluation and limitations.
* [RESUME.md](RESUME.md): pinned reproduction environment and entry commands.
* [Compiler guide](compiler/README.md): pass source, original GQA export and
  operator-level studies.
* [GPU/DRAM guide](dram/README.md): final application-range profiling protocol,
  numerical gates, timing method and recorded outputs.
* [Pretrained-model guide](real_model/README.md): original export and short-context
  protocol; the repaired long-context evaluation uses the GPU/DRAM guide.
* [Upstream audit](audit/README.md): bounded source/flag investigation and its
  independently tested compiler versions.

## Experiment records

Reports under `compiler/research`, `compiler/research2` and `real_model` describe
specific study stages. Their CPU, compile-only or original numerical findings
retain that scope even when a later experiment is available. Final model results
are under `results/dram`; other campaign directories retain earlier results,
failures and controls. Immutable protocol snapshots inside result directories
and compressed evidence archives preserve the original experiment wording.

The root tables and default plot/summary tools show IREE baseline versus optimized
IREE. Raw observations and correctness references remain intact. GPU compilation,
physical allocation, DRAM traffic and execution latency require different evidence.
No failed numerical case is treated as a performance baseline.

## Validation and compatibility

Use `scripts/unpack_artifacts.py` before accessing archived IR, codegen, source
snapshots and larger measurements. The root source manifest verifies committed
files; historical release manifests describe their original packages.
`scripts/reproduce_results.py` checks 112 final profile exports and regenerates
the IREE-only model aggregates without GPU execution.

The tested final GPU environment is TeslaT4, Torch2.11 CUDA and IREE3.13. Other
architectures and runtime versions require fresh numerical validation. The PBS
guide documents site-specific prerequisites; the old glibc2.17 host cannot run
the modern wheels directly. Profile access requires permitted NVIDIA counters.

The [per-study tables](report/study_results/per_study_results.md) map measured gains to individual or combined transformations. Regenerate them with `python3 report/summarize_studies.py` after unpacking. New DRAM plans collect IREE metrics; original multi-backend records remain unchanged.
