# Project report

* **Course submission:** [PDF](CS6886_Report_Concise.pdf) and [LaTeX](CS6886_Report_Concise.tex), two main pages plus a reference page. This is the report to use for the course's two-page body limit.
* **Expanded explanation:** [PDF](CS6886_Report.pdf) and [editable LaTeX](CS6886_Report.tex), five pages including references. Each transformation has an example, use case and validity limits. Page 1 traces both IREE GQA flags.
* **Per-study measurements:** [readable tables](study_results/per_study_results.md), [CSV](study_results/per_study_results.csv) and [JSON](study_results/per_study_results.json). Includes 182 IREE comparisons, individual variants, combined-pass ablations and regressions.

The submission report covers the problem and baseline investigation, prior work cited alongside each method, the implementation/integration, matched-baseline evaluation and references including the repository URL. The expanded version is supplementary.

Both LaTeX files are self-contained; no external figures or BibTeX build are required.

Upload `CS6886_Report_Concise.tex` to a standard Overleaf project, or compile locally:

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error CS6886_Report_Concise.tex
pdflatex -interaction=nonstopmode -halt-on-error CS6886_Report_Concise.tex
```

The source uses standard TeX Live packages, including Latin Modern, TikZ,
tabularx, tcolorbox and hyperref. Edit `\projectauthors` to update the author line.
Results are tied to the experiment-source snapshot cited in the references.
The reference page distinguishes originating papers, classical principles and
related systems; it does not claim that the array identities are new.

## Runnable examples and IR evidence

From the repository root, with the CPU dependencies installed:

```bash
python report/examples.py
python report/reproduce_flags.py --output reproduced/flag_evidence
```

The first command checks 14 concrete NumPy examples of the displayed array laws,
including both winner conditions and invalid-operation counterexamples. These
are explanatory semantic checks, not C++ pass tests or GPU measurements. The
second command recompiles the original fixture under all four flag combinations;
run the root artifact unpacker first. Saved Flow IR and the changed-pass trace
are in `flag_evidence/`. Dispatch-name counting is specific to the pinned compiler.

| Report transformation | C++ implementation under `compiler/src/` |
|---|---|
| Original grouped head reuse | `GQAHeadReuse.cpp` |
| Conditional forwarding | `CacheReadForwarding.cpp` |
| Masked gather forwarding | `CacheGatherForwarding.cpp` |
| Winner factoring | `CacheCommitFactoring.cpp`; concat extension under `real_model/compiler/src/ConcatCommit.inc` |
| Update coalescing | `CacheUpdateCoalescing.cpp` |
| Gather-index normalization | `GatherIndexProjection.cpp` |
| Snapshot and borrowed-buffer COW | `CacheSnapshotProtection.cpp` |
| Projection and partial RoPE | `AttentionProjection.cpp` |
| Native reduction guard/fallback | `OnlineAttentionGuard.cpp`, `AttentionProjection.cpp` |
| Stable trig | `StableTrig.cpp` |

## Regenerate the study tables

From the repository root:

```bash
python3 scripts/unpack_artifacts.py
python3 report/summarize_studies.py
```

Only the Python standard library is needed. This recalculates recorded ratios; it does not rerun benchmarks. Synthetic CPU, GPU operator and pretrained-model timing contracts differ and are not pooled. Protection/normalization/phase guards are correctness repairs; combined forwarding gains are not attributed to a repair alone. No compiled-PyTorch metric is included.
