# Project report

Nishanth Senthil Kumar (EE23B049) and Dhruv Prasad (EE23B130), IIT Madras.

[Final report (PDF)](CS6886_GQA_Attention_Report.pdf) · [LaTeX source](CS6886_GQA_Attention_Report.tex)

Compile from this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error CS6886_GQA_Attention_Report.tex
pdflatex -interaction=nonstopmode -halt-on-error CS6886_GQA_Attention_Report.tex
```

## Supporting files

- `flag_evidence/`: saved IR and results for the IREE flag ablation.
- `study_results/`: per-study measurements in Markdown, CSV and JSON.
- `examples.py`: NumPy checks of the array identities.
- `reproduce_flags.py`: rerun the flag ablation.
- `summarize_studies.py`: regenerate tables from saved measurements.

From the repository root:

```bash
python3 scripts/unpack_artifacts.py
python3 report/examples.py
python3 report/reproduce_flags.py --output reproduced/flag_evidence
python3 report/summarize_studies.py
```

The examples check semantics; the summary script recalculates recorded ratios. Neither creates new GPU timing measurements. See [the main README](../README.md) for dependencies and benchmark instructions.
