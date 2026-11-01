# Project report

* [PDF](CS6886_Report.pdf): two main pages plus one reference page.
* [Editable LaTeX](CS6886_Report.tex): self-contained source, including the diagram,
  tables and bibliography. No external figures or BibTeX build are required.

Upload `CS6886_Report.tex` to a standard Overleaf project, or compile locally:

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error CS6886_Report.tex
pdflatex -interaction=nonstopmode -halt-on-error CS6886_Report.tex
```

The source uses standard TeX Live packages, including Latin Modern, TikZ,
tabularx, tcolorbox and hyperref. Edit `\projectauthors` to update the author line.
Results are tied to the experiment-source snapshot cited in the references.
The reference page distinguishes originating papers, classical principles and
related systems; it does not claim that the array identities are new.
