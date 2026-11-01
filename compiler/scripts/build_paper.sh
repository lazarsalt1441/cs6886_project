#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python research/make_paper.py > research/logs/paper-data.log 2>&1
cd research/paper
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex > build.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex >> build.log 2>&1
if rg -q 'Overfull|undefined references' manuscript.log; then
  rg 'Overfull|undefined references' manuscript.log
  exit 1
fi
