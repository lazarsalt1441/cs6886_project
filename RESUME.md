# Continuation checkpoint

Repository: https://github.com/lazarsalt1441/cs6886_project ; default branch main.
Authoring workspace: /workspace/scratch/4088b657d6f5/cs6886_project .
Saved experiment paths are intentionally preserved under compiler/, real_model/, audit/ and results/. Absolute paths in historical logs identify original runs; current scripts use repository-relative inputs.

Pinned experiment environment: Python3.12 exports; IREE compiler/runtime3.13.0, embedded compiler89ece87b89852100c0a4ac3d515e14842d090967, LLVM24; standalone MLIR19.1.1 SDK; Torch2.11.0 CUDA13.0 on TeslaT4(sm75), driver580.82.07; NumPy2.3.5; Transformers4.57.1; Turbine3.9.0; model revision12fd25f77366fa6b3b4b768ec3050bf629380bac. Original study environments remain in their own evidence; do not silently pool them.

Next command on a new checkout:

```
python3 scripts/unpack_artifacts.py
python3 scripts/check_submission.py
python3 scripts/reproduce_results.py
```

Local authoring dependencies available outside this repository at ../followup_deps . Successful no-install CPU validation uses PYTHONPATH=../followup_deps and ../followup_deps/bin/iree-compile; that path is not required on a fresh install. See root README for normal venv installation and C++ build.

Next three tasks:
1. Re-run independent final B GPU session; examine clock telemetry and no-rewrite controls before strengthening causal/performance claims.
2. Analyze generated kernel differences and range read/write balance; do not assert full-model cache-copy elimination.
3. When authorized, produce the course report (<=2 pages excluding references). Do not treat historical long reports as the course submission.

Failures already diagnosed: no compute-node Internet; glibc2.17 wheel incompatibility; unavailable old Torch2.14.1 pin; Colab ensurepip failure; Nsight raw export needs --print-units base and must not include details-only --print-metric-name. Fixed final runner and failure history are retained.

All returned final A/B profiles and raw timing samples are in results/dram. B zip and tar contained the same experiment; do not count twice. Original ncu/nsys binary reports and pretrained weights are excluded from Git; full exported metrics and commands remain available. SHA256 original archive provenance is retained.
