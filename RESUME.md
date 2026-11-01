# Reproduction environment

Repository: https://github.com/lazarsalt1441/cs6886_project (branch `main`).

The final experiments use Python 3.12 exports, IREE compiler/runtime 3.13.0
(compiler `89ece87b89852100c0a4ac3d515e14842d090967`, LLVM 24), standalone
MLIR/LLVM 19.1.1, Torch 2.11.0 CUDA 13.0, Tesla T4 (`sm_75`), driver 580.82.07,
NumPy 2.3.5, Transformers 4.57.1 and Turbine 3.9.0. The checkpoint revision is
`12fd25f77366fa6b3b4b768ec3050bf629380bac`.

From a fresh checkout:

```bash
python3 scripts/unpack_artifacts.py
python3 scripts/check_submission.py
python3 scripts/reproduce_results.py
```

The root README contains dependency installation, C++ build, numerical tests
and GPU execution commands. Each experiment retains its own environment record;
absolute paths inside historical logs identify the original runs, while current
instructions use checkout-relative paths.

Final A/B profiles and raw timing samples are under `results/dram`. The B ZIP
and tar uploads contained the same experiment and are counted once. Original
profiler report binaries, downloaded weights and SDKs are excluded from Git;
exported metrics, commands and original archive SHA256s are retained.

Environment compatibility: modern wheels need a newer glibc than 2.17. GPU
benchmarks require a CUDA compute node or Colab GPU, with profiler counter
permission for DRAM collection. Nsight raw CSV export uses `--print-units base`
and omits the details-only `--print-metric-name` option.
