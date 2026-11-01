# Compiler reproduction notes

The driver is built from `src/` with MLIR/LLVM19.1.1. The final project uses
IREE3.13.0, compiler `89ece87b89852100c0a4ac3d515e14842d090967` (LLVM24).
Original experiment-specific versions remain in their saved environment records.
Use the root requirements for a fresh installation.

From the repository root:

```bash
python3 scripts/unpack_artifacts.py
bash scripts/build_compilers.sh
python compiler/tests/test_structural.py
```

The root build script shares the downloaded SDK with `real_model/compiler`.
The source-patch proposals under `research2/patches` are distinct from the
compiled standalone passes; a complete patched IREE compiler was not built.
The failed source-build attempt and partial object-build evidence remain in the
historical logs. This limitation does not affect the saved standalone pipeline.

For operator-level GPU reproduction, run `research2/gpu_suite.py` and
`research2/extension_gpu_suite.py` from this directory; `--compile-only` separates
offline compilation from device execution. For the final pretrained-model DRAM
study, use the root README and `dram/Colab_DRAM_B.ipynb`.

Original CPU metrics, errors, allocations and codegen are retained under
`results/`, `research/results/` and `research2/results/` after extraction.
No incorrect stock output is used as a speedup baseline. Global novelty and
universal GQA applicability are not established by these experiments.
