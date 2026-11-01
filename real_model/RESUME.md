# Pretrained-model reproduction notes

The model is SmolLM2-135M-Instruct, revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. Export uses Python3.12, Torch2.11,
Transformers4.57.1, Turbine3.9.0, IREE compiler/runtime3.13.0 and MLIR19.1.1.
Checkpoint and SDK hashes are included. SetA has zero concat-commit matches;
the custom two-candidate SetB has60. Prepared IR is restored by the root unpacker.

From this directory after the root dependency installation and C++ build:

```bash
python scripts/download_model.py
python scripts/prepare.py --set A
python scripts/prepare.py --set B
python tests/test_concat_factor.py
python tests/test_concat_numerical.py
python tests/validate_pretrained.py
python tests/trace_fullgraph.py
python tests/validate_sdpa.py
python tests/test_runner_protocol.py
python tests/validate_fresh_parameters.py
```

The original short-context protocol is available through Colab_Set_A.ipynb and
Colab_Set_B.ipynb. The final long-context evaluation uses the bounded trig repair
and `../dram` runner. See the root README for those results and commands.
Model weights and generated IRPA files are downloaded/regenerated rather than
committed. Parameter identities, exported guards, tolerances and compiler flags
remain fixed for a matched comparison.
