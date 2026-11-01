# DRAM reproduction notes

Pinned versions: Torch2.11 CUDA, IREE compiler/runtime3.13.0, Turbine3.9.0,
Transformers4.57.1; model revision12fd25f77366fa6b3b4b768ec3050bf629380bac.
Recorded profiler: Nsight Compute2025.3.1, TeslaT4 sm_75.

In Colab, clone the repository, restore its artifacts, mount Drive and run
COLAB_RUN.py with `DRAM_KIT_ROOT='/content/cs6886_project/dram'`, `TRACK='B'`
and `PLAN='main'`. TrackA is the ordinary-decode/no-match control. Full commands
are in README.md. The driver runs smoke before main and saves runs under
`runs/RUN_ID/`; backups use `MyDrive/Attention_DRAM_Results`.

From this directory on a configured GPU host:

```bash
python -u study.py --track B --plan main --output results_B_main
python analyze.py results_B_main
python plot_results.py results_B_main
```

For local validation:

```bash
python -m unittest discover -s tests -v
python tests/cpu_integration.py --cache cpu_test_cache
python tests/compile_gpu.py --cache compile_test_cache --arch sm_75
```

Completed final datasets are under `../results/dram/A` and `../results/dram/B`.
A resumed cached profile is not a new repetition. Use a new run ID for an
independent session and retain numerical failures, clock telemetry and commands.
