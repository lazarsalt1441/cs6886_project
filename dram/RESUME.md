# Continuation

Kit root in Colab: /content/Attention_DRAM_Tests
Optional reused model: /content/attention_real_llm/model
Expected Torch: 2.11 CUDA; IREE compiler/runtime 3.13.0; Turbine 3.9.0;
Transformers 4.57.1. Model revision is recorded in README and model adapter.
Observed user profiler: Nsight Compute 2025.3.1, Tesla T4 sm_75.

Start in the notebook with TRACK='B', PLAN='main', then execute COLAB_RUN.py.
The driver runs smoke first and then main, saving results in runs/RUN_ID/.
Cache/model/compiled modules are local; only result archives are backed up.
Drive location: MyDrive/Attention_DRAM_Results/attention_dram_RUN_ID.tar.gz.
DRAM_RUN_ID in the notebook permits intentional resumption. A new independent
session should use a new ID, not count cached profiles as fresh measurements.

Command line from kit directory:

    python -u study.py --track B --plan main --output results_B_main
    python analyze.py results_B_main
    python plot_results.py results_B_main

Local validation commands:

    python -m unittest discover -s tests -v
    python tests/cpu_integration.py --cache cpu_test_cache
    python tests/compile_gpu.py --cache compile_test_cache --arch sm_75

Authoring workspace: /workspace/scratch/4088b657d6f5/dram_work/
Actual local compiler invoked through followup_deps/bin/iree-compile, version
3.13.0 at 89ece87b89852100c0a4ac3d515e14842d090967. Python 3.12.14.
GPU compilation was tested, GPU execution/profiling was unavailable.

Next three tasks:
1. Inspect smoke ncu.log, metrics.csv, worker.json and numerical gates.
2. Complete matched range-counter and separate latency runs with Torch controls.
3. Check copy traces, cache sensitivity and repeated-session results before
   updating the consolidated report or making a physical traffic claim.

Latest recovery: returned B run 20261101_074854_B_main captured its streaming
range correctly, then rejected --units during CSV export. The updated kit uses
--print-units base and reads both Nsight CSV layouts. Preserve this run ID and
cache to resume. Matching .ncu-rep and worker.json can be re-exported directly.
24 unit tests pass after the fix. No fresh GPU inference/profile result is claimed.

Second returned export diagnosis: Nsight 2025.3.1 rejects --print-metric-name on the raw page. Removed that details-only option; raw CSV already names metric columns. The 24-test suite now explicitly rejects both obsolete --units and raw-page --print-metric-name. Capture and numerical computation are unchanged.
