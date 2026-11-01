# IITM PBS GPU tests: four optimization families

This is a self-contained source/IR bundle. Do not build or benchmark on the login node. All Python installation, IREE CUDA compilation, references, validation and timing run through PBS. No account credentials are needed or included. The jobs run in the submission directory on shared storage, avoiding scratch cleanup/7-day deletion. Generated directories belong to the benchmark; do not delete them until results are retrieved.

## What is tested

1. Cache read forwarding: nine saved attention shapes (different pools, read density, heads, ratios, dimensions, update positions and branch counts), plus two synthetic full transformer blocks.
2. Adjacent update coalescing + forwarding: 1-token control, 4, 8 and 16-token chains; additional original-cache snapshot-preservation test.
3. Winner commit factoring + forwarding: runtime winner derived from attention, checks candidate attention, selected cache and decision; multi-seed validation alternates update biases to exercise both winner directions. Inspect reported results before asserting both decisions occurred.
4. Latent projection placement: all saved decomposed, native and partial-RoPE cases (27 shapes total), comparing optimized to strong unabsorbed baselines. These are not pretrained models or comparisons against manually tuned MLA.

The main cache/winner suite includes strong, aggressive-fusion, normalized-gather, normalized/aggressive, conditional and masked forms. Chains include a fused baseline. A masked irregular form is known to fail CUDA compilation; its error must remain visible and a passing conditional alternative is required. Projection failures are recorded rather than assumed correct. No failed program is a valid timing baseline.

Default: 101 randomized-order synchronized host timings per variant, after 10 warmups, plus 10 fresh validation seeds per compiled binary. Reference models use CPU float64. Winner returned cache/decision and snapshot outputs have extra exact bit checks. Fresh uploads per validation invocation prevent earlier invocations contaminating the next test. Run a second independent job for repeatability. Finite shapes/seeds do not prove universal correctness.

The initial timing suite and additional correctness suite are separate: only use a variant's timings if *all* its later validation checks pass. CPU reference tolerance is atol=5e-5, rtol=2e-3 (block atol=2e-4); report max errors, do not loosen tolerances to make failures pass. Native attention may expose previously unknown GPU defects.

## Site requirements and execution

The legacy glibc2.17 login-node environment is incompatible with the modern
experiment wheels. A compatible compute-node image/container and Python3.12
are required. Compute-node Internet is also required by the default setup;
where unavailable, prepare an approved offline wheelhouse and adapt the pip
commands before submission. The Colab procedure in the root README is the
validated GPU reproduction path.

1. Clone the repository and restore its artifacts. On the login node, restrict activity to download/file management; compile and benchmark through PBS:

```bash
git clone https://github.com/lazarsalt1441/cs6886_project.git "$HOME/cs6886_project"
cd "$HOME/cs6886_project"
python3 scripts/unpack_artifacts.py
cd compiler
qsub pbs_gpu/setup.cmd
qstat -an
```

2. Wait for setup to finish successfully. Inspect PBS's `attention_setup.o<jobid>` log, `gpu_environment.txt`, and `gpu_setup_hardware.txt`. Setup needs internet access on the compute node and Python 3.12 plus `venv`. If your site uses modules, insert the actual Python/CUDA module commands near the top of **both** PBS scripts. Do not guess module names. If internet is disabled, ask the administrators for an approved wheelhouse and replace pip's index with that wheelhouse. Do not run setup on the login node.

3. Submit the complete suite:

```bash
qsub pbs_gpu/tests.cmd
qstat -an
```

The script detects NVIDIA compute capability. If `nvidia-smi` cannot report it, specify the real architecture explicitly, for example A100:

```bash
qsub -v GPU_ARCH=sm_80 pbs_gpu/tests.cmd
```

H100 is sm_90; do not assume your cluster has either GPU. The cluster example requested one GPU in `gpuq`, which these jobs follow. Adjust the walltime/resource request only to the site's permitted limits. Default walltime is 8 hours, not a guarantee of completion. To cancel: `qdel JOBID`.

4. If the full job exceeds the queue walltime, submit families **sequentially**, waiting for each to finish in this directory:

```bash
qsub -v FAMILY=cache pbs_gpu/tests.cmd
qsub -v FAMILY=chain pbs_gpu/tests.cmd
qsub -v FAMILY=winner pbs_gpu/tests.cmd
qsub -v FAMILY=projection pbs_gpu/tests.cmd
```

Do not execute those four commands at once in one working directory: compiled result paths overlap. Separate copies of the unpacked bundle are required for concurrent jobs. Default TRIALS=101 and SEEDS=10 can be overridden using `qsub -v FAMILY=cache,TRIALS=201,SEEDS=20 ...`. Setup is needed once per directory/environment. Validation failures are retained; an archive's existence is not success.

For sm_80 the snapshot suite uses bundled phase-local COW-repaired IR. Other architectures use `--skip-root-cow` and still test tensor-level snapshot protection and both optimized forms. Regenerating the phase-local COW case on other architectures requires the C++ toolkit; it is deliberately not silently reused from sm_80.

## Recorded outputs

After each job finishes, download **`gpu_results_JOBID.tar.gz`**, plus its PBS stdout/stderr log (`attention_gpu.oJOBID` and any `.eJOBID`). The archive includes:

- `arch.txt`, `nvidia-smi.txt`, `compiler.txt`, `packages.txt`, `system.txt`;
- `commands.tsv` with each command's exit status and `step_*.log` with complete failures;
- `validation.json`: every binary hash, seed, correctness result and numerical error;
- timing JSON with all raw observations and compilation times for all four families;
- generated compile logs/PTX and available repaired IR for investigation.

Retain raw samples, partial archives and PBS logs, including failures. If walltime/kill prevents archive creation, download `pbs_results/job_JOBID` and the PBS log manually. Keep generated VMFB files locally on the cluster until analysis is complete; they are excluded from the small return archive and can be requested by path/hash later. Old bundled CPU results are not new GPU evidence. Collected artifacts can include previous family runs on the same architecture; `commands.tsv` and directory job ID identify what ran in this job.

## Timing, memory and optional profiling

Timing here includes synchronized IREE invocation + GPU execution + output download, and excludes input upload and compilation. It is NOT kernel-only latency. Compilation time is separately recorded. CPU PyTorch is used for float64 references; these default jobs do NOT report eager CUDA PyTorch performance. That avoids installing a large or driver-incompatible CUDA Torch wheel just for reference generation.

Allocator statistics, when the runtime exposes them, measure invocation allocation requests; they are not peak GPU VRAM or hardware memory traffic. `nvidia-smi` records hardware/process context, not per-kernel DRAM bytes. No missing counter is replaced with an estimate.

Optional Nsight Systems profiling should run in a separate PBS GPU job, outside timing measurements. If installed and permitted by the site, activate the same environment, enter the project directory, and run e.g. `nsys profile --trace=cuda,nvtx -o cache_profile python research2/gpu_suite.py --arch sm_80 --family attention --limit 1 --trials 5` (replace architecture). Retain the `.nsys-rep`. It helps inspect launches/copies; hardware DRAM counters require a separate suitable Nsight Compute experiment and site profiling permissions. Do not claim traffic reduction from host latency alone.

## Local validation of this bundle

Shell syntax and Python byte-compilation checks are included in validation. No GPU is attached to the preparation environment; this site-specific PBS path is not the source of the final Colab GPU results. Prepared IR is used so the GPU host need not build MLIR/IREE or run the C++ passes again. Source passes and original project reports remain included for inspection. This package does not introduce a universal GQA pass and does not test the artificial scalar-read example as a natural model optimization.
