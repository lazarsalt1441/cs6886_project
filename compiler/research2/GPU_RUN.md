# Run the saved experiments on an NVIDIA GPU

Use the supplied optimized tensor IR: rebuilding MLIR/IREE from source is unnecessary for these GPU runs. Choose the actual architecture; the commands below use an A100(sm_80). H100 uses sm_90, but only sm_80 offline compilation was tested here.

From the unpacked repository root, in a Python3.12 environment:

```bash
nvidia-smi
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install iree-base-compiler==3.13.0 iree-base-runtime==3.13.0 iree-turbine==3.9.0 numpy==2.3.5 ml_dtypes==0.6.0
python research2/gpu_suite.py --arch sm_80
python research2/extension_gpu_suite.py --arch sm_80 --prepared-ir
```

CPU PyTorch supplies references; IREE uses the GPU through the NVIDIA driver. For eager CUDA PyTorch comparisons, install a CUDA-enabled PyTorch build appropriate for the host instead of the CPU wheel, then add `--torch-cuda` to `gpu_suite.py`. The optional eager comparison currently covers the main and winner suite.

The first suite covers nine attention shapes, two synthetic transformer blocks and the runtime-winner case. The second covers four per-token update lengths and the observable-snapshot case, including the phase-local COW repair. Both check numerical results before timing. Snapshot results require exact cache bits.

`--prepared-ir` reuses the bundled sm_80 COW-repaired IR, so the second command does not invoke the standalone C++ tool. For a different architecture without rebuilding that tool, run:

```bash
python research2/extension_gpu_suite.py --arch sm_90 --skip-root-cow
```

This still tests the protected baseline and conditional/masked optimized forms. To regenerate the architecture-specific phase-local COW form, build `gqa-opt` using the main README and omit both flags. The SDK bootstrap targets Ubuntu24.04 x86_64; the prepared-IR route avoids that SDK dependency.

For CPU-only offline checks, append `--compile-only`. Recorded here:43/44 main forms,4/4 winner forms and21/21 extension forms compile for sm_80. The main irregular masked-gather case fails GPU bufferization; the script retains that failure and the conditional alternative. No GPU runtime result was collected here.

Outputs are JSON under `research2/results/gpu-*.json`, with logs and executables in the GPU result directories. GPU timing includes synchronized invocation and all output downloads, excludes input upload, and is not kernel-only latency. CPU references and compilation are outside timing. Use device profiling separately for hardware traffic; scheduled HAL copies are not DRAM counters.

Keep the generated runtime JSON, the GPU model/driver output from nvidia-smi, and the exact compiler version when sharing results. Do not compare against the unprotected stock snapshot diagnostic: it returns incorrect cache values.
