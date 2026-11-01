# M1: evidence before optimization

Compiler: IREE 3.13.0rc20261016 (89ece87b89852100c0a4ac3d515e14842d090967).
The input is an actual Turbine export from explicit PyTorch repeat_interleave.

For B=1, Hq=8, Hkv=2, M=1, N=128, D=32:
- Torch IR has two aten.repeat_interleave.self_int operations.
- Linalg lowering has two rank-five pure-copy broadcast generic operations,
  output shape 1x2x4x128x32, then reshape/transpose/batch_matmul chains.
- Baseline IREE executable source includes a standalone dispatch named
  elementwise_broadcast_2x4x4096_f32. Its output is a writable storage binding
  of shape 2x4x4096; it performs dispatch.tensor.store on the broadcast result.
- This is a physical intermediate crossing dispatch boundaries, rather than
  just a logical reshape. Inspect ir/baseline-dumps/*dispatch_0.mlir.
- Optimized IR replaces QK and PV with rank-five grouped contractions;
  all indexing maps are projected permutations. No floordiv is inserted into
  Linalg indexing maps (which can impede tiling/inversion).
- The identity h = kv_head * ratio + within_group implements kv_head=h//ratio
  through splitting the batch/head loop. This keeps the input head extent Hkv.

Baseline LLVM IR and assembly were dumped with IREE's executable-files flag.
See later results for allocation sizes and performance; source-level traffic
is not a measurement of hardware cache misses or DRAM traffic.
