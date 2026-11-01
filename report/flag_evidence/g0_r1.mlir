module @module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "znver4", cpu_features = "+adx,+aes,+avx,+avx2,+avx512bf16,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi,+bmi2,+clflushopt,+clwb,+clzero,+crc32,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+gfni,+invpcid,+lzcnt,+mmx,+movbe,+mwaitx,+pclmul,+pku,+popcnt,+prfchw,+rdpid,+rdpru,+rdrnd,+rdseed,+sahf,+sha,+shstk,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+sse4a,+ssse3,+vaes,+vpclmulqdq,+wbnoinvd,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,+64bit,+adx,+aes,-amx-avx512,-amx-bf16,-amx-complex,-amx-fp16,-amx-fp8,-amx-int8,-amx-movrs,-amx-tile,+avx,-avx10.1,-avx10.2,-avx10v2aux,+avx2,+avx512bf16,+avx512bitalg,-avx512bmm,+avx512bw,+avx512cd,+avx512dq,+avx512f,-avx512fp16,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,-avx512vp2intersect,+avx512vpopcntdq,-avxifma,-avxneconvert,-avxvnni,-avxvnniint16,-avxvnniint8,+bmi,+bmi2,-ccmp,-cf,-cldemote,+clflushopt,+clwb,+clzero,+cmov,-cmpccxadd,+crc32,+cx16,+cx8,-egpr,-enqcmd,+f16c,+fma,-fma4,+fsgsbase,+fxsr,+gfni,-hreset,+invpcid,-jmpabs,-kl,-lwp,+lzcnt,+mmx,+movbe,-movdir64b,-movdiri,-movrs,-mwaitx,-ndd,-nf,+pclmul,-pconfig,-pku,+popcnt,-ppx,-prefetchi,+prfchw,-ptwrite,-push2pop2,-raoint,+rdpid,-rdpru,+rdrnd,+rdseed,-rtm,+sahf,-serialize,-sgx,+sha,-sha512,+shstk,-sm3,-sm4,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+sse4a,+ssse3,-tbm,-tsxldtrk,-uintr,-usermsr,+vaes,+vpclmulqdq,-waitpkg,-wbnoinvd,-widekl,-xop,+xsave,+xsavec,+xsaveopt,+xsaves,-zu", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 64 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>]> : !hal.device
  flow.executable private @main_dispatch_0 {
    flow.executable.export public @main_dispatch_0_elementwise_broadcast_2x4x4096_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_0_elementwise_broadcast_2x4x4096_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4096xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x4x4096xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [2, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4096xf32>> -> tensor<2x4096xf32>
        %1 = tensor.empty() : tensor<2x4x4096xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<2x4096xf32>) outs(%1 : tensor<2x4x4096xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<2x4x4096xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [2, 4, 4096], strides = [1, 1, 1] : tensor<2x4x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x4x4096xf32>>
        return
      }
    }
  }
  flow.executable private @main_dispatch_2 {
    flow.executable.export public @main_dispatch_2_batch_matmul_8x1x128x32_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_2_batch_matmul_8x1x128x32_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x1x32xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x128x32xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x1x128xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 0.176776692 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [8, 1, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x1x32xf32>> -> tensor<8x1x32xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0], sizes = [8, 128, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x128x32xf32>> -> tensor<8x128x32xf32>
        %2 = tensor.empty() : tensor<8x1x128xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<8x1x128xf32>) -> tensor<8x1x128xf32>
        %4 = linalg.batch_matmul indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>] ins(%0, %1 : tensor<8x1x32xf32>, tensor<8x128x32xf32>) outs(%3 : tensor<8x1x128xf32>) -> tensor<8x1x128xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<8x1x128xf32>) outs(%2 : tensor<8x1x128xf32>) {
        ^bb0(%in: f32, %out: f32):
          %6 = arith.mulf %in, %cst_0 : f32
          linalg.yield %6 : f32
        } -> tensor<8x1x128xf32>
        iree_tensor_ext.dispatch.tensor.store %5, %arg2, offsets = [0, 0, 0], sizes = [8, 1, 128], strides = [1, 1, 1] : tensor<8x1x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x1x128xf32>>
        return
      }
    }
  }
  flow.executable private @main_dispatch_3 {
    flow.executable.export public @main_dispatch_3_softmax_8x128xf32_dispatch_tensor_store workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_3_softmax_8x128xf32_dispatch_tensor_store(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x128xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x128xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [8, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x128xf32>> -> tensor<8x128xf32>
        %1 = tensor.empty() : tensor<8x128xf32>
        %2 = linalg.softmax dimension(1) ins(%0 : tensor<8x128xf32>) outs(%1 : tensor<8x128xf32>) -> tensor<8x128xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0, 0], sizes = [8, 128], strides = [1, 1] : tensor<8x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x128xf32>>
        return
      }
    }
  }
  flow.executable private @main_dispatch_4 {
    flow.executable.export public @main_dispatch_4_batch_matmul_8x1x32x128_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_4_batch_matmul_8x1x32x128_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x1x128xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x128x32xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x1x32xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [8, 1, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x1x128xf32>> -> tensor<8x1x128xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0], sizes = [8, 128, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x128x32xf32>> -> tensor<8x128x32xf32>
        %2 = tensor.empty() : tensor<8x1x32xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<8x1x32xf32>) -> tensor<8x1x32xf32>
        %4 = linalg.batch_matmul ins(%0, %1 : tensor<8x1x128xf32>, tensor<8x128x32xf32>) outs(%3 : tensor<8x1x32xf32>) -> tensor<8x1x32xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0], sizes = [8, 1, 32], strides = [1, 1, 1] : tensor<8x1x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x1x32xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @main(%input0: tensor<1x8x1x32xf32>, %input1: tensor<1x2x128x32xf32>, %input2: tensor<1x2x128x32xf32>) -> (%output0: tensor<1x8x1x32xf32>)"}} {
    %0 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<1x8x1x32xf32>
    %1 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<1x2x128x32xf32>
    %2 = hal.tensor.import %arg2 "input2" : !hal.buffer_view -> tensor<1x2x128x32xf32>
    %3 = flow.tensor.reshape %1 : tensor<1x2x128x32xf32> -> tensor<2x4096xf32>
    %4 = flow.dispatch @main_dispatch_0::@main_dispatch_0_elementwise_broadcast_2x4x4096_f32(%3) : (tensor<2x4096xf32>) -> tensor<2x4x4096xf32>
    %5 = flow.tensor.reshape %2 : tensor<1x2x128x32xf32> -> tensor<2x4096xf32>
    %6 = flow.dispatch @main_dispatch_0::@main_dispatch_0_elementwise_broadcast_2x4x4096_f32(%5) : (tensor<2x4096xf32>) -> tensor<2x4x4096xf32>
    %7 = flow.tensor.reshape %4 : tensor<2x4x4096xf32> -> tensor<8x128x32xf32>
    %8 = flow.tensor.reshape %0 : tensor<1x8x1x32xf32> -> tensor<8x1x32xf32>
    %9 = flow.dispatch @main_dispatch_2::@main_dispatch_2_batch_matmul_8x1x128x32_f32(%8, %7) : (tensor<8x1x32xf32>, tensor<8x128x32xf32>) -> tensor<8x1x128xf32>
    %10 = flow.tensor.reshape %9 : tensor<8x1x128xf32> -> tensor<8x128xf32>
    %11 = flow.dispatch @main_dispatch_3::@main_dispatch_3_softmax_8x128xf32_dispatch_tensor_store(%10) : (tensor<8x128xf32>) -> tensor<8x128xf32>
    %12 = flow.tensor.reshape %11 : tensor<8x128xf32> -> tensor<8x1x128xf32>
    %13 = flow.tensor.reshape %6 : tensor<2x4x4096xf32> -> tensor<8x128x32xf32>
    %14 = flow.dispatch @main_dispatch_4::@main_dispatch_4_batch_matmul_8x1x32x128_f32(%12, %13) : (tensor<8x1x128xf32>, tensor<8x128x32xf32>) -> tensor<8x1x32xf32>
    %15 = flow.tensor.reshape %14 : tensor<8x1x32xf32> -> tensor<1x8x1x32xf32>
    %16 = hal.tensor.export %15 "output0" : tensor<1x8x1x32xf32> -> !hal.buffer_view
    util.return %16 : !hal.buffer_view
  }
}