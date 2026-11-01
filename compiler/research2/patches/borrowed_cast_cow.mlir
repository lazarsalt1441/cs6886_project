// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-stream-materialize-copy-on-write))' %s | FileCheck %s
// Proposed in-tree regression. The stock pinned compiler fails the first case.
// The separately compiled phase-local C++ repair is end-to-end validated.
// CHECK-LABEL: @borrowed_cast
util.func public @borrowed_cast(%buffer: !hal.buffer_view, %update: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %import = stream.tensor.import %buffer : !hal.buffer_view -> tensor<64xf32> in !stream.resource<external>{%c256}
  // CHECK: %[[CAST:.+]] = stream.async.cast
  %cast = stream.async.cast %import : !stream.resource<external>{%c256} -> !stream.resource<*>{%c256}
  // CHECK: %[[COPY:.+]] = stream.async.clone %[[CAST]]
  // CHECK: stream.async.update {{.*}}, %[[COPY]]
  %result = stream.async.update %update, %cast[%c0 to %c8] : !stream.resource<*>{%c8} -> %cast as !stream.resource<*>{%c256}
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @owned_cast
util.func public @owned_cast(%buffer: !hal.buffer_view, %update: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %import = stream.tensor.import consume %buffer : !hal.buffer_view -> tensor<64xf32> in !stream.resource<external>{%c256}
  %cast = stream.async.cast %import : !stream.resource<external>{%c256} -> !stream.resource<*>{%c256}
  // CHECK-NOT: stream.async.clone
  %result = stream.async.update %update, %cast[%c0 to %c8] : !stream.resource<*>{%c8} -> %cast as !stream.resource<*>{%c256}
  util.return %result : !stream.resource<*>
}
