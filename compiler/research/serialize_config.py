from iree.compiler import ir
from pathlib import Path
with ir.Context():
 m=ir.Module.parse(Path("research/ir/backend-configured.mlir").read_text())
 Path("research/ir/backend-configured-generic.mlir").write_text(m.operation.get_asm(print_generic_op_form=True))
