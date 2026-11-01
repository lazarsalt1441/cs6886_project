"""Compile with the C++ feature-reduction guard at the configured-IR boundary."""
from pathlib import Path
from guard_config import guard
from run_suite import run
from projection_study import STRONG
FLAGS=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host','--iree-llvmcpu-stack-allocation-limit=131072']
def compile_guarded(source,destination,work,name='model',baseline_destination=None):
 work=Path(work);configured=work/(name+'-configured.mlir');fixed=work/(name+'-guarded.mlir')
 run(['iree-compile',source,*FLAGS,*STRONG,'--compile-to=executable-configurations','-o',configured],work/(name+'-config.log'))
 changed=guard(configured,fixed,work/(name+'-bridge'))
 if baseline_destination:
  run(['iree-compile',configured,*FLAGS,'--compile-from=executable-configurations','-o',baseline_destination],work/(name+'-baseline-compile.log'))
 run(['iree-compile',fixed,*FLAGS,'--compile-from=executable-configurations','-o',destination],work/(name+'-guarded-compile.log'))
 return changed
