"""Inject the compiled C++ phase-local COW repair into an actual IREE pipeline.
The IR capture is at the verifier immediately preceding COW materialization.
"""
import sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import ir,run,STRONG

def repair(source,dest,log):
 with ir.Context():mod=ir.Module.parse(Path(source).read_text());temp=Path(dest).with_name(Path(dest).stem+'-input.mlir');temp.write_text(mod.operation.get_asm(print_generic_op_form=True,use_local_scope=True))
 run(['build/gqa-opt',temp,'--allow-unregistered-dialect','--attention-materialize-borrowed-cache-cow','-o',dest],log)
 with ir.Context():ir.Module.parse(Path(dest).read_text()).operation.verify()
def compile_repaired(source,output,target='cpu',arch='sm_80'):
 w=Path(output).with_suffix('.cow-artifacts');w.mkdir(parents=True,exist_ok=True);flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host'] if target=='cpu' else ['--iree-hal-target-backends=cuda','--iree-cuda-target='+arch]
 # Print only this single module checkpoint, not all compiler passes.
 run(['iree-compile',source,*flags,*STRONG,'--compile-to=stream','--mlir-disable-threading','--mlir-print-ir-after=iree-stream-verify-lowering-to-async-resources','--mlir-print-ir-module-scope','-o',w/'discarded-stream.mlir'],w/'checkpoint.log')
 text=(w/'checkpoint.log').read_text();start=text.find('\n',text.index('IR Dump After'));captured=text[start+1:].strip();assert captured.startswith('#') or captured.startswith('module'),captured[:200];(w/'pre-cow.mlir').write_text(captured)
 repair(w/'pre-cow.mlir',w/'repaired.mlir',w/'repair.log');count=(w/'repaired.mlir').read_text().count('borrowed_cache_cow')
 run(['iree-compile',w/'repaired.mlir','--compile-from=flow',*flags,*STRONG,'-o',output],w/'compile.log');return count
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--target',choices=['cpu','cuda'],default='cpu');ap.add_argument('--arch',default='sm_80');a=ap.parse_args();print(json.dumps({'inserted_clones':compile_repaired(a.source,a.output,a.target,a.arch)}))
