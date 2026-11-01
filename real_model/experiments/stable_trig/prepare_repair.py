import json,os,subprocess,sys,hashlib
from pathlib import Path
from iree.compiler import ir
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'prepared';OUT.mkdir(exist_ok=True)
sdk=ROOT/'compiler/toolchain/usr/lib';env=os.environ.copy();env['LD_LIBRARY_PATH']=str(sdk/'llvm-19/lib')+':'+str(sdk/'x86_64-linux-gnu')
rows=[]
for which,base,label in [('A','baseline','A_stable'),('B','baseline','B_stable'),('B','optimized','B_stable_commit')]:
 source=ROOT/'prepared'/which/(base+'.mlir')
 with ir.Context():m=ir.Module.parse(source.read_text());(OUT/(label+'_generic.mlir')).write_text(m.operation.get_asm(print_generic_op_form=True))
 subprocess.run([str(ROOT/'compiler/build/gqa-opt'),str(OUT/(label+'_generic.mlir')),'--allow-unregistered-dialect','--attention-stable-trig','--canonicalize','-o',str(OUT/(label+'_changed.mlir'))],env=env,check=True)
 with ir.Context():m=ir.Module.parse((OUT/(label+'_changed.mlir')).read_text());text=str(m)
 target=OUT/(label+'.mlir');target.write_text(text);rows.append(dict(set=which,label=label,sha256=hashlib.sha256(text.encode()).hexdigest(),stable_trig_markers=text.count('attention.stable_trig'),commit_markers=text.count('attention.concat_commit_factored')))
(OUT/'manifest.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
