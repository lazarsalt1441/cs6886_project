"""Add aggressive-fusion controls to full-block and escaping-cache studies."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
from block_study import BranchBlock,block_inputs

def main():
 torch.set_num_threads(1)
 for family,file,folder in [('block','block-study.json','blocks'),('commit','commit-study.json','commit')]:
  path=Path('research2/results')/file;rows=json.loads(path.read_text())
  for row in rows:
   if family=='block':model=BranchBlock(row['branches'],row['pool']);x=block_inputs(row['branches'],row['pool'])
   else:model=BranchAttention(row['shape'],returned='one' if row['case']=='one' else True);x=inputs(row['shape'])
   with torch.inference_mode():
    ref=model(*[a.double() if a.is_floating_point() else a for a in x]);refs=[a.numpy() for a in ref] if isinstance(ref,(tuple,list)) else [ref.numpy()]
   w=Path('research2/results')/folder/row['case']
   for v,source in [('aggressive','baseline'),('lifted-aggressive','lifted-normalized')]:
    flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'--iree-dispatch-creation-enable-aggressive-fusion']
    cs=run(['iree-compile',w/(source+'.mlir'),*flags,'-o',w/(v+'.vmfb')],w/(v+'-compile.log'));run(['iree-compile',w/(source+'.mlir'),*flags,'--compile-to=hal','-o',w/(v+'-hal.mlir')],w/(v+'-hal.log'))
    cfg,ctx,fn,args=load(w/(v+'.vmfb'),[a.clone() for a in x]);before=cfg.device.allocator.statistics;y=fn(*args);after=cfg.device.allocator.statistics;ys=[y] if hasattr(y,'to_host') else list(y);assert len(ys)==len(refs);errors=[]
    for a,b in zip(ys,refs):actual=a.to_host();np.testing.assert_allclose(actual,b,atol=1e-4 if family=='block' else 4e-5,rtol=2e-3);errors.append(float(np.max(np.abs(actual-b))))
    text=(w/(v+'-hal.mlir')).read_text();row['variants'][v]={'compiled':True,'correct':True,'compile_s':cs,'max_abs':max(errors),'allocated_bytes':after['device_bytes_allocated']-before['device_bytes_allocated'],'dispatches':text.count('hal.command_buffer.dispatch'),'copy_command_bytes':copy_stats(text),'vmfb_sha256':hashlib.sha256((w/(v+'.vmfb')).read_bytes()).hexdigest()}
    print(family,row['case'],v,row['variants'][v]['allocated_bytes'],flush=True)
   path.write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
