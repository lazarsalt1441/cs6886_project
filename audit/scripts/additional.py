from pathlib import Path
import os,subprocess,json,argparse,importlib.util
P=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);a=ap.parse_args();proj=Path(a.project).resolve();env=os.environ.copy();env['LD_LIBRARY_PATH']=str(proj/'compiler/toolchain/usr/lib/llvm-19/lib')+':'+str(proj/'compiler/toolchain/usr/lib/x86_64-linux-gnu');tool=proj/'compiler/build/gqa-opt';cases=json.loads((P/'reproducers/cases.json').read_text());rows=[]
def add(name,text,meta,flags=None):
 src=P/'reproducers'/(name+'.mlir');src.write_text(text);cases.append(dict(meta,name=name))
 if flags:
  name2=name.replace('_stock','_actual');dest=P/'reproducers'/(name2+'.mlir');cmd=[str(tool),str(src),'--canonicalize',*flags,'--canonicalize','-o',str(dest)];p=subprocess.run(cmd,env=env,capture_output=True,text=True);(P/'results/actual_pass'/(name2+'.log')).write_text(p.stderr);p.check_returncode();cases.append(dict(meta,name=name2,custom=True));rows.append({'case':name2,'command':cmd,'coalesce':dest.read_text().count('attention.cache_updates_coalesced'),'projection':dest.read_text().count('attention.projection_placement')})
s=(P/'reproducers/gather_forks_512_stock.mlir').read_text();s=s.replace('%ua = tensor.insert_slice %a into %base[128,0] [4,16] [1,1] : tensor<4x16xf32> into tensor<512x16xf32>','%u0 = tensor.insert_slice %a into %base[128,0] [4,16] [1,1] : tensor<4x16xf32> into tensor<512x16xf32>\n%ua = tensor.insert_slice %a into %u0[132,0] [4,16] [1,1] : tensor<4x16xf32> into tensor<512x16xf32>');meta=next(c for c in cases if c['name']=='gather_forks_512_stock');add('coalesce_stock',s,dict(meta,kind='coalesce'),['--attention-coalesce-cache-updates'])
def t(shape):return 'tensor<'+'x'.join(map(str,shape))+'xf32>'
def mm(n,x,y,sx,sy):
 so=[sx[0],sx[1],sy[2]]
 return f'%{n}empty = tensor.empty() : {t(so)}\n%{n}init = linalg.fill ins(%zero : f32) outs(%{n}empty : {t(so)}) -> {t(so)}\n%{n} = linalg.batch_matmul ins(%{x},%{y} : {t(sx)},{t(sy)}) outs(%{n}init : {t(so)}) -> {t(so)}\n'
for rope in [False,True]:
 N,R,D,C=32,8,16,4 if rope else 0;qs=[1,1,D+C];ls=[1,N,R];ws=[1,R,D];ks=[1,N,D+C]
 args=f'%q: {t(qs)},%l: {t(ls)},%w: {t(ws)}'+(f',%kr: {t([1,N,C])}' if rope else '')
 b='%zero = arith.constant 0.0 : f32\n'+mm('content','l','w',ls,ws)
 if rope:b+=f'%key = tensor.concat dim(2) %content,%kr : ({t([1,N,D])},{t([1,N,C])}) -> {t(ks)}\n'
 key='key' if rope else 'content';ts=[1,D+C,N]
 b+=f'%te = tensor.empty() : {t(ts)}\n%kt = linalg.transpose ins(%{key} : {t(ks)}) outs(%te : {t(ts)}) permutation = [0,2,1]\n'+mm('r','q','kt',qs,ts)
 text='module {\nfunc.func @main('+args+') -> tensor<1x1x32xf32> attributes {attention.allow_reassociation} {\n'+b+'return %r : tensor<1x1x32xf32>\n}\n}'
 shapes=[qs,ls,ws]+([[1,N,C]] if rope else []);add('original_'+('rope' if rope else 'projection')+'_stock',text,dict(shapes=shapes,kind='original_projection',rope=rope,custom=False),['--attention-project-reassociate'])
spec=importlib.util.spec_from_file_location('concat_source',proj/'tests/test_concat_factor.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
add('concat_stock',mod.source(8192),dict(shapes=[[1,2,8192,4],[1,2,8192,4],[1,2,1,4],[1,2,1,4],[]],kind='concat',custom=False),['--attention-factor-cache-commit'])
# Idempotent manifest update.
merged={c['name']:c for c in cases};(P/'reproducers/cases.json').write_text(json.dumps(list(merged.values()),indent=2));(P/'results/actual_pass/additional_transform.json').write_text(json.dumps(rows,indent=2));print(rows)
