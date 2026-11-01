from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'reproducers';P.mkdir(exist_ok=True)
cases=[]
def t(s):return 'tensor<'+'x'.join(map(str,s))+'xf32>'
def save(name,args,shape,body,shapes,kind,**kw):
 (P/(name+'.mlir')).write_text('module {\nfunc.func @main('+args+') -> '+t(shape)+' {\n'+body+'\nreturn %r : '+t(shape)+'\n}\n}\n')
 cases.append(dict(name=name,shapes=shapes,kind=kind,**kw))
def mm(n,a,b,sa,sb,tr=False):
 so=[sa[0],sb[0] if tr else sb[1]]
 maps=' indexing_maps = [affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(n,k)>,affine_map<(m,n,k)->(m,n)>]' if tr else ''
 return f'%{n}z = arith.constant dense<0.0> : {t(so)}\n%{n} = linalg.matmul{maps} ins(%{a},%{b} : {t(sa)}, {t(sb)}) outs(%{n}z : {t(so)}) -> {t(so)}\n'
def softmax(score,N,D):
 return f'''%scale = arith.constant dense<{D**-.5:.12e}> : tensor<1x{N}xf32>
%scaled = arith.mulf %{score}, %scale : tensor<1x{N}xf32>
%neg = arith.constant dense<0xFF800000> : tensor<1xf32>
%mx = linalg.generic {{indexing_maps = [affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m)>], iterator_types = ["parallel","reduction"]}} ins(%scaled : tensor<1x{N}xf32>) outs(%neg : tensor<1xf32>) {{
^bb0(%x:f32,%o:f32):
%v = arith.maximumf %x,%o : f32
linalg.yield %v : f32
}} -> tensor<1xf32>
%ee = tensor.empty() : tensor<1x{N}xf32>
%ex = linalg.generic {{indexing_maps = [affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m)>,affine_map<(m,n)->(m,n)>], iterator_types = ["parallel","parallel"]}} ins(%scaled,%mx : tensor<1x{N}xf32>,tensor<1xf32>) outs(%ee : tensor<1x{N}xf32>) {{
^bb0(%x:f32,%m:f32,%o:f32):
%d = arith.subf %x,%m : f32
%e = math.exp %d : f32
linalg.yield %e : f32
}} -> tensor<1x{N}xf32>
%zero = arith.constant dense<0.0> : tensor<1xf32>
%sum = linalg.generic {{indexing_maps = [affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m)>], iterator_types = ["parallel","reduction"]}} ins(%ex : tensor<1x{N}xf32>) outs(%zero : tensor<1xf32>) {{
^bb0(%x:f32,%o:f32):
%v = arith.addf %x,%o : f32
linalg.yield %v : f32
}} -> tensor<1xf32>
%pe = tensor.empty() : tensor<1x{N}xf32>
%p = linalg.generic {{indexing_maps = [affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m)>,affine_map<(m,n)->(m,n)>], iterator_types = ["parallel","parallel"]}} ins(%ex,%sum : tensor<1x{N}xf32>,tensor<1xf32>) outs(%pe : tensor<1x{N}xf32>) {{
^bb0(%x:f32,%s:f32,%o:f32):
%v = arith.divf %x,%s : f32
linalg.yield %v : f32
}} -> tensor<1x{N}xf32>
'''
def read(n,base,patch,start,read,U,N):
 return f'''%{n}end = arith.addi %{start}, %U : index
%{n}ge = arith.cmpi uge, %{read}, %{start} : index
%{n}lt = arith.cmpi ult, %{read}, %{n}end : index
%{n}in = arith.andi %{n}ge, %{n}lt : i1
%{n} = scf.if %{n}in -> (f32) {{
%j = arith.subi %{read}, %{start} : index
%v = tensor.extract %{patch}[%j] : tensor<{U}xf32>
scf.yield %v : f32
}} else {{
%v = tensor.extract %{base}[%{read}] : tensor<{N}xf32>
scf.yield %v : f32
}}
'''
for N,U in [(64,4),(4096,16)]:
 for opt in [False,True]:
  args=f'%base: tensor<{N}xf32>, %patch: tensor<{U}xf32>, %idx: tensor<2xi64>'
  b='%c0 = arith.constant 0 : index\n%c1 = arith.constant 1 : index\n%s0 = tensor.extract %idx[%c0] : tensor<2xi64>\n%i0 = tensor.extract %idx[%c1] : tensor<2xi64>\n%s = arith.index_cast %s0 : i64 to index\n%i = arith.index_cast %i0 : i64 to index\n'
  if opt:b+=f'%U = arith.constant {U} : index\n'+read('x','base','patch','s','i',U,N)
  else:b+=f'%updated = tensor.insert_slice %patch into %base[%s] [{U}] [1] : tensor<{U}xf32> into tensor<{N}xf32>\n%x = tensor.extract %updated[%i] : tensor<{N}xf32>\n'
  save(f'forward_{N}_'+('custom' if opt else 'stock'),args,[1],b+'%r = tensor.from_elements %x : tensor<1xf32>',[[N],[U],[2]],'forward',N=N,U=U,custom=opt)
for kind in ['same_slice','overwrite','disjoint_chain']:
 b='%u = tensor.insert_slice %a into %base[8] [4] [1] : tensor<4xf32> into tensor<64xf32>\n'
 if kind=='same_slice':b+='%r = tensor.extract_slice %u[8] [4] [1] : tensor<64xf32> to tensor<4xf32>';s=[4]
 elif kind=='overwrite':b+='%r = tensor.insert_slice %b into %u[8] [4] [1] : tensor<4xf32> into tensor<64xf32>';s=[64]
 else:b+='%v = tensor.insert_slice %b into %u[16] [4] [1] : tensor<4xf32> into tensor<64xf32>\n%r = tensor.extract_slice %v[8] [4] [1] : tensor<64xf32> to tensor<4xf32>';s=[4]
 save(kind,'%base: tensor<64xf32>, %a: tensor<4xf32>, %b: tensor<4xf32>',s,b,[[64],[4],[4]],kind)
for opt in [False,True]:
 b='%c0 = arith.constant 0 : index\n%c = tensor.extract %cond[%c0] : tensor<1xi32>\n%z = arith.constant 0 : i32\n%yes = arith.cmpi ne, %c, %z : i32\n'
 if opt:b+='%p = arith.select %yes, %a, %b : tensor<16xf32>\n%r = tensor.insert_slice %p into %base[128] [16] [1] : tensor<16xf32> into tensor<4096xf32>'
 else:b+='%u = tensor.insert_slice %a into %base[128] [16] [1] : tensor<16xf32> into tensor<4096xf32>\n%v = tensor.insert_slice %b into %base[128] [16] [1] : tensor<16xf32> into tensor<4096xf32>\n%r = arith.select %yes, %u, %v : tensor<4096xf32>'
 save('winner_'+('custom' if opt else 'stock'),'%base: tensor<4096xf32>, %a: tensor<16xf32>, %b: tensor<16xf32>, %cond: tensor<1xi32>',[4096],b,[[4096],[16],[16],[1]],'winner',custom=opt)
# Two dynamic patches with arbitrary overlap; the later patch wins.
for opt in [False,True]:
 b='%c0 = arith.constant 0 : index\n%c1 = arith.constant 1 : index\n%c2 = arith.constant 2 : index\n%a0 = tensor.extract %idx[%c0] : tensor<3xi64>\n%b0 = tensor.extract %idx[%c1] : tensor<3xi64>\n%i0 = tensor.extract %idx[%c2] : tensor<3xi64>\n%s = arith.index_cast %a0 : i64 to index\n%t = arith.index_cast %b0 : i64 to index\n%i = arith.index_cast %i0 : i64 to index\n'
 if not opt:b+='%u = tensor.insert_slice %a into %base[%s] [4] [1] : tensor<4xf32> into tensor<64xf32>\n%v = tensor.insert_slice %b into %u[%t] [4] [1] : tensor<4xf32> into tensor<64xf32>\n%x = tensor.extract %v[%i] : tensor<64xf32>\n'
 else:
  b+='%U = arith.constant 4 : index\n%end = arith.addi %t,%U : index\n%ge = arith.cmpi uge, %i,%t : index\n%lt = arith.cmpi ult, %i,%end : index\n%in = arith.andi %ge,%lt : i1\n%x = scf.if %in -> (f32) {\n%j = arith.subi %i,%t : index\n%v = tensor.extract %b[%j] : tensor<4xf32>\nscf.yield %v : f32\n} else {\n'+read('early','base','a','s','i',4,64)+'scf.yield %early : f32\n}\n'
 save('chain_'+('custom' if opt else 'stock'),'%base: tensor<64xf32>, %a: tensor<4xf32>, %b: tensor<4xf32>, %idx: tensor<3xi64>',[1],b+'%r = tensor.from_elements %x : tensor<1xf32>',[[64],[4],[4],[3]],'chain',custom=opt)
for N,R,D in [(32,8,16),(512,32,128)]:
 for opt in [False,True]:
  args=f'%q: {t([1,D])}, %l: {t([N,R])}, %wk: {t([R,D])}, %wv: {t([R,D])}'
  if opt:b=mm('ql','q','wk',[1,D],[R,D],True)+mm('score','ql','l',[1,R],[N,R],True)
  else:b=mm('key','l','wk',[N,R],[R,D])+mm('score','q','key',[1,D],[N,D],True)
  b+=softmax('score',N,D)
  if opt:b+=mm('vl','p','l',[1,N],[N,R])+mm('r','vl','wv',[1,R],[R,D])
  else:b+=mm('val','l','wv',[N,R],[R,D])+mm('r','p','val',[1,N],[N,D])
  save(f'projection_{N}_'+('custom' if opt else 'stock'),args,[1,D],b,[[1,D],[N,R],[R,D],[R,D]],'projection',N=N,R=R,D=D,custom=opt)
for opt in [False,True]:
 N,R,D,C=32,8,16,4
 args=f'%q: {t([1,D])}, %l: {t([N,R])}, %wk: {t([R,D])}, %qr: {t([1,C])}, %kr: {t([N,C])}, %wv: {t([R,D])}'
 if opt:b=mm('ql','q','wk',[1,D],[R,D],True)+mm('content','ql','l',[1,R],[N,R],True)
 else:b=mm('key','l','wk',[N,R],[R,D])+mm('content','q','key',[1,D],[N,D],True)
 b+=mm('position','qr','kr',[1,C],[N,C],True)+f'%score = arith.addf %content,%position : {t([1,N])}\n'+softmax('score',N,D+C)
 if opt:b+=mm('vl','p','l',[1,N],[N,R])+mm('r','vl','wv',[1,R],[R,D])
 else:b+=mm('val','l','wv',[N,R],[R,D])+mm('r','p','val',[1,N],[N,D])
 save('partial_rope_'+('custom' if opt else 'stock'),args,[1,D],b,[[1,D],[N,R],[R,D],[1,C],[N,C],[R,D]],'partial_rope',custom=opt)
for H,G in [(2,4),(1,4),(2,1)]:
 N,D=32,16;QH=H*G
 b=f'''%ke = tensor.expand_shape %k [[0,1],[2],[3]] output_shape [{H},1,{N},{D}] : {t([H,N,D])} into {t([H,1,N,D])}
%empty = tensor.empty() : {t([H,G,N,D])}
%kb = linalg.generic {{indexing_maps = [affine_map<(h,g,n,d)->(h,0,n,d)>, affine_map<(h,g,n,d)->(h,g,n,d)>], iterator_types = ["parallel","parallel","parallel","parallel"]}} ins(%ke : {t([H,1,N,D])}) outs(%empty : {t([H,G,N,D])}) {{
^bb0(%x:f32,%o:f32):
linalg.yield %x : f32
}} -> {t([H,G,N,D])}
%kc = tensor.collapse_shape %kb [[0,1],[2],[3]] : {t([H,G,N,D])} into {t([QH,N,D])}
%zero = arith.constant dense<0.0> : {t([QH,1,N])}
%r = linalg.batch_matmul indexing_maps = [affine_map<(b,m,n,k)->(b,m,k)>,affine_map<(b,m,n,k)->(b,n,k)>,affine_map<(b,m,n,k)->(b,m,n)>] ins(%q,%kc : {t([QH,1,D])}, {t([QH,N,D])}) outs(%zero : {t([QH,1,N])}) -> {t([QH,1,N])}
'''
 save(f'gqa_{H}_{G}',f'%q: {t([QH,1,D])}, %k: {t([H,N,D])}',[QH,1,N],b,[[QH,1,D],[H,N,D]],'gqa',H=H,G=G)
for D,V in [(64,64),(128,128),(256,128),(512,128),(128,512),(512,512)]:
 N=32
 args=f'%q: {t([1,1,D])}, %k: {t([1,N,D])}, %v: {t([1,N,V])}'
 maps='[affine_map<(b,m,k,n,v)->(b,m,k)>,affine_map<(b,m,k,n,v)->(b,n,k)>,affine_map<(b,m,k,n,v)->(b,n,v)>,affine_map<(b,m,k,n,v)->()>,affine_map<(b,m,k,n,v)->(b,m,v)>]'
 b=f'%scale = arith.constant {D**-.5:.12e} : f32\n%e = tensor.empty() : {t([1,1,V])}\n%r = iree_linalg_ext.attention <indexing_maps = {maps}> ins(%q,%k,%v,%scale : {t([1,1,D])}, {t([1,N,D])}, {t([1,N,V])}, f32) outs(%e : {t([1,1,V])}) {{\n^bb0(%score:f32):\niree_linalg_ext.yield %score : f32\n}} -> {t([1,1,V])}'
 save(f'native_{D}_{V}',args,[1,1,V],b,[[1,1,D],[1,N,D],[1,N,V]],'native',D=D,V=V)
# A pattern the original C++ pass actually selects: two competing versions,
# each consumed only by a small gather (not the single-use scalar controls).
for N in [512,8192]:
 for opt in [False,True]:
  b=''
  if not opt:
   b+=f'%ua = tensor.insert_slice %a into %base[128,0] [4,16] [1,1] : tensor<4x16xf32> into tensor<{N}x16xf32>\n%ub = tensor.insert_slice %b into %base[256,0] [4,16] [1,1] : tensor<4x16xf32> into tensor<{N}x16xf32>\n'
  for name,patch,off in [('ga','a',128),('gb','b',256)]:
   b+=f'''%{name}e = tensor.empty() : tensor<8x16xf32>
%{name} = linalg.generic {{indexing_maps=[affine_map<(i,j)->(i)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}} ins(%idx : tensor<8xi64>) outs(%{name}e : tensor<8x16xf32>) {{
^bb0(%raw:i64,%unused:f32):
%row = arith.index_cast %raw : i64 to index
%col = linalg.index 1 : index
'''
   if opt:
    b+=f'''%start = arith.constant {off} : index
%end = arith.constant {off+4} : index
%ge = arith.cmpi uge, %row,%start : index
%lt = arith.cmpi ult, %row,%end : index
%inside = arith.andi %ge,%lt : i1
%x = scf.if %inside -> (f32) {{
%local = arith.subi %row,%start : index
%v = tensor.extract %{patch}[%local,%col] : tensor<4x16xf32>
scf.yield %v : f32
}} else {{
%v = tensor.extract %base[%row,%col] : tensor<{N}x16xf32>
scf.yield %v : f32
}}
'''
   else:b+=f'%x = tensor.extract %u{patch}[%row,%col] : tensor<{N}x16xf32>\n'
   b+='linalg.yield %x : f32\n} -> tensor<8x16xf32>\n'
  b+='%r = tensor.concat dim(0) %ga,%gb : (tensor<8x16xf32>,tensor<8x16xf32>) -> tensor<16x16xf32>'
  save(f'gather_forks_{N}_'+('custom' if opt else 'stock'),f'%base: tensor<{N}x16xf32>, %a: tensor<4x16xf32>, %b: tensor<4x16xf32>, %idx: tensor<8xi64>',[16,16],b,[[N,16],[4,16],[4,16],[8]],'gather_forks',N=N,custom=opt)
(P/'cases.json').write_text(json.dumps(cases,indent=2));print('Generated',len(cases),'programs')
