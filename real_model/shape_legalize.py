"""Prove exporter symbolic guard minima ordered over declared integer ranges.
No tensor/cache math is changed; unproved min expressions fail rather than guess.
"""
import operator,torch,sympy
from torch.utils._sympy.value_ranges import bound_sympy
def legalize(ep):
 count=0
 for node in list(ep.graph.nodes):
  if node.op!='call_function' or node.target is not torch.sym_min:continue
  a,b=node.args
  def expr(x):
   v=x.meta.get('val') if isinstance(x,torch.fx.Node) else x
   return v.node.expr if isinstance(v,torch.SymInt) else sympy.Integer(v)
  delta=sympy.expand(expr(b)-expr(a));r=bound_sympy(delta,ep.range_constraints)
  chosen=a if r.lower>=0 else b if r.upper<=0 else None
  if chosen is None:raise RuntimeError(f'Unproved symbolic min: {delta}, {r}')
  node.replace_all_uses_with(chosen);ep.graph.erase_node(node);count+=1
 for node in list(ep.graph.nodes):
  if node.op=='call_function' and node.target is operator.eq and node.args[0] is node.args[1]:
   for user in list(node.users):
    if user.target==torch.ops.aten._assert_scalar.default:ep.graph.erase_node(user)
   if not node.users:ep.graph.erase_node(node)
 ep.graph.eliminate_dead_code();ep.graph_module.recompile();return count
