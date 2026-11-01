#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>
#map5 = affine_map<(d0, d1) -> (d0, 0)>
#map6 = affine_map<(d0, d1) -> (0, 0)>
#map7 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> ()>
#map10 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map11 = affine_map<(d0, d1, d2) -> (d2)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map15 = affine_map<()[s0] -> (s0 + 1)>
#map16 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map17 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module @module {
  util.global private @__auto.model.lm_head.weight = #flow.parameter.named<"model"::"model.lm_head.weight"> : tensor<49152x576xf32>
  util.global private @__auto.model.model.layers.0.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.0.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.0.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.0.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.0.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.0.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.0.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.0.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.0.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.0.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.0.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.0.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.0.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.0.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.0.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.0.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.0.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.0.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.1.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.1.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.1.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.1.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.1.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.1.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.1.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.1.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.1.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.1.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.1.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.1.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.1.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.1.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.1.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.1.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.1.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.1.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.2.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.2.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.2.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.2.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.2.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.2.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.2.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.2.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.2.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.2.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.2.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.2.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.2.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.2.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.2.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.2.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.2.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.2.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.3.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.3.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.3.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.3.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.3.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.3.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.3.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.3.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.3.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.3.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.3.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.3.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.3.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.3.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.3.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.3.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.3.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.3.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.4.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.4.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.4.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.4.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.4.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.4.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.4.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.4.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.4.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.4.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.4.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.4.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.4.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.4.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.4.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.4.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.4.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.4.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.5.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.5.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.5.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.5.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.5.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.5.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.5.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.5.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.5.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.5.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.5.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.5.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.5.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.5.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.5.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.5.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.5.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.5.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.6.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.6.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.6.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.6.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.6.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.6.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.6.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.6.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.6.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.6.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.6.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.6.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.6.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.6.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.6.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.6.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.6.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.6.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.7.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.7.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.7.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.7.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.7.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.7.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.7.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.7.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.7.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.7.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.7.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.7.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.7.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.7.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.7.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.7.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.7.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.7.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.8.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.8.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.8.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.8.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.8.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.8.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.8.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.8.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.8.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.8.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.8.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.8.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.8.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.8.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.8.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.8.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.8.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.8.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.9.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.9.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.9.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.9.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.9.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.9.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.9.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.9.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.9.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.9.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.9.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.9.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.9.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.9.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.9.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.9.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.9.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.9.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.10.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.10.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.10.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.10.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.10.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.10.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.10.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.10.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.10.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.10.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.10.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.10.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.10.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.10.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.10.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.10.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.10.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.10.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.11.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.11.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.11.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.11.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.11.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.11.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.11.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.11.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.11.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.11.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.11.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.11.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.11.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.11.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.11.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.11.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.11.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.11.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.12.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.12.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.12.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.12.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.12.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.12.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.12.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.12.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.12.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.12.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.12.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.12.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.12.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.12.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.12.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.12.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.12.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.12.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.13.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.13.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.13.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.13.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.13.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.13.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.13.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.13.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.13.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.13.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.13.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.13.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.13.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.13.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.13.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.13.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.13.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.13.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.14.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.14.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.14.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.14.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.14.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.14.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.14.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.14.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.14.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.14.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.14.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.14.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.14.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.14.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.14.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.14.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.14.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.14.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.15.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.15.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.15.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.15.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.15.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.15.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.15.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.15.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.15.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.15.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.15.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.15.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.15.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.15.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.15.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.15.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.15.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.15.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.16.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.16.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.16.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.16.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.16.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.16.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.16.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.16.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.16.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.16.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.16.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.16.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.16.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.16.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.16.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.16.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.16.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.16.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.17.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.17.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.17.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.17.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.17.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.17.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.17.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.17.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.17.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.17.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.17.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.17.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.17.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.17.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.17.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.17.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.17.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.17.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.18.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.18.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.18.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.18.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.18.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.18.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.18.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.18.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.18.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.18.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.18.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.18.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.18.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.18.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.18.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.18.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.18.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.18.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.19.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.19.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.19.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.19.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.19.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.19.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.19.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.19.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.19.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.19.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.19.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.19.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.19.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.19.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.19.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.19.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.19.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.19.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.20.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.20.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.20.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.20.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.20.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.20.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.20.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.20.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.20.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.20.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.20.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.20.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.20.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.20.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.20.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.20.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.20.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.20.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.21.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.21.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.21.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.21.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.21.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.21.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.21.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.21.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.21.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.21.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.21.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.21.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.21.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.21.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.21.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.21.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.21.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.21.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.22.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.22.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.22.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.22.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.22.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.22.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.22.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.22.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.22.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.22.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.22.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.22.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.22.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.22.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.22.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.22.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.22.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.22.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.23.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.23.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.23.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.23.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.23.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.23.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.23.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.23.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.23.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.23.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.23.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.23.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.23.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.23.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.23.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.23.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.23.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.23.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.24.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.24.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.24.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.24.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.24.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.24.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.24.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.24.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.24.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.24.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.24.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.24.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.24.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.24.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.24.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.24.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.24.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.24.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.25.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.25.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.25.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.25.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.25.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.25.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.25.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.25.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.25.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.25.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.25.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.25.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.25.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.25.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.25.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.25.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.25.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.25.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.26.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.26.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.26.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.26.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.26.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.26.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.26.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.26.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.26.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.26.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.26.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.26.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.26.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.26.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.26.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.26.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.26.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.26.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.27.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.27.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.27.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.27.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.27.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.27.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.27.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.27.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.27.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.27.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.27.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.27.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.27.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.27.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.27.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.27.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.27.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.27.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.28.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.28.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.28.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.28.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.28.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.28.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.28.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.28.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.28.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.28.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.28.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.28.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.28.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.28.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.28.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.28.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.28.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.28.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.layers.29.input_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.29.input_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.29.self_attn.q_proj.weight = #flow.parameter.named<"model"::"model.model.layers.29.self_attn.q_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.29.self_attn.k_proj.weight = #flow.parameter.named<"model"::"model.model.layers.29.self_attn.k_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.29.self_attn.v_proj.weight = #flow.parameter.named<"model"::"model.model.layers.29.self_attn.v_proj.weight"> : tensor<192x576xf32>
  util.global private @__auto.model.model.layers.29.self_attn.o_proj.weight = #flow.parameter.named<"model"::"model.model.layers.29.self_attn.o_proj.weight"> : tensor<576x576xf32>
  util.global private @__auto.model.model.layers.29.post_attention_layernorm.weight = #flow.parameter.named<"model"::"model.model.layers.29.post_attention_layernorm.weight"> : tensor<576xf32>
  util.global private @__auto.model.model.layers.29.mlp.gate_proj.weight = #flow.parameter.named<"model"::"model.model.layers.29.mlp.gate_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.29.mlp.up_proj.weight = #flow.parameter.named<"model"::"model.model.layers.29.mlp.up_proj.weight"> : tensor<1536x576xf32>
  util.global private @__auto.model.model.layers.29.mlp.down_proj.weight = #flow.parameter.named<"model"::"model.model.layers.29.mlp.down_proj.weight"> : tensor<576x1536xf32>
  util.global private @__auto.model.model.norm.weight = #flow.parameter.named<"model"::"model.model.norm.weight"> : tensor<576xf32>
  util.func public @main$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.buffer_view, %arg7: !hal.buffer_view, %arg8: !hal.buffer_view, %arg9: !hal.buffer_view, %arg10: !hal.buffer_view, %arg11: !hal.buffer_view, %arg12: !hal.buffer_view, %arg13: !hal.buffer_view, %arg14: !hal.buffer_view, %arg15: !hal.buffer_view, %arg16: !hal.buffer_view, %arg17: !hal.buffer_view, %arg18: !hal.buffer_view, %arg19: !hal.buffer_view, %arg20: !hal.buffer_view, %arg21: !hal.buffer_view, %arg22: !hal.buffer_view, %arg23: !hal.buffer_view, %arg24: !hal.buffer_view, %arg25: !hal.buffer_view, %arg26: !hal.buffer_view, %arg27: !hal.buffer_view, %arg28: !hal.buffer_view, %arg29: !hal.buffer_view, %arg30: !hal.buffer_view, %arg31: !hal.buffer_view, %arg32: !hal.buffer_view, %arg33: !hal.buffer_view, %arg34: !hal.buffer_view, %arg35: !hal.buffer_view, %arg36: !hal.buffer_view, %arg37: !hal.buffer_view, %arg38: !hal.buffer_view, %arg39: !hal.buffer_view, %arg40: !hal.buffer_view, %arg41: !hal.buffer_view, %arg42: !hal.buffer_view, %arg43: !hal.buffer_view, %arg44: !hal.buffer_view, %arg45: !hal.buffer_view, %arg46: !hal.buffer_view, %arg47: !hal.buffer_view, %arg48: !hal.buffer_view, %arg49: !hal.buffer_view, %arg50: !hal.buffer_view, %arg51: !hal.buffer_view, %arg52: !hal.buffer_view, %arg53: !hal.buffer_view, %arg54: !hal.buffer_view, %arg55: !hal.buffer_view, %arg56: !hal.buffer_view, %arg57: !hal.buffer_view, %arg58: !hal.buffer_view, %arg59: !hal.buffer_view, %arg60: !hal.buffer_view, %arg61: !hal.buffer_view, %arg62: !hal.buffer_view, %arg63: !hal.fence, %arg64: !hal.fence) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub} {
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c49152 = arith.constant 49152 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c-1 = arith.constant -1 : index
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant dense<-3.40282347E+38> : tensor<f32>
    %cst_3 = arith.constant dense<true> : tensor<1x1xi1>
    %cst_4 = arith.constant 9.99999974E-6 : f32
    %cst_5 = arith.constant dense_resource<torch_tensor_32_torch.float32> : tensor<32xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_7 = arith.constant 5.760000e+02 : f32
    %cst_8 = arith.constant 1.250000e-01 : f32
    %0 = hal.tensor.import wait(%arg63) => %arg0 : !hal.buffer_view -> tensor<1x1xi64>
    %1 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
    %2 = hal.tensor.import wait(%arg63) => %arg1 : !hal.buffer_view -> tensor<1x?xi64>{%1}
    %3 = hal.tensor.import wait(%arg63) => %arg2 : !hal.buffer_view -> tensor<1x1xi64>
    %4 = hal.buffer_view.dim<%arg3 : !hal.buffer_view>[2] : index
    %5 = hal.tensor.import wait(%arg63) => %arg3 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%4}
    %6 = hal.buffer_view.dim<%arg4 : !hal.buffer_view>[2] : index
    %7 = hal.tensor.import wait(%arg63) => %arg4 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%6}
    %8 = hal.buffer_view.dim<%arg5 : !hal.buffer_view>[2] : index
    %9 = hal.tensor.import wait(%arg63) => %arg5 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%8}
    %10 = hal.buffer_view.dim<%arg6 : !hal.buffer_view>[2] : index
    %11 = hal.tensor.import wait(%arg63) => %arg6 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%10}
    %12 = hal.buffer_view.dim<%arg7 : !hal.buffer_view>[2] : index
    %13 = hal.tensor.import wait(%arg63) => %arg7 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%12}
    %14 = hal.buffer_view.dim<%arg8 : !hal.buffer_view>[2] : index
    %15 = hal.tensor.import wait(%arg63) => %arg8 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%14}
    %16 = hal.buffer_view.dim<%arg9 : !hal.buffer_view>[2] : index
    %17 = hal.tensor.import wait(%arg63) => %arg9 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%16}
    %18 = hal.buffer_view.dim<%arg10 : !hal.buffer_view>[2] : index
    %19 = hal.tensor.import wait(%arg63) => %arg10 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%18}
    %20 = hal.buffer_view.dim<%arg11 : !hal.buffer_view>[2] : index
    %21 = hal.tensor.import wait(%arg63) => %arg11 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%20}
    %22 = hal.buffer_view.dim<%arg12 : !hal.buffer_view>[2] : index
    %23 = hal.tensor.import wait(%arg63) => %arg12 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%22}
    %24 = hal.buffer_view.dim<%arg13 : !hal.buffer_view>[2] : index
    %25 = hal.tensor.import wait(%arg63) => %arg13 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%24}
    %26 = hal.buffer_view.dim<%arg14 : !hal.buffer_view>[2] : index
    %27 = hal.tensor.import wait(%arg63) => %arg14 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%26}
    %28 = hal.buffer_view.dim<%arg15 : !hal.buffer_view>[2] : index
    %29 = hal.tensor.import wait(%arg63) => %arg15 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%28}
    %30 = hal.buffer_view.dim<%arg16 : !hal.buffer_view>[2] : index
    %31 = hal.tensor.import wait(%arg63) => %arg16 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%30}
    %32 = hal.buffer_view.dim<%arg17 : !hal.buffer_view>[2] : index
    %33 = hal.tensor.import wait(%arg63) => %arg17 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%32}
    %34 = hal.buffer_view.dim<%arg18 : !hal.buffer_view>[2] : index
    %35 = hal.tensor.import wait(%arg63) => %arg18 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%34}
    %36 = hal.buffer_view.dim<%arg19 : !hal.buffer_view>[2] : index
    %37 = hal.tensor.import wait(%arg63) => %arg19 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%36}
    %38 = hal.buffer_view.dim<%arg20 : !hal.buffer_view>[2] : index
    %39 = hal.tensor.import wait(%arg63) => %arg20 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%38}
    %40 = hal.buffer_view.dim<%arg21 : !hal.buffer_view>[2] : index
    %41 = hal.tensor.import wait(%arg63) => %arg21 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%40}
    %42 = hal.buffer_view.dim<%arg22 : !hal.buffer_view>[2] : index
    %43 = hal.tensor.import wait(%arg63) => %arg22 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%42}
    %44 = hal.buffer_view.dim<%arg23 : !hal.buffer_view>[2] : index
    %45 = hal.tensor.import wait(%arg63) => %arg23 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%44}
    %46 = hal.buffer_view.dim<%arg24 : !hal.buffer_view>[2] : index
    %47 = hal.tensor.import wait(%arg63) => %arg24 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%46}
    %48 = hal.buffer_view.dim<%arg25 : !hal.buffer_view>[2] : index
    %49 = hal.tensor.import wait(%arg63) => %arg25 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%48}
    %50 = hal.buffer_view.dim<%arg26 : !hal.buffer_view>[2] : index
    %51 = hal.tensor.import wait(%arg63) => %arg26 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%50}
    %52 = hal.buffer_view.dim<%arg27 : !hal.buffer_view>[2] : index
    %53 = hal.tensor.import wait(%arg63) => %arg27 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%52}
    %54 = hal.buffer_view.dim<%arg28 : !hal.buffer_view>[2] : index
    %55 = hal.tensor.import wait(%arg63) => %arg28 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%54}
    %56 = hal.buffer_view.dim<%arg29 : !hal.buffer_view>[2] : index
    %57 = hal.tensor.import wait(%arg63) => %arg29 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%56}
    %58 = hal.buffer_view.dim<%arg30 : !hal.buffer_view>[2] : index
    %59 = hal.tensor.import wait(%arg63) => %arg30 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%58}
    %60 = hal.buffer_view.dim<%arg31 : !hal.buffer_view>[2] : index
    %61 = hal.tensor.import wait(%arg63) => %arg31 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%60}
    %62 = hal.buffer_view.dim<%arg32 : !hal.buffer_view>[2] : index
    %63 = hal.tensor.import wait(%arg63) => %arg32 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%62}
    %64 = hal.buffer_view.dim<%arg33 : !hal.buffer_view>[2] : index
    %65 = hal.tensor.import wait(%arg63) => %arg33 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%64}
    %66 = hal.buffer_view.dim<%arg34 : !hal.buffer_view>[2] : index
    %67 = hal.tensor.import wait(%arg63) => %arg34 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%66}
    %68 = hal.buffer_view.dim<%arg35 : !hal.buffer_view>[2] : index
    %69 = hal.tensor.import wait(%arg63) => %arg35 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%68}
    %70 = hal.buffer_view.dim<%arg36 : !hal.buffer_view>[2] : index
    %71 = hal.tensor.import wait(%arg63) => %arg36 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%70}
    %72 = hal.buffer_view.dim<%arg37 : !hal.buffer_view>[2] : index
    %73 = hal.tensor.import wait(%arg63) => %arg37 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%72}
    %74 = hal.buffer_view.dim<%arg38 : !hal.buffer_view>[2] : index
    %75 = hal.tensor.import wait(%arg63) => %arg38 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%74}
    %76 = hal.buffer_view.dim<%arg39 : !hal.buffer_view>[2] : index
    %77 = hal.tensor.import wait(%arg63) => %arg39 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%76}
    %78 = hal.buffer_view.dim<%arg40 : !hal.buffer_view>[2] : index
    %79 = hal.tensor.import wait(%arg63) => %arg40 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%78}
    %80 = hal.buffer_view.dim<%arg41 : !hal.buffer_view>[2] : index
    %81 = hal.tensor.import wait(%arg63) => %arg41 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%80}
    %82 = hal.buffer_view.dim<%arg42 : !hal.buffer_view>[2] : index
    %83 = hal.tensor.import wait(%arg63) => %arg42 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%82}
    %84 = hal.buffer_view.dim<%arg43 : !hal.buffer_view>[2] : index
    %85 = hal.tensor.import wait(%arg63) => %arg43 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%84}
    %86 = hal.buffer_view.dim<%arg44 : !hal.buffer_view>[2] : index
    %87 = hal.tensor.import wait(%arg63) => %arg44 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%86}
    %88 = hal.buffer_view.dim<%arg45 : !hal.buffer_view>[2] : index
    %89 = hal.tensor.import wait(%arg63) => %arg45 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%88}
    %90 = hal.buffer_view.dim<%arg46 : !hal.buffer_view>[2] : index
    %91 = hal.tensor.import wait(%arg63) => %arg46 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%90}
    %92 = hal.buffer_view.dim<%arg47 : !hal.buffer_view>[2] : index
    %93 = hal.tensor.import wait(%arg63) => %arg47 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%92}
    %94 = hal.buffer_view.dim<%arg48 : !hal.buffer_view>[2] : index
    %95 = hal.tensor.import wait(%arg63) => %arg48 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%94}
    %96 = hal.buffer_view.dim<%arg49 : !hal.buffer_view>[2] : index
    %97 = hal.tensor.import wait(%arg63) => %arg49 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%96}
    %98 = hal.buffer_view.dim<%arg50 : !hal.buffer_view>[2] : index
    %99 = hal.tensor.import wait(%arg63) => %arg50 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%98}
    %100 = hal.buffer_view.dim<%arg51 : !hal.buffer_view>[2] : index
    %101 = hal.tensor.import wait(%arg63) => %arg51 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%100}
    %102 = hal.buffer_view.dim<%arg52 : !hal.buffer_view>[2] : index
    %103 = hal.tensor.import wait(%arg63) => %arg52 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%102}
    %104 = hal.buffer_view.dim<%arg53 : !hal.buffer_view>[2] : index
    %105 = hal.tensor.import wait(%arg63) => %arg53 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%104}
    %106 = hal.buffer_view.dim<%arg54 : !hal.buffer_view>[2] : index
    %107 = hal.tensor.import wait(%arg63) => %arg54 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%106}
    %108 = hal.buffer_view.dim<%arg55 : !hal.buffer_view>[2] : index
    %109 = hal.tensor.import wait(%arg63) => %arg55 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%108}
    %110 = hal.buffer_view.dim<%arg56 : !hal.buffer_view>[2] : index
    %111 = hal.tensor.import wait(%arg63) => %arg56 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%110}
    %112 = hal.buffer_view.dim<%arg57 : !hal.buffer_view>[2] : index
    %113 = hal.tensor.import wait(%arg63) => %arg57 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%112}
    %114 = hal.buffer_view.dim<%arg58 : !hal.buffer_view>[2] : index
    %115 = hal.tensor.import wait(%arg63) => %arg58 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%114}
    %116 = hal.buffer_view.dim<%arg59 : !hal.buffer_view>[2] : index
    %117 = hal.tensor.import wait(%arg63) => %arg59 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%116}
    %118 = hal.buffer_view.dim<%arg60 : !hal.buffer_view>[2] : index
    %119 = hal.tensor.import wait(%arg63) => %arg60 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%118}
    %120 = hal.buffer_view.dim<%arg61 : !hal.buffer_view>[2] : index
    %121 = hal.tensor.import wait(%arg63) => %arg61 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%120}
    %122 = hal.buffer_view.dim<%arg62 : !hal.buffer_view>[2] : index
    %123 = hal.tensor.import wait(%arg63) => %arg62 : !hal.buffer_view -> tensor<1x3x?x64xf32>{%122}
    %dim = tensor.dim %2, %c1 : tensor<1x?xi64>
    %124 = arith.index_cast %dim : index to i64
    %dim_9 = tensor.dim %5, %c2 : tensor<1x3x?x64xf32>
    %125 = arith.index_cast %dim_9 : index to i64
    %dim_10 = tensor.dim %7, %c2 : tensor<1x3x?x64xf32>
    %126 = arith.index_cast %dim_10 : index to i64
    %dim_11 = tensor.dim %9, %c2 : tensor<1x3x?x64xf32>
    %127 = arith.index_cast %dim_11 : index to i64
    %dim_12 = tensor.dim %11, %c2 : tensor<1x3x?x64xf32>
    %128 = arith.index_cast %dim_12 : index to i64
    %dim_13 = tensor.dim %13, %c2 : tensor<1x3x?x64xf32>
    %129 = arith.index_cast %dim_13 : index to i64
    %dim_14 = tensor.dim %15, %c2 : tensor<1x3x?x64xf32>
    %130 = arith.index_cast %dim_14 : index to i64
    %dim_15 = tensor.dim %17, %c2 : tensor<1x3x?x64xf32>
    %131 = arith.index_cast %dim_15 : index to i64
    %dim_16 = tensor.dim %19, %c2 : tensor<1x3x?x64xf32>
    %132 = arith.index_cast %dim_16 : index to i64
    %dim_17 = tensor.dim %21, %c2 : tensor<1x3x?x64xf32>
    %133 = arith.index_cast %dim_17 : index to i64
    %dim_18 = tensor.dim %23, %c2 : tensor<1x3x?x64xf32>
    %134 = arith.index_cast %dim_18 : index to i64
    %dim_19 = tensor.dim %25, %c2 : tensor<1x3x?x64xf32>
    %135 = arith.index_cast %dim_19 : index to i64
    %dim_20 = tensor.dim %27, %c2 : tensor<1x3x?x64xf32>
    %136 = arith.index_cast %dim_20 : index to i64
    %dim_21 = tensor.dim %29, %c2 : tensor<1x3x?x64xf32>
    %137 = arith.index_cast %dim_21 : index to i64
    %dim_22 = tensor.dim %31, %c2 : tensor<1x3x?x64xf32>
    %138 = arith.index_cast %dim_22 : index to i64
    %dim_23 = tensor.dim %33, %c2 : tensor<1x3x?x64xf32>
    %139 = arith.index_cast %dim_23 : index to i64
    %dim_24 = tensor.dim %35, %c2 : tensor<1x3x?x64xf32>
    %140 = arith.index_cast %dim_24 : index to i64
    %dim_25 = tensor.dim %37, %c2 : tensor<1x3x?x64xf32>
    %141 = arith.index_cast %dim_25 : index to i64
    %dim_26 = tensor.dim %39, %c2 : tensor<1x3x?x64xf32>
    %142 = arith.index_cast %dim_26 : index to i64
    %dim_27 = tensor.dim %41, %c2 : tensor<1x3x?x64xf32>
    %143 = arith.index_cast %dim_27 : index to i64
    %dim_28 = tensor.dim %43, %c2 : tensor<1x3x?x64xf32>
    %144 = arith.index_cast %dim_28 : index to i64
    %dim_29 = tensor.dim %45, %c2 : tensor<1x3x?x64xf32>
    %145 = arith.index_cast %dim_29 : index to i64
    %dim_30 = tensor.dim %47, %c2 : tensor<1x3x?x64xf32>
    %146 = arith.index_cast %dim_30 : index to i64
    %dim_31 = tensor.dim %49, %c2 : tensor<1x3x?x64xf32>
    %147 = arith.index_cast %dim_31 : index to i64
    %dim_32 = tensor.dim %51, %c2 : tensor<1x3x?x64xf32>
    %148 = arith.index_cast %dim_32 : index to i64
    %dim_33 = tensor.dim %53, %c2 : tensor<1x3x?x64xf32>
    %149 = arith.index_cast %dim_33 : index to i64
    %dim_34 = tensor.dim %55, %c2 : tensor<1x3x?x64xf32>
    %150 = arith.index_cast %dim_34 : index to i64
    %dim_35 = tensor.dim %57, %c2 : tensor<1x3x?x64xf32>
    %151 = arith.index_cast %dim_35 : index to i64
    %dim_36 = tensor.dim %59, %c2 : tensor<1x3x?x64xf32>
    %152 = arith.index_cast %dim_36 : index to i64
    %dim_37 = tensor.dim %61, %c2 : tensor<1x3x?x64xf32>
    %153 = arith.index_cast %dim_37 : index to i64
    %dim_38 = tensor.dim %63, %c2 : tensor<1x3x?x64xf32>
    %154 = arith.index_cast %dim_38 : index to i64
    %dim_39 = tensor.dim %65, %c2 : tensor<1x3x?x64xf32>
    %155 = arith.index_cast %dim_39 : index to i64
    %dim_40 = tensor.dim %67, %c2 : tensor<1x3x?x64xf32>
    %156 = arith.index_cast %dim_40 : index to i64
    %dim_41 = tensor.dim %69, %c2 : tensor<1x3x?x64xf32>
    %157 = arith.index_cast %dim_41 : index to i64
    %dim_42 = tensor.dim %71, %c2 : tensor<1x3x?x64xf32>
    %158 = arith.index_cast %dim_42 : index to i64
    %dim_43 = tensor.dim %73, %c2 : tensor<1x3x?x64xf32>
    %159 = arith.index_cast %dim_43 : index to i64
    %dim_44 = tensor.dim %75, %c2 : tensor<1x3x?x64xf32>
    %160 = arith.index_cast %dim_44 : index to i64
    %dim_45 = tensor.dim %77, %c2 : tensor<1x3x?x64xf32>
    %161 = arith.index_cast %dim_45 : index to i64
    %dim_46 = tensor.dim %79, %c2 : tensor<1x3x?x64xf32>
    %162 = arith.index_cast %dim_46 : index to i64
    %dim_47 = tensor.dim %81, %c2 : tensor<1x3x?x64xf32>
    %163 = arith.index_cast %dim_47 : index to i64
    %dim_48 = tensor.dim %83, %c2 : tensor<1x3x?x64xf32>
    %164 = arith.index_cast %dim_48 : index to i64
    %dim_49 = tensor.dim %85, %c2 : tensor<1x3x?x64xf32>
    %165 = arith.index_cast %dim_49 : index to i64
    %dim_50 = tensor.dim %87, %c2 : tensor<1x3x?x64xf32>
    %166 = arith.index_cast %dim_50 : index to i64
    %dim_51 = tensor.dim %89, %c2 : tensor<1x3x?x64xf32>
    %167 = arith.index_cast %dim_51 : index to i64
    %dim_52 = tensor.dim %91, %c2 : tensor<1x3x?x64xf32>
    %168 = arith.index_cast %dim_52 : index to i64
    %dim_53 = tensor.dim %93, %c2 : tensor<1x3x?x64xf32>
    %169 = arith.index_cast %dim_53 : index to i64
    %dim_54 = tensor.dim %95, %c2 : tensor<1x3x?x64xf32>
    %170 = arith.index_cast %dim_54 : index to i64
    %dim_55 = tensor.dim %97, %c2 : tensor<1x3x?x64xf32>
    %171 = arith.index_cast %dim_55 : index to i64
    %dim_56 = tensor.dim %99, %c2 : tensor<1x3x?x64xf32>
    %172 = arith.index_cast %dim_56 : index to i64
    %dim_57 = tensor.dim %101, %c2 : tensor<1x3x?x64xf32>
    %173 = arith.index_cast %dim_57 : index to i64
    %dim_58 = tensor.dim %103, %c2 : tensor<1x3x?x64xf32>
    %174 = arith.index_cast %dim_58 : index to i64
    %dim_59 = tensor.dim %105, %c2 : tensor<1x3x?x64xf32>
    %175 = arith.index_cast %dim_59 : index to i64
    %dim_60 = tensor.dim %107, %c2 : tensor<1x3x?x64xf32>
    %176 = arith.index_cast %dim_60 : index to i64
    %dim_61 = tensor.dim %109, %c2 : tensor<1x3x?x64xf32>
    %177 = arith.index_cast %dim_61 : index to i64
    %dim_62 = tensor.dim %111, %c2 : tensor<1x3x?x64xf32>
    %178 = arith.index_cast %dim_62 : index to i64
    %dim_63 = tensor.dim %113, %c2 : tensor<1x3x?x64xf32>
    %179 = arith.index_cast %dim_63 : index to i64
    %dim_64 = tensor.dim %115, %c2 : tensor<1x3x?x64xf32>
    %180 = arith.index_cast %dim_64 : index to i64
    %dim_65 = tensor.dim %117, %c2 : tensor<1x3x?x64xf32>
    %181 = arith.index_cast %dim_65 : index to i64
    %dim_66 = tensor.dim %119, %c2 : tensor<1x3x?x64xf32>
    %182 = arith.index_cast %dim_66 : index to i64
    %dim_67 = tensor.dim %121, %c2 : tensor<1x3x?x64xf32>
    %183 = arith.index_cast %dim_67 : index to i64
    %dim_68 = tensor.dim %123, %c2 : tensor<1x3x?x64xf32>
    %184 = arith.index_cast %dim_68 : index to i64
    %185 = arith.addi %125, %c1_i64 : i64
    %186 = arith.cmpi sle, %185, %124 : i64
    cf.assert %186, "Runtime assertion failed for expression s2 + 1 <= s64 on node 'le_33'"
    %187 = arith.cmpi eq, %126, %125 : i64
    cf.assert %187, "Runtime assertion failed for expression Eq(s29, s2) on node 'eq_2358'"
    %188 = arith.cmpi sle, %127, %125 : i64
    cf.assert %188, "Runtime assertion failed for expression s65 <= s2 on node 'le_34'"
    %189 = arith.cmpi eq, %128, %127 : i64
    cf.assert %189, "Runtime assertion failed for expression Eq(s87, s65) on node 'eq_2361'"
    %190 = arith.cmpi sle, %129, %125 : i64
    cf.assert %190, "Runtime assertion failed for expression s6 <= s2 on node 'le_35'"
    %191 = arith.cmpi eq, %130, %129 : i64
    cf.assert %191, "Runtime assertion failed for expression Eq(s13, s6) on node 'eq_2364'"
    %192 = arith.cmpi sle, %131, %125 : i64
    cf.assert %192, "Runtime assertion failed for expression s17 <= s2 on node 'le_36'"
    %193 = arith.cmpi eq, %132, %131 : i64
    cf.assert %193, "Runtime assertion failed for expression Eq(s14, s17) on node 'eq_2367'"
    %194 = arith.cmpi sle, %133, %125 : i64
    cf.assert %194, "Runtime assertion failed for expression s82 <= s2 on node 'le_37'"
    %195 = arith.cmpi eq, %134, %133 : i64
    cf.assert %195, "Runtime assertion failed for expression Eq(s69, s82) on node 'eq_2370'"
    %196 = arith.cmpi sle, %135, %125 : i64
    cf.assert %196, "Runtime assertion failed for expression s40 <= s2 on node 'le_38'"
    %197 = arith.cmpi eq, %136, %135 : i64
    cf.assert %197, "Runtime assertion failed for expression Eq(s5, s40) on node 'eq_2373'"
    %198 = arith.cmpi sle, %137, %125 : i64
    cf.assert %198, "Runtime assertion failed for expression s67 <= s2 on node 'le_39'"
    %199 = arith.cmpi eq, %138, %137 : i64
    cf.assert %199, "Runtime assertion failed for expression Eq(s46, s67) on node 'eq_2376'"
    %200 = arith.cmpi sle, %139, %125 : i64
    cf.assert %200, "Runtime assertion failed for expression s8 <= s2 on node 'le_40'"
    %201 = arith.cmpi eq, %140, %139 : i64
    cf.assert %201, "Runtime assertion failed for expression Eq(s25, s8) on node 'eq_2379'"
    %202 = arith.cmpi sle, %141, %125 : i64
    cf.assert %202, "Runtime assertion failed for expression s66 <= s2 on node 'le_41'"
    %203 = arith.cmpi eq, %142, %141 : i64
    cf.assert %203, "Runtime assertion failed for expression Eq(s81, s66) on node 'eq_2382'"
    %204 = arith.cmpi sle, %143, %125 : i64
    cf.assert %204, "Runtime assertion failed for expression s47 <= s2 on node 'le_42'"
    %205 = arith.cmpi eq, %144, %143 : i64
    cf.assert %205, "Runtime assertion failed for expression Eq(s62, s47) on node 'eq_2385'"
    %206 = arith.cmpi sle, %145, %125 : i64
    cf.assert %206, "Runtime assertion failed for expression s32 <= s2 on node 'le_43'"
    %207 = arith.cmpi eq, %146, %145 : i64
    cf.assert %207, "Runtime assertion failed for expression Eq(s7, s32) on node 'eq_2388'"
    %208 = arith.cmpi sle, %147, %125 : i64
    cf.assert %208, "Runtime assertion failed for expression s77 <= s2 on node 'le_44'"
    %209 = arith.cmpi eq, %148, %147 : i64
    cf.assert %209, "Runtime assertion failed for expression Eq(s15, s77) on node 'eq_2391'"
    %210 = arith.cmpi sle, %149, %125 : i64
    cf.assert %210, "Runtime assertion failed for expression s4 <= s2 on node 'le_45'"
    %211 = arith.cmpi eq, %150, %149 : i64
    cf.assert %211, "Runtime assertion failed for expression Eq(s9, s4) on node 'eq_2394'"
    %212 = arith.cmpi sle, %151, %125 : i64
    cf.assert %212, "Runtime assertion failed for expression s48 <= s2 on node 'le_46'"
    %213 = arith.cmpi eq, %152, %151 : i64
    cf.assert %213, "Runtime assertion failed for expression Eq(s26, s48) on node 'eq_2397'"
    %214 = arith.cmpi sle, %153, %125 : i64
    cf.assert %214, "Runtime assertion failed for expression s91 <= s2 on node 'le_47'"
    %215 = arith.cmpi eq, %154, %153 : i64
    cf.assert %215, "Runtime assertion failed for expression Eq(s93, s91) on node 'eq_2400'"
    %216 = arith.cmpi sle, %155, %125 : i64
    cf.assert %216, "Runtime assertion failed for expression s88 <= s2 on node 'le_48'"
    %217 = arith.cmpi eq, %156, %155 : i64
    cf.assert %217, "Runtime assertion failed for expression Eq(s99, s88) on node 'eq_2403'"
    %218 = arith.cmpi sle, %157, %125 : i64
    cf.assert %218, "Runtime assertion failed for expression s16 <= s2 on node 'le_49'"
    %219 = arith.cmpi eq, %158, %157 : i64
    cf.assert %219, "Runtime assertion failed for expression Eq(s43, s16) on node 'eq_2406'"
    %220 = arith.cmpi sle, %159, %125 : i64
    cf.assert %220, "Runtime assertion failed for expression s89 <= s2 on node 'le_50'"
    %221 = arith.cmpi eq, %160, %159 : i64
    cf.assert %221, "Runtime assertion failed for expression Eq(s78, s89) on node 'eq_2409'"
    %222 = arith.cmpi sle, %161, %125 : i64
    cf.assert %222, "Runtime assertion failed for expression s79 <= s2 on node 'le_51'"
    %223 = arith.cmpi eq, %162, %161 : i64
    cf.assert %223, "Runtime assertion failed for expression Eq(s10, s79) on node 'eq_2412'"
    %224 = arith.cmpi sle, %163, %125 : i64
    cf.assert %224, "Runtime assertion failed for expression s92 <= s2 on node 'le_52'"
    %225 = arith.cmpi eq, %164, %163 : i64
    cf.assert %225, "Runtime assertion failed for expression Eq(s45, s92) on node 'eq_2415'"
    %226 = arith.cmpi sle, %165, %125 : i64
    cf.assert %226, "Runtime assertion failed for expression s98 <= s2 on node 'le_53'"
    %227 = arith.cmpi eq, %166, %165 : i64
    cf.assert %227, "Runtime assertion failed for expression Eq(s80, s98) on node 'eq_2418'"
    %228 = arith.cmpi sle, %167, %125 : i64
    cf.assert %228, "Runtime assertion failed for expression s55 <= s2 on node 'le_54'"
    %229 = arith.cmpi eq, %168, %167 : i64
    cf.assert %229, "Runtime assertion failed for expression Eq(s68, s55) on node 'eq_2421'"
    %230 = arith.cmpi sle, %169, %125 : i64
    cf.assert %230, "Runtime assertion failed for expression s18 <= s2 on node 'le_55'"
    %231 = arith.cmpi eq, %170, %169 : i64
    cf.assert %231, "Runtime assertion failed for expression Eq(s41, s18) on node 'eq_2424'"
    %232 = arith.cmpi sle, %171, %125 : i64
    cf.assert %232, "Runtime assertion failed for expression s19 <= s2 on node 'le_56'"
    %233 = arith.cmpi eq, %172, %171 : i64
    cf.assert %233, "Runtime assertion failed for expression Eq(s95, s19) on node 'eq_2427'"
    %234 = arith.cmpi sle, %173, %125 : i64
    cf.assert %234, "Runtime assertion failed for expression s20 <= s2 on node 'le_57'"
    %235 = arith.cmpi eq, %174, %173 : i64
    cf.assert %235, "Runtime assertion failed for expression Eq(s74, s20) on node 'eq_2430'"
    %236 = arith.cmpi sle, %175, %125 : i64
    cf.assert %236, "Runtime assertion failed for expression s83 <= s2 on node 'le_58'"
    %237 = arith.cmpi eq, %176, %175 : i64
    cf.assert %237, "Runtime assertion failed for expression Eq(s84, s83) on node 'eq_2433'"
    %238 = arith.cmpi sle, %177, %125 : i64
    cf.assert %238, "Runtime assertion failed for expression s21 <= s2 on node 'le_59'"
    %239 = arith.cmpi eq, %178, %177 : i64
    cf.assert %239, "Runtime assertion failed for expression Eq(s70, s21) on node 'eq_2436'"
    %240 = arith.cmpi sle, %179, %125 : i64
    cf.assert %240, "Runtime assertion failed for expression s71 <= s2 on node 'le_60'"
    %241 = arith.cmpi eq, %180, %179 : i64
    cf.assert %241, "Runtime assertion failed for expression Eq(s94, s71) on node 'eq_2439'"
    %242 = arith.cmpi sle, %181, %125 : i64
    cf.assert %242, "Runtime assertion failed for expression s85 <= s2 on node 'le_61'"
    %243 = arith.cmpi eq, %182, %181 : i64
    cf.assert %243, "Runtime assertion failed for expression Eq(s34, s85) on node 'eq_2442'"
    %244 = arith.cmpi sle, %183, %125 : i64
    cf.assert %244, "Runtime assertion failed for expression s73 <= s2 on node 'le_62'"
    %245 = arith.cmpi eq, %184, %183 : i64
    cf.assert %245, "Runtime assertion failed for expression Eq(s28, s73) on node 'eq_2445'"
    %__auto.model.lm_head.weight = util.global.load @__auto.model.lm_head.weight : tensor<49152x576xf32>
    %246 = tensor.empty() : tensor<1x1x576xf32>
    %247 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<1x1xi64>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: i64, %out: f32):
      %1997 = arith.index_cast %in : i64 to index
      %1998 = linalg.index 2 : index
      %1999 = arith.cmpi slt, %1997, %c49152 : index
      cf.assert %1999, "index must be smaller than dim size"
      %2000 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %2000, "index must be larger or equal to 0"
      %extracted = tensor.extract %__auto.model.lm_head.weight[%1997, %1998] : tensor<49152x576xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x1x576xf32>
    %248 = tensor.empty() : tensor<1xi64>
    %249 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%248 : tensor<1xi64>) {
    ^bb0(%out: i64):
      linalg.yield %162 : i64
    } -> tensor<1xi64>
    %250 = tensor.empty(%dim) : tensor<1x?xi1>
    %251 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<1x?xi64>) outs(%250 : tensor<1x?xi1>) {
    ^bb0(%in: i64, %out: i1):
      %1997 = arith.cmpi ne, %in, %c0_i64 : i64
      linalg.yield %1997 : i1
    } -> tensor<1x?xi1>
    %252 = tensor.empty(%dim) : tensor<?xi64>
    %253 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%252 : tensor<?xi64>) {
    ^bb0(%out: i64):
      %1997 = linalg.index 0 : index
      %1998 = arith.index_cast %1997 : index to i64
      linalg.yield %1998 : i64
    } -> tensor<?xi64>
    %254 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%248 : tensor<1xi64>) {
    ^bb0(%out: i64):
      linalg.yield %c0_i64 : i64
    } -> tensor<1xi64>
    %expanded = tensor.expand_shape %249 [[0, 1]] output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>
    %255 = linalg.generic {indexing_maps = [#map4, #map5, #map3], iterator_types = ["parallel", "parallel"]} ins(%253, %expanded : tensor<?xi64>, tensor<1x1xi64>) outs(%250 : tensor<1x?xi1>) {
    ^bb0(%in: i64, %in_1308: i64, %out: i1):
      %1997 = arith.cmpi sle, %in, %in_1308 : i64
      linalg.yield %1997 : i1
    } -> tensor<1x?xi1>
    %256 = linalg.generic {indexing_maps = [#map5, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%cst_3, %255 : tensor<1x1xi1>, tensor<1x?xi1>) outs(%250 : tensor<1x?xi1>) {
    ^bb0(%in: i1, %in_1308: i1, %out: i1):
      %1997 = arith.andi %in, %in_1308 : i1
      linalg.yield %1997 : i1
    } -> tensor<1x?xi1>
    %expanded_69 = tensor.expand_shape %254 [[0, 1]] output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>
    %257 = linalg.generic {indexing_maps = [#map6, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%expanded_69, %253 : tensor<1x1xi64>, tensor<?xi64>) outs(%250 : tensor<1x?xi1>) {
    ^bb0(%in: i64, %in_1308: i64, %out: i1):
      %1997 = arith.cmpi slt, %in, %c0_i64 : i64
      %1998 = arith.addi %in, %c1_i64 : i64
      %1999 = arith.select %1997, %1998, %in : i64
      %2000 = arith.index_cast %1999 : i64 to index
      %2001 = arith.cmpi slt, %in_1308, %c0_i64 : i64
      %2002 = arith.addi %in_1308, %124 : i64
      %2003 = arith.select %2001, %2002, %in_1308 : i64
      %2004 = arith.index_cast %2003 : i64 to index
      %extracted = tensor.extract %251[%2000, %2004] : tensor<1x?xi1>
      linalg.yield %extracted : i1
    } -> tensor<1x?xi1>
    %expanded_70 = tensor.expand_shape %257 [[0], [1, 2]] output_shape [1, 1, %dim] : tensor<1x?xi1> into tensor<1x1x?xi1>
    %258 = tensor.empty(%dim) : tensor<1x1x?xi1>
    %259 = linalg.generic {indexing_maps = [#map7, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%256, %expanded_70 : tensor<1x?xi1>, tensor<1x1x?xi1>) outs(%258 : tensor<1x1x?xi1>) {
    ^bb0(%in: i1, %in_1308: i1, %out: i1):
      %1997 = arith.andi %in, %in_1308 : i1
      linalg.yield %1997 : i1
    } -> tensor<1x1x?xi1>
    %expanded_71 = tensor.expand_shape %259 [[0], [1], [2, 3]] output_shape [1, 1, 1, %dim] : tensor<1x1x?xi1> into tensor<1x1x1x?xi1>
    %260 = tensor.empty(%dim) : tensor<1x1x1x?xf32>
    %261 = linalg.generic {indexing_maps = [#map8, #map9, #map9, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_71, %cst_6, %cst_2 : tensor<1x1x1x?xi1>, tensor<f32>, tensor<f32>) outs(%260 : tensor<1x1x1x?xf32>) {
    ^bb0(%in: i1, %in_1308: f32, %in_1309: f32, %out: f32):
      %1997 = arith.select %in, %in_1308, %in_1309 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1x?xf32>
    %expanded_72 = tensor.expand_shape %cst_5 [[0, 1, 2]] output_shape [1, 32, 1] : tensor<32xf32> into tensor<1x32x1xf32>
    %expanded_73 = tensor.expand_shape %3 [[0], [1, 2]] output_shape [1, 1, 1] : tensor<1x1xi64> into tensor<1x1x1xi64>
    %262 = tensor.empty() : tensor<1x1x1xf32>
    %263 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_73 : tensor<1x1x1xi64>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: i64, %out: f32):
      %1997 = arith.sitofp %in : i64 to f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %264 = tensor.empty() : tensor<1x32x1xf32>
    %265 = linalg.fill ins(%cst : f32) outs(%264 : tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %266 = linalg.batch_matmul ins(%expanded_72, %263 : tensor<1x32x1xf32>, tensor<1x1x1xf32>) outs(%265 : tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %267 = tensor.empty() : tensor<1x1x32xf32>
    %transposed = linalg.transpose ins(%266 : tensor<1x32x1xf32>) outs(%267 : tensor<1x1x32xf32>) permutation = [0, 2, 1] 
    %concat = tensor.concat dim(2) %transposed, %transposed : (tensor<1x1x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x64xf32>
    %268 = tensor.empty() : tensor<1x1x64xf32>
    %269 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%concat : tensor<1x1x64xf32>) outs(%268 : tensor<1x1x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.cos %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x64xf32>
    %270 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%concat : tensor<1x1x64xf32>) outs(%268 : tensor<1x1x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.sin %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x64xf32>
    %271 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%247 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %272 = linalg.fill ins(%cst : f32) outs(%262 : tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %273 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%271 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %274 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%273 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %275 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%274 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %276 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%275 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %277 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%247, %276 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.0.input_layernorm.weight = util.global.load @__auto.model.model.layers.0.input_layernorm.weight : tensor<576xf32>
    %278 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.0.input_layernorm.weight, %277 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.0.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.0.self_attn.q_proj.weight : tensor<576x576xf32>
    %279 = tensor.empty() : tensor<576x576xf32>
    %transposed_74 = linalg.transpose ins(%__auto.model.model.layers.0.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed = tensor.collapse_shape %278 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %280 = tensor.empty() : tensor<1x576xf32>
    %281 = linalg.fill ins(%cst : f32) outs(%280 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %282 = linalg.matmul ins(%collapsed, %transposed_74 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_75 = tensor.expand_shape %282 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %283 = tensor.empty() : tensor<1x9x1x64xf32>
    %transposed_76 = linalg.transpose ins(%expanded_75 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.0.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.0.self_attn.k_proj.weight : tensor<192x576xf32>
    %284 = tensor.empty() : tensor<576x192xf32>
    %transposed_77 = linalg.transpose ins(%__auto.model.model.layers.0.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %285 = tensor.empty() : tensor<1x192xf32>
    %286 = linalg.fill ins(%cst : f32) outs(%285 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %287 = linalg.matmul ins(%collapsed, %transposed_77 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_78 = tensor.expand_shape %287 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %288 = tensor.empty() : tensor<1x3x1x64xf32>
    %transposed_79 = linalg.transpose ins(%expanded_78 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.0.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.0.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_80 = linalg.transpose ins(%__auto.model.model.layers.0.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %289 = linalg.matmul ins(%collapsed, %transposed_80 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_81 = tensor.expand_shape %289 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_82 = linalg.transpose ins(%expanded_81 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_83 = tensor.expand_shape %269 [[0], [1, 2], [3]] output_shape [1, 1, 1, 64] : tensor<1x1x64xf32> into tensor<1x1x1x64xf32>
    %expanded_84 = tensor.expand_shape %270 [[0], [1, 2], [3]] output_shape [1, 1, 1, 64] : tensor<1x1x64xf32> into tensor<1x1x1x64xf32>
    %290 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_76, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice = tensor.extract_slice %transposed_76[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_85 = tensor.extract_slice %transposed_76[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %291 = tensor.empty() : tensor<1x9x1x32xf32>
    %292 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_85 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_86 = tensor.concat dim(3) %292, %extracted_slice : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %293 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_86, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %294 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%290, %293 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %295 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_79, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_87 = tensor.extract_slice %transposed_79[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_88 = tensor.extract_slice %transposed_79[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %296 = tensor.empty() : tensor<1x3x1x32xf32>
    %297 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_88 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_89 = tensor.concat dim(3) %297, %extracted_slice_87 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %298 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_89, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %299 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%295, %298 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_90 = tensor.concat dim(2) %5, %299 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_91 = tensor.concat dim(2) %7, %transposed_82 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %300 = tensor.empty(%dim) : tensor<1x3x3x?x64xf32>
    %301 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_90 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_92 = tensor.collapse_shape %301 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %302 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_91 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %303 = affine.apply #map15()[%dim_9]
    %304 = tensor.empty(%303) : tensor<1x9x64x?xf32>
    %transposed_93 = linalg.transpose ins(%collapsed_92 : tensor<1x9x?x64xf32>) outs(%304 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_94 = tensor.collapse_shape %294 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_95 = tensor.collapse_shape %transposed_93 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %305 = tensor.empty(%303) : tensor<9x1x?xf32>
    %306 = linalg.fill ins(%cst : f32) outs(%305 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %307 = linalg.batch_matmul ins(%collapsed_94, %collapsed_95 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%306 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_96 = tensor.expand_shape %307 [[0, 1], [2], [3]] output_shape [1, 9, 1, %303] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %308 = tensor.empty(%303) : tensor<1x9x1x?xf32>
    %309 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_96 : tensor<1x9x1x?xf32>) outs(%308 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %310 = arith.cmpi slt, %dim, %c0 : index
    %311 = arith.addi %dim, %dim : index
    %312 = arith.select %310, %311, %dim : index
    %313 = arith.cmpi slt, %312, %c0 : index
    %314 = arith.select %313, %c-1, %312 : index
    %315 = arith.minsi %314, %dim : index
    %316 = arith.maxsi %315, %c0 : index
    %extracted_slice_97 = tensor.extract_slice %261[0, 0, 0, 0] [1, 1, 1, %316] [1, 1, 1, 1] : tensor<1x1x1x?xf32> to tensor<1x1x1x?xf32>
    %317 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%309, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%308 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %318 = tensor.empty() : tensor<1x9x1xi64>
    %319 = linalg.fill ins(%c0_i64 : i64) outs(%318 : tensor<1x9x1xi64>) -> tensor<1x9x1xi64>
    %320 = tensor.empty() : tensor<1x9x1xf32>
    %321 = linalg.fill ins(%cst_0 : f32) outs(%320 : tensor<1x9x1xf32>) -> tensor<1x9x1xf32>
    %322:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%317 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_98 = tensor.expand_shape %322#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %323 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%317, %expanded_98 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%308 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %324 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%323 : tensor<1x9x1x?xf32>) outs(%308 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %325 = tensor.empty() : tensor<1x9x1x1xf32>
    %326 = linalg.fill ins(%cst : f32) outs(%325 : tensor<1x9x1x1xf32>) -> tensor<1x9x1x1xf32>
    %327 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%324 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %328 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%324, %327 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%308 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_99 = tensor.collapse_shape %328 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_100 = tensor.collapse_shape %302 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %329 = affine.apply #map15()[%dim_10]
    %330 = arith.index_cast %303 : index to i64
    %331 = arith.index_cast %329 : index to i64
    %332 = arith.cmpi eq, %330, %331 : i64
    cf.assert %332, "mismatching contracting dimension"
    %333 = tensor.empty() : tensor<9x1x64xf32>
    %334 = linalg.fill ins(%cst : f32) outs(%333 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %335 = linalg.batch_matmul ins(%collapsed_99, %collapsed_100 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_101 = tensor.expand_shape %335 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %336 = tensor.empty() : tensor<1x1x9x64xf32>
    %transposed_102 = linalg.transpose ins(%expanded_101 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.0.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.0.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_103 = linalg.transpose ins(%__auto.model.model.layers.0.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_104 = tensor.collapse_shape %transposed_102 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %337 = linalg.matmul ins(%collapsed_104, %transposed_103 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_105 = tensor.expand_shape %337 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %338 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%247, %expanded_105 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %339 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%338 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %340 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%339 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %341 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%340 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %342 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%341 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %343 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%342 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %344 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%338, %343 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.0.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.0.post_attention_layernorm.weight : tensor<576xf32>
    %345 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.0.post_attention_layernorm.weight, %344 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.0.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.0.mlp.gate_proj.weight : tensor<1536x576xf32>
    %346 = tensor.empty() : tensor<576x1536xf32>
    %transposed_106 = linalg.transpose ins(%__auto.model.model.layers.0.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_107 = tensor.collapse_shape %345 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %347 = tensor.empty() : tensor<1x1536xf32>
    %348 = linalg.fill ins(%cst : f32) outs(%347 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %349 = linalg.matmul ins(%collapsed_107, %transposed_106 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_108 = tensor.expand_shape %349 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %350 = tensor.empty() : tensor<1x1x1536xf32>
    %351 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_108 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %352 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_108, %351 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.0.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.0.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_109 = linalg.transpose ins(%__auto.model.model.layers.0.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %353 = linalg.matmul ins(%collapsed_107, %transposed_109 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_110 = tensor.expand_shape %353 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %354 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%352, %expanded_110 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.0.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.0.mlp.down_proj.weight : tensor<576x1536xf32>
    %355 = tensor.empty() : tensor<1536x576xf32>
    %transposed_111 = linalg.transpose ins(%__auto.model.model.layers.0.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_112 = tensor.collapse_shape %354 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %356 = linalg.matmul ins(%collapsed_112, %transposed_111 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_113 = tensor.expand_shape %356 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %357 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%338, %expanded_113 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %358 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%357 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %359 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%358 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %360 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%359 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %361 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%360 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %362 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%361 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %363 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%357, %362 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.1.input_layernorm.weight = util.global.load @__auto.model.model.layers.1.input_layernorm.weight : tensor<576xf32>
    %364 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.1.input_layernorm.weight, %363 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.1.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.1.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_114 = linalg.transpose ins(%__auto.model.model.layers.1.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_115 = tensor.collapse_shape %364 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %365 = linalg.matmul ins(%collapsed_115, %transposed_114 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_116 = tensor.expand_shape %365 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_117 = linalg.transpose ins(%expanded_116 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.1.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.1.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_118 = linalg.transpose ins(%__auto.model.model.layers.1.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %366 = linalg.matmul ins(%collapsed_115, %transposed_118 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_119 = tensor.expand_shape %366 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_120 = linalg.transpose ins(%expanded_119 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.1.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.1.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_121 = linalg.transpose ins(%__auto.model.model.layers.1.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %367 = linalg.matmul ins(%collapsed_115, %transposed_121 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_122 = tensor.expand_shape %367 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_123 = linalg.transpose ins(%expanded_122 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %368 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_117, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_124 = tensor.extract_slice %transposed_117[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_125 = tensor.extract_slice %transposed_117[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %369 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_125 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_126 = tensor.concat dim(3) %369, %extracted_slice_124 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %370 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_126, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %371 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%368, %370 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %372 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_120, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_127 = tensor.extract_slice %transposed_120[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_128 = tensor.extract_slice %transposed_120[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %373 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_128 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_129 = tensor.concat dim(3) %373, %extracted_slice_127 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %374 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_129, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %375 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%372, %374 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_130 = tensor.concat dim(2) %9, %375 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_131 = tensor.concat dim(2) %11, %transposed_123 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %376 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_130 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_132 = tensor.collapse_shape %376 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %377 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_131 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %378 = affine.apply #map15()[%dim_11]
    %379 = tensor.empty(%378) : tensor<1x9x64x?xf32>
    %transposed_133 = linalg.transpose ins(%collapsed_132 : tensor<1x9x?x64xf32>) outs(%379 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_134 = tensor.collapse_shape %371 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_135 = tensor.collapse_shape %transposed_133 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %380 = tensor.empty(%378) : tensor<9x1x?xf32>
    %381 = linalg.fill ins(%cst : f32) outs(%380 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %382 = linalg.batch_matmul ins(%collapsed_134, %collapsed_135 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%381 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_136 = tensor.expand_shape %382 [[0, 1], [2], [3]] output_shape [1, 9, 1, %378] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %383 = tensor.empty(%378) : tensor<1x9x1x?xf32>
    %384 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_136 : tensor<1x9x1x?xf32>) outs(%383 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %385 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%384, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%383 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %386:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%385 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_137 = tensor.expand_shape %386#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %387 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%385, %expanded_137 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%383 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %388 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%387 : tensor<1x9x1x?xf32>) outs(%383 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %389 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%388 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %390 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%388, %389 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%383 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_138 = tensor.collapse_shape %390 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_139 = tensor.collapse_shape %377 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %391 = affine.apply #map15()[%dim_12]
    %392 = arith.index_cast %378 : index to i64
    %393 = arith.index_cast %391 : index to i64
    %394 = arith.cmpi eq, %392, %393 : i64
    cf.assert %394, "mismatching contracting dimension"
    %395 = linalg.batch_matmul ins(%collapsed_138, %collapsed_139 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_140 = tensor.expand_shape %395 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_141 = linalg.transpose ins(%expanded_140 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.1.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.1.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_142 = linalg.transpose ins(%__auto.model.model.layers.1.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_143 = tensor.collapse_shape %transposed_141 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %396 = linalg.matmul ins(%collapsed_143, %transposed_142 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_144 = tensor.expand_shape %396 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %397 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%357, %expanded_144 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %398 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%397 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %399 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%398 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %400 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%399 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %401 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%400 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %402 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%401 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %403 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%397, %402 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.1.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.1.post_attention_layernorm.weight : tensor<576xf32>
    %404 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.1.post_attention_layernorm.weight, %403 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.1.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.1.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_145 = linalg.transpose ins(%__auto.model.model.layers.1.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_146 = tensor.collapse_shape %404 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %405 = linalg.matmul ins(%collapsed_146, %transposed_145 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_147 = tensor.expand_shape %405 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %406 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_147 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %407 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_147, %406 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.1.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.1.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_148 = linalg.transpose ins(%__auto.model.model.layers.1.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %408 = linalg.matmul ins(%collapsed_146, %transposed_148 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_149 = tensor.expand_shape %408 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %409 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%407, %expanded_149 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.1.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.1.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_150 = linalg.transpose ins(%__auto.model.model.layers.1.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_151 = tensor.collapse_shape %409 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %410 = linalg.matmul ins(%collapsed_151, %transposed_150 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_152 = tensor.expand_shape %410 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %411 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%397, %expanded_152 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %412 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%411 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %413 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%412 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %414 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%413 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %415 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%414 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %416 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%415 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %417 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%411, %416 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.2.input_layernorm.weight = util.global.load @__auto.model.model.layers.2.input_layernorm.weight : tensor<576xf32>
    %418 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.2.input_layernorm.weight, %417 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.2.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.2.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_153 = linalg.transpose ins(%__auto.model.model.layers.2.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_154 = tensor.collapse_shape %418 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %419 = linalg.matmul ins(%collapsed_154, %transposed_153 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_155 = tensor.expand_shape %419 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_156 = linalg.transpose ins(%expanded_155 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.2.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.2.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_157 = linalg.transpose ins(%__auto.model.model.layers.2.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %420 = linalg.matmul ins(%collapsed_154, %transposed_157 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_158 = tensor.expand_shape %420 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_159 = linalg.transpose ins(%expanded_158 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.2.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.2.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_160 = linalg.transpose ins(%__auto.model.model.layers.2.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %421 = linalg.matmul ins(%collapsed_154, %transposed_160 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_161 = tensor.expand_shape %421 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_162 = linalg.transpose ins(%expanded_161 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %422 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_156, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_163 = tensor.extract_slice %transposed_156[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_164 = tensor.extract_slice %transposed_156[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %423 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_164 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_165 = tensor.concat dim(3) %423, %extracted_slice_163 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %424 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_165, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %425 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%422, %424 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %426 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_159, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_166 = tensor.extract_slice %transposed_159[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_167 = tensor.extract_slice %transposed_159[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %427 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_167 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_168 = tensor.concat dim(3) %427, %extracted_slice_166 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %428 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_168, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %429 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%426, %428 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_169 = tensor.concat dim(2) %13, %429 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_170 = tensor.concat dim(2) %15, %transposed_162 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %430 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_169 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_171 = tensor.collapse_shape %430 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %431 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_170 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %432 = affine.apply #map15()[%dim_13]
    %433 = tensor.empty(%432) : tensor<1x9x64x?xf32>
    %transposed_172 = linalg.transpose ins(%collapsed_171 : tensor<1x9x?x64xf32>) outs(%433 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_173 = tensor.collapse_shape %425 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_174 = tensor.collapse_shape %transposed_172 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %434 = tensor.empty(%432) : tensor<9x1x?xf32>
    %435 = linalg.fill ins(%cst : f32) outs(%434 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %436 = linalg.batch_matmul ins(%collapsed_173, %collapsed_174 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%435 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_175 = tensor.expand_shape %436 [[0, 1], [2], [3]] output_shape [1, 9, 1, %432] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %437 = tensor.empty(%432) : tensor<1x9x1x?xf32>
    %438 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_175 : tensor<1x9x1x?xf32>) outs(%437 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %439 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%438, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%437 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %440:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%439 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_176 = tensor.expand_shape %440#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %441 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%439, %expanded_176 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%437 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %442 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%441 : tensor<1x9x1x?xf32>) outs(%437 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %443 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%442 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %444 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%442, %443 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%437 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_177 = tensor.collapse_shape %444 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_178 = tensor.collapse_shape %431 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %445 = affine.apply #map15()[%dim_14]
    %446 = arith.index_cast %432 : index to i64
    %447 = arith.index_cast %445 : index to i64
    %448 = arith.cmpi eq, %446, %447 : i64
    cf.assert %448, "mismatching contracting dimension"
    %449 = linalg.batch_matmul ins(%collapsed_177, %collapsed_178 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_179 = tensor.expand_shape %449 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_180 = linalg.transpose ins(%expanded_179 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.2.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.2.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_181 = linalg.transpose ins(%__auto.model.model.layers.2.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_182 = tensor.collapse_shape %transposed_180 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %450 = linalg.matmul ins(%collapsed_182, %transposed_181 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_183 = tensor.expand_shape %450 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %451 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%411, %expanded_183 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %452 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%451 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %453 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%452 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %454 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%453 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %455 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%454 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %456 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%455 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %457 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%451, %456 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.2.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.2.post_attention_layernorm.weight : tensor<576xf32>
    %458 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.2.post_attention_layernorm.weight, %457 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.2.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.2.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_184 = linalg.transpose ins(%__auto.model.model.layers.2.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_185 = tensor.collapse_shape %458 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %459 = linalg.matmul ins(%collapsed_185, %transposed_184 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_186 = tensor.expand_shape %459 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %460 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_186 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %461 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_186, %460 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.2.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.2.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_187 = linalg.transpose ins(%__auto.model.model.layers.2.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %462 = linalg.matmul ins(%collapsed_185, %transposed_187 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_188 = tensor.expand_shape %462 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %463 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%461, %expanded_188 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.2.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.2.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_189 = linalg.transpose ins(%__auto.model.model.layers.2.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_190 = tensor.collapse_shape %463 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %464 = linalg.matmul ins(%collapsed_190, %transposed_189 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_191 = tensor.expand_shape %464 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %465 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%451, %expanded_191 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %466 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%465 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %467 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%466 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %468 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%467 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %469 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%468 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %470 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%469 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %471 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%465, %470 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.3.input_layernorm.weight = util.global.load @__auto.model.model.layers.3.input_layernorm.weight : tensor<576xf32>
    %472 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.3.input_layernorm.weight, %471 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.3.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.3.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_192 = linalg.transpose ins(%__auto.model.model.layers.3.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_193 = tensor.collapse_shape %472 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %473 = linalg.matmul ins(%collapsed_193, %transposed_192 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_194 = tensor.expand_shape %473 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_195 = linalg.transpose ins(%expanded_194 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.3.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.3.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_196 = linalg.transpose ins(%__auto.model.model.layers.3.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %474 = linalg.matmul ins(%collapsed_193, %transposed_196 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_197 = tensor.expand_shape %474 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_198 = linalg.transpose ins(%expanded_197 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.3.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.3.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_199 = linalg.transpose ins(%__auto.model.model.layers.3.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %475 = linalg.matmul ins(%collapsed_193, %transposed_199 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_200 = tensor.expand_shape %475 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_201 = linalg.transpose ins(%expanded_200 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %476 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_195, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_202 = tensor.extract_slice %transposed_195[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_203 = tensor.extract_slice %transposed_195[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %477 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_203 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_204 = tensor.concat dim(3) %477, %extracted_slice_202 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %478 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_204, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %479 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%476, %478 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %480 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_198, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_205 = tensor.extract_slice %transposed_198[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_206 = tensor.extract_slice %transposed_198[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %481 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_206 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_207 = tensor.concat dim(3) %481, %extracted_slice_205 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %482 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_207, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %483 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%480, %482 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_208 = tensor.concat dim(2) %17, %483 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_209 = tensor.concat dim(2) %19, %transposed_201 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %484 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_208 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_210 = tensor.collapse_shape %484 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %485 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_209 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %486 = affine.apply #map15()[%dim_15]
    %487 = tensor.empty(%486) : tensor<1x9x64x?xf32>
    %transposed_211 = linalg.transpose ins(%collapsed_210 : tensor<1x9x?x64xf32>) outs(%487 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_212 = tensor.collapse_shape %479 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_213 = tensor.collapse_shape %transposed_211 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %488 = tensor.empty(%486) : tensor<9x1x?xf32>
    %489 = linalg.fill ins(%cst : f32) outs(%488 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %490 = linalg.batch_matmul ins(%collapsed_212, %collapsed_213 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%489 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_214 = tensor.expand_shape %490 [[0, 1], [2], [3]] output_shape [1, 9, 1, %486] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %491 = tensor.empty(%486) : tensor<1x9x1x?xf32>
    %492 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_214 : tensor<1x9x1x?xf32>) outs(%491 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %493 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%492, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%491 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %494:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%493 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_215 = tensor.expand_shape %494#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %495 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%493, %expanded_215 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%491 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %496 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%495 : tensor<1x9x1x?xf32>) outs(%491 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %497 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%496 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %498 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%496, %497 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%491 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_216 = tensor.collapse_shape %498 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_217 = tensor.collapse_shape %485 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %499 = affine.apply #map15()[%dim_16]
    %500 = arith.index_cast %486 : index to i64
    %501 = arith.index_cast %499 : index to i64
    %502 = arith.cmpi eq, %500, %501 : i64
    cf.assert %502, "mismatching contracting dimension"
    %503 = linalg.batch_matmul ins(%collapsed_216, %collapsed_217 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_218 = tensor.expand_shape %503 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_219 = linalg.transpose ins(%expanded_218 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.3.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.3.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_220 = linalg.transpose ins(%__auto.model.model.layers.3.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_221 = tensor.collapse_shape %transposed_219 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %504 = linalg.matmul ins(%collapsed_221, %transposed_220 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_222 = tensor.expand_shape %504 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %505 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%465, %expanded_222 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %506 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%505 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %507 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%506 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %508 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%507 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %509 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%508 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %510 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%509 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %511 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%505, %510 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.3.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.3.post_attention_layernorm.weight : tensor<576xf32>
    %512 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.3.post_attention_layernorm.weight, %511 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.3.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.3.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_223 = linalg.transpose ins(%__auto.model.model.layers.3.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_224 = tensor.collapse_shape %512 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %513 = linalg.matmul ins(%collapsed_224, %transposed_223 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_225 = tensor.expand_shape %513 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %514 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_225 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %515 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_225, %514 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.3.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.3.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_226 = linalg.transpose ins(%__auto.model.model.layers.3.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %516 = linalg.matmul ins(%collapsed_224, %transposed_226 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_227 = tensor.expand_shape %516 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %517 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%515, %expanded_227 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.3.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.3.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_228 = linalg.transpose ins(%__auto.model.model.layers.3.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_229 = tensor.collapse_shape %517 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %518 = linalg.matmul ins(%collapsed_229, %transposed_228 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_230 = tensor.expand_shape %518 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %519 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%505, %expanded_230 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %520 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%519 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %521 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%520 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %522 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%521 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %523 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%522 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %524 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%523 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %525 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%519, %524 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.4.input_layernorm.weight = util.global.load @__auto.model.model.layers.4.input_layernorm.weight : tensor<576xf32>
    %526 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.4.input_layernorm.weight, %525 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.4.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.4.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_231 = linalg.transpose ins(%__auto.model.model.layers.4.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_232 = tensor.collapse_shape %526 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %527 = linalg.matmul ins(%collapsed_232, %transposed_231 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_233 = tensor.expand_shape %527 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_234 = linalg.transpose ins(%expanded_233 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.4.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.4.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_235 = linalg.transpose ins(%__auto.model.model.layers.4.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %528 = linalg.matmul ins(%collapsed_232, %transposed_235 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_236 = tensor.expand_shape %528 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_237 = linalg.transpose ins(%expanded_236 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.4.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.4.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_238 = linalg.transpose ins(%__auto.model.model.layers.4.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %529 = linalg.matmul ins(%collapsed_232, %transposed_238 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_239 = tensor.expand_shape %529 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_240 = linalg.transpose ins(%expanded_239 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %530 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_234, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_241 = tensor.extract_slice %transposed_234[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_242 = tensor.extract_slice %transposed_234[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %531 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_242 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_243 = tensor.concat dim(3) %531, %extracted_slice_241 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %532 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_243, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %533 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%530, %532 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %534 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_237, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_244 = tensor.extract_slice %transposed_237[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_245 = tensor.extract_slice %transposed_237[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %535 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_245 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_246 = tensor.concat dim(3) %535, %extracted_slice_244 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %536 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_246, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %537 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%534, %536 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_247 = tensor.concat dim(2) %21, %537 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_248 = tensor.concat dim(2) %23, %transposed_240 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %538 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_247 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_249 = tensor.collapse_shape %538 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %539 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_248 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %540 = affine.apply #map15()[%dim_17]
    %541 = tensor.empty(%540) : tensor<1x9x64x?xf32>
    %transposed_250 = linalg.transpose ins(%collapsed_249 : tensor<1x9x?x64xf32>) outs(%541 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_251 = tensor.collapse_shape %533 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_252 = tensor.collapse_shape %transposed_250 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %542 = tensor.empty(%540) : tensor<9x1x?xf32>
    %543 = linalg.fill ins(%cst : f32) outs(%542 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %544 = linalg.batch_matmul ins(%collapsed_251, %collapsed_252 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%543 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_253 = tensor.expand_shape %544 [[0, 1], [2], [3]] output_shape [1, 9, 1, %540] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %545 = tensor.empty(%540) : tensor<1x9x1x?xf32>
    %546 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_253 : tensor<1x9x1x?xf32>) outs(%545 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %547 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%546, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%545 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %548:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%547 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_254 = tensor.expand_shape %548#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %549 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%547, %expanded_254 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%545 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %550 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%549 : tensor<1x9x1x?xf32>) outs(%545 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %551 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%550 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %552 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%550, %551 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%545 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_255 = tensor.collapse_shape %552 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_256 = tensor.collapse_shape %539 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %553 = affine.apply #map15()[%dim_18]
    %554 = arith.index_cast %540 : index to i64
    %555 = arith.index_cast %553 : index to i64
    %556 = arith.cmpi eq, %554, %555 : i64
    cf.assert %556, "mismatching contracting dimension"
    %557 = linalg.batch_matmul ins(%collapsed_255, %collapsed_256 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_257 = tensor.expand_shape %557 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_258 = linalg.transpose ins(%expanded_257 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.4.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.4.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_259 = linalg.transpose ins(%__auto.model.model.layers.4.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_260 = tensor.collapse_shape %transposed_258 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %558 = linalg.matmul ins(%collapsed_260, %transposed_259 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_261 = tensor.expand_shape %558 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %559 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%519, %expanded_261 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %560 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%559 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %561 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%560 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %562 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%561 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %563 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%562 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %564 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%563 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %565 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%559, %564 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.4.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.4.post_attention_layernorm.weight : tensor<576xf32>
    %566 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.4.post_attention_layernorm.weight, %565 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.4.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.4.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_262 = linalg.transpose ins(%__auto.model.model.layers.4.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_263 = tensor.collapse_shape %566 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %567 = linalg.matmul ins(%collapsed_263, %transposed_262 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_264 = tensor.expand_shape %567 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %568 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_264 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %569 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_264, %568 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.4.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.4.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_265 = linalg.transpose ins(%__auto.model.model.layers.4.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %570 = linalg.matmul ins(%collapsed_263, %transposed_265 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_266 = tensor.expand_shape %570 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %571 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%569, %expanded_266 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.4.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.4.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_267 = linalg.transpose ins(%__auto.model.model.layers.4.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_268 = tensor.collapse_shape %571 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %572 = linalg.matmul ins(%collapsed_268, %transposed_267 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_269 = tensor.expand_shape %572 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %573 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%559, %expanded_269 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %574 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%573 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %575 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%574 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %576 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%575 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %577 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%576 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %578 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%577 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %579 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%573, %578 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.5.input_layernorm.weight = util.global.load @__auto.model.model.layers.5.input_layernorm.weight : tensor<576xf32>
    %580 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.5.input_layernorm.weight, %579 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.5.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.5.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_270 = linalg.transpose ins(%__auto.model.model.layers.5.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_271 = tensor.collapse_shape %580 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %581 = linalg.matmul ins(%collapsed_271, %transposed_270 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_272 = tensor.expand_shape %581 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_273 = linalg.transpose ins(%expanded_272 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.5.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.5.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_274 = linalg.transpose ins(%__auto.model.model.layers.5.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %582 = linalg.matmul ins(%collapsed_271, %transposed_274 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_275 = tensor.expand_shape %582 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_276 = linalg.transpose ins(%expanded_275 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.5.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.5.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_277 = linalg.transpose ins(%__auto.model.model.layers.5.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %583 = linalg.matmul ins(%collapsed_271, %transposed_277 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_278 = tensor.expand_shape %583 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_279 = linalg.transpose ins(%expanded_278 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %584 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_273, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_280 = tensor.extract_slice %transposed_273[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_281 = tensor.extract_slice %transposed_273[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %585 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_281 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_282 = tensor.concat dim(3) %585, %extracted_slice_280 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %586 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_282, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %587 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%584, %586 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %588 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_276, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_283 = tensor.extract_slice %transposed_276[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_284 = tensor.extract_slice %transposed_276[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %589 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_284 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_285 = tensor.concat dim(3) %589, %extracted_slice_283 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %590 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_285, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %591 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%588, %590 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_286 = tensor.concat dim(2) %25, %591 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_287 = tensor.concat dim(2) %27, %transposed_279 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %592 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_286 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_288 = tensor.collapse_shape %592 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %593 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_287 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %594 = affine.apply #map15()[%dim_19]
    %595 = tensor.empty(%594) : tensor<1x9x64x?xf32>
    %transposed_289 = linalg.transpose ins(%collapsed_288 : tensor<1x9x?x64xf32>) outs(%595 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_290 = tensor.collapse_shape %587 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_291 = tensor.collapse_shape %transposed_289 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %596 = tensor.empty(%594) : tensor<9x1x?xf32>
    %597 = linalg.fill ins(%cst : f32) outs(%596 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %598 = linalg.batch_matmul ins(%collapsed_290, %collapsed_291 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%597 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_292 = tensor.expand_shape %598 [[0, 1], [2], [3]] output_shape [1, 9, 1, %594] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %599 = tensor.empty(%594) : tensor<1x9x1x?xf32>
    %600 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_292 : tensor<1x9x1x?xf32>) outs(%599 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %601 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%600, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%599 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %602:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%601 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_293 = tensor.expand_shape %602#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %603 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%601, %expanded_293 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%599 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %604 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%603 : tensor<1x9x1x?xf32>) outs(%599 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %605 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%604 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %606 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%604, %605 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%599 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_294 = tensor.collapse_shape %606 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_295 = tensor.collapse_shape %593 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %607 = affine.apply #map15()[%dim_20]
    %608 = arith.index_cast %594 : index to i64
    %609 = arith.index_cast %607 : index to i64
    %610 = arith.cmpi eq, %608, %609 : i64
    cf.assert %610, "mismatching contracting dimension"
    %611 = linalg.batch_matmul ins(%collapsed_294, %collapsed_295 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_296 = tensor.expand_shape %611 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_297 = linalg.transpose ins(%expanded_296 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.5.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.5.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_298 = linalg.transpose ins(%__auto.model.model.layers.5.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_299 = tensor.collapse_shape %transposed_297 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %612 = linalg.matmul ins(%collapsed_299, %transposed_298 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_300 = tensor.expand_shape %612 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %613 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%573, %expanded_300 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %614 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%613 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %615 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%614 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %616 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%615 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %617 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%616 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %618 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%617 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %619 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%613, %618 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.5.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.5.post_attention_layernorm.weight : tensor<576xf32>
    %620 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.5.post_attention_layernorm.weight, %619 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.5.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.5.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_301 = linalg.transpose ins(%__auto.model.model.layers.5.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_302 = tensor.collapse_shape %620 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %621 = linalg.matmul ins(%collapsed_302, %transposed_301 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_303 = tensor.expand_shape %621 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %622 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_303 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %623 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_303, %622 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.5.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.5.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_304 = linalg.transpose ins(%__auto.model.model.layers.5.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %624 = linalg.matmul ins(%collapsed_302, %transposed_304 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_305 = tensor.expand_shape %624 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %625 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%623, %expanded_305 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.5.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.5.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_306 = linalg.transpose ins(%__auto.model.model.layers.5.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_307 = tensor.collapse_shape %625 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %626 = linalg.matmul ins(%collapsed_307, %transposed_306 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_308 = tensor.expand_shape %626 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %627 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%613, %expanded_308 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %628 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%627 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %629 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%628 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %630 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%629 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %631 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%630 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %632 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%631 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %633 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%627, %632 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.6.input_layernorm.weight = util.global.load @__auto.model.model.layers.6.input_layernorm.weight : tensor<576xf32>
    %634 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.6.input_layernorm.weight, %633 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.6.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.6.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_309 = linalg.transpose ins(%__auto.model.model.layers.6.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_310 = tensor.collapse_shape %634 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %635 = linalg.matmul ins(%collapsed_310, %transposed_309 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_311 = tensor.expand_shape %635 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_312 = linalg.transpose ins(%expanded_311 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.6.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.6.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_313 = linalg.transpose ins(%__auto.model.model.layers.6.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %636 = linalg.matmul ins(%collapsed_310, %transposed_313 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_314 = tensor.expand_shape %636 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_315 = linalg.transpose ins(%expanded_314 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.6.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.6.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_316 = linalg.transpose ins(%__auto.model.model.layers.6.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %637 = linalg.matmul ins(%collapsed_310, %transposed_316 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_317 = tensor.expand_shape %637 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_318 = linalg.transpose ins(%expanded_317 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %638 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_312, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_319 = tensor.extract_slice %transposed_312[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_320 = tensor.extract_slice %transposed_312[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %639 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_320 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_321 = tensor.concat dim(3) %639, %extracted_slice_319 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %640 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_321, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %641 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%638, %640 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %642 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_315, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_322 = tensor.extract_slice %transposed_315[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_323 = tensor.extract_slice %transposed_315[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %643 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_323 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_324 = tensor.concat dim(3) %643, %extracted_slice_322 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %644 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_324, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %645 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%642, %644 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_325 = tensor.concat dim(2) %29, %645 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_326 = tensor.concat dim(2) %31, %transposed_318 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %646 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_325 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_327 = tensor.collapse_shape %646 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %647 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_326 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %648 = affine.apply #map15()[%dim_21]
    %649 = tensor.empty(%648) : tensor<1x9x64x?xf32>
    %transposed_328 = linalg.transpose ins(%collapsed_327 : tensor<1x9x?x64xf32>) outs(%649 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_329 = tensor.collapse_shape %641 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_330 = tensor.collapse_shape %transposed_328 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %650 = tensor.empty(%648) : tensor<9x1x?xf32>
    %651 = linalg.fill ins(%cst : f32) outs(%650 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %652 = linalg.batch_matmul ins(%collapsed_329, %collapsed_330 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%651 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_331 = tensor.expand_shape %652 [[0, 1], [2], [3]] output_shape [1, 9, 1, %648] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %653 = tensor.empty(%648) : tensor<1x9x1x?xf32>
    %654 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_331 : tensor<1x9x1x?xf32>) outs(%653 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %655 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%654, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%653 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %656:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%655 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_332 = tensor.expand_shape %656#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %657 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%655, %expanded_332 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%653 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %658 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%657 : tensor<1x9x1x?xf32>) outs(%653 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %659 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%658 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %660 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%658, %659 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%653 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_333 = tensor.collapse_shape %660 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_334 = tensor.collapse_shape %647 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %661 = affine.apply #map15()[%dim_22]
    %662 = arith.index_cast %648 : index to i64
    %663 = arith.index_cast %661 : index to i64
    %664 = arith.cmpi eq, %662, %663 : i64
    cf.assert %664, "mismatching contracting dimension"
    %665 = linalg.batch_matmul ins(%collapsed_333, %collapsed_334 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_335 = tensor.expand_shape %665 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_336 = linalg.transpose ins(%expanded_335 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.6.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.6.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_337 = linalg.transpose ins(%__auto.model.model.layers.6.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_338 = tensor.collapse_shape %transposed_336 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %666 = linalg.matmul ins(%collapsed_338, %transposed_337 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_339 = tensor.expand_shape %666 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %667 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%627, %expanded_339 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %668 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%667 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %669 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%668 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %670 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%669 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %671 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%670 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %672 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%671 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %673 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%667, %672 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.6.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.6.post_attention_layernorm.weight : tensor<576xf32>
    %674 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.6.post_attention_layernorm.weight, %673 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.6.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.6.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_340 = linalg.transpose ins(%__auto.model.model.layers.6.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_341 = tensor.collapse_shape %674 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %675 = linalg.matmul ins(%collapsed_341, %transposed_340 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_342 = tensor.expand_shape %675 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %676 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_342 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %677 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_342, %676 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.6.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.6.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_343 = linalg.transpose ins(%__auto.model.model.layers.6.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %678 = linalg.matmul ins(%collapsed_341, %transposed_343 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_344 = tensor.expand_shape %678 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %679 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%677, %expanded_344 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.6.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.6.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_345 = linalg.transpose ins(%__auto.model.model.layers.6.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_346 = tensor.collapse_shape %679 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %680 = linalg.matmul ins(%collapsed_346, %transposed_345 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_347 = tensor.expand_shape %680 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %681 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%667, %expanded_347 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %682 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%681 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %683 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%682 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %684 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%683 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %685 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%684 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %686 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%685 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %687 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%681, %686 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.7.input_layernorm.weight = util.global.load @__auto.model.model.layers.7.input_layernorm.weight : tensor<576xf32>
    %688 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.7.input_layernorm.weight, %687 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.7.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.7.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_348 = linalg.transpose ins(%__auto.model.model.layers.7.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_349 = tensor.collapse_shape %688 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %689 = linalg.matmul ins(%collapsed_349, %transposed_348 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_350 = tensor.expand_shape %689 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_351 = linalg.transpose ins(%expanded_350 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.7.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.7.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_352 = linalg.transpose ins(%__auto.model.model.layers.7.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %690 = linalg.matmul ins(%collapsed_349, %transposed_352 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_353 = tensor.expand_shape %690 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_354 = linalg.transpose ins(%expanded_353 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.7.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.7.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_355 = linalg.transpose ins(%__auto.model.model.layers.7.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %691 = linalg.matmul ins(%collapsed_349, %transposed_355 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_356 = tensor.expand_shape %691 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_357 = linalg.transpose ins(%expanded_356 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %692 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_351, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_358 = tensor.extract_slice %transposed_351[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_359 = tensor.extract_slice %transposed_351[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %693 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_359 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_360 = tensor.concat dim(3) %693, %extracted_slice_358 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %694 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_360, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %695 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%692, %694 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %696 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_354, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_361 = tensor.extract_slice %transposed_354[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_362 = tensor.extract_slice %transposed_354[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %697 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_362 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_363 = tensor.concat dim(3) %697, %extracted_slice_361 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %698 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_363, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %699 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%696, %698 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_364 = tensor.concat dim(2) %33, %699 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_365 = tensor.concat dim(2) %35, %transposed_357 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %700 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_364 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_366 = tensor.collapse_shape %700 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %701 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_365 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %702 = affine.apply #map15()[%dim_23]
    %703 = tensor.empty(%702) : tensor<1x9x64x?xf32>
    %transposed_367 = linalg.transpose ins(%collapsed_366 : tensor<1x9x?x64xf32>) outs(%703 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_368 = tensor.collapse_shape %695 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_369 = tensor.collapse_shape %transposed_367 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %704 = tensor.empty(%702) : tensor<9x1x?xf32>
    %705 = linalg.fill ins(%cst : f32) outs(%704 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %706 = linalg.batch_matmul ins(%collapsed_368, %collapsed_369 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%705 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_370 = tensor.expand_shape %706 [[0, 1], [2], [3]] output_shape [1, 9, 1, %702] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %707 = tensor.empty(%702) : tensor<1x9x1x?xf32>
    %708 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_370 : tensor<1x9x1x?xf32>) outs(%707 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %709 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%708, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%707 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %710:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%709 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_371 = tensor.expand_shape %710#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %711 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%709, %expanded_371 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%707 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %712 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%711 : tensor<1x9x1x?xf32>) outs(%707 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %713 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%712 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %714 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%712, %713 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%707 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_372 = tensor.collapse_shape %714 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_373 = tensor.collapse_shape %701 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %715 = affine.apply #map15()[%dim_24]
    %716 = arith.index_cast %702 : index to i64
    %717 = arith.index_cast %715 : index to i64
    %718 = arith.cmpi eq, %716, %717 : i64
    cf.assert %718, "mismatching contracting dimension"
    %719 = linalg.batch_matmul ins(%collapsed_372, %collapsed_373 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_374 = tensor.expand_shape %719 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_375 = linalg.transpose ins(%expanded_374 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.7.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.7.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_376 = linalg.transpose ins(%__auto.model.model.layers.7.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_377 = tensor.collapse_shape %transposed_375 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %720 = linalg.matmul ins(%collapsed_377, %transposed_376 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_378 = tensor.expand_shape %720 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %721 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%681, %expanded_378 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %722 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%721 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %723 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%722 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %724 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%723 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %725 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%724 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %726 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%725 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %727 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%721, %726 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.7.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.7.post_attention_layernorm.weight : tensor<576xf32>
    %728 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.7.post_attention_layernorm.weight, %727 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.7.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.7.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_379 = linalg.transpose ins(%__auto.model.model.layers.7.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_380 = tensor.collapse_shape %728 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %729 = linalg.matmul ins(%collapsed_380, %transposed_379 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_381 = tensor.expand_shape %729 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %730 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_381 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %731 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_381, %730 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.7.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.7.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_382 = linalg.transpose ins(%__auto.model.model.layers.7.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %732 = linalg.matmul ins(%collapsed_380, %transposed_382 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_383 = tensor.expand_shape %732 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %733 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%731, %expanded_383 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.7.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.7.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_384 = linalg.transpose ins(%__auto.model.model.layers.7.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_385 = tensor.collapse_shape %733 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %734 = linalg.matmul ins(%collapsed_385, %transposed_384 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_386 = tensor.expand_shape %734 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %735 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%721, %expanded_386 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %736 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%735 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %737 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%736 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %738 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%737 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %739 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%738 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %740 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%739 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %741 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%735, %740 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.8.input_layernorm.weight = util.global.load @__auto.model.model.layers.8.input_layernorm.weight : tensor<576xf32>
    %742 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.8.input_layernorm.weight, %741 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.8.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.8.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_387 = linalg.transpose ins(%__auto.model.model.layers.8.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_388 = tensor.collapse_shape %742 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %743 = linalg.matmul ins(%collapsed_388, %transposed_387 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_389 = tensor.expand_shape %743 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_390 = linalg.transpose ins(%expanded_389 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.8.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.8.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_391 = linalg.transpose ins(%__auto.model.model.layers.8.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %744 = linalg.matmul ins(%collapsed_388, %transposed_391 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_392 = tensor.expand_shape %744 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_393 = linalg.transpose ins(%expanded_392 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.8.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.8.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_394 = linalg.transpose ins(%__auto.model.model.layers.8.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %745 = linalg.matmul ins(%collapsed_388, %transposed_394 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_395 = tensor.expand_shape %745 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_396 = linalg.transpose ins(%expanded_395 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %746 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_390, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_397 = tensor.extract_slice %transposed_390[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_398 = tensor.extract_slice %transposed_390[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %747 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_398 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_399 = tensor.concat dim(3) %747, %extracted_slice_397 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %748 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_399, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %749 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%746, %748 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %750 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_393, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_400 = tensor.extract_slice %transposed_393[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_401 = tensor.extract_slice %transposed_393[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %751 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_401 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_402 = tensor.concat dim(3) %751, %extracted_slice_400 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %752 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_402, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %753 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%750, %752 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_403 = tensor.concat dim(2) %37, %753 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_404 = tensor.concat dim(2) %39, %transposed_396 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %754 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_403 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_405 = tensor.collapse_shape %754 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %755 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_404 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %756 = affine.apply #map15()[%dim_25]
    %757 = tensor.empty(%756) : tensor<1x9x64x?xf32>
    %transposed_406 = linalg.transpose ins(%collapsed_405 : tensor<1x9x?x64xf32>) outs(%757 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_407 = tensor.collapse_shape %749 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_408 = tensor.collapse_shape %transposed_406 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %758 = tensor.empty(%756) : tensor<9x1x?xf32>
    %759 = linalg.fill ins(%cst : f32) outs(%758 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %760 = linalg.batch_matmul ins(%collapsed_407, %collapsed_408 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%759 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_409 = tensor.expand_shape %760 [[0, 1], [2], [3]] output_shape [1, 9, 1, %756] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %761 = tensor.empty(%756) : tensor<1x9x1x?xf32>
    %762 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_409 : tensor<1x9x1x?xf32>) outs(%761 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %763 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%762, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%761 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %764:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%763 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_410 = tensor.expand_shape %764#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %765 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%763, %expanded_410 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%761 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %766 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%765 : tensor<1x9x1x?xf32>) outs(%761 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %767 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%766 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %768 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%766, %767 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%761 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_411 = tensor.collapse_shape %768 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_412 = tensor.collapse_shape %755 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %769 = affine.apply #map15()[%dim_26]
    %770 = arith.index_cast %756 : index to i64
    %771 = arith.index_cast %769 : index to i64
    %772 = arith.cmpi eq, %770, %771 : i64
    cf.assert %772, "mismatching contracting dimension"
    %773 = linalg.batch_matmul ins(%collapsed_411, %collapsed_412 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_413 = tensor.expand_shape %773 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_414 = linalg.transpose ins(%expanded_413 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.8.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.8.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_415 = linalg.transpose ins(%__auto.model.model.layers.8.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_416 = tensor.collapse_shape %transposed_414 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %774 = linalg.matmul ins(%collapsed_416, %transposed_415 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_417 = tensor.expand_shape %774 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %775 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%735, %expanded_417 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %776 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%775 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %777 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%776 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%777 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %779 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%778 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %780 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%779 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %781 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%775, %780 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.8.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.8.post_attention_layernorm.weight : tensor<576xf32>
    %782 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.8.post_attention_layernorm.weight, %781 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.8.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.8.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_418 = linalg.transpose ins(%__auto.model.model.layers.8.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_419 = tensor.collapse_shape %782 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %783 = linalg.matmul ins(%collapsed_419, %transposed_418 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_420 = tensor.expand_shape %783 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %784 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_420 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %785 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_420, %784 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.8.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.8.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_421 = linalg.transpose ins(%__auto.model.model.layers.8.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %786 = linalg.matmul ins(%collapsed_419, %transposed_421 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_422 = tensor.expand_shape %786 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %787 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%785, %expanded_422 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.8.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.8.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_423 = linalg.transpose ins(%__auto.model.model.layers.8.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_424 = tensor.collapse_shape %787 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %788 = linalg.matmul ins(%collapsed_424, %transposed_423 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_425 = tensor.expand_shape %788 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %789 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%775, %expanded_425 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %790 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%789 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %791 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%790 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %792 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%791 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %793 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%792 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %794 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%793 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %795 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%789, %794 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.9.input_layernorm.weight = util.global.load @__auto.model.model.layers.9.input_layernorm.weight : tensor<576xf32>
    %796 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.9.input_layernorm.weight, %795 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.9.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.9.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_426 = linalg.transpose ins(%__auto.model.model.layers.9.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_427 = tensor.collapse_shape %796 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %797 = linalg.matmul ins(%collapsed_427, %transposed_426 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_428 = tensor.expand_shape %797 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_429 = linalg.transpose ins(%expanded_428 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.9.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.9.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_430 = linalg.transpose ins(%__auto.model.model.layers.9.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %798 = linalg.matmul ins(%collapsed_427, %transposed_430 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_431 = tensor.expand_shape %798 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_432 = linalg.transpose ins(%expanded_431 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.9.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.9.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_433 = linalg.transpose ins(%__auto.model.model.layers.9.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %799 = linalg.matmul ins(%collapsed_427, %transposed_433 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_434 = tensor.expand_shape %799 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_435 = linalg.transpose ins(%expanded_434 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %800 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_429, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_436 = tensor.extract_slice %transposed_429[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_437 = tensor.extract_slice %transposed_429[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %801 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_437 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_438 = tensor.concat dim(3) %801, %extracted_slice_436 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %802 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_438, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %803 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%800, %802 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %804 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_432, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_439 = tensor.extract_slice %transposed_432[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_440 = tensor.extract_slice %transposed_432[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %805 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_440 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_441 = tensor.concat dim(3) %805, %extracted_slice_439 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %806 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_441, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %807 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%804, %806 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_442 = tensor.concat dim(2) %41, %807 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_443 = tensor.concat dim(2) %43, %transposed_435 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %808 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_442 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_444 = tensor.collapse_shape %808 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %809 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_443 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %810 = affine.apply #map15()[%dim_27]
    %811 = tensor.empty(%810) : tensor<1x9x64x?xf32>
    %transposed_445 = linalg.transpose ins(%collapsed_444 : tensor<1x9x?x64xf32>) outs(%811 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_446 = tensor.collapse_shape %803 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_447 = tensor.collapse_shape %transposed_445 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %812 = tensor.empty(%810) : tensor<9x1x?xf32>
    %813 = linalg.fill ins(%cst : f32) outs(%812 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %814 = linalg.batch_matmul ins(%collapsed_446, %collapsed_447 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%813 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_448 = tensor.expand_shape %814 [[0, 1], [2], [3]] output_shape [1, 9, 1, %810] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %815 = tensor.empty(%810) : tensor<1x9x1x?xf32>
    %816 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_448 : tensor<1x9x1x?xf32>) outs(%815 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %817 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%816, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%815 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %818:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%817 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_449 = tensor.expand_shape %818#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %819 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%817, %expanded_449 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%815 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %820 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%819 : tensor<1x9x1x?xf32>) outs(%815 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %821 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%820 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %822 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%820, %821 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%815 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_450 = tensor.collapse_shape %822 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_451 = tensor.collapse_shape %809 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %823 = affine.apply #map15()[%dim_28]
    %824 = arith.index_cast %810 : index to i64
    %825 = arith.index_cast %823 : index to i64
    %826 = arith.cmpi eq, %824, %825 : i64
    cf.assert %826, "mismatching contracting dimension"
    %827 = linalg.batch_matmul ins(%collapsed_450, %collapsed_451 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_452 = tensor.expand_shape %827 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_453 = linalg.transpose ins(%expanded_452 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.9.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.9.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_454 = linalg.transpose ins(%__auto.model.model.layers.9.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_455 = tensor.collapse_shape %transposed_453 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %828 = linalg.matmul ins(%collapsed_455, %transposed_454 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_456 = tensor.expand_shape %828 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %829 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%789, %expanded_456 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %830 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%829 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %831 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%830 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %832 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%831 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %833 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%832 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %834 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%833 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %835 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%829, %834 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.9.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.9.post_attention_layernorm.weight : tensor<576xf32>
    %836 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.9.post_attention_layernorm.weight, %835 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.9.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.9.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_457 = linalg.transpose ins(%__auto.model.model.layers.9.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_458 = tensor.collapse_shape %836 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %837 = linalg.matmul ins(%collapsed_458, %transposed_457 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_459 = tensor.expand_shape %837 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %838 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_459 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %839 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_459, %838 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.9.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.9.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_460 = linalg.transpose ins(%__auto.model.model.layers.9.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %840 = linalg.matmul ins(%collapsed_458, %transposed_460 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_461 = tensor.expand_shape %840 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %841 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%839, %expanded_461 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.9.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.9.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_462 = linalg.transpose ins(%__auto.model.model.layers.9.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_463 = tensor.collapse_shape %841 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %842 = linalg.matmul ins(%collapsed_463, %transposed_462 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_464 = tensor.expand_shape %842 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %843 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%829, %expanded_464 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %844 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%843 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %845 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%844 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %846 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%845 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %847 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%846 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %848 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%847 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %849 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%843, %848 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.10.input_layernorm.weight = util.global.load @__auto.model.model.layers.10.input_layernorm.weight : tensor<576xf32>
    %850 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.10.input_layernorm.weight, %849 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.10.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.10.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_465 = linalg.transpose ins(%__auto.model.model.layers.10.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_466 = tensor.collapse_shape %850 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %851 = linalg.matmul ins(%collapsed_466, %transposed_465 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_467 = tensor.expand_shape %851 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_468 = linalg.transpose ins(%expanded_467 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.10.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.10.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_469 = linalg.transpose ins(%__auto.model.model.layers.10.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %852 = linalg.matmul ins(%collapsed_466, %transposed_469 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_470 = tensor.expand_shape %852 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_471 = linalg.transpose ins(%expanded_470 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.10.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.10.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_472 = linalg.transpose ins(%__auto.model.model.layers.10.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %853 = linalg.matmul ins(%collapsed_466, %transposed_472 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_473 = tensor.expand_shape %853 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_474 = linalg.transpose ins(%expanded_473 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %854 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_468, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_475 = tensor.extract_slice %transposed_468[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_476 = tensor.extract_slice %transposed_468[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %855 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_476 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_477 = tensor.concat dim(3) %855, %extracted_slice_475 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %856 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_477, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %857 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%854, %856 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %858 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_471, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_478 = tensor.extract_slice %transposed_471[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_479 = tensor.extract_slice %transposed_471[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %859 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_479 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_480 = tensor.concat dim(3) %859, %extracted_slice_478 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %860 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_480, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %861 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%858, %860 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_481 = tensor.concat dim(2) %45, %861 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_482 = tensor.concat dim(2) %47, %transposed_474 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %862 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_481 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_483 = tensor.collapse_shape %862 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %863 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_482 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %864 = affine.apply #map15()[%dim_29]
    %865 = tensor.empty(%864) : tensor<1x9x64x?xf32>
    %transposed_484 = linalg.transpose ins(%collapsed_483 : tensor<1x9x?x64xf32>) outs(%865 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_485 = tensor.collapse_shape %857 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_486 = tensor.collapse_shape %transposed_484 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %866 = tensor.empty(%864) : tensor<9x1x?xf32>
    %867 = linalg.fill ins(%cst : f32) outs(%866 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %868 = linalg.batch_matmul ins(%collapsed_485, %collapsed_486 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%867 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_487 = tensor.expand_shape %868 [[0, 1], [2], [3]] output_shape [1, 9, 1, %864] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %869 = tensor.empty(%864) : tensor<1x9x1x?xf32>
    %870 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_487 : tensor<1x9x1x?xf32>) outs(%869 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %871 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%870, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%869 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %872:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%871 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_488 = tensor.expand_shape %872#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %873 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%871, %expanded_488 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%869 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %874 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%873 : tensor<1x9x1x?xf32>) outs(%869 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %875 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%874 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %876 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%874, %875 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%869 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_489 = tensor.collapse_shape %876 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_490 = tensor.collapse_shape %863 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %877 = affine.apply #map15()[%dim_30]
    %878 = arith.index_cast %864 : index to i64
    %879 = arith.index_cast %877 : index to i64
    %880 = arith.cmpi eq, %878, %879 : i64
    cf.assert %880, "mismatching contracting dimension"
    %881 = linalg.batch_matmul ins(%collapsed_489, %collapsed_490 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_491 = tensor.expand_shape %881 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_492 = linalg.transpose ins(%expanded_491 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.10.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.10.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_493 = linalg.transpose ins(%__auto.model.model.layers.10.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_494 = tensor.collapse_shape %transposed_492 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %882 = linalg.matmul ins(%collapsed_494, %transposed_493 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_495 = tensor.expand_shape %882 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %883 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%843, %expanded_495 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %884 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%883 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %885 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%884 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %886 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%885 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %887 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%886 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %888 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%887 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %889 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%883, %888 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.10.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.10.post_attention_layernorm.weight : tensor<576xf32>
    %890 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.10.post_attention_layernorm.weight, %889 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.10.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.10.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_496 = linalg.transpose ins(%__auto.model.model.layers.10.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_497 = tensor.collapse_shape %890 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %891 = linalg.matmul ins(%collapsed_497, %transposed_496 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_498 = tensor.expand_shape %891 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %892 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_498 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %893 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_498, %892 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.10.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.10.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_499 = linalg.transpose ins(%__auto.model.model.layers.10.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %894 = linalg.matmul ins(%collapsed_497, %transposed_499 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_500 = tensor.expand_shape %894 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %895 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%893, %expanded_500 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.10.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.10.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_501 = linalg.transpose ins(%__auto.model.model.layers.10.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_502 = tensor.collapse_shape %895 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %896 = linalg.matmul ins(%collapsed_502, %transposed_501 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_503 = tensor.expand_shape %896 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %897 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%883, %expanded_503 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %898 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%897 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %899 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%898 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %900 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%899 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %901 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%900 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %902 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%901 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %903 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%897, %902 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.11.input_layernorm.weight = util.global.load @__auto.model.model.layers.11.input_layernorm.weight : tensor<576xf32>
    %904 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.11.input_layernorm.weight, %903 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.11.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.11.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_504 = linalg.transpose ins(%__auto.model.model.layers.11.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_505 = tensor.collapse_shape %904 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %905 = linalg.matmul ins(%collapsed_505, %transposed_504 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_506 = tensor.expand_shape %905 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_507 = linalg.transpose ins(%expanded_506 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.11.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.11.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_508 = linalg.transpose ins(%__auto.model.model.layers.11.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %906 = linalg.matmul ins(%collapsed_505, %transposed_508 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_509 = tensor.expand_shape %906 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_510 = linalg.transpose ins(%expanded_509 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.11.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.11.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_511 = linalg.transpose ins(%__auto.model.model.layers.11.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %907 = linalg.matmul ins(%collapsed_505, %transposed_511 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_512 = tensor.expand_shape %907 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_513 = linalg.transpose ins(%expanded_512 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %908 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_507, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_514 = tensor.extract_slice %transposed_507[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_515 = tensor.extract_slice %transposed_507[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %909 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_515 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_516 = tensor.concat dim(3) %909, %extracted_slice_514 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %910 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_516, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %911 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%908, %910 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %912 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_510, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_517 = tensor.extract_slice %transposed_510[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_518 = tensor.extract_slice %transposed_510[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %913 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_518 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_519 = tensor.concat dim(3) %913, %extracted_slice_517 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %914 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_519, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %915 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%912, %914 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_520 = tensor.concat dim(2) %49, %915 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_521 = tensor.concat dim(2) %51, %transposed_513 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %916 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_520 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_522 = tensor.collapse_shape %916 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %917 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_521 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %918 = affine.apply #map15()[%dim_31]
    %919 = tensor.empty(%918) : tensor<1x9x64x?xf32>
    %transposed_523 = linalg.transpose ins(%collapsed_522 : tensor<1x9x?x64xf32>) outs(%919 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_524 = tensor.collapse_shape %911 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_525 = tensor.collapse_shape %transposed_523 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %920 = tensor.empty(%918) : tensor<9x1x?xf32>
    %921 = linalg.fill ins(%cst : f32) outs(%920 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %922 = linalg.batch_matmul ins(%collapsed_524, %collapsed_525 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%921 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_526 = tensor.expand_shape %922 [[0, 1], [2], [3]] output_shape [1, 9, 1, %918] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %923 = tensor.empty(%918) : tensor<1x9x1x?xf32>
    %924 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_526 : tensor<1x9x1x?xf32>) outs(%923 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %925 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%924, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%923 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %926:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%925 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_527 = tensor.expand_shape %926#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %927 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%925, %expanded_527 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%923 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %928 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%927 : tensor<1x9x1x?xf32>) outs(%923 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %929 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%928 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %930 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%928, %929 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%923 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_528 = tensor.collapse_shape %930 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_529 = tensor.collapse_shape %917 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %931 = affine.apply #map15()[%dim_32]
    %932 = arith.index_cast %918 : index to i64
    %933 = arith.index_cast %931 : index to i64
    %934 = arith.cmpi eq, %932, %933 : i64
    cf.assert %934, "mismatching contracting dimension"
    %935 = linalg.batch_matmul ins(%collapsed_528, %collapsed_529 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_530 = tensor.expand_shape %935 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_531 = linalg.transpose ins(%expanded_530 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.11.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.11.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_532 = linalg.transpose ins(%__auto.model.model.layers.11.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_533 = tensor.collapse_shape %transposed_531 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %936 = linalg.matmul ins(%collapsed_533, %transposed_532 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_534 = tensor.expand_shape %936 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %937 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%897, %expanded_534 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %938 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%937 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %939 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%938 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %940 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%939 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %941 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%940 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %942 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%941 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %943 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%937, %942 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.11.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.11.post_attention_layernorm.weight : tensor<576xf32>
    %944 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.11.post_attention_layernorm.weight, %943 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.11.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.11.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_535 = linalg.transpose ins(%__auto.model.model.layers.11.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_536 = tensor.collapse_shape %944 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %945 = linalg.matmul ins(%collapsed_536, %transposed_535 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_537 = tensor.expand_shape %945 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %946 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_537 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %947 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_537, %946 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.11.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.11.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_538 = linalg.transpose ins(%__auto.model.model.layers.11.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %948 = linalg.matmul ins(%collapsed_536, %transposed_538 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_539 = tensor.expand_shape %948 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %949 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%947, %expanded_539 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.11.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.11.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_540 = linalg.transpose ins(%__auto.model.model.layers.11.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_541 = tensor.collapse_shape %949 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %950 = linalg.matmul ins(%collapsed_541, %transposed_540 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_542 = tensor.expand_shape %950 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %951 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%937, %expanded_542 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %952 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%951 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %953 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%952 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %954 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%953 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %955 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%954 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %956 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%955 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %957 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%951, %956 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.12.input_layernorm.weight = util.global.load @__auto.model.model.layers.12.input_layernorm.weight : tensor<576xf32>
    %958 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.12.input_layernorm.weight, %957 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.12.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.12.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_543 = linalg.transpose ins(%__auto.model.model.layers.12.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_544 = tensor.collapse_shape %958 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %959 = linalg.matmul ins(%collapsed_544, %transposed_543 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_545 = tensor.expand_shape %959 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_546 = linalg.transpose ins(%expanded_545 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.12.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.12.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_547 = linalg.transpose ins(%__auto.model.model.layers.12.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %960 = linalg.matmul ins(%collapsed_544, %transposed_547 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_548 = tensor.expand_shape %960 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_549 = linalg.transpose ins(%expanded_548 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.12.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.12.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_550 = linalg.transpose ins(%__auto.model.model.layers.12.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %961 = linalg.matmul ins(%collapsed_544, %transposed_550 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_551 = tensor.expand_shape %961 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_552 = linalg.transpose ins(%expanded_551 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %962 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_546, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_553 = tensor.extract_slice %transposed_546[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_554 = tensor.extract_slice %transposed_546[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %963 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_554 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_555 = tensor.concat dim(3) %963, %extracted_slice_553 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %964 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_555, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %965 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%962, %964 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %966 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_549, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_556 = tensor.extract_slice %transposed_549[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_557 = tensor.extract_slice %transposed_549[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %967 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_557 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_558 = tensor.concat dim(3) %967, %extracted_slice_556 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %968 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_558, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %969 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%966, %968 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_559 = tensor.concat dim(2) %53, %969 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_560 = tensor.concat dim(2) %55, %transposed_552 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %970 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_559 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_561 = tensor.collapse_shape %970 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %971 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_560 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %972 = affine.apply #map15()[%dim_33]
    %973 = tensor.empty(%972) : tensor<1x9x64x?xf32>
    %transposed_562 = linalg.transpose ins(%collapsed_561 : tensor<1x9x?x64xf32>) outs(%973 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_563 = tensor.collapse_shape %965 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_564 = tensor.collapse_shape %transposed_562 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %974 = tensor.empty(%972) : tensor<9x1x?xf32>
    %975 = linalg.fill ins(%cst : f32) outs(%974 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %976 = linalg.batch_matmul ins(%collapsed_563, %collapsed_564 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%975 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_565 = tensor.expand_shape %976 [[0, 1], [2], [3]] output_shape [1, 9, 1, %972] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %977 = tensor.empty(%972) : tensor<1x9x1x?xf32>
    %978 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_565 : tensor<1x9x1x?xf32>) outs(%977 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %979 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%978, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%977 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %980:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%979 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_566 = tensor.expand_shape %980#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %981 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%979, %expanded_566 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%977 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %982 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%981 : tensor<1x9x1x?xf32>) outs(%977 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %983 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%982 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %984 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%982, %983 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%977 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_567 = tensor.collapse_shape %984 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_568 = tensor.collapse_shape %971 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %985 = affine.apply #map15()[%dim_34]
    %986 = arith.index_cast %972 : index to i64
    %987 = arith.index_cast %985 : index to i64
    %988 = arith.cmpi eq, %986, %987 : i64
    cf.assert %988, "mismatching contracting dimension"
    %989 = linalg.batch_matmul ins(%collapsed_567, %collapsed_568 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_569 = tensor.expand_shape %989 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_570 = linalg.transpose ins(%expanded_569 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.12.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.12.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_571 = linalg.transpose ins(%__auto.model.model.layers.12.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_572 = tensor.collapse_shape %transposed_570 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %990 = linalg.matmul ins(%collapsed_572, %transposed_571 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_573 = tensor.expand_shape %990 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %991 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%951, %expanded_573 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %992 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%991 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %993 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%992 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %994 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%993 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %995 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%994 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %996 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%995 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %997 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%991, %996 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.12.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.12.post_attention_layernorm.weight : tensor<576xf32>
    %998 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.12.post_attention_layernorm.weight, %997 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.12.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.12.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_574 = linalg.transpose ins(%__auto.model.model.layers.12.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_575 = tensor.collapse_shape %998 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %999 = linalg.matmul ins(%collapsed_575, %transposed_574 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_576 = tensor.expand_shape %999 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1000 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_576 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1001 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_576, %1000 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.12.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.12.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_577 = linalg.transpose ins(%__auto.model.model.layers.12.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1002 = linalg.matmul ins(%collapsed_575, %transposed_577 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_578 = tensor.expand_shape %1002 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1003 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1001, %expanded_578 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.12.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.12.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_579 = linalg.transpose ins(%__auto.model.model.layers.12.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_580 = tensor.collapse_shape %1003 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1004 = linalg.matmul ins(%collapsed_580, %transposed_579 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_581 = tensor.expand_shape %1004 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1005 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%991, %expanded_581 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1006 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1005 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1007 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1006 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1008 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1007 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1009 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1008 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1010 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1009 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1011 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1005, %1010 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.13.input_layernorm.weight = util.global.load @__auto.model.model.layers.13.input_layernorm.weight : tensor<576xf32>
    %1012 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.13.input_layernorm.weight, %1011 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.13.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.13.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_582 = linalg.transpose ins(%__auto.model.model.layers.13.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_583 = tensor.collapse_shape %1012 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1013 = linalg.matmul ins(%collapsed_583, %transposed_582 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_584 = tensor.expand_shape %1013 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_585 = linalg.transpose ins(%expanded_584 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.13.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.13.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_586 = linalg.transpose ins(%__auto.model.model.layers.13.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1014 = linalg.matmul ins(%collapsed_583, %transposed_586 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_587 = tensor.expand_shape %1014 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_588 = linalg.transpose ins(%expanded_587 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.13.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.13.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_589 = linalg.transpose ins(%__auto.model.model.layers.13.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1015 = linalg.matmul ins(%collapsed_583, %transposed_589 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_590 = tensor.expand_shape %1015 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_591 = linalg.transpose ins(%expanded_590 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %1016 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_585, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_592 = tensor.extract_slice %transposed_585[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_593 = tensor.extract_slice %transposed_585[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %1017 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_593 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_594 = tensor.concat dim(3) %1017, %extracted_slice_592 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %1018 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_594, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1019 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1016, %1018 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1020 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_588, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_595 = tensor.extract_slice %transposed_588[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_596 = tensor.extract_slice %transposed_588[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %1021 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_596 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_597 = tensor.concat dim(3) %1021, %extracted_slice_595 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %1022 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_597, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %1023 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1020, %1022 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_598 = tensor.concat dim(2) %57, %1023 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_599 = tensor.concat dim(2) %59, %transposed_591 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %1024 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_598 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_600 = tensor.collapse_shape %1024 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %1025 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_599 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %1026 = affine.apply #map15()[%dim_35]
    %1027 = tensor.empty(%1026) : tensor<1x9x64x?xf32>
    %transposed_601 = linalg.transpose ins(%collapsed_600 : tensor<1x9x?x64xf32>) outs(%1027 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_602 = tensor.collapse_shape %1019 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_603 = tensor.collapse_shape %transposed_601 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %1028 = tensor.empty(%1026) : tensor<9x1x?xf32>
    %1029 = linalg.fill ins(%cst : f32) outs(%1028 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %1030 = linalg.batch_matmul ins(%collapsed_602, %collapsed_603 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%1029 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_604 = tensor.expand_shape %1030 [[0, 1], [2], [3]] output_shape [1, 9, 1, %1026] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %1031 = tensor.empty(%1026) : tensor<1x9x1x?xf32>
    %1032 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_604 : tensor<1x9x1x?xf32>) outs(%1031 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1033 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1032, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%1031 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1034:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1033 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_605 = tensor.expand_shape %1034#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %1035 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1033, %expanded_605 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1031 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1036 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1035 : tensor<1x9x1x?xf32>) outs(%1031 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1037 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1036 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %1038 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1036, %1037 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1031 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_606 = tensor.collapse_shape %1038 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_607 = tensor.collapse_shape %1025 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %1039 = affine.apply #map15()[%dim_36]
    %1040 = arith.index_cast %1026 : index to i64
    %1041 = arith.index_cast %1039 : index to i64
    %1042 = arith.cmpi eq, %1040, %1041 : i64
    cf.assert %1042, "mismatching contracting dimension"
    %1043 = linalg.batch_matmul ins(%collapsed_606, %collapsed_607 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_608 = tensor.expand_shape %1043 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_609 = linalg.transpose ins(%expanded_608 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.13.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.13.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_610 = linalg.transpose ins(%__auto.model.model.layers.13.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_611 = tensor.collapse_shape %transposed_609 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %1044 = linalg.matmul ins(%collapsed_611, %transposed_610 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_612 = tensor.expand_shape %1044 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1045 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1005, %expanded_612 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1046 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1045 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1047 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1046 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1048 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1047 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1049 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1048 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1050 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1049 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1051 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1045, %1050 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.13.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.13.post_attention_layernorm.weight : tensor<576xf32>
    %1052 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.13.post_attention_layernorm.weight, %1051 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.13.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.13.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_613 = linalg.transpose ins(%__auto.model.model.layers.13.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_614 = tensor.collapse_shape %1052 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1053 = linalg.matmul ins(%collapsed_614, %transposed_613 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_615 = tensor.expand_shape %1053 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1054 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_615 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1055 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_615, %1054 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.13.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.13.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_616 = linalg.transpose ins(%__auto.model.model.layers.13.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1056 = linalg.matmul ins(%collapsed_614, %transposed_616 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_617 = tensor.expand_shape %1056 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1057 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1055, %expanded_617 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.13.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.13.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_618 = linalg.transpose ins(%__auto.model.model.layers.13.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_619 = tensor.collapse_shape %1057 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1058 = linalg.matmul ins(%collapsed_619, %transposed_618 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_620 = tensor.expand_shape %1058 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1059 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1045, %expanded_620 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1060 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1059 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1061 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1060 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1062 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1061 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1063 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1062 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1064 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1063 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1065 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1059, %1064 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.14.input_layernorm.weight = util.global.load @__auto.model.model.layers.14.input_layernorm.weight : tensor<576xf32>
    %1066 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.14.input_layernorm.weight, %1065 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.14.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.14.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_621 = linalg.transpose ins(%__auto.model.model.layers.14.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_622 = tensor.collapse_shape %1066 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1067 = linalg.matmul ins(%collapsed_622, %transposed_621 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_623 = tensor.expand_shape %1067 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_624 = linalg.transpose ins(%expanded_623 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.14.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.14.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_625 = linalg.transpose ins(%__auto.model.model.layers.14.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1068 = linalg.matmul ins(%collapsed_622, %transposed_625 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_626 = tensor.expand_shape %1068 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_627 = linalg.transpose ins(%expanded_626 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.14.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.14.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_628 = linalg.transpose ins(%__auto.model.model.layers.14.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1069 = linalg.matmul ins(%collapsed_622, %transposed_628 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_629 = tensor.expand_shape %1069 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_630 = linalg.transpose ins(%expanded_629 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %1070 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_624, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_631 = tensor.extract_slice %transposed_624[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_632 = tensor.extract_slice %transposed_624[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %1071 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_632 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_633 = tensor.concat dim(3) %1071, %extracted_slice_631 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %1072 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_633, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1073 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1070, %1072 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1074 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_627, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_634 = tensor.extract_slice %transposed_627[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_635 = tensor.extract_slice %transposed_627[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %1075 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_635 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_636 = tensor.concat dim(3) %1075, %extracted_slice_634 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %1076 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_636, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %1077 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1074, %1076 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_637 = tensor.concat dim(2) %61, %1077 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_638 = tensor.concat dim(2) %63, %transposed_630 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %1078 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_637 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_639 = tensor.collapse_shape %1078 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %1079 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_638 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %1080 = affine.apply #map15()[%dim_37]
    %1081 = tensor.empty(%1080) : tensor<1x9x64x?xf32>
    %transposed_640 = linalg.transpose ins(%collapsed_639 : tensor<1x9x?x64xf32>) outs(%1081 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_641 = tensor.collapse_shape %1073 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_642 = tensor.collapse_shape %transposed_640 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %1082 = tensor.empty(%1080) : tensor<9x1x?xf32>
    %1083 = linalg.fill ins(%cst : f32) outs(%1082 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %1084 = linalg.batch_matmul ins(%collapsed_641, %collapsed_642 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%1083 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_643 = tensor.expand_shape %1084 [[0, 1], [2], [3]] output_shape [1, 9, 1, %1080] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %1085 = tensor.empty(%1080) : tensor<1x9x1x?xf32>
    %1086 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_643 : tensor<1x9x1x?xf32>) outs(%1085 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1087 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1086, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%1085 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1088:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1087 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_644 = tensor.expand_shape %1088#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %1089 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1087, %expanded_644 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1085 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1090 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1089 : tensor<1x9x1x?xf32>) outs(%1085 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1091 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1090 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %1092 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1090, %1091 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1085 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_645 = tensor.collapse_shape %1092 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_646 = tensor.collapse_shape %1079 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %1093 = affine.apply #map15()[%dim_38]
    %1094 = arith.index_cast %1080 : index to i64
    %1095 = arith.index_cast %1093 : index to i64
    %1096 = arith.cmpi eq, %1094, %1095 : i64
    cf.assert %1096, "mismatching contracting dimension"
    %1097 = linalg.batch_matmul ins(%collapsed_645, %collapsed_646 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_647 = tensor.expand_shape %1097 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_648 = linalg.transpose ins(%expanded_647 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.14.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.14.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_649 = linalg.transpose ins(%__auto.model.model.layers.14.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_650 = tensor.collapse_shape %transposed_648 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %1098 = linalg.matmul ins(%collapsed_650, %transposed_649 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_651 = tensor.expand_shape %1098 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1099 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1059, %expanded_651 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1100 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1099 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1101 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1100 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1102 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1101 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1103 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1102 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1104 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1103 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1105 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1099, %1104 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.14.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.14.post_attention_layernorm.weight : tensor<576xf32>
    %1106 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.14.post_attention_layernorm.weight, %1105 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.14.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.14.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_652 = linalg.transpose ins(%__auto.model.model.layers.14.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_653 = tensor.collapse_shape %1106 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1107 = linalg.matmul ins(%collapsed_653, %transposed_652 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_654 = tensor.expand_shape %1107 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1108 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_654 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1109 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_654, %1108 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.14.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.14.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_655 = linalg.transpose ins(%__auto.model.model.layers.14.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1110 = linalg.matmul ins(%collapsed_653, %transposed_655 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_656 = tensor.expand_shape %1110 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1111 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1109, %expanded_656 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.14.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.14.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_657 = linalg.transpose ins(%__auto.model.model.layers.14.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_658 = tensor.collapse_shape %1111 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1112 = linalg.matmul ins(%collapsed_658, %transposed_657 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_659 = tensor.expand_shape %1112 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1113 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1099, %expanded_659 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1114 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1113 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1115 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1114 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1116 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1115 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1117 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1116 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1118 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1117 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1119 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1113, %1118 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.15.input_layernorm.weight = util.global.load @__auto.model.model.layers.15.input_layernorm.weight : tensor<576xf32>
    %1120 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.15.input_layernorm.weight, %1119 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.15.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.15.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_660 = linalg.transpose ins(%__auto.model.model.layers.15.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_661 = tensor.collapse_shape %1120 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1121 = linalg.matmul ins(%collapsed_661, %transposed_660 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_662 = tensor.expand_shape %1121 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_663 = linalg.transpose ins(%expanded_662 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.15.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.15.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_664 = linalg.transpose ins(%__auto.model.model.layers.15.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1122 = linalg.matmul ins(%collapsed_661, %transposed_664 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_665 = tensor.expand_shape %1122 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_666 = linalg.transpose ins(%expanded_665 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.15.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.15.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_667 = linalg.transpose ins(%__auto.model.model.layers.15.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1123 = linalg.matmul ins(%collapsed_661, %transposed_667 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_668 = tensor.expand_shape %1123 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_669 = linalg.transpose ins(%expanded_668 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %1124 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_663, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_670 = tensor.extract_slice %transposed_663[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_671 = tensor.extract_slice %transposed_663[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %1125 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_671 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_672 = tensor.concat dim(3) %1125, %extracted_slice_670 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %1126 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_672, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1127 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1124, %1126 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1128 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_666, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_673 = tensor.extract_slice %transposed_666[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_674 = tensor.extract_slice %transposed_666[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %1129 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_674 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_675 = tensor.concat dim(3) %1129, %extracted_slice_673 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %1130 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_675, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %1131 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1128, %1130 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_676 = tensor.concat dim(2) %65, %1131 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_677 = tensor.concat dim(2) %67, %transposed_669 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %1132 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_676 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_678 = tensor.collapse_shape %1132 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %1133 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_677 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %1134 = affine.apply #map15()[%dim_39]
    %1135 = tensor.empty(%1134) : tensor<1x9x64x?xf32>
    %transposed_679 = linalg.transpose ins(%collapsed_678 : tensor<1x9x?x64xf32>) outs(%1135 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_680 = tensor.collapse_shape %1127 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_681 = tensor.collapse_shape %transposed_679 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %1136 = tensor.empty(%1134) : tensor<9x1x?xf32>
    %1137 = linalg.fill ins(%cst : f32) outs(%1136 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %1138 = linalg.batch_matmul ins(%collapsed_680, %collapsed_681 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%1137 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_682 = tensor.expand_shape %1138 [[0, 1], [2], [3]] output_shape [1, 9, 1, %1134] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %1139 = tensor.empty(%1134) : tensor<1x9x1x?xf32>
    %1140 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_682 : tensor<1x9x1x?xf32>) outs(%1139 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1141 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1140, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%1139 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1142:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1141 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_683 = tensor.expand_shape %1142#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %1143 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1141, %expanded_683 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1139 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1144 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1143 : tensor<1x9x1x?xf32>) outs(%1139 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1145 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1144 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %1146 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1144, %1145 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1139 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_684 = tensor.collapse_shape %1146 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_685 = tensor.collapse_shape %1133 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %1147 = affine.apply #map15()[%dim_40]
    %1148 = arith.index_cast %1134 : index to i64
    %1149 = arith.index_cast %1147 : index to i64
    %1150 = arith.cmpi eq, %1148, %1149 : i64
    cf.assert %1150, "mismatching contracting dimension"
    %1151 = linalg.batch_matmul ins(%collapsed_684, %collapsed_685 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_686 = tensor.expand_shape %1151 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_687 = linalg.transpose ins(%expanded_686 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.15.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.15.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_688 = linalg.transpose ins(%__auto.model.model.layers.15.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_689 = tensor.collapse_shape %transposed_687 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %1152 = linalg.matmul ins(%collapsed_689, %transposed_688 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_690 = tensor.expand_shape %1152 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1153 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1113, %expanded_690 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1154 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1153 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1155 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1154 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1156 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1155 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1157 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1156 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1158 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1157 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1159 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1153, %1158 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.15.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.15.post_attention_layernorm.weight : tensor<576xf32>
    %1160 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.15.post_attention_layernorm.weight, %1159 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.15.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.15.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_691 = linalg.transpose ins(%__auto.model.model.layers.15.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_692 = tensor.collapse_shape %1160 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1161 = linalg.matmul ins(%collapsed_692, %transposed_691 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_693 = tensor.expand_shape %1161 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1162 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_693 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1163 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_693, %1162 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.15.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.15.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_694 = linalg.transpose ins(%__auto.model.model.layers.15.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1164 = linalg.matmul ins(%collapsed_692, %transposed_694 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_695 = tensor.expand_shape %1164 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1165 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1163, %expanded_695 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.15.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.15.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_696 = linalg.transpose ins(%__auto.model.model.layers.15.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_697 = tensor.collapse_shape %1165 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1166 = linalg.matmul ins(%collapsed_697, %transposed_696 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_698 = tensor.expand_shape %1166 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1167 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1153, %expanded_698 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1168 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1167 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1169 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1168 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1170 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1169 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1171 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1170 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1172 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1171 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1173 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1167, %1172 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.16.input_layernorm.weight = util.global.load @__auto.model.model.layers.16.input_layernorm.weight : tensor<576xf32>
    %1174 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.16.input_layernorm.weight, %1173 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.16.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.16.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_699 = linalg.transpose ins(%__auto.model.model.layers.16.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_700 = tensor.collapse_shape %1174 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1175 = linalg.matmul ins(%collapsed_700, %transposed_699 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_701 = tensor.expand_shape %1175 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_702 = linalg.transpose ins(%expanded_701 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.16.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.16.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_703 = linalg.transpose ins(%__auto.model.model.layers.16.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1176 = linalg.matmul ins(%collapsed_700, %transposed_703 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_704 = tensor.expand_shape %1176 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_705 = linalg.transpose ins(%expanded_704 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.16.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.16.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_706 = linalg.transpose ins(%__auto.model.model.layers.16.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1177 = linalg.matmul ins(%collapsed_700, %transposed_706 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_707 = tensor.expand_shape %1177 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_708 = linalg.transpose ins(%expanded_707 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %1178 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_702, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_709 = tensor.extract_slice %transposed_702[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_710 = tensor.extract_slice %transposed_702[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %1179 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_710 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_711 = tensor.concat dim(3) %1179, %extracted_slice_709 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %1180 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_711, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1181 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1178, %1180 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1182 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_705, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_712 = tensor.extract_slice %transposed_705[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_713 = tensor.extract_slice %transposed_705[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %1183 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_713 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_714 = tensor.concat dim(3) %1183, %extracted_slice_712 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %1184 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_714, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %1185 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1182, %1184 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_715 = tensor.concat dim(2) %69, %1185 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_716 = tensor.concat dim(2) %71, %transposed_708 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %1186 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_715 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_717 = tensor.collapse_shape %1186 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %1187 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_716 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %1188 = affine.apply #map15()[%dim_41]
    %1189 = tensor.empty(%1188) : tensor<1x9x64x?xf32>
    %transposed_718 = linalg.transpose ins(%collapsed_717 : tensor<1x9x?x64xf32>) outs(%1189 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_719 = tensor.collapse_shape %1181 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_720 = tensor.collapse_shape %transposed_718 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %1190 = tensor.empty(%1188) : tensor<9x1x?xf32>
    %1191 = linalg.fill ins(%cst : f32) outs(%1190 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %1192 = linalg.batch_matmul ins(%collapsed_719, %collapsed_720 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%1191 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_721 = tensor.expand_shape %1192 [[0, 1], [2], [3]] output_shape [1, 9, 1, %1188] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %1193 = tensor.empty(%1188) : tensor<1x9x1x?xf32>
    %1194 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_721 : tensor<1x9x1x?xf32>) outs(%1193 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1195 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1194, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%1193 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1196:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1195 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_722 = tensor.expand_shape %1196#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %1197 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1195, %expanded_722 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1193 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1198 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1197 : tensor<1x9x1x?xf32>) outs(%1193 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1199 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1198 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %1200 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1198, %1199 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1193 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_723 = tensor.collapse_shape %1200 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_724 = tensor.collapse_shape %1187 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %1201 = affine.apply #map15()[%dim_42]
    %1202 = arith.index_cast %1188 : index to i64
    %1203 = arith.index_cast %1201 : index to i64
    %1204 = arith.cmpi eq, %1202, %1203 : i64
    cf.assert %1204, "mismatching contracting dimension"
    %1205 = linalg.batch_matmul ins(%collapsed_723, %collapsed_724 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_725 = tensor.expand_shape %1205 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_726 = linalg.transpose ins(%expanded_725 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.16.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.16.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_727 = linalg.transpose ins(%__auto.model.model.layers.16.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_728 = tensor.collapse_shape %transposed_726 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %1206 = linalg.matmul ins(%collapsed_728, %transposed_727 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_729 = tensor.expand_shape %1206 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1207 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1167, %expanded_729 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1208 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1207 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1209 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1208 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1210 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1209 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1211 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1210 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1212 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1211 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1213 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1207, %1212 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.16.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.16.post_attention_layernorm.weight : tensor<576xf32>
    %1214 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.16.post_attention_layernorm.weight, %1213 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.16.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.16.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_730 = linalg.transpose ins(%__auto.model.model.layers.16.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_731 = tensor.collapse_shape %1214 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1215 = linalg.matmul ins(%collapsed_731, %transposed_730 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_732 = tensor.expand_shape %1215 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1216 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_732 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1217 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_732, %1216 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.16.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.16.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_733 = linalg.transpose ins(%__auto.model.model.layers.16.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1218 = linalg.matmul ins(%collapsed_731, %transposed_733 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_734 = tensor.expand_shape %1218 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1219 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1217, %expanded_734 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.16.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.16.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_735 = linalg.transpose ins(%__auto.model.model.layers.16.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_736 = tensor.collapse_shape %1219 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1220 = linalg.matmul ins(%collapsed_736, %transposed_735 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_737 = tensor.expand_shape %1220 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1221 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1207, %expanded_737 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1222 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1221 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1223 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1222 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1224 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1223 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1225 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1224 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1226 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1225 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1227 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1221, %1226 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.17.input_layernorm.weight = util.global.load @__auto.model.model.layers.17.input_layernorm.weight : tensor<576xf32>
    %1228 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.17.input_layernorm.weight, %1227 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.17.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.17.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_738 = linalg.transpose ins(%__auto.model.model.layers.17.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_739 = tensor.collapse_shape %1228 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1229 = linalg.matmul ins(%collapsed_739, %transposed_738 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_740 = tensor.expand_shape %1229 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_741 = linalg.transpose ins(%expanded_740 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.17.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.17.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_742 = linalg.transpose ins(%__auto.model.model.layers.17.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1230 = linalg.matmul ins(%collapsed_739, %transposed_742 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_743 = tensor.expand_shape %1230 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_744 = linalg.transpose ins(%expanded_743 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.17.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.17.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_745 = linalg.transpose ins(%__auto.model.model.layers.17.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1231 = linalg.matmul ins(%collapsed_739, %transposed_745 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_746 = tensor.expand_shape %1231 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_747 = linalg.transpose ins(%expanded_746 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %1232 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_741, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_748 = tensor.extract_slice %transposed_741[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_749 = tensor.extract_slice %transposed_741[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %1233 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_749 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_750 = tensor.concat dim(3) %1233, %extracted_slice_748 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %1234 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_750, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1235 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1232, %1234 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1236 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_744, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_751 = tensor.extract_slice %transposed_744[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_752 = tensor.extract_slice %transposed_744[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %1237 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_752 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_753 = tensor.concat dim(3) %1237, %extracted_slice_751 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %1238 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_753, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %1239 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1236, %1238 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_754 = tensor.concat dim(2) %73, %1239 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_755 = tensor.concat dim(2) %75, %transposed_747 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %1240 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_754 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_756 = tensor.collapse_shape %1240 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %1241 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_755 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %1242 = affine.apply #map15()[%dim_43]
    %1243 = tensor.empty(%1242) : tensor<1x9x64x?xf32>
    %transposed_757 = linalg.transpose ins(%collapsed_756 : tensor<1x9x?x64xf32>) outs(%1243 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_758 = tensor.collapse_shape %1235 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_759 = tensor.collapse_shape %transposed_757 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %1244 = tensor.empty(%1242) : tensor<9x1x?xf32>
    %1245 = linalg.fill ins(%cst : f32) outs(%1244 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %1246 = linalg.batch_matmul ins(%collapsed_758, %collapsed_759 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%1245 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_760 = tensor.expand_shape %1246 [[0, 1], [2], [3]] output_shape [1, 9, 1, %1242] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %1247 = tensor.empty(%1242) : tensor<1x9x1x?xf32>
    %1248 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_760 : tensor<1x9x1x?xf32>) outs(%1247 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1249 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1248, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%1247 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1250:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1249 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_761 = tensor.expand_shape %1250#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %1251 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1249, %expanded_761 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1247 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1252 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1251 : tensor<1x9x1x?xf32>) outs(%1247 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1253 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1252 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %1254 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1252, %1253 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1247 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_762 = tensor.collapse_shape %1254 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_763 = tensor.collapse_shape %1241 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %1255 = affine.apply #map15()[%dim_44]
    %1256 = arith.index_cast %1242 : index to i64
    %1257 = arith.index_cast %1255 : index to i64
    %1258 = arith.cmpi eq, %1256, %1257 : i64
    cf.assert %1258, "mismatching contracting dimension"
    %1259 = linalg.batch_matmul ins(%collapsed_762, %collapsed_763 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_764 = tensor.expand_shape %1259 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_765 = linalg.transpose ins(%expanded_764 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.17.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.17.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_766 = linalg.transpose ins(%__auto.model.model.layers.17.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_767 = tensor.collapse_shape %transposed_765 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %1260 = linalg.matmul ins(%collapsed_767, %transposed_766 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_768 = tensor.expand_shape %1260 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1261 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1221, %expanded_768 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1262 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1261 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1263 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1262 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1264 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1263 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1265 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1264 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1266 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1265 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1267 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1261, %1266 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.17.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.17.post_attention_layernorm.weight : tensor<576xf32>
    %1268 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.17.post_attention_layernorm.weight, %1267 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.17.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.17.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_769 = linalg.transpose ins(%__auto.model.model.layers.17.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_770 = tensor.collapse_shape %1268 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1269 = linalg.matmul ins(%collapsed_770, %transposed_769 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_771 = tensor.expand_shape %1269 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1270 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_771 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1271 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_771, %1270 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.17.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.17.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_772 = linalg.transpose ins(%__auto.model.model.layers.17.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1272 = linalg.matmul ins(%collapsed_770, %transposed_772 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_773 = tensor.expand_shape %1272 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1273 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1271, %expanded_773 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.17.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.17.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_774 = linalg.transpose ins(%__auto.model.model.layers.17.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_775 = tensor.collapse_shape %1273 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1274 = linalg.matmul ins(%collapsed_775, %transposed_774 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_776 = tensor.expand_shape %1274 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1275 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1261, %expanded_776 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1276 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1275 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1277 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1276 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1278 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1277 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1279 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1278 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1280 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1279 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1281 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1275, %1280 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.18.input_layernorm.weight = util.global.load @__auto.model.model.layers.18.input_layernorm.weight : tensor<576xf32>
    %1282 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.18.input_layernorm.weight, %1281 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.18.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.18.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_777 = linalg.transpose ins(%__auto.model.model.layers.18.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_778 = tensor.collapse_shape %1282 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1283 = linalg.matmul ins(%collapsed_778, %transposed_777 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_779 = tensor.expand_shape %1283 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_780 = linalg.transpose ins(%expanded_779 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.18.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.18.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_781 = linalg.transpose ins(%__auto.model.model.layers.18.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1284 = linalg.matmul ins(%collapsed_778, %transposed_781 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_782 = tensor.expand_shape %1284 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_783 = linalg.transpose ins(%expanded_782 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.18.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.18.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_784 = linalg.transpose ins(%__auto.model.model.layers.18.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1285 = linalg.matmul ins(%collapsed_778, %transposed_784 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_785 = tensor.expand_shape %1285 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_786 = linalg.transpose ins(%expanded_785 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %1286 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_780, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_787 = tensor.extract_slice %transposed_780[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_788 = tensor.extract_slice %transposed_780[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %1287 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_788 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_789 = tensor.concat dim(3) %1287, %extracted_slice_787 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %1288 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_789, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1289 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1286, %1288 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1290 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_783, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_790 = tensor.extract_slice %transposed_783[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_791 = tensor.extract_slice %transposed_783[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %1291 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_791 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_792 = tensor.concat dim(3) %1291, %extracted_slice_790 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %1292 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_792, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %1293 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1290, %1292 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_793 = tensor.concat dim(2) %77, %1293 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_794 = tensor.concat dim(2) %79, %transposed_786 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %1294 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_793 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_795 = tensor.collapse_shape %1294 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %1295 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_794 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %1296 = affine.apply #map15()[%dim_45]
    %1297 = tensor.empty(%1296) : tensor<1x9x64x?xf32>
    %transposed_796 = linalg.transpose ins(%collapsed_795 : tensor<1x9x?x64xf32>) outs(%1297 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_797 = tensor.collapse_shape %1289 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_798 = tensor.collapse_shape %transposed_796 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %1298 = tensor.empty(%1296) : tensor<9x1x?xf32>
    %1299 = linalg.fill ins(%cst : f32) outs(%1298 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %1300 = linalg.batch_matmul ins(%collapsed_797, %collapsed_798 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%1299 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_799 = tensor.expand_shape %1300 [[0, 1], [2], [3]] output_shape [1, 9, 1, %1296] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %1301 = tensor.empty(%1296) : tensor<1x9x1x?xf32>
    %1302 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_799 : tensor<1x9x1x?xf32>) outs(%1301 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1303 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1302, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%1301 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1304:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1303 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_800 = tensor.expand_shape %1304#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %1305 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1303, %expanded_800 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1301 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1306 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1305 : tensor<1x9x1x?xf32>) outs(%1301 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1307 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1306 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %1308 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1306, %1307 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1301 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_801 = tensor.collapse_shape %1308 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_802 = tensor.collapse_shape %1295 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %1309 = affine.apply #map15()[%dim_46]
    %1310 = arith.index_cast %1296 : index to i64
    %1311 = arith.index_cast %1309 : index to i64
    %1312 = arith.cmpi eq, %1310, %1311 : i64
    cf.assert %1312, "mismatching contracting dimension"
    %1313 = linalg.batch_matmul ins(%collapsed_801, %collapsed_802 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_803 = tensor.expand_shape %1313 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_804 = linalg.transpose ins(%expanded_803 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.18.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.18.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_805 = linalg.transpose ins(%__auto.model.model.layers.18.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_806 = tensor.collapse_shape %transposed_804 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %1314 = linalg.matmul ins(%collapsed_806, %transposed_805 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_807 = tensor.expand_shape %1314 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1315 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1275, %expanded_807 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1316 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1315 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1317 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1316 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1318 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1317 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1319 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1318 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1320 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1319 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1321 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1315, %1320 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.18.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.18.post_attention_layernorm.weight : tensor<576xf32>
    %1322 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.18.post_attention_layernorm.weight, %1321 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.18.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.18.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_808 = linalg.transpose ins(%__auto.model.model.layers.18.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_809 = tensor.collapse_shape %1322 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1323 = linalg.matmul ins(%collapsed_809, %transposed_808 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_810 = tensor.expand_shape %1323 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1324 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_810 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1325 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_810, %1324 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.18.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.18.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_811 = linalg.transpose ins(%__auto.model.model.layers.18.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1326 = linalg.matmul ins(%collapsed_809, %transposed_811 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_812 = tensor.expand_shape %1326 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1327 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1325, %expanded_812 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.18.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.18.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_813 = linalg.transpose ins(%__auto.model.model.layers.18.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_814 = tensor.collapse_shape %1327 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1328 = linalg.matmul ins(%collapsed_814, %transposed_813 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_815 = tensor.expand_shape %1328 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1329 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1315, %expanded_815 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1330 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1329 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1331 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1330 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1332 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1331 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1333 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1332 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1334 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1333 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1335 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1329, %1334 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.19.input_layernorm.weight = util.global.load @__auto.model.model.layers.19.input_layernorm.weight : tensor<576xf32>
    %1336 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.19.input_layernorm.weight, %1335 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.19.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.19.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_816 = linalg.transpose ins(%__auto.model.model.layers.19.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_817 = tensor.collapse_shape %1336 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1337 = linalg.matmul ins(%collapsed_817, %transposed_816 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_818 = tensor.expand_shape %1337 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_819 = linalg.transpose ins(%expanded_818 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.19.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.19.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_820 = linalg.transpose ins(%__auto.model.model.layers.19.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1338 = linalg.matmul ins(%collapsed_817, %transposed_820 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_821 = tensor.expand_shape %1338 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_822 = linalg.transpose ins(%expanded_821 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.19.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.19.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_823 = linalg.transpose ins(%__auto.model.model.layers.19.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1339 = linalg.matmul ins(%collapsed_817, %transposed_823 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_824 = tensor.expand_shape %1339 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_825 = linalg.transpose ins(%expanded_824 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %1340 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_819, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_826 = tensor.extract_slice %transposed_819[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_827 = tensor.extract_slice %transposed_819[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %1341 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_827 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_828 = tensor.concat dim(3) %1341, %extracted_slice_826 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %1342 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_828, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1343 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1340, %1342 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1344 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_822, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_829 = tensor.extract_slice %transposed_822[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_830 = tensor.extract_slice %transposed_822[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %1345 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_830 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_831 = tensor.concat dim(3) %1345, %extracted_slice_829 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %1346 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_831, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %1347 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1344, %1346 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_832 = tensor.concat dim(2) %81, %1347 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_833 = tensor.concat dim(2) %83, %transposed_825 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %1348 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_832 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_834 = tensor.collapse_shape %1348 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %1349 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_833 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %1350 = affine.apply #map15()[%dim_47]
    %1351 = tensor.empty(%1350) : tensor<1x9x64x?xf32>
    %transposed_835 = linalg.transpose ins(%collapsed_834 : tensor<1x9x?x64xf32>) outs(%1351 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_836 = tensor.collapse_shape %1343 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_837 = tensor.collapse_shape %transposed_835 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %1352 = tensor.empty(%1350) : tensor<9x1x?xf32>
    %1353 = linalg.fill ins(%cst : f32) outs(%1352 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %1354 = linalg.batch_matmul ins(%collapsed_836, %collapsed_837 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%1353 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_838 = tensor.expand_shape %1354 [[0, 1], [2], [3]] output_shape [1, 9, 1, %1350] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %1355 = tensor.empty(%1350) : tensor<1x9x1x?xf32>
    %1356 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_838 : tensor<1x9x1x?xf32>) outs(%1355 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1357 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1356, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%1355 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1358:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1357 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_839 = tensor.expand_shape %1358#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %1359 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1357, %expanded_839 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1355 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1360 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1359 : tensor<1x9x1x?xf32>) outs(%1355 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1361 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1360 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %1362 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1360, %1361 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1355 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_840 = tensor.collapse_shape %1362 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_841 = tensor.collapse_shape %1349 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %1363 = affine.apply #map15()[%dim_48]
    %1364 = arith.index_cast %1350 : index to i64
    %1365 = arith.index_cast %1363 : index to i64
    %1366 = arith.cmpi eq, %1364, %1365 : i64
    cf.assert %1366, "mismatching contracting dimension"
    %1367 = linalg.batch_matmul ins(%collapsed_840, %collapsed_841 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_842 = tensor.expand_shape %1367 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_843 = linalg.transpose ins(%expanded_842 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.19.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.19.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_844 = linalg.transpose ins(%__auto.model.model.layers.19.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_845 = tensor.collapse_shape %transposed_843 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %1368 = linalg.matmul ins(%collapsed_845, %transposed_844 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_846 = tensor.expand_shape %1368 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1369 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1329, %expanded_846 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1370 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1369 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1371 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1370 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1372 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1371 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1373 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1372 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1374 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1373 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1375 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1369, %1374 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.19.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.19.post_attention_layernorm.weight : tensor<576xf32>
    %1376 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.19.post_attention_layernorm.weight, %1375 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.19.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.19.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_847 = linalg.transpose ins(%__auto.model.model.layers.19.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_848 = tensor.collapse_shape %1376 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1377 = linalg.matmul ins(%collapsed_848, %transposed_847 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_849 = tensor.expand_shape %1377 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1378 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_849 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1379 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_849, %1378 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.19.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.19.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_850 = linalg.transpose ins(%__auto.model.model.layers.19.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1380 = linalg.matmul ins(%collapsed_848, %transposed_850 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_851 = tensor.expand_shape %1380 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1381 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1379, %expanded_851 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.19.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.19.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_852 = linalg.transpose ins(%__auto.model.model.layers.19.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_853 = tensor.collapse_shape %1381 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1382 = linalg.matmul ins(%collapsed_853, %transposed_852 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_854 = tensor.expand_shape %1382 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1383 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1369, %expanded_854 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1384 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1383 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1385 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1384 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1386 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1385 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1387 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1386 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1388 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1387 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1389 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1383, %1388 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.20.input_layernorm.weight = util.global.load @__auto.model.model.layers.20.input_layernorm.weight : tensor<576xf32>
    %1390 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.20.input_layernorm.weight, %1389 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.20.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.20.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_855 = linalg.transpose ins(%__auto.model.model.layers.20.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_856 = tensor.collapse_shape %1390 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1391 = linalg.matmul ins(%collapsed_856, %transposed_855 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_857 = tensor.expand_shape %1391 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_858 = linalg.transpose ins(%expanded_857 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.20.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.20.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_859 = linalg.transpose ins(%__auto.model.model.layers.20.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1392 = linalg.matmul ins(%collapsed_856, %transposed_859 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_860 = tensor.expand_shape %1392 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_861 = linalg.transpose ins(%expanded_860 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.20.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.20.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_862 = linalg.transpose ins(%__auto.model.model.layers.20.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1393 = linalg.matmul ins(%collapsed_856, %transposed_862 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_863 = tensor.expand_shape %1393 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_864 = linalg.transpose ins(%expanded_863 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %1394 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_858, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_865 = tensor.extract_slice %transposed_858[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_866 = tensor.extract_slice %transposed_858[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %1395 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_866 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_867 = tensor.concat dim(3) %1395, %extracted_slice_865 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %1396 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_867, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1397 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1394, %1396 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1398 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_861, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_868 = tensor.extract_slice %transposed_861[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_869 = tensor.extract_slice %transposed_861[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %1399 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_869 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_870 = tensor.concat dim(3) %1399, %extracted_slice_868 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %1400 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_870, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %1401 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1398, %1400 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_871 = tensor.concat dim(2) %85, %1401 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_872 = tensor.concat dim(2) %87, %transposed_864 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %1402 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_871 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_873 = tensor.collapse_shape %1402 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %1403 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_872 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %1404 = affine.apply #map15()[%dim_49]
    %1405 = tensor.empty(%1404) : tensor<1x9x64x?xf32>
    %transposed_874 = linalg.transpose ins(%collapsed_873 : tensor<1x9x?x64xf32>) outs(%1405 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_875 = tensor.collapse_shape %1397 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_876 = tensor.collapse_shape %transposed_874 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %1406 = tensor.empty(%1404) : tensor<9x1x?xf32>
    %1407 = linalg.fill ins(%cst : f32) outs(%1406 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %1408 = linalg.batch_matmul ins(%collapsed_875, %collapsed_876 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%1407 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_877 = tensor.expand_shape %1408 [[0, 1], [2], [3]] output_shape [1, 9, 1, %1404] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %1409 = tensor.empty(%1404) : tensor<1x9x1x?xf32>
    %1410 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_877 : tensor<1x9x1x?xf32>) outs(%1409 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1411 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1410, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%1409 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1412:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1411 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_878 = tensor.expand_shape %1412#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %1413 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1411, %expanded_878 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1409 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1414 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1413 : tensor<1x9x1x?xf32>) outs(%1409 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1415 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1414 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %1416 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1414, %1415 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1409 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_879 = tensor.collapse_shape %1416 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_880 = tensor.collapse_shape %1403 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %1417 = affine.apply #map15()[%dim_50]
    %1418 = arith.index_cast %1404 : index to i64
    %1419 = arith.index_cast %1417 : index to i64
    %1420 = arith.cmpi eq, %1418, %1419 : i64
    cf.assert %1420, "mismatching contracting dimension"
    %1421 = linalg.batch_matmul ins(%collapsed_879, %collapsed_880 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_881 = tensor.expand_shape %1421 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_882 = linalg.transpose ins(%expanded_881 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.20.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.20.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_883 = linalg.transpose ins(%__auto.model.model.layers.20.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_884 = tensor.collapse_shape %transposed_882 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %1422 = linalg.matmul ins(%collapsed_884, %transposed_883 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_885 = tensor.expand_shape %1422 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1423 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1383, %expanded_885 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1424 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1423 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1425 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1424 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1426 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1425 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1427 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1426 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1428 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1427 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1429 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1423, %1428 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.20.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.20.post_attention_layernorm.weight : tensor<576xf32>
    %1430 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.20.post_attention_layernorm.weight, %1429 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.20.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.20.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_886 = linalg.transpose ins(%__auto.model.model.layers.20.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_887 = tensor.collapse_shape %1430 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1431 = linalg.matmul ins(%collapsed_887, %transposed_886 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_888 = tensor.expand_shape %1431 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1432 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_888 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1433 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_888, %1432 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.20.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.20.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_889 = linalg.transpose ins(%__auto.model.model.layers.20.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1434 = linalg.matmul ins(%collapsed_887, %transposed_889 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_890 = tensor.expand_shape %1434 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1435 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1433, %expanded_890 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.20.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.20.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_891 = linalg.transpose ins(%__auto.model.model.layers.20.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_892 = tensor.collapse_shape %1435 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1436 = linalg.matmul ins(%collapsed_892, %transposed_891 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_893 = tensor.expand_shape %1436 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1437 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1423, %expanded_893 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1438 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1437 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1439 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1438 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1440 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1439 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1441 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1440 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1442 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1441 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1443 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1437, %1442 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.21.input_layernorm.weight = util.global.load @__auto.model.model.layers.21.input_layernorm.weight : tensor<576xf32>
    %1444 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.21.input_layernorm.weight, %1443 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.21.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.21.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_894 = linalg.transpose ins(%__auto.model.model.layers.21.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_895 = tensor.collapse_shape %1444 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1445 = linalg.matmul ins(%collapsed_895, %transposed_894 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_896 = tensor.expand_shape %1445 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_897 = linalg.transpose ins(%expanded_896 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.21.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.21.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_898 = linalg.transpose ins(%__auto.model.model.layers.21.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1446 = linalg.matmul ins(%collapsed_895, %transposed_898 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_899 = tensor.expand_shape %1446 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_900 = linalg.transpose ins(%expanded_899 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.21.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.21.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_901 = linalg.transpose ins(%__auto.model.model.layers.21.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1447 = linalg.matmul ins(%collapsed_895, %transposed_901 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_902 = tensor.expand_shape %1447 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_903 = linalg.transpose ins(%expanded_902 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %1448 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_897, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_904 = tensor.extract_slice %transposed_897[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_905 = tensor.extract_slice %transposed_897[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %1449 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_905 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_906 = tensor.concat dim(3) %1449, %extracted_slice_904 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %1450 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_906, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1451 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1448, %1450 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1452 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_900, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_907 = tensor.extract_slice %transposed_900[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_908 = tensor.extract_slice %transposed_900[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %1453 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_908 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_909 = tensor.concat dim(3) %1453, %extracted_slice_907 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %1454 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_909, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %1455 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1452, %1454 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_910 = tensor.concat dim(2) %89, %1455 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_911 = tensor.concat dim(2) %91, %transposed_903 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %1456 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_910 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_912 = tensor.collapse_shape %1456 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %1457 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_911 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %1458 = affine.apply #map15()[%dim_51]
    %1459 = tensor.empty(%1458) : tensor<1x9x64x?xf32>
    %transposed_913 = linalg.transpose ins(%collapsed_912 : tensor<1x9x?x64xf32>) outs(%1459 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_914 = tensor.collapse_shape %1451 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_915 = tensor.collapse_shape %transposed_913 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %1460 = tensor.empty(%1458) : tensor<9x1x?xf32>
    %1461 = linalg.fill ins(%cst : f32) outs(%1460 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %1462 = linalg.batch_matmul ins(%collapsed_914, %collapsed_915 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%1461 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_916 = tensor.expand_shape %1462 [[0, 1], [2], [3]] output_shape [1, 9, 1, %1458] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %1463 = tensor.empty(%1458) : tensor<1x9x1x?xf32>
    %1464 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_916 : tensor<1x9x1x?xf32>) outs(%1463 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1465 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1464, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%1463 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1466:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1465 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_917 = tensor.expand_shape %1466#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %1467 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1465, %expanded_917 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1463 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1468 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1467 : tensor<1x9x1x?xf32>) outs(%1463 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1469 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1468 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %1470 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1468, %1469 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1463 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_918 = tensor.collapse_shape %1470 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_919 = tensor.collapse_shape %1457 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %1471 = affine.apply #map15()[%dim_52]
    %1472 = arith.index_cast %1458 : index to i64
    %1473 = arith.index_cast %1471 : index to i64
    %1474 = arith.cmpi eq, %1472, %1473 : i64
    cf.assert %1474, "mismatching contracting dimension"
    %1475 = linalg.batch_matmul ins(%collapsed_918, %collapsed_919 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_920 = tensor.expand_shape %1475 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_921 = linalg.transpose ins(%expanded_920 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.21.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.21.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_922 = linalg.transpose ins(%__auto.model.model.layers.21.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_923 = tensor.collapse_shape %transposed_921 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %1476 = linalg.matmul ins(%collapsed_923, %transposed_922 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_924 = tensor.expand_shape %1476 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1477 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1437, %expanded_924 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1478 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1477 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1479 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1478 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1480 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1479 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1481 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1480 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1482 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1481 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1483 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1477, %1482 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.21.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.21.post_attention_layernorm.weight : tensor<576xf32>
    %1484 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.21.post_attention_layernorm.weight, %1483 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.21.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.21.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_925 = linalg.transpose ins(%__auto.model.model.layers.21.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_926 = tensor.collapse_shape %1484 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1485 = linalg.matmul ins(%collapsed_926, %transposed_925 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_927 = tensor.expand_shape %1485 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1486 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_927 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1487 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_927, %1486 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.21.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.21.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_928 = linalg.transpose ins(%__auto.model.model.layers.21.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1488 = linalg.matmul ins(%collapsed_926, %transposed_928 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_929 = tensor.expand_shape %1488 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1489 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1487, %expanded_929 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.21.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.21.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_930 = linalg.transpose ins(%__auto.model.model.layers.21.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_931 = tensor.collapse_shape %1489 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1490 = linalg.matmul ins(%collapsed_931, %transposed_930 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_932 = tensor.expand_shape %1490 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1491 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1477, %expanded_932 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1492 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1491 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1493 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1492 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1494 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1493 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1495 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1494 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1496 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1495 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1497 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1491, %1496 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.22.input_layernorm.weight = util.global.load @__auto.model.model.layers.22.input_layernorm.weight : tensor<576xf32>
    %1498 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.22.input_layernorm.weight, %1497 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.22.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.22.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_933 = linalg.transpose ins(%__auto.model.model.layers.22.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_934 = tensor.collapse_shape %1498 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1499 = linalg.matmul ins(%collapsed_934, %transposed_933 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_935 = tensor.expand_shape %1499 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_936 = linalg.transpose ins(%expanded_935 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.22.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.22.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_937 = linalg.transpose ins(%__auto.model.model.layers.22.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1500 = linalg.matmul ins(%collapsed_934, %transposed_937 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_938 = tensor.expand_shape %1500 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_939 = linalg.transpose ins(%expanded_938 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.22.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.22.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_940 = linalg.transpose ins(%__auto.model.model.layers.22.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1501 = linalg.matmul ins(%collapsed_934, %transposed_940 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_941 = tensor.expand_shape %1501 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_942 = linalg.transpose ins(%expanded_941 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %1502 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_936, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_943 = tensor.extract_slice %transposed_936[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_944 = tensor.extract_slice %transposed_936[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %1503 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_944 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_945 = tensor.concat dim(3) %1503, %extracted_slice_943 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %1504 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_945, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1505 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1502, %1504 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1506 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_939, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_946 = tensor.extract_slice %transposed_939[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_947 = tensor.extract_slice %transposed_939[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %1507 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_947 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_948 = tensor.concat dim(3) %1507, %extracted_slice_946 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %1508 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_948, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %1509 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1506, %1508 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_949 = tensor.concat dim(2) %93, %1509 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_950 = tensor.concat dim(2) %95, %transposed_942 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %1510 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_949 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_951 = tensor.collapse_shape %1510 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %1511 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_950 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %1512 = affine.apply #map15()[%dim_53]
    %1513 = tensor.empty(%1512) : tensor<1x9x64x?xf32>
    %transposed_952 = linalg.transpose ins(%collapsed_951 : tensor<1x9x?x64xf32>) outs(%1513 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_953 = tensor.collapse_shape %1505 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_954 = tensor.collapse_shape %transposed_952 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %1514 = tensor.empty(%1512) : tensor<9x1x?xf32>
    %1515 = linalg.fill ins(%cst : f32) outs(%1514 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %1516 = linalg.batch_matmul ins(%collapsed_953, %collapsed_954 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%1515 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_955 = tensor.expand_shape %1516 [[0, 1], [2], [3]] output_shape [1, 9, 1, %1512] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %1517 = tensor.empty(%1512) : tensor<1x9x1x?xf32>
    %1518 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_955 : tensor<1x9x1x?xf32>) outs(%1517 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1519 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1518, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%1517 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1520:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1519 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_956 = tensor.expand_shape %1520#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %1521 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1519, %expanded_956 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1517 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1522 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1521 : tensor<1x9x1x?xf32>) outs(%1517 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1523 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1522 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %1524 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1522, %1523 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1517 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_957 = tensor.collapse_shape %1524 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_958 = tensor.collapse_shape %1511 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %1525 = affine.apply #map15()[%dim_54]
    %1526 = arith.index_cast %1512 : index to i64
    %1527 = arith.index_cast %1525 : index to i64
    %1528 = arith.cmpi eq, %1526, %1527 : i64
    cf.assert %1528, "mismatching contracting dimension"
    %1529 = linalg.batch_matmul ins(%collapsed_957, %collapsed_958 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_959 = tensor.expand_shape %1529 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_960 = linalg.transpose ins(%expanded_959 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.22.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.22.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_961 = linalg.transpose ins(%__auto.model.model.layers.22.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_962 = tensor.collapse_shape %transposed_960 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %1530 = linalg.matmul ins(%collapsed_962, %transposed_961 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_963 = tensor.expand_shape %1530 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1531 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1491, %expanded_963 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1532 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1531 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1533 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1532 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1534 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1533 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1535 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1534 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1536 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1535 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1537 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1531, %1536 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.22.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.22.post_attention_layernorm.weight : tensor<576xf32>
    %1538 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.22.post_attention_layernorm.weight, %1537 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.22.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.22.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_964 = linalg.transpose ins(%__auto.model.model.layers.22.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_965 = tensor.collapse_shape %1538 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1539 = linalg.matmul ins(%collapsed_965, %transposed_964 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_966 = tensor.expand_shape %1539 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1540 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_966 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1541 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_966, %1540 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.22.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.22.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_967 = linalg.transpose ins(%__auto.model.model.layers.22.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1542 = linalg.matmul ins(%collapsed_965, %transposed_967 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_968 = tensor.expand_shape %1542 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1543 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1541, %expanded_968 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.22.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.22.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_969 = linalg.transpose ins(%__auto.model.model.layers.22.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_970 = tensor.collapse_shape %1543 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1544 = linalg.matmul ins(%collapsed_970, %transposed_969 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_971 = tensor.expand_shape %1544 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1545 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1531, %expanded_971 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1546 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1545 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1547 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1546 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1548 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1547 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1549 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1548 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1550 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1549 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1551 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1545, %1550 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.23.input_layernorm.weight = util.global.load @__auto.model.model.layers.23.input_layernorm.weight : tensor<576xf32>
    %1552 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.23.input_layernorm.weight, %1551 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.23.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.23.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_972 = linalg.transpose ins(%__auto.model.model.layers.23.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_973 = tensor.collapse_shape %1552 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1553 = linalg.matmul ins(%collapsed_973, %transposed_972 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_974 = tensor.expand_shape %1553 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_975 = linalg.transpose ins(%expanded_974 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.23.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.23.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_976 = linalg.transpose ins(%__auto.model.model.layers.23.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1554 = linalg.matmul ins(%collapsed_973, %transposed_976 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_977 = tensor.expand_shape %1554 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_978 = linalg.transpose ins(%expanded_977 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.23.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.23.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_979 = linalg.transpose ins(%__auto.model.model.layers.23.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1555 = linalg.matmul ins(%collapsed_973, %transposed_979 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_980 = tensor.expand_shape %1555 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_981 = linalg.transpose ins(%expanded_980 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %1556 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_975, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_982 = tensor.extract_slice %transposed_975[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_983 = tensor.extract_slice %transposed_975[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %1557 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_983 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_984 = tensor.concat dim(3) %1557, %extracted_slice_982 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %1558 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_984, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1559 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1556, %1558 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1560 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_978, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_985 = tensor.extract_slice %transposed_978[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_986 = tensor.extract_slice %transposed_978[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %1561 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_986 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_987 = tensor.concat dim(3) %1561, %extracted_slice_985 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %1562 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_987, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %1563 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1560, %1562 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_988 = tensor.concat dim(2) %97, %1563 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_989 = tensor.concat dim(2) %99, %transposed_981 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %1564 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_988 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_990 = tensor.collapse_shape %1564 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %1565 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_989 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %1566 = affine.apply #map15()[%dim_55]
    %1567 = tensor.empty(%1566) : tensor<1x9x64x?xf32>
    %transposed_991 = linalg.transpose ins(%collapsed_990 : tensor<1x9x?x64xf32>) outs(%1567 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_992 = tensor.collapse_shape %1559 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_993 = tensor.collapse_shape %transposed_991 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %1568 = tensor.empty(%1566) : tensor<9x1x?xf32>
    %1569 = linalg.fill ins(%cst : f32) outs(%1568 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %1570 = linalg.batch_matmul ins(%collapsed_992, %collapsed_993 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%1569 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_994 = tensor.expand_shape %1570 [[0, 1], [2], [3]] output_shape [1, 9, 1, %1566] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %1571 = tensor.empty(%1566) : tensor<1x9x1x?xf32>
    %1572 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_994 : tensor<1x9x1x?xf32>) outs(%1571 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1573 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1572, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%1571 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1574:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1573 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_995 = tensor.expand_shape %1574#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %1575 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1573, %expanded_995 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1571 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1576 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1575 : tensor<1x9x1x?xf32>) outs(%1571 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1577 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1576 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %1578 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1576, %1577 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1571 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_996 = tensor.collapse_shape %1578 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_997 = tensor.collapse_shape %1565 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %1579 = affine.apply #map15()[%dim_56]
    %1580 = arith.index_cast %1566 : index to i64
    %1581 = arith.index_cast %1579 : index to i64
    %1582 = arith.cmpi eq, %1580, %1581 : i64
    cf.assert %1582, "mismatching contracting dimension"
    %1583 = linalg.batch_matmul ins(%collapsed_996, %collapsed_997 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_998 = tensor.expand_shape %1583 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_999 = linalg.transpose ins(%expanded_998 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.23.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.23.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_1000 = linalg.transpose ins(%__auto.model.model.layers.23.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_1001 = tensor.collapse_shape %transposed_999 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %1584 = linalg.matmul ins(%collapsed_1001, %transposed_1000 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1002 = tensor.expand_shape %1584 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1585 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1545, %expanded_1002 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1586 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1585 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1587 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1586 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1588 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1587 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1589 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1588 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1590 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1589 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1591 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1585, %1590 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.23.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.23.post_attention_layernorm.weight : tensor<576xf32>
    %1592 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.23.post_attention_layernorm.weight, %1591 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.23.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.23.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_1003 = linalg.transpose ins(%__auto.model.model.layers.23.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_1004 = tensor.collapse_shape %1592 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1593 = linalg.matmul ins(%collapsed_1004, %transposed_1003 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_1005 = tensor.expand_shape %1593 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1594 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_1005 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1595 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_1005, %1594 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.23.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.23.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_1006 = linalg.transpose ins(%__auto.model.model.layers.23.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1596 = linalg.matmul ins(%collapsed_1004, %transposed_1006 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_1007 = tensor.expand_shape %1596 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1597 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1595, %expanded_1007 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.23.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.23.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_1008 = linalg.transpose ins(%__auto.model.model.layers.23.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_1009 = tensor.collapse_shape %1597 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1598 = linalg.matmul ins(%collapsed_1009, %transposed_1008 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1010 = tensor.expand_shape %1598 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1599 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1585, %expanded_1010 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1600 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1599 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1601 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1600 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1602 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1601 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1603 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1602 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1604 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1603 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1605 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1599, %1604 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.24.input_layernorm.weight = util.global.load @__auto.model.model.layers.24.input_layernorm.weight : tensor<576xf32>
    %1606 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.24.input_layernorm.weight, %1605 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.24.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.24.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_1011 = linalg.transpose ins(%__auto.model.model.layers.24.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_1012 = tensor.collapse_shape %1606 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1607 = linalg.matmul ins(%collapsed_1012, %transposed_1011 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1013 = tensor.expand_shape %1607 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_1014 = linalg.transpose ins(%expanded_1013 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.24.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.24.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_1015 = linalg.transpose ins(%__auto.model.model.layers.24.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1608 = linalg.matmul ins(%collapsed_1012, %transposed_1015 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_1016 = tensor.expand_shape %1608 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_1017 = linalg.transpose ins(%expanded_1016 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.24.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.24.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_1018 = linalg.transpose ins(%__auto.model.model.layers.24.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1609 = linalg.matmul ins(%collapsed_1012, %transposed_1018 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_1019 = tensor.expand_shape %1609 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_1020 = linalg.transpose ins(%expanded_1019 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %1610 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_1014, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_1021 = tensor.extract_slice %transposed_1014[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_1022 = tensor.extract_slice %transposed_1014[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %1611 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1022 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_1023 = tensor.concat dim(3) %1611, %extracted_slice_1021 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %1612 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_1023, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1613 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1610, %1612 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1614 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_1017, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_1024 = tensor.extract_slice %transposed_1017[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_1025 = tensor.extract_slice %transposed_1017[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %1615 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1025 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_1026 = tensor.concat dim(3) %1615, %extracted_slice_1024 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %1616 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_1026, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %1617 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1614, %1616 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_1027 = tensor.concat dim(2) %101, %1617 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_1028 = tensor.concat dim(2) %103, %transposed_1020 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %1618 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_1027 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_1029 = tensor.collapse_shape %1618 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %1619 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_1028 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %1620 = affine.apply #map15()[%dim_57]
    %1621 = tensor.empty(%1620) : tensor<1x9x64x?xf32>
    %transposed_1030 = linalg.transpose ins(%collapsed_1029 : tensor<1x9x?x64xf32>) outs(%1621 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_1031 = tensor.collapse_shape %1613 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_1032 = tensor.collapse_shape %transposed_1030 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %1622 = tensor.empty(%1620) : tensor<9x1x?xf32>
    %1623 = linalg.fill ins(%cst : f32) outs(%1622 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %1624 = linalg.batch_matmul ins(%collapsed_1031, %collapsed_1032 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%1623 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_1033 = tensor.expand_shape %1624 [[0, 1], [2], [3]] output_shape [1, 9, 1, %1620] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %1625 = tensor.empty(%1620) : tensor<1x9x1x?xf32>
    %1626 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_1033 : tensor<1x9x1x?xf32>) outs(%1625 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1627 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1626, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%1625 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1628:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1627 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_1034 = tensor.expand_shape %1628#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %1629 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1627, %expanded_1034 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1625 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1630 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1629 : tensor<1x9x1x?xf32>) outs(%1625 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1631 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1630 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %1632 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1630, %1631 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1625 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_1035 = tensor.collapse_shape %1632 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_1036 = tensor.collapse_shape %1619 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %1633 = affine.apply #map15()[%dim_58]
    %1634 = arith.index_cast %1620 : index to i64
    %1635 = arith.index_cast %1633 : index to i64
    %1636 = arith.cmpi eq, %1634, %1635 : i64
    cf.assert %1636, "mismatching contracting dimension"
    %1637 = linalg.batch_matmul ins(%collapsed_1035, %collapsed_1036 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_1037 = tensor.expand_shape %1637 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_1038 = linalg.transpose ins(%expanded_1037 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.24.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.24.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_1039 = linalg.transpose ins(%__auto.model.model.layers.24.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_1040 = tensor.collapse_shape %transposed_1038 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %1638 = linalg.matmul ins(%collapsed_1040, %transposed_1039 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1041 = tensor.expand_shape %1638 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1639 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1599, %expanded_1041 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1640 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1639 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1641 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1640 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1642 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1641 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1643 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1642 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1644 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1643 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1645 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1639, %1644 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.24.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.24.post_attention_layernorm.weight : tensor<576xf32>
    %1646 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.24.post_attention_layernorm.weight, %1645 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.24.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.24.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_1042 = linalg.transpose ins(%__auto.model.model.layers.24.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_1043 = tensor.collapse_shape %1646 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1647 = linalg.matmul ins(%collapsed_1043, %transposed_1042 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_1044 = tensor.expand_shape %1647 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1648 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_1044 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1649 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_1044, %1648 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.24.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.24.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_1045 = linalg.transpose ins(%__auto.model.model.layers.24.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1650 = linalg.matmul ins(%collapsed_1043, %transposed_1045 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_1046 = tensor.expand_shape %1650 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1651 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1649, %expanded_1046 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.24.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.24.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_1047 = linalg.transpose ins(%__auto.model.model.layers.24.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_1048 = tensor.collapse_shape %1651 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1652 = linalg.matmul ins(%collapsed_1048, %transposed_1047 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1049 = tensor.expand_shape %1652 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1653 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1639, %expanded_1049 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1654 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1653 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1655 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1654 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1656 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1655 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1657 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1656 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1658 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1657 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1659 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1653, %1658 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.25.input_layernorm.weight = util.global.load @__auto.model.model.layers.25.input_layernorm.weight : tensor<576xf32>
    %1660 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.25.input_layernorm.weight, %1659 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.25.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.25.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_1050 = linalg.transpose ins(%__auto.model.model.layers.25.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_1051 = tensor.collapse_shape %1660 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1661 = linalg.matmul ins(%collapsed_1051, %transposed_1050 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1052 = tensor.expand_shape %1661 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_1053 = linalg.transpose ins(%expanded_1052 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.25.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.25.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_1054 = linalg.transpose ins(%__auto.model.model.layers.25.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1662 = linalg.matmul ins(%collapsed_1051, %transposed_1054 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_1055 = tensor.expand_shape %1662 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_1056 = linalg.transpose ins(%expanded_1055 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.25.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.25.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_1057 = linalg.transpose ins(%__auto.model.model.layers.25.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1663 = linalg.matmul ins(%collapsed_1051, %transposed_1057 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_1058 = tensor.expand_shape %1663 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_1059 = linalg.transpose ins(%expanded_1058 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %1664 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_1053, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_1060 = tensor.extract_slice %transposed_1053[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_1061 = tensor.extract_slice %transposed_1053[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %1665 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1061 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_1062 = tensor.concat dim(3) %1665, %extracted_slice_1060 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %1666 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_1062, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1667 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1664, %1666 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1668 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_1056, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_1063 = tensor.extract_slice %transposed_1056[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_1064 = tensor.extract_slice %transposed_1056[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %1669 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1064 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_1065 = tensor.concat dim(3) %1669, %extracted_slice_1063 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %1670 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_1065, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %1671 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1668, %1670 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_1066 = tensor.concat dim(2) %105, %1671 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_1067 = tensor.concat dim(2) %107, %transposed_1059 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %1672 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_1066 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_1068 = tensor.collapse_shape %1672 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %1673 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_1067 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %1674 = affine.apply #map15()[%dim_59]
    %1675 = tensor.empty(%1674) : tensor<1x9x64x?xf32>
    %transposed_1069 = linalg.transpose ins(%collapsed_1068 : tensor<1x9x?x64xf32>) outs(%1675 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_1070 = tensor.collapse_shape %1667 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_1071 = tensor.collapse_shape %transposed_1069 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %1676 = tensor.empty(%1674) : tensor<9x1x?xf32>
    %1677 = linalg.fill ins(%cst : f32) outs(%1676 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %1678 = linalg.batch_matmul ins(%collapsed_1070, %collapsed_1071 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%1677 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_1072 = tensor.expand_shape %1678 [[0, 1], [2], [3]] output_shape [1, 9, 1, %1674] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %1679 = tensor.empty(%1674) : tensor<1x9x1x?xf32>
    %1680 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_1072 : tensor<1x9x1x?xf32>) outs(%1679 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1681 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1680, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%1679 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1682:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1681 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_1073 = tensor.expand_shape %1682#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %1683 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1681, %expanded_1073 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1679 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1684 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1683 : tensor<1x9x1x?xf32>) outs(%1679 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1685 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1684 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %1686 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1684, %1685 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1679 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_1074 = tensor.collapse_shape %1686 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_1075 = tensor.collapse_shape %1673 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %1687 = affine.apply #map15()[%dim_60]
    %1688 = arith.index_cast %1674 : index to i64
    %1689 = arith.index_cast %1687 : index to i64
    %1690 = arith.cmpi eq, %1688, %1689 : i64
    cf.assert %1690, "mismatching contracting dimension"
    %1691 = linalg.batch_matmul ins(%collapsed_1074, %collapsed_1075 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_1076 = tensor.expand_shape %1691 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_1077 = linalg.transpose ins(%expanded_1076 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.25.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.25.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_1078 = linalg.transpose ins(%__auto.model.model.layers.25.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_1079 = tensor.collapse_shape %transposed_1077 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %1692 = linalg.matmul ins(%collapsed_1079, %transposed_1078 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1080 = tensor.expand_shape %1692 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1693 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1653, %expanded_1080 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1694 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1693 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1695 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1694 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1696 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1695 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1697 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1696 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1698 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1697 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1699 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1693, %1698 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.25.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.25.post_attention_layernorm.weight : tensor<576xf32>
    %1700 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.25.post_attention_layernorm.weight, %1699 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.25.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.25.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_1081 = linalg.transpose ins(%__auto.model.model.layers.25.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_1082 = tensor.collapse_shape %1700 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1701 = linalg.matmul ins(%collapsed_1082, %transposed_1081 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_1083 = tensor.expand_shape %1701 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1702 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_1083 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1703 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_1083, %1702 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.25.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.25.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_1084 = linalg.transpose ins(%__auto.model.model.layers.25.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1704 = linalg.matmul ins(%collapsed_1082, %transposed_1084 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_1085 = tensor.expand_shape %1704 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1705 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1703, %expanded_1085 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.25.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.25.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_1086 = linalg.transpose ins(%__auto.model.model.layers.25.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_1087 = tensor.collapse_shape %1705 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1706 = linalg.matmul ins(%collapsed_1087, %transposed_1086 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1088 = tensor.expand_shape %1706 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1707 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1693, %expanded_1088 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1708 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1707 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1709 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1708 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1710 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1709 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1711 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1710 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1712 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1711 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1713 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1707, %1712 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.26.input_layernorm.weight = util.global.load @__auto.model.model.layers.26.input_layernorm.weight : tensor<576xf32>
    %1714 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.26.input_layernorm.weight, %1713 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.26.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.26.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_1089 = linalg.transpose ins(%__auto.model.model.layers.26.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_1090 = tensor.collapse_shape %1714 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1715 = linalg.matmul ins(%collapsed_1090, %transposed_1089 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1091 = tensor.expand_shape %1715 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_1092 = linalg.transpose ins(%expanded_1091 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.26.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.26.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_1093 = linalg.transpose ins(%__auto.model.model.layers.26.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1716 = linalg.matmul ins(%collapsed_1090, %transposed_1093 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_1094 = tensor.expand_shape %1716 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_1095 = linalg.transpose ins(%expanded_1094 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.26.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.26.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_1096 = linalg.transpose ins(%__auto.model.model.layers.26.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1717 = linalg.matmul ins(%collapsed_1090, %transposed_1096 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_1097 = tensor.expand_shape %1717 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_1098 = linalg.transpose ins(%expanded_1097 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %1718 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_1092, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_1099 = tensor.extract_slice %transposed_1092[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_1100 = tensor.extract_slice %transposed_1092[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %1719 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1100 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_1101 = tensor.concat dim(3) %1719, %extracted_slice_1099 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %1720 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_1101, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1721 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1718, %1720 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1722 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_1095, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_1102 = tensor.extract_slice %transposed_1095[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_1103 = tensor.extract_slice %transposed_1095[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %1723 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1103 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_1104 = tensor.concat dim(3) %1723, %extracted_slice_1102 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %1724 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_1104, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %1725 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1722, %1724 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_1105 = tensor.concat dim(2) %109, %1725 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_1106 = tensor.concat dim(2) %111, %transposed_1098 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %1726 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_1105 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_1107 = tensor.collapse_shape %1726 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %1727 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_1106 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %1728 = affine.apply #map15()[%dim_61]
    %1729 = tensor.empty(%1728) : tensor<1x9x64x?xf32>
    %transposed_1108 = linalg.transpose ins(%collapsed_1107 : tensor<1x9x?x64xf32>) outs(%1729 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_1109 = tensor.collapse_shape %1721 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_1110 = tensor.collapse_shape %transposed_1108 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %1730 = tensor.empty(%1728) : tensor<9x1x?xf32>
    %1731 = linalg.fill ins(%cst : f32) outs(%1730 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %1732 = linalg.batch_matmul ins(%collapsed_1109, %collapsed_1110 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%1731 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_1111 = tensor.expand_shape %1732 [[0, 1], [2], [3]] output_shape [1, 9, 1, %1728] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %1733 = tensor.empty(%1728) : tensor<1x9x1x?xf32>
    %1734 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_1111 : tensor<1x9x1x?xf32>) outs(%1733 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1735 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1734, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%1733 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1736:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1735 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_1112 = tensor.expand_shape %1736#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %1737 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1735, %expanded_1112 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1733 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1738 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1737 : tensor<1x9x1x?xf32>) outs(%1733 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1739 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1738 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %1740 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1738, %1739 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1733 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_1113 = tensor.collapse_shape %1740 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_1114 = tensor.collapse_shape %1727 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %1741 = affine.apply #map15()[%dim_62]
    %1742 = arith.index_cast %1728 : index to i64
    %1743 = arith.index_cast %1741 : index to i64
    %1744 = arith.cmpi eq, %1742, %1743 : i64
    cf.assert %1744, "mismatching contracting dimension"
    %1745 = linalg.batch_matmul ins(%collapsed_1113, %collapsed_1114 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_1115 = tensor.expand_shape %1745 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_1116 = linalg.transpose ins(%expanded_1115 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.26.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.26.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_1117 = linalg.transpose ins(%__auto.model.model.layers.26.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_1118 = tensor.collapse_shape %transposed_1116 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %1746 = linalg.matmul ins(%collapsed_1118, %transposed_1117 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1119 = tensor.expand_shape %1746 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1747 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1707, %expanded_1119 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1748 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1747 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1749 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1748 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1750 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1749 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1751 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1750 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1752 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1751 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1753 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1747, %1752 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.26.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.26.post_attention_layernorm.weight : tensor<576xf32>
    %1754 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.26.post_attention_layernorm.weight, %1753 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.26.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.26.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_1120 = linalg.transpose ins(%__auto.model.model.layers.26.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_1121 = tensor.collapse_shape %1754 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1755 = linalg.matmul ins(%collapsed_1121, %transposed_1120 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_1122 = tensor.expand_shape %1755 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1756 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_1122 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1757 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_1122, %1756 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.26.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.26.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_1123 = linalg.transpose ins(%__auto.model.model.layers.26.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1758 = linalg.matmul ins(%collapsed_1121, %transposed_1123 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_1124 = tensor.expand_shape %1758 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1759 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1757, %expanded_1124 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.26.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.26.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_1125 = linalg.transpose ins(%__auto.model.model.layers.26.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_1126 = tensor.collapse_shape %1759 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1760 = linalg.matmul ins(%collapsed_1126, %transposed_1125 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1127 = tensor.expand_shape %1760 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1761 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1747, %expanded_1127 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1762 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1761 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1763 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1762 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1764 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1763 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1765 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1764 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1766 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1765 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1767 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1761, %1766 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.27.input_layernorm.weight = util.global.load @__auto.model.model.layers.27.input_layernorm.weight : tensor<576xf32>
    %1768 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.27.input_layernorm.weight, %1767 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.27.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.27.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_1128 = linalg.transpose ins(%__auto.model.model.layers.27.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_1129 = tensor.collapse_shape %1768 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1769 = linalg.matmul ins(%collapsed_1129, %transposed_1128 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1130 = tensor.expand_shape %1769 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_1131 = linalg.transpose ins(%expanded_1130 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.27.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.27.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_1132 = linalg.transpose ins(%__auto.model.model.layers.27.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1770 = linalg.matmul ins(%collapsed_1129, %transposed_1132 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_1133 = tensor.expand_shape %1770 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_1134 = linalg.transpose ins(%expanded_1133 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.27.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.27.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_1135 = linalg.transpose ins(%__auto.model.model.layers.27.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1771 = linalg.matmul ins(%collapsed_1129, %transposed_1135 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_1136 = tensor.expand_shape %1771 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_1137 = linalg.transpose ins(%expanded_1136 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %1772 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_1131, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_1138 = tensor.extract_slice %transposed_1131[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_1139 = tensor.extract_slice %transposed_1131[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %1773 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1139 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_1140 = tensor.concat dim(3) %1773, %extracted_slice_1138 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %1774 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_1140, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1775 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1772, %1774 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1776 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_1134, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_1141 = tensor.extract_slice %transposed_1134[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_1142 = tensor.extract_slice %transposed_1134[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %1777 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1142 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_1143 = tensor.concat dim(3) %1777, %extracted_slice_1141 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %1778 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_1143, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %1779 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1776, %1778 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_1144 = tensor.concat dim(2) %113, %1779 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_1145 = tensor.concat dim(2) %115, %transposed_1137 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %1780 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_1144 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_1146 = tensor.collapse_shape %1780 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %1781 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_1145 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %1782 = affine.apply #map15()[%dim_63]
    %1783 = tensor.empty(%1782) : tensor<1x9x64x?xf32>
    %transposed_1147 = linalg.transpose ins(%collapsed_1146 : tensor<1x9x?x64xf32>) outs(%1783 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_1148 = tensor.collapse_shape %1775 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_1149 = tensor.collapse_shape %transposed_1147 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %1784 = tensor.empty(%1782) : tensor<9x1x?xf32>
    %1785 = linalg.fill ins(%cst : f32) outs(%1784 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %1786 = linalg.batch_matmul ins(%collapsed_1148, %collapsed_1149 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%1785 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_1150 = tensor.expand_shape %1786 [[0, 1], [2], [3]] output_shape [1, 9, 1, %1782] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %1787 = tensor.empty(%1782) : tensor<1x9x1x?xf32>
    %1788 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_1150 : tensor<1x9x1x?xf32>) outs(%1787 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1789 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1788, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%1787 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1790:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1789 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_1151 = tensor.expand_shape %1790#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %1791 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1789, %expanded_1151 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1787 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1792 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1791 : tensor<1x9x1x?xf32>) outs(%1787 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1793 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1792 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %1794 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1792, %1793 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1787 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_1152 = tensor.collapse_shape %1794 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_1153 = tensor.collapse_shape %1781 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %1795 = affine.apply #map15()[%dim_64]
    %1796 = arith.index_cast %1782 : index to i64
    %1797 = arith.index_cast %1795 : index to i64
    %1798 = arith.cmpi eq, %1796, %1797 : i64
    cf.assert %1798, "mismatching contracting dimension"
    %1799 = linalg.batch_matmul ins(%collapsed_1152, %collapsed_1153 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_1154 = tensor.expand_shape %1799 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_1155 = linalg.transpose ins(%expanded_1154 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.27.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.27.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_1156 = linalg.transpose ins(%__auto.model.model.layers.27.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_1157 = tensor.collapse_shape %transposed_1155 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %1800 = linalg.matmul ins(%collapsed_1157, %transposed_1156 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1158 = tensor.expand_shape %1800 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1801 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1761, %expanded_1158 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1802 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1801 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1803 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1802 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1804 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1803 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1805 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1804 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1806 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1805 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1807 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1801, %1806 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.27.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.27.post_attention_layernorm.weight : tensor<576xf32>
    %1808 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.27.post_attention_layernorm.weight, %1807 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.27.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.27.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_1159 = linalg.transpose ins(%__auto.model.model.layers.27.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_1160 = tensor.collapse_shape %1808 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1809 = linalg.matmul ins(%collapsed_1160, %transposed_1159 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_1161 = tensor.expand_shape %1809 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1810 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_1161 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1811 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_1161, %1810 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.27.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.27.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_1162 = linalg.transpose ins(%__auto.model.model.layers.27.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1812 = linalg.matmul ins(%collapsed_1160, %transposed_1162 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_1163 = tensor.expand_shape %1812 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1813 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1811, %expanded_1163 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.27.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.27.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_1164 = linalg.transpose ins(%__auto.model.model.layers.27.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_1165 = tensor.collapse_shape %1813 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1814 = linalg.matmul ins(%collapsed_1165, %transposed_1164 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1166 = tensor.expand_shape %1814 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1815 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1801, %expanded_1166 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1816 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1815 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1817 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1816 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1818 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1817 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1819 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1818 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1820 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1819 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1821 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1815, %1820 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.28.input_layernorm.weight = util.global.load @__auto.model.model.layers.28.input_layernorm.weight : tensor<576xf32>
    %1822 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.28.input_layernorm.weight, %1821 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.28.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.28.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_1167 = linalg.transpose ins(%__auto.model.model.layers.28.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_1168 = tensor.collapse_shape %1822 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1823 = linalg.matmul ins(%collapsed_1168, %transposed_1167 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1169 = tensor.expand_shape %1823 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_1170 = linalg.transpose ins(%expanded_1169 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.28.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.28.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_1171 = linalg.transpose ins(%__auto.model.model.layers.28.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1824 = linalg.matmul ins(%collapsed_1168, %transposed_1171 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_1172 = tensor.expand_shape %1824 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_1173 = linalg.transpose ins(%expanded_1172 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.28.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.28.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_1174 = linalg.transpose ins(%__auto.model.model.layers.28.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1825 = linalg.matmul ins(%collapsed_1168, %transposed_1174 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_1175 = tensor.expand_shape %1825 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_1176 = linalg.transpose ins(%expanded_1175 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %1826 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_1170, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_1177 = tensor.extract_slice %transposed_1170[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_1178 = tensor.extract_slice %transposed_1170[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %1827 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1178 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_1179 = tensor.concat dim(3) %1827, %extracted_slice_1177 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %1828 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_1179, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1829 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1826, %1828 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1830 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_1173, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_1180 = tensor.extract_slice %transposed_1173[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_1181 = tensor.extract_slice %transposed_1173[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %1831 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1181 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_1182 = tensor.concat dim(3) %1831, %extracted_slice_1180 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %1832 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_1182, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %1833 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1830, %1832 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_1183 = tensor.concat dim(2) %117, %1833 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_1184 = tensor.concat dim(2) %119, %transposed_1176 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %1834 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_1183 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_1185 = tensor.collapse_shape %1834 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %1835 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_1184 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %1836 = affine.apply #map15()[%dim_65]
    %1837 = tensor.empty(%1836) : tensor<1x9x64x?xf32>
    %transposed_1186 = linalg.transpose ins(%collapsed_1185 : tensor<1x9x?x64xf32>) outs(%1837 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_1187 = tensor.collapse_shape %1829 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_1188 = tensor.collapse_shape %transposed_1186 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %1838 = tensor.empty(%1836) : tensor<9x1x?xf32>
    %1839 = linalg.fill ins(%cst : f32) outs(%1838 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %1840 = linalg.batch_matmul ins(%collapsed_1187, %collapsed_1188 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%1839 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_1189 = tensor.expand_shape %1840 [[0, 1], [2], [3]] output_shape [1, 9, 1, %1836] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %1841 = tensor.empty(%1836) : tensor<1x9x1x?xf32>
    %1842 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_1189 : tensor<1x9x1x?xf32>) outs(%1841 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1843 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1842, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%1841 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1844:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1843 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_1190 = tensor.expand_shape %1844#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %1845 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1843, %expanded_1190 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1841 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1846 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1845 : tensor<1x9x1x?xf32>) outs(%1841 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1847 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1846 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %1848 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1846, %1847 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1841 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_1191 = tensor.collapse_shape %1848 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_1192 = tensor.collapse_shape %1835 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %1849 = affine.apply #map15()[%dim_66]
    %1850 = arith.index_cast %1836 : index to i64
    %1851 = arith.index_cast %1849 : index to i64
    %1852 = arith.cmpi eq, %1850, %1851 : i64
    cf.assert %1852, "mismatching contracting dimension"
    %1853 = linalg.batch_matmul ins(%collapsed_1191, %collapsed_1192 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_1193 = tensor.expand_shape %1853 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_1194 = linalg.transpose ins(%expanded_1193 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.28.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.28.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_1195 = linalg.transpose ins(%__auto.model.model.layers.28.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_1196 = tensor.collapse_shape %transposed_1194 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %1854 = linalg.matmul ins(%collapsed_1196, %transposed_1195 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1197 = tensor.expand_shape %1854 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1855 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1815, %expanded_1197 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1856 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1855 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1857 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1856 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1858 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1857 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1859 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1858 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1860 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1859 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1861 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1855, %1860 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.28.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.28.post_attention_layernorm.weight : tensor<576xf32>
    %1862 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.28.post_attention_layernorm.weight, %1861 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.28.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.28.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_1198 = linalg.transpose ins(%__auto.model.model.layers.28.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_1199 = tensor.collapse_shape %1862 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1863 = linalg.matmul ins(%collapsed_1199, %transposed_1198 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_1200 = tensor.expand_shape %1863 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1864 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_1200 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1865 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_1200, %1864 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.28.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.28.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_1201 = linalg.transpose ins(%__auto.model.model.layers.28.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1866 = linalg.matmul ins(%collapsed_1199, %transposed_1201 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_1202 = tensor.expand_shape %1866 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1867 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1865, %expanded_1202 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.28.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.28.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_1203 = linalg.transpose ins(%__auto.model.model.layers.28.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_1204 = tensor.collapse_shape %1867 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1868 = linalg.matmul ins(%collapsed_1204, %transposed_1203 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1205 = tensor.expand_shape %1868 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1869 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1855, %expanded_1205 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1870 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1869 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1871 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1870 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1872 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1871 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1873 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1872 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1874 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1873 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1875 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1869, %1874 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.29.input_layernorm.weight = util.global.load @__auto.model.model.layers.29.input_layernorm.weight : tensor<576xf32>
    %1876 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.29.input_layernorm.weight, %1875 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.29.self_attn.q_proj.weight = util.global.load @__auto.model.model.layers.29.self_attn.q_proj.weight : tensor<576x576xf32>
    %transposed_1206 = linalg.transpose ins(%__auto.model.model.layers.29.self_attn.q_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_1207 = tensor.collapse_shape %1876 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1877 = linalg.matmul ins(%collapsed_1207, %transposed_1206 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1208 = tensor.expand_shape %1877 [[0], [1, 2, 3]] output_shape [1, 1, 9, 64] : tensor<1x576xf32> into tensor<1x1x9x64xf32>
    %transposed_1209 = linalg.transpose ins(%expanded_1208 : tensor<1x1x9x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.29.self_attn.k_proj.weight = util.global.load @__auto.model.model.layers.29.self_attn.k_proj.weight : tensor<192x576xf32>
    %transposed_1210 = linalg.transpose ins(%__auto.model.model.layers.29.self_attn.k_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1878 = linalg.matmul ins(%collapsed_1207, %transposed_1210 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_1211 = tensor.expand_shape %1878 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_1212 = linalg.transpose ins(%expanded_1211 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.29.self_attn.v_proj.weight = util.global.load @__auto.model.model.layers.29.self_attn.v_proj.weight : tensor<192x576xf32>
    %transposed_1213 = linalg.transpose ins(%__auto.model.model.layers.29.self_attn.v_proj.weight : tensor<192x576xf32>) outs(%284 : tensor<576x192xf32>) permutation = [1, 0] 
    %1879 = linalg.matmul ins(%collapsed_1207, %transposed_1213 : tensor<1x576xf32>, tensor<576x192xf32>) outs(%286 : tensor<1x192xf32>) -> tensor<1x192xf32>
    %expanded_1214 = tensor.expand_shape %1879 [[0], [1, 2, 3]] output_shape [1, 1, 3, 64] : tensor<1x192xf32> into tensor<1x1x3x64xf32>
    %transposed_1215 = linalg.transpose ins(%expanded_1214 : tensor<1x1x3x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) permutation = [0, 2, 1, 3] 
    %1880 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_1209, %expanded_83 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %extracted_slice_1216 = tensor.extract_slice %transposed_1209[0, 0, 0, 0] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %extracted_slice_1217 = tensor.extract_slice %transposed_1209[0, 0, 0, 32] [1, 9, 1, 32] [1, 1, 1, 1] : tensor<1x9x1x64xf32> to tensor<1x9x1x32xf32>
    %1881 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1217 : tensor<1x9x1x32xf32>) outs(%291 : tensor<1x9x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x32xf32>
    %concat_1218 = tensor.concat dim(3) %1881, %extracted_slice_1216 : (tensor<1x9x1x32xf32>, tensor<1x9x1x32xf32>) -> tensor<1x9x1x64xf32>
    %1882 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_1218, %expanded_84 : tensor<1x9x1x64xf32>, tensor<1x1x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1883 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1880, %1882 : tensor<1x9x1x64xf32>, tensor<1x9x1x64xf32>) outs(%283 : tensor<1x9x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x64xf32>
    %1884 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_1212, %expanded_83 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %extracted_slice_1219 = tensor.extract_slice %transposed_1212[0, 0, 0, 0] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %extracted_slice_1220 = tensor.extract_slice %transposed_1212[0, 0, 0, 32] [1, 3, 1, 32] [1, 1, 1, 1] : tensor<1x3x1x64xf32> to tensor<1x3x1x32xf32>
    %1885 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1220 : tensor<1x3x1x32xf32>) outs(%296 : tensor<1x3x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x32xf32>
    %concat_1221 = tensor.concat dim(3) %1885, %extracted_slice_1219 : (tensor<1x3x1x32xf32>, tensor<1x3x1x32xf32>) -> tensor<1x3x1x64xf32>
    %1886 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_1221, %expanded_84 : tensor<1x3x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %1887 = linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1884, %1886 : tensor<1x3x1x64xf32>, tensor<1x3x1x64xf32>) outs(%288 : tensor<1x3x1x64xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x3x1x64xf32>
    %concat_1222 = tensor.concat dim(2) %121, %1887 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %concat_1223 = tensor.concat dim(2) %123, %transposed_1215 : (tensor<1x3x?x64xf32>, tensor<1x3x1x64xf32>) -> tensor<1x3x?x64xf32>
    %1888 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_1222 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %collapsed_1224 = tensor.collapse_shape %1888 [[0], [1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<1x9x?x64xf32>
    %1889 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_1223 : tensor<1x3x?x64xf32>) outs(%300 : tensor<1x3x3x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3x3x?x64xf32>
    %1890 = affine.apply #map15()[%dim_67]
    %1891 = tensor.empty(%1890) : tensor<1x9x64x?xf32>
    %transposed_1225 = linalg.transpose ins(%collapsed_1224 : tensor<1x9x?x64xf32>) outs(%1891 : tensor<1x9x64x?xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_1226 = tensor.collapse_shape %1883 [[0, 1], [2], [3]] : tensor<1x9x1x64xf32> into tensor<9x1x64xf32>
    %collapsed_1227 = tensor.collapse_shape %transposed_1225 [[0, 1], [2], [3]] : tensor<1x9x64x?xf32> into tensor<9x64x?xf32>
    %1892 = tensor.empty(%1890) : tensor<9x1x?xf32>
    %1893 = linalg.fill ins(%cst : f32) outs(%1892 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %1894 = linalg.batch_matmul ins(%collapsed_1226, %collapsed_1227 : tensor<9x1x64xf32>, tensor<9x64x?xf32>) outs(%1893 : tensor<9x1x?xf32>) -> tensor<9x1x?xf32>
    %expanded_1228 = tensor.expand_shape %1894 [[0, 1], [2], [3]] output_shape [1, 9, 1, %1890] : tensor<9x1x?xf32> into tensor<1x9x1x?xf32>
    %1895 = tensor.empty(%1890) : tensor<1x9x1x?xf32>
    %1896 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_1228 : tensor<1x9x1x?xf32>) outs(%1895 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.mulf %in, %cst_8 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1897 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1896, %extracted_slice_97 : tensor<1x9x1x?xf32>, tensor<1x1x1x?xf32>) outs(%1895 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1898:2 = linalg.generic {indexing_maps = [#map8, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1897 : tensor<1x9x1x?xf32>) outs(%321, %319 : tensor<1x9x1xf32>, tensor<1x9x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1308: i64):
      %1997 = linalg.index 3 : index
      %1998 = arith.index_cast %1997 : index to i64
      %1999 = arith.maximumf %in, %out : f32
      %2000 = arith.cmpf ogt, %in, %out : f32
      %2001 = arith.select %2000, %1998, %out_1308 : i64
      linalg.yield %1999, %2001 : f32, i64
    } -> (tensor<1x9x1xf32>, tensor<1x9x1xi64>)
    %expanded_1229 = tensor.expand_shape %1898#0 [[0], [1], [2, 3]] output_shape [1, 9, 1, 1] : tensor<1x9x1xf32> into tensor<1x9x1x1xf32>
    %1899 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1897, %expanded_1229 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1895 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.subf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1900 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1899 : tensor<1x9x1x?xf32>) outs(%1895 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.exp %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %1901 = linalg.generic {indexing_maps = [#map8, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1900 : tensor<1x9x1x?xf32>) outs(%326 : tensor<1x9x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x1xf32>
    %1902 = linalg.generic {indexing_maps = [#map8, #map17, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1900, %1901 : tensor<1x9x1x?xf32>, tensor<1x9x1x1xf32>) outs(%1895 : tensor<1x9x1x?xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.divf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x9x1x?xf32>
    %collapsed_1230 = tensor.collapse_shape %1902 [[0, 1], [2], [3]] : tensor<1x9x1x?xf32> into tensor<9x1x?xf32>
    %collapsed_1231 = tensor.collapse_shape %1889 [[0, 1, 2], [3], [4]] : tensor<1x3x3x?x64xf32> into tensor<9x?x64xf32>
    %1903 = affine.apply #map15()[%dim_68]
    %1904 = arith.index_cast %1890 : index to i64
    %1905 = arith.index_cast %1903 : index to i64
    %1906 = arith.cmpi eq, %1904, %1905 : i64
    cf.assert %1906, "mismatching contracting dimension"
    %1907 = linalg.batch_matmul ins(%collapsed_1230, %collapsed_1231 : tensor<9x1x?xf32>, tensor<9x?x64xf32>) outs(%334 : tensor<9x1x64xf32>) -> tensor<9x1x64xf32>
    %expanded_1232 = tensor.expand_shape %1907 [[0, 1], [2], [3]] output_shape [1, 9, 1, 64] : tensor<9x1x64xf32> into tensor<1x9x1x64xf32>
    %transposed_1233 = linalg.transpose ins(%expanded_1232 : tensor<1x9x1x64xf32>) outs(%336 : tensor<1x1x9x64xf32>) permutation = [0, 2, 1, 3] 
    %__auto.model.model.layers.29.self_attn.o_proj.weight = util.global.load @__auto.model.model.layers.29.self_attn.o_proj.weight : tensor<576x576xf32>
    %transposed_1234 = linalg.transpose ins(%__auto.model.model.layers.29.self_attn.o_proj.weight : tensor<576x576xf32>) outs(%279 : tensor<576x576xf32>) permutation = [1, 0] 
    %collapsed_1235 = tensor.collapse_shape %transposed_1233 [[0], [1, 2, 3]] : tensor<1x1x9x64xf32> into tensor<1x576xf32>
    %1908 = linalg.matmul ins(%collapsed_1235, %transposed_1234 : tensor<1x576xf32>, tensor<576x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1236 = tensor.expand_shape %1908 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1909 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1869, %expanded_1236 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1910 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1909 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1911 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1910 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1912 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1911 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1913 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1912 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1914 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1913 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1915 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1909, %1914 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.29.post_attention_layernorm.weight = util.global.load @__auto.model.model.layers.29.post_attention_layernorm.weight : tensor<576xf32>
    %1916 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.layers.29.post_attention_layernorm.weight, %1915 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.layers.29.mlp.gate_proj.weight = util.global.load @__auto.model.model.layers.29.mlp.gate_proj.weight : tensor<1536x576xf32>
    %transposed_1237 = linalg.transpose ins(%__auto.model.model.layers.29.mlp.gate_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %collapsed_1238 = tensor.collapse_shape %1916 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1917 = linalg.matmul ins(%collapsed_1238, %transposed_1237 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_1239 = tensor.expand_shape %1917 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1918 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_1239 : tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.negf %in : f32
      %1998 = math.exp %1997 : f32
      %1999 = arith.addf %1998, %cst_1 : f32
      %2000 = arith.divf %cst_1, %1999 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x1x1536xf32>
    %1919 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_1239, %1918 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.29.mlp.up_proj.weight = util.global.load @__auto.model.model.layers.29.mlp.up_proj.weight : tensor<1536x576xf32>
    %transposed_1240 = linalg.transpose ins(%__auto.model.model.layers.29.mlp.up_proj.weight : tensor<1536x576xf32>) outs(%346 : tensor<576x1536xf32>) permutation = [1, 0] 
    %1920 = linalg.matmul ins(%collapsed_1238, %transposed_1240 : tensor<1x576xf32>, tensor<576x1536xf32>) outs(%348 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %expanded_1241 = tensor.expand_shape %1920 [[0], [1, 2]] output_shape [1, 1, 1536] : tensor<1x1536xf32> into tensor<1x1x1536xf32>
    %1921 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1919, %expanded_1241 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%350 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1536xf32>
    %__auto.model.model.layers.29.mlp.down_proj.weight = util.global.load @__auto.model.model.layers.29.mlp.down_proj.weight : tensor<576x1536xf32>
    %transposed_1242 = linalg.transpose ins(%__auto.model.model.layers.29.mlp.down_proj.weight : tensor<576x1536xf32>) outs(%355 : tensor<1536x576xf32>) permutation = [1, 0] 
    %collapsed_1243 = tensor.collapse_shape %1921 [[0], [1, 2]] : tensor<1x1x1536xf32> into tensor<1x1536xf32>
    %1922 = linalg.matmul ins(%collapsed_1243, %transposed_1242 : tensor<1x1536xf32>, tensor<1536x576xf32>) outs(%281 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %expanded_1244 = tensor.expand_shape %1922 [[0], [1, 2]] output_shape [1, 1, 576] : tensor<1x576xf32> into tensor<1x1x576xf32>
    %1923 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1909, %expanded_1244 : tensor<1x1x576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.addf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1924 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1923 : tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1925 = linalg.generic {indexing_maps = [#map1, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1924 : tensor<1x1x576xf32>) outs(%272 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %out : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1926 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1925 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.divf %in, %cst_7 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1927 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1926 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = arith.addf %in, %cst_4 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1928 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1927 : tensor<1x1x1xf32>) outs(%262 : tensor<1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1997 = math.rsqrt %in : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x1xf32>
    %1929 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1923, %1928 : tensor<1x1x576xf32>, tensor<1x1x1xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %__auto.model.model.norm.weight = util.global.load @__auto.model.model.norm.weight : tensor<576xf32>
    %1930 = linalg.generic {indexing_maps = [#map11, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%__auto.model.model.norm.weight, %1929 : tensor<576xf32>, tensor<1x1x576xf32>) outs(%246 : tensor<1x1x576xf32>) {
    ^bb0(%in: f32, %in_1308: f32, %out: f32):
      %1997 = arith.mulf %in, %in_1308 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1x576xf32>
    %1931 = tensor.empty() : tensor<576x49152xf32>
    %transposed_1245 = linalg.transpose ins(%__auto.model.lm_head.weight : tensor<49152x576xf32>) outs(%1931 : tensor<576x49152xf32>) permutation = [1, 0] 
    %collapsed_1246 = tensor.collapse_shape %1930 [[0], [1, 2]] : tensor<1x1x576xf32> into tensor<1x576xf32>
    %1932 = tensor.empty() : tensor<1x49152xf32>
    %1933 = linalg.fill ins(%cst : f32) outs(%1932 : tensor<1x49152xf32>) -> tensor<1x49152xf32>
    %1934 = linalg.matmul ins(%collapsed_1246, %transposed_1245 : tensor<1x576xf32>, tensor<576x49152xf32>) outs(%1933 : tensor<1x49152xf32>) -> tensor<1x49152xf32>
    %expanded_1247 = tensor.expand_shape %1934 [[0], [1, 2]] output_shape [1, 1, 49152] : tensor<1x49152xf32> into tensor<1x1x49152xf32>
    %1935:61 = hal.tensor.barrier join(%expanded_1247, %concat_90, %concat_91, %concat_130, %concat_131, %concat_169, %concat_170, %concat_208, %concat_209, %concat_247, %concat_248, %concat_286, %concat_287, %concat_325, %concat_326, %concat_364, %concat_365, %concat_403, %concat_404, %concat_442, %concat_443, %concat_481, %concat_482, %concat_520, %concat_521, %concat_559, %concat_560, %concat_598, %concat_599, %concat_637, %concat_638, %concat_676, %concat_677, %concat_715, %concat_716, %concat_754, %concat_755, %concat_793, %concat_794, %concat_832, %concat_833, %concat_871, %concat_872, %concat_910, %concat_911, %concat_949, %concat_950, %concat_988, %concat_989, %concat_1027, %concat_1028, %concat_1066, %concat_1067, %concat_1105, %concat_1106, %concat_1144, %concat_1145, %concat_1183, %concat_1184, %concat_1222, %concat_1223 : tensor<1x1x49152xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>, tensor<1x3x?x64xf32>) => %arg64 : !hal.fence
    %1936 = hal.tensor.export %1935#0 : tensor<1x1x49152xf32> -> !hal.buffer_view
    %dim_1248 = tensor.dim %1935#1, %c2 : tensor<1x3x?x64xf32>
    %1937 = hal.tensor.export %1935#1 : tensor<1x3x?x64xf32>{%dim_1248} -> !hal.buffer_view
    %dim_1249 = tensor.dim %1935#2, %c2 : tensor<1x3x?x64xf32>
    %1938 = hal.tensor.export %1935#2 : tensor<1x3x?x64xf32>{%dim_1249} -> !hal.buffer_view
    %dim_1250 = tensor.dim %1935#3, %c2 : tensor<1x3x?x64xf32>
    %1939 = hal.tensor.export %1935#3 : tensor<1x3x?x64xf32>{%dim_1250} -> !hal.buffer_view
    %dim_1251 = tensor.dim %1935#4, %c2 : tensor<1x3x?x64xf32>
    %1940 = hal.tensor.export %1935#4 : tensor<1x3x?x64xf32>{%dim_1251} -> !hal.buffer_view
    %dim_1252 = tensor.dim %1935#5, %c2 : tensor<1x3x?x64xf32>
    %1941 = hal.tensor.export %1935#5 : tensor<1x3x?x64xf32>{%dim_1252} -> !hal.buffer_view
    %dim_1253 = tensor.dim %1935#6, %c2 : tensor<1x3x?x64xf32>
    %1942 = hal.tensor.export %1935#6 : tensor<1x3x?x64xf32>{%dim_1253} -> !hal.buffer_view
    %dim_1254 = tensor.dim %1935#7, %c2 : tensor<1x3x?x64xf32>
    %1943 = hal.tensor.export %1935#7 : tensor<1x3x?x64xf32>{%dim_1254} -> !hal.buffer_view
    %dim_1255 = tensor.dim %1935#8, %c2 : tensor<1x3x?x64xf32>
    %1944 = hal.tensor.export %1935#8 : tensor<1x3x?x64xf32>{%dim_1255} -> !hal.buffer_view
    %dim_1256 = tensor.dim %1935#9, %c2 : tensor<1x3x?x64xf32>
    %1945 = hal.tensor.export %1935#9 : tensor<1x3x?x64xf32>{%dim_1256} -> !hal.buffer_view
    %dim_1257 = tensor.dim %1935#10, %c2 : tensor<1x3x?x64xf32>
    %1946 = hal.tensor.export %1935#10 : tensor<1x3x?x64xf32>{%dim_1257} -> !hal.buffer_view
    %dim_1258 = tensor.dim %1935#11, %c2 : tensor<1x3x?x64xf32>
    %1947 = hal.tensor.export %1935#11 : tensor<1x3x?x64xf32>{%dim_1258} -> !hal.buffer_view
    %dim_1259 = tensor.dim %1935#12, %c2 : tensor<1x3x?x64xf32>
    %1948 = hal.tensor.export %1935#12 : tensor<1x3x?x64xf32>{%dim_1259} -> !hal.buffer_view
    %dim_1260 = tensor.dim %1935#13, %c2 : tensor<1x3x?x64xf32>
    %1949 = hal.tensor.export %1935#13 : tensor<1x3x?x64xf32>{%dim_1260} -> !hal.buffer_view
    %dim_1261 = tensor.dim %1935#14, %c2 : tensor<1x3x?x64xf32>
    %1950 = hal.tensor.export %1935#14 : tensor<1x3x?x64xf32>{%dim_1261} -> !hal.buffer_view
    %dim_1262 = tensor.dim %1935#15, %c2 : tensor<1x3x?x64xf32>
    %1951 = hal.tensor.export %1935#15 : tensor<1x3x?x64xf32>{%dim_1262} -> !hal.buffer_view
    %dim_1263 = tensor.dim %1935#16, %c2 : tensor<1x3x?x64xf32>
    %1952 = hal.tensor.export %1935#16 : tensor<1x3x?x64xf32>{%dim_1263} -> !hal.buffer_view
    %dim_1264 = tensor.dim %1935#17, %c2 : tensor<1x3x?x64xf32>
    %1953 = hal.tensor.export %1935#17 : tensor<1x3x?x64xf32>{%dim_1264} -> !hal.buffer_view
    %dim_1265 = tensor.dim %1935#18, %c2 : tensor<1x3x?x64xf32>
    %1954 = hal.tensor.export %1935#18 : tensor<1x3x?x64xf32>{%dim_1265} -> !hal.buffer_view
    %dim_1266 = tensor.dim %1935#19, %c2 : tensor<1x3x?x64xf32>
    %1955 = hal.tensor.export %1935#19 : tensor<1x3x?x64xf32>{%dim_1266} -> !hal.buffer_view
    %dim_1267 = tensor.dim %1935#20, %c2 : tensor<1x3x?x64xf32>
    %1956 = hal.tensor.export %1935#20 : tensor<1x3x?x64xf32>{%dim_1267} -> !hal.buffer_view
    %dim_1268 = tensor.dim %1935#21, %c2 : tensor<1x3x?x64xf32>
    %1957 = hal.tensor.export %1935#21 : tensor<1x3x?x64xf32>{%dim_1268} -> !hal.buffer_view
    %dim_1269 = tensor.dim %1935#22, %c2 : tensor<1x3x?x64xf32>
    %1958 = hal.tensor.export %1935#22 : tensor<1x3x?x64xf32>{%dim_1269} -> !hal.buffer_view
    %dim_1270 = tensor.dim %1935#23, %c2 : tensor<1x3x?x64xf32>
    %1959 = hal.tensor.export %1935#23 : tensor<1x3x?x64xf32>{%dim_1270} -> !hal.buffer_view
    %dim_1271 = tensor.dim %1935#24, %c2 : tensor<1x3x?x64xf32>
    %1960 = hal.tensor.export %1935#24 : tensor<1x3x?x64xf32>{%dim_1271} -> !hal.buffer_view
    %dim_1272 = tensor.dim %1935#25, %c2 : tensor<1x3x?x64xf32>
    %1961 = hal.tensor.export %1935#25 : tensor<1x3x?x64xf32>{%dim_1272} -> !hal.buffer_view
    %dim_1273 = tensor.dim %1935#26, %c2 : tensor<1x3x?x64xf32>
    %1962 = hal.tensor.export %1935#26 : tensor<1x3x?x64xf32>{%dim_1273} -> !hal.buffer_view
    %dim_1274 = tensor.dim %1935#27, %c2 : tensor<1x3x?x64xf32>
    %1963 = hal.tensor.export %1935#27 : tensor<1x3x?x64xf32>{%dim_1274} -> !hal.buffer_view
    %dim_1275 = tensor.dim %1935#28, %c2 : tensor<1x3x?x64xf32>
    %1964 = hal.tensor.export %1935#28 : tensor<1x3x?x64xf32>{%dim_1275} -> !hal.buffer_view
    %dim_1276 = tensor.dim %1935#29, %c2 : tensor<1x3x?x64xf32>
    %1965 = hal.tensor.export %1935#29 : tensor<1x3x?x64xf32>{%dim_1276} -> !hal.buffer_view
    %dim_1277 = tensor.dim %1935#30, %c2 : tensor<1x3x?x64xf32>
    %1966 = hal.tensor.export %1935#30 : tensor<1x3x?x64xf32>{%dim_1277} -> !hal.buffer_view
    %dim_1278 = tensor.dim %1935#31, %c2 : tensor<1x3x?x64xf32>
    %1967 = hal.tensor.export %1935#31 : tensor<1x3x?x64xf32>{%dim_1278} -> !hal.buffer_view
    %dim_1279 = tensor.dim %1935#32, %c2 : tensor<1x3x?x64xf32>
    %1968 = hal.tensor.export %1935#32 : tensor<1x3x?x64xf32>{%dim_1279} -> !hal.buffer_view
    %dim_1280 = tensor.dim %1935#33, %c2 : tensor<1x3x?x64xf32>
    %1969 = hal.tensor.export %1935#33 : tensor<1x3x?x64xf32>{%dim_1280} -> !hal.buffer_view
    %dim_1281 = tensor.dim %1935#34, %c2 : tensor<1x3x?x64xf32>
    %1970 = hal.tensor.export %1935#34 : tensor<1x3x?x64xf32>{%dim_1281} -> !hal.buffer_view
    %dim_1282 = tensor.dim %1935#35, %c2 : tensor<1x3x?x64xf32>
    %1971 = hal.tensor.export %1935#35 : tensor<1x3x?x64xf32>{%dim_1282} -> !hal.buffer_view
    %dim_1283 = tensor.dim %1935#36, %c2 : tensor<1x3x?x64xf32>
    %1972 = hal.tensor.export %1935#36 : tensor<1x3x?x64xf32>{%dim_1283} -> !hal.buffer_view
    %dim_1284 = tensor.dim %1935#37, %c2 : tensor<1x3x?x64xf32>
    %1973 = hal.tensor.export %1935#37 : tensor<1x3x?x64xf32>{%dim_1284} -> !hal.buffer_view
    %dim_1285 = tensor.dim %1935#38, %c2 : tensor<1x3x?x64xf32>
    %1974 = hal.tensor.export %1935#38 : tensor<1x3x?x64xf32>{%dim_1285} -> !hal.buffer_view
    %dim_1286 = tensor.dim %1935#39, %c2 : tensor<1x3x?x64xf32>
    %1975 = hal.tensor.export %1935#39 : tensor<1x3x?x64xf32>{%dim_1286} -> !hal.buffer_view
    %dim_1287 = tensor.dim %1935#40, %c2 : tensor<1x3x?x64xf32>
    %1976 = hal.tensor.export %1935#40 : tensor<1x3x?x64xf32>{%dim_1287} -> !hal.buffer_view
    %dim_1288 = tensor.dim %1935#41, %c2 : tensor<1x3x?x64xf32>
    %1977 = hal.tensor.export %1935#41 : tensor<1x3x?x64xf32>{%dim_1288} -> !hal.buffer_view
    %dim_1289 = tensor.dim %1935#42, %c2 : tensor<1x3x?x64xf32>
    %1978 = hal.tensor.export %1935#42 : tensor<1x3x?x64xf32>{%dim_1289} -> !hal.buffer_view
    %dim_1290 = tensor.dim %1935#43, %c2 : tensor<1x3x?x64xf32>
    %1979 = hal.tensor.export %1935#43 : tensor<1x3x?x64xf32>{%dim_1290} -> !hal.buffer_view
    %dim_1291 = tensor.dim %1935#44, %c2 : tensor<1x3x?x64xf32>
    %1980 = hal.tensor.export %1935#44 : tensor<1x3x?x64xf32>{%dim_1291} -> !hal.buffer_view
    %dim_1292 = tensor.dim %1935#45, %c2 : tensor<1x3x?x64xf32>
    %1981 = hal.tensor.export %1935#45 : tensor<1x3x?x64xf32>{%dim_1292} -> !hal.buffer_view
    %dim_1293 = tensor.dim %1935#46, %c2 : tensor<1x3x?x64xf32>
    %1982 = hal.tensor.export %1935#46 : tensor<1x3x?x64xf32>{%dim_1293} -> !hal.buffer_view
    %dim_1294 = tensor.dim %1935#47, %c2 : tensor<1x3x?x64xf32>
    %1983 = hal.tensor.export %1935#47 : tensor<1x3x?x64xf32>{%dim_1294} -> !hal.buffer_view
    %dim_1295 = tensor.dim %1935#48, %c2 : tensor<1x3x?x64xf32>
    %1984 = hal.tensor.export %1935#48 : tensor<1x3x?x64xf32>{%dim_1295} -> !hal.buffer_view
    %dim_1296 = tensor.dim %1935#49, %c2 : tensor<1x3x?x64xf32>
    %1985 = hal.tensor.export %1935#49 : tensor<1x3x?x64xf32>{%dim_1296} -> !hal.buffer_view
    %dim_1297 = tensor.dim %1935#50, %c2 : tensor<1x3x?x64xf32>
    %1986 = hal.tensor.export %1935#50 : tensor<1x3x?x64xf32>{%dim_1297} -> !hal.buffer_view
    %dim_1298 = tensor.dim %1935#51, %c2 : tensor<1x3x?x64xf32>
    %1987 = hal.tensor.export %1935#51 : tensor<1x3x?x64xf32>{%dim_1298} -> !hal.buffer_view
    %dim_1299 = tensor.dim %1935#52, %c2 : tensor<1x3x?x64xf32>
    %1988 = hal.tensor.export %1935#52 : tensor<1x3x?x64xf32>{%dim_1299} -> !hal.buffer_view
    %dim_1300 = tensor.dim %1935#53, %c2 : tensor<1x3x?x64xf32>
    %1989 = hal.tensor.export %1935#53 : tensor<1x3x?x64xf32>{%dim_1300} -> !hal.buffer_view
    %dim_1301 = tensor.dim %1935#54, %c2 : tensor<1x3x?x64xf32>
    %1990 = hal.tensor.export %1935#54 : tensor<1x3x?x64xf32>{%dim_1301} -> !hal.buffer_view
    %dim_1302 = tensor.dim %1935#55, %c2 : tensor<1x3x?x64xf32>
    %1991 = hal.tensor.export %1935#55 : tensor<1x3x?x64xf32>{%dim_1302} -> !hal.buffer_view
    %dim_1303 = tensor.dim %1935#56, %c2 : tensor<1x3x?x64xf32>
    %1992 = hal.tensor.export %1935#56 : tensor<1x3x?x64xf32>{%dim_1303} -> !hal.buffer_view
    %dim_1304 = tensor.dim %1935#57, %c2 : tensor<1x3x?x64xf32>
    %1993 = hal.tensor.export %1935#57 : tensor<1x3x?x64xf32>{%dim_1304} -> !hal.buffer_view
    %dim_1305 = tensor.dim %1935#58, %c2 : tensor<1x3x?x64xf32>
    %1994 = hal.tensor.export %1935#58 : tensor<1x3x?x64xf32>{%dim_1305} -> !hal.buffer_view
    %dim_1306 = tensor.dim %1935#59, %c2 : tensor<1x3x?x64xf32>
    %1995 = hal.tensor.export %1935#59 : tensor<1x3x?x64xf32>{%dim_1306} -> !hal.buffer_view
    %dim_1307 = tensor.dim %1935#60, %c2 : tensor<1x3x?x64xf32>
    %1996 = hal.tensor.export %1935#60 : tensor<1x3x?x64xf32>{%dim_1307} -> !hal.buffer_view
    util.return %1936, %1937, %1938, %1939, %1940, %1941, %1942, %1943, %1944, %1945, %1946, %1947, %1948, %1949, %1950, %1951, %1952, %1953, %1954, %1955, %1956, %1957, %1958, %1959, %1960, %1961, %1962, %1963, %1964, %1965, %1966, %1967, %1968, %1969, %1970, %1971, %1972, %1973, %1974, %1975, %1976, %1977, %1978, %1979, %1980, %1981, %1982, %1983, %1984, %1985, %1986, %1987, %1988, %1989, %1990, %1991, %1992, %1993, %1994, %1995, %1996 : !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view
  }
  util.func public @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.buffer_view, %arg7: !hal.buffer_view, %arg8: !hal.buffer_view, %arg9: !hal.buffer_view, %arg10: !hal.buffer_view, %arg11: !hal.buffer_view, %arg12: !hal.buffer_view, %arg13: !hal.buffer_view, %arg14: !hal.buffer_view, %arg15: !hal.buffer_view, %arg16: !hal.buffer_view, %arg17: !hal.buffer_view, %arg18: !hal.buffer_view, %arg19: !hal.buffer_view, %arg20: !hal.buffer_view, %arg21: !hal.buffer_view, %arg22: !hal.buffer_view, %arg23: !hal.buffer_view, %arg24: !hal.buffer_view, %arg25: !hal.buffer_view, %arg26: !hal.buffer_view, %arg27: !hal.buffer_view, %arg28: !hal.buffer_view, %arg29: !hal.buffer_view, %arg30: !hal.buffer_view, %arg31: !hal.buffer_view, %arg32: !hal.buffer_view, %arg33: !hal.buffer_view, %arg34: !hal.buffer_view, %arg35: !hal.buffer_view, %arg36: !hal.buffer_view, %arg37: !hal.buffer_view, %arg38: !hal.buffer_view, %arg39: !hal.buffer_view, %arg40: !hal.buffer_view, %arg41: !hal.buffer_view, %arg42: !hal.buffer_view, %arg43: !hal.buffer_view, %arg44: !hal.buffer_view, %arg45: !hal.buffer_view, %arg46: !hal.buffer_view, %arg47: !hal.buffer_view, %arg48: !hal.buffer_view, %arg49: !hal.buffer_view, %arg50: !hal.buffer_view, %arg51: !hal.buffer_view, %arg52: !hal.buffer_view, %arg53: !hal.buffer_view, %arg54: !hal.buffer_view, %arg55: !hal.buffer_view, %arg56: !hal.buffer_view, %arg57: !hal.buffer_view, %arg58: !hal.buffer_view, %arg59: !hal.buffer_view, %arg60: !hal.buffer_view, %arg61: !hal.buffer_view, %arg62: !hal.buffer_view) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) attributes {iree.abi.stub} {
    %c0 = arith.constant 0 : index
    %c-1_i32 = arith.constant -1 : i32
    %0 = util.null : !hal.fence
    %device_0 = hal.devices.get %c0 : !hal.device
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    %1:61 = util.call @main$async(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %0, %fence) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.fence, !hal.fence) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view)
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) flags("None") : i32
    util.return %1#0, %1#1, %1#2, %1#3, %1#4, %1#5, %1#6, %1#7, %1#8, %1#9, %1#10, %1#11, %1#12, %1#13, %1#14, %1#15, %1#16, %1#17, %1#18, %1#19, %1#20, %1#21, %1#22, %1#23, %1#24, %1#25, %1#26, %1#27, %1#28, %1#29, %1#30, %1#31, %1#32, %1#33, %1#34, %1#35, %1#36, %1#37, %1#38, %1#39, %1#40, %1#41, %1#42, %1#43, %1#44, %1#45, %1#46, %1#47, %1#48, %1#49, %1#50, %1#51, %1#52, %1#53, %1#54, %1#55, %1#56, %1#57, %1#58, %1#59, %1#60 : !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_32_torch.float32: "0x040000000000803F06A5323FCF53F93EFFFCAD3E23D4723E0974293ED67FEC3D5609A53DC255663D1DBC203DD254E03C978B9C3CF27B5A3C0577183C15CAD43BAE7D943B383E4F3BDB9E103B5CD7C93ADBD98C3A8C94443A022E093AD774BF39A99A853953773A392B1F02391B9BB538D4757D3852DF303896DAF6371D43AC376D6B7037"
    }
  }
#-}
