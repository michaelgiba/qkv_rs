use std::result;

use crate::base_types::{LogicalGraph, LogicalOp, LogicalTensor, LogicalValueType};
use crate::ops::basic::inputs::plan_new_weights;
use crate::ops::basic::math::plan_concat;
use crate::ops::basic::math::{plan_dot_product, plan_mat_mul};
use crate::ops::nn::activations::plan_softmax;
use crate::ops::nn::position_embed::plan_rope;

#[derive(Debug)]
pub struct LogicalAttentionHeadOp {
    input_embed_dim: usize,
    output_head_dim: usize,
}

impl LogicalOp for LogicalAttentionHeadOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        let x = inputs[0];

        let q_weights = plan_new_weights(
            graph,
            &[self.input_embed_dim, self.output_head_dim],
            LogicalValueType::F64,
        );
        let k_weights = plan_new_weights(
            graph,
            &[self.input_embed_dim, self.output_head_dim],
            LogicalValueType::F64,
        );
        let v_weights = plan_new_weights(
            graph,
            &[self.input_embed_dim, self.output_head_dim],
            LogicalValueType::F64,
        );

        let q_proj = plan_mat_mul(graph, &x, &q_weights); // [seq_n, output_head_dim]
        let k_proj = plan_mat_mul(graph, &x, &k_weights); // [seq_n, output_head_dim]
        let v_proj = plan_mat_mul(graph, &x, &v_weights); // [seq_n, output_head_dim]

        // apply rope just before attention on Q and K

        let positions = graph.new_tensor(q_proj.shape.clone(), q_proj.value_type);

        let q_proj = plan_rope(graph, &q_proj, &positions, self.output_head_dim);
        let k_proj = plan_rope(graph, &k_proj, &positions, self.output_head_dim);

        // Compute attention scores
        let attention_logits = plan_dot_product(graph, &q_proj, &k_proj); // [seq_n, seq_n]
        let attention_activations = plan_softmax(graph, &attention_logits);

        // Attend values based on scores
        let attended_v_proj = plan_mat_mul(graph, &attention_activations, &v_proj); // [seq_n, output_head_dim]

        attended_v_proj
    }
}

pub fn plan_attention_head(
    graph: &mut LogicalGraph,
    input_seq: &LogicalTensor,
    input_embed_dim: usize,
    output_head_dim: usize,
) -> LogicalTensor {
    let head_op = LogicalAttentionHeadOp {
        input_embed_dim: input_embed_dim,
        output_head_dim: output_head_dim,
    };

    graph.register_computation(Box::new(head_op), &[input_seq])
}

pub fn plan_multihead_attention(
    graph: &mut LogicalGraph,
    input_seq: &LogicalTensor,
    input_embed_dim: usize,
    output_head_dim: usize,
    num_heads: usize,
) -> LogicalTensor {
    // for each of the heads, we need to run the attention head and append to a list which we will concat
    let head_outputs: Vec<LogicalTensor> = (0..num_heads)
        .map(|_| plan_attention_head(graph, input_seq, input_embed_dim, output_head_dim))
        .collect();

    let head_refs = head_outputs.iter().collect::<Vec<&LogicalTensor>>();

    let concatted_head_outputs = plan_concat(graph, head_refs.as_slice(), 1); // [seq_n, n_heads * output_head_dim]

    let weights_out = plan_new_weights(
        graph,
        &[num_heads * output_head_dim, input_embed_dim],
        LogicalValueType::F64,
    ); // [n_heads * output_head_dim, input_embed_dim]

    plan_mat_mul(graph, &concatted_head_outputs, &weights_out) // [seq_n, input_embed_dim]
}
