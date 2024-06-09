use crate::base_types::{LogicalGraph, LogicalOp, LogicalTensor, LogicalValueType};
use crate::ops::basic::plan_concat;
use crate::ops::basic::{plan_dot_product, plan_mat_mul, plan_new_weights};
use crate::ops::position_embed::plan_rope;

#[derive(Debug)]
pub struct LogicalAttentionHeadOp {
    input_sequence_len: usize,
    input_embed_dim: usize,
    output_head_dim: usize,
    q_weights: LogicalTensor,
    k_weights: LogicalTensor,
    v_weights: LogicalTensor,
}

impl LogicalOp for LogicalAttentionHeadOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        let x = inputs[0];

        let q_proj = plan_mat_mul(graph, &x, &self.q_weights); // [N, output_head_dim]
        let k_proj = plan_mat_mul(graph, &x, &self.k_weights); // [N, output_head_dim]
        let v_proj = plan_mat_mul(graph, &x, &self.v_weights); // [N, output_head_dim]

        // apply rope just before attention on Q and K

        let positions = graph.empty_tensor(); // TODO: get positions

        let q_proj = plan_rope(graph, &q_proj, &positions, self.output_head_dim);
        let k_proj = plan_rope(graph, &k_proj, &positions, self.output_head_dim);

        let attention_scores = plan_dot_product(graph, &q_proj, &k_proj); // [N, N]

        let attended_v_proj = plan_mat_mul(graph, &attention_scores, &v_proj); // [N, output_head_dim]

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
        input_sequence_len: input_embed_dim,
        input_embed_dim: input_embed_dim,
        output_head_dim: output_head_dim,
        q_weights: plan_new_weights(
            graph,
            &[input_embed_dim, output_head_dim],
            LogicalValueType::F64,
        ),
        k_weights: plan_new_weights(
            graph,
            &[input_embed_dim, output_head_dim],
            LogicalValueType::F64,
        ),
        v_weights: plan_new_weights(
            graph,
            &[input_embed_dim, output_head_dim],
            LogicalValueType::F64,
        ),
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
    let head_outputs = plan_attention_head(graph, input_seq, input_embed_dim, output_head_dim);
    let concatted_head_outputs = plan_concat(graph, &[&head_outputs]);

    concatted_head_outputs
}
