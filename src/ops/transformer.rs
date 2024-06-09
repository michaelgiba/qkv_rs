use crate::base_types::LogicalGraph;
use crate::base_types::{LogicalOp, LogicalTensor};
use crate::ops::attention::plan_multihead_attention;
use crate::ops::basic::plan_add;
use crate::ops::dense::plan_dense_op;
use crate::ops::norm::plan_rms_norm;

pub struct LogicalTransformerBlockOp {
    embed_dim: usize,
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
}

impl LogicalOp for LogicalTransformerBlockOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        // 1. An input is provided as an N item array of embeddings (N, D_emb)
        // 2. We compute Q, K, V for each head. Each of them have shape (N, D_head)
        // 3. We compute the dot product of Q and K. The output of the dot product is shape (N, N)
        // 4. The attention scores are multiplied by the V outputs giving a new attention scaled output of (N, D_head)
        // 5. The attention scaled values are passed to a feed forward layer and converted first to (N, D_hidden) then back to (N, D_emb)

        let residual_stream_t0 = inputs[0];

        // 1. Apply layer normalization
        let normed_input = plan_rms_norm(graph, residual_stream_t0);

        // 2. Apply multi-head attention
        let multi_head_attention_output = plan_multihead_attention(
            graph,
            &normed_input,
            self.embed_dim,
            self.head_dim,
            self.num_heads,
        );

        // 3. Join back with residual stream
        let residual_stream_t1 = plan_add(graph, residual_stream_t0, &multi_head_attention_output);

        // 4. Peform normalization before feed forward
        let normed_pre_ffw = plan_rms_norm(graph, &residual_stream_t1);

        // 5. Apply feed forward layer
        let dense_ffw_op = plan_dense_op(
            graph,
            &[&normed_pre_ffw],
            self.head_dim,
            self.hidden_dim,
            self.embed_dim,
        );

        dense_ffw_op
    }
}

pub fn plan_transformer_block(
    graph: &mut LogicalGraph,
    input_seq: &LogicalTensor,
    embed_dim: usize,
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
) -> LogicalTensor {
    let op = LogicalTransformerBlockOp {
        embed_dim: embed_dim,
        hidden_dim: hidden_dim,
        num_heads: num_heads,
        head_dim: head_dim,
    };

    op.plan_forward(graph, &[input_seq])
}
