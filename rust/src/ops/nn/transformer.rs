use crate::logical::LogicalGraph;
use crate::logical::{LogicalOp, LogicalTensor};
use crate::opcode::OpCode;
use crate::ops::basic::math::plan_add;
use crate::ops::nn::attention::plan_multihead_attention;
use crate::ops::nn::dense::plan_dense_op;
use crate::ops::nn::norm::plan_rms_norm;

#[derive(Debug, Clone)]
pub struct LogicalTransformerBlockOp {
    embed_dim: usize,
    mha_num_heads: usize,
    mha_head_dim: usize,
    ff_hidden_dim: usize,
    ff_output_dim: usize,
}

impl LogicalOp for LogicalTransformerBlockOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        _name: String,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
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
            self.mha_head_dim,
            self.mha_num_heads,
        );

        // // 3. Join back with residual stream
        let residual_stream_t1 = plan_add(graph, residual_stream_t0, &multi_head_attention_output);

        // // 4. Peform normalization before feed forward
        let normed_pre_ffw = plan_rms_norm(graph, &residual_stream_t1);
        // // 5. Apply feed forward layer
        let dense_ffw_op = plan_dense_op(
            graph,
            &normed_pre_ffw,
            self.embed_dim,
            self.ff_hidden_dim,
            self.ff_output_dim,
        );
        dense_ffw_op
    }
}

pub fn plan_transformer_block(
    graph: &mut LogicalGraph,
    input_seq: &LogicalTensor,
    embed_dim: usize,
    mha_num_heads: usize,
    mha_head_dim: usize,
    ff_hidden_dim: usize,
    ff_output_dim: usize,
) -> LogicalTensor {
    let op = LogicalTransformerBlockOp {
        embed_dim: embed_dim,
        mha_num_heads: mha_num_heads,
        mha_head_dim: mha_head_dim,
        ff_hidden_dim: ff_hidden_dim,
        ff_output_dim: ff_output_dim,
    };

    graph.register_call(OpCode::NnTransformer(op), &[input_seq])
}
