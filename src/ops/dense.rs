use crate::base_types::LogicalGraph;
use crate::base_types::{LogicalOp, LogicalTensor, LogicalValueType};
use crate::ops::basic::{plan_mat_mul, plan_new_weights};

pub struct LogicalDenseOp {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    w1: LogicalTensor,
    w2: LogicalTensor,
}

impl LogicalOp for LogicalDenseOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        let input = inputs[0];

        let ff1_hidden = plan_mat_mul(graph, &input, &self.w1);
        let ff1_output = plan_mat_mul(graph, &ff1_hidden, &self.w2);

        ff1_output
    }
}

pub fn plan_dense_op(
    graph: &mut LogicalGraph,
    inputs: &[&LogicalTensor],
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
) -> LogicalTensor {
    let op = LogicalDenseOp {
        input_dim: input_dim,
        hidden_dim: hidden_dim,
        output_dim: output_dim,
        w1: plan_new_weights(graph, &[input_dim, hidden_dim], LogicalValueType::F64),
        w2: plan_new_weights(graph, &[hidden_dim, output_dim], LogicalValueType::F64),
    };

    op.plan_forward(graph, inputs)
}
