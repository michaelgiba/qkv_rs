use crate::logical::LogicalGraph;
use crate::logical::{LogicalOp, LogicalTensor, LogicalValueType};
use crate::opcode::OpCode;
use crate::ops::basic::inputs::plan_new_weights;
use crate::ops::basic::math::plan_mat_mul;

#[derive(Debug, Clone)]
pub struct LogicalDenseOp {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
}

impl LogicalOp for LogicalDenseOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        let input = inputs[0];

        let w1 = plan_new_weights(
            graph,
            &[self.input_dim, self.hidden_dim],
            LogicalValueType::F64,
        );
        let w2 = plan_new_weights(
            graph,
            &[self.hidden_dim, self.output_dim],
            LogicalValueType::F64,
        );

        let ff1_hidden = plan_mat_mul(graph, &input, &w1);
        let ff1_output = plan_mat_mul(graph, &ff1_hidden, &w2);

        ff1_output
    }
}

pub fn plan_dense_op(
    graph: &mut LogicalGraph,
    input: &LogicalTensor,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
) -> LogicalTensor {
    let op = LogicalDenseOp {
        input_dim: input_dim,
        hidden_dim: hidden_dim,
        output_dim: output_dim,
    };

    graph.register_call(OpCode::NnDense(op), &[input])
}
