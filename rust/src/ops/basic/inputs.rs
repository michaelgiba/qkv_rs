use crate::logical::{LogicalGraph, LogicalOp, LogicalTensor, LogicalValueType};
use crate::opcode::OpCode;

#[derive(Debug, Clone)]
pub struct LogicalPlaceholderOp {
    shape: Vec<usize>,
    value_type: LogicalValueType,
}
impl LogicalOp for LogicalPlaceholderOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        assert_eq!(inputs.len(), 0);
        graph.new_tensor(self.shape.clone(), self.value_type)
    }
}

pub fn plan_input_placeholder(
    graph: &mut LogicalGraph,
    shape: &[usize],
    value_type: LogicalValueType,
) -> LogicalTensor {
    graph.register_call(
        OpCode::BasicPlaceholder(LogicalPlaceholderOp {
            shape: shape.to_vec(),
            value_type,
        }),
        &[],
    )
}

pub fn plan_new_weights(
    graph: &mut LogicalGraph,
    shape: &[usize],
    value_type: LogicalValueType,
) -> LogicalTensor {
    plan_input_placeholder(graph, shape, value_type)
}
