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
        name: String,
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
    name: String,
) -> LogicalTensor {
    let output = graph.register_call_with_name(
        OpCode::BasicPlaceholder(LogicalPlaceholderOp {
            shape: shape.to_vec(),
            value_type,
        }),
        &[],
        name.to_string(),
    );
    output
}

pub fn plan_new_weights(
    graph: &mut LogicalGraph,
    shape: &[usize],
    value_type: LogicalValueType,
    name: String,
) -> LogicalTensor {
    plan_input_placeholder(graph, shape, value_type, name)
}
