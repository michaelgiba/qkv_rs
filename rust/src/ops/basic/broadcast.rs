use crate::{
    logical::{LogicalGraph, LogicalOp, LogicalTensor},
    opcode::OpCode,
};

#[derive(Debug, Clone)]
pub struct LogicalBroadcastOp {}

impl LogicalOp for LogicalBroadcastOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        _name: String,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        assert_eq!(inputs.len(), 2);
        let x = inputs[0];
        let y_to_match = inputs[1];

        assert!(x.is_scalar());
        assert!(!y_to_match.is_scalar());

        graph.new_tensor(y_to_match.shape.clone(), x.value_type)
    }
}

pub fn plan_broadcast(
    graph: &mut LogicalGraph,
    a: &LogicalTensor,
    b_to_match: &LogicalTensor,
) -> LogicalTensor {
    graph.register_call(OpCode::Broadcast(LogicalBroadcastOp {}), &[a, b_to_match])
}
