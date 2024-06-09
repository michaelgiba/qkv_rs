use crate::logical::LogicalGraph;
use crate::logical::{LogicalOp, LogicalTensor};
use crate::opcode::OpCode;

#[derive(Debug, Clone)]
pub struct LogicalSoftmaxOp {}

impl LogicalOp for LogicalSoftmaxOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        assert_eq!(inputs.len(), 1);
        let x = inputs[0];
        graph.new_tensor(x.shape.clone(), x.value_type)
    }
}

pub fn plan_softmax(graph: &mut LogicalGraph, input: &LogicalTensor) -> LogicalTensor {
    graph.register_call(OpCode::NnSoftmax(LogicalSoftmaxOp {}), &[input])
}
