use crate::base_types::LogicalGraph;
use crate::base_types::{LogicalOp, LogicalTensor};

#[derive(Debug)]
pub struct LogicalSoftmaxOp {}

impl LogicalOp for LogicalSoftmaxOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        assert_eq!(inputs.len(), 1);
        let x = inputs[0];
        graph.new_tensor(x.shape.clone(), x.value_type)
    }
}

pub fn plan_softmax(graph: &mut LogicalGraph, input: &LogicalTensor) -> LogicalTensor {
    graph.register_computation(Box::new(LogicalSoftmaxOp {}), &[input])
}
