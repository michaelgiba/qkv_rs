
use crate::base_types::{LogicalGraph, LogicalOp, LogicalTensor, LogicalValueType};


pub fn plan_input_placeholder(
    graph: &mut LogicalGraph,
    shape: &[usize],
    value_type: LogicalValueType,
) -> LogicalTensor {

    graph.new_tensor(shape.to_vec(), value_type)
}

pub fn plan_new_weights(
    graph: &mut LogicalGraph,
    shape: &[usize],
    value_type: LogicalValueType,
) -> LogicalTensor {
    plan_input_placeholder(graph, shape, value_type)
}
