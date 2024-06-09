use crate::base_types::{LogicalGraph, LogicalOp, LogicalTensor, LogicalValueType, LOGICAL_EMPTY};

// struct PhysicalNoOp;
// impl PhysicalOp for PhysicalNoOp {
//     fn compute(inputs: &[PhysicalOp]) -> PhysicalOp {
//         PHYSICAL_EMPTY
//     }
// }

struct LogicalNoOp;
impl LogicalOp for LogicalNoOp {
    // fn to_physical() -> PhysicalOp {
    //     PhysicalNoOp {}
    // }
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        LOGICAL_EMPTY
    }
}

pub fn plan_new_weights(
    graph: &mut LogicalGraph,
    shape: &[usize],
    value_type: LogicalValueType,
) -> LogicalTensor {
    LOGICAL_EMPTY
}

pub fn plan_input_placeholder(
    graph: &mut LogicalGraph,
    shape: &[usize],
    value_type: LogicalValueType,
) -> LogicalTensor {
    // TODO
    LOGICAL_EMPTY
}

pub fn plan_no_op(graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
    // TODO
    LOGICAL_EMPTY
}

pub fn plan_mat_mul(
    graph: &mut LogicalGraph,
    a: &LogicalTensor,
    b: &LogicalTensor,
) -> LogicalTensor {
    // TODO
    LOGICAL_EMPTY
}

pub fn plan_weights(graph: &mut LogicalGraph, shape: &[usize]) -> LogicalTensor {
    // TODO
    LOGICAL_EMPTY
}

pub fn plan_dot_product(
    graph: &mut LogicalGraph,
    a: &LogicalTensor,
    b: &LogicalTensor,
) -> LogicalTensor {
    // TODO
    LOGICAL_EMPTY
}

pub fn plan_concat(graph: &mut LogicalGraph, tensors: &[&LogicalTensor]) -> LogicalTensor {
    // TODO
    LOGICAL_EMPTY
}

pub fn plan_add(graph: &mut LogicalGraph, a: &LogicalTensor, b: &LogicalTensor) -> LogicalTensor {
    // TODO
    LOGICAL_EMPTY
}
