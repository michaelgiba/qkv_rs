use crate::base_types::{LogicalGraph, LogicalOp, LogicalTensor, LogicalValueType, LOGICAL_EMPTY};
use crate::ops::basic::{plan_dot_product, plan_mat_mul, plan_new_weights};

use super::basic::plan_concat;

pub struct RotaryPositionEmbeddingOp {
    head_dim: usize,
}

impl LogicalOp for RotaryPositionEmbeddingOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        // TODO
        LOGICAL_EMPTY
    }
}

pub fn plan_rope(
    graph: &mut LogicalGraph,
    inputs: &LogicalTensor,
    positions: &LogicalTensor,
    head_dim: usize,
) -> LogicalTensor {
    let rope_op = RotaryPositionEmbeddingOp { head_dim: head_dim };

    rope_op.plan_forward(graph, &[inputs, positions])
}
