use crate::base_types::{LogicalGraph, LogicalOp, LogicalTensor};

#[derive(Debug)]
pub struct RotaryPositionEmbeddingOp {
    head_dim: usize,
}

impl LogicalOp for RotaryPositionEmbeddingOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        unimplemented!()
    }
}

pub fn plan_rope(
    graph: &mut LogicalGraph,
    inputs: &LogicalTensor,
    positions: &LogicalTensor,
    head_dim: usize,
) -> LogicalTensor {
    graph.register_computation(
        Box::new(RotaryPositionEmbeddingOp { head_dim: head_dim }),
        &[inputs, positions],
    )
}
