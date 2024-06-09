use crate::base_types::{LogicalGraph, LogicalOp, LogicalTensor};
use crate::ops::basic::math::{plan_divide, plan_mul, plan_sqrt, plan_square, plan_sum};

#[derive(Debug)]
pub struct LogicalRmsNormOp;
impl LogicalOp for LogicalRmsNormOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        let input = inputs[0];

        let num_elem = graph.scalar_u64(input.num_elements() as u64);
        let scalar_one = graph.scalar_f64(1.0);

        let squared = plan_square(graph, input);
        let sum_squares = plan_sum(graph, &squared);
        let mean_sum_square = plan_divide(graph, &sum_squares, &num_elem);

        let sum_squares_sqrt = plan_sqrt(graph, &mean_sum_square);

        let inverse_sum_squares_sqrt = plan_divide(graph, &scalar_one, &sum_squares_sqrt);
        let normed_input = plan_mul(graph, input, &inverse_sum_squares_sqrt);

        normed_input
    }
}

pub fn plan_rms_norm(graph: &mut LogicalGraph, tensor: &LogicalTensor) -> LogicalTensor {
    graph.register_computation(Box::new(LogicalRmsNormOp {}), &[tensor])
}