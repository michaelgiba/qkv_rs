use crate::base_types::{LogicalGraph, LogicalOp, LogicalTensor, LOGICAL_EMPTY};

// struct PhysicalRmsNormOp;
// impl PhysicalOp for PhysicalRmsNormOp {
//     fn compute(&self, inputs: &[PhysicalOp]) -> PhysicalOp {
//         PHYSICAL_EMPTY
//     }
// }

pub struct LogicalRmsNormOp;
impl LogicalOp for LogicalRmsNormOp {
    // fn to_physical(&self) -> PhysicalOp {
    //     PhysicalRmsNormOp {}
    // }

    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        // let input = inputs[0];
        // let sum_squares: f32 = input.iter().map(|x| x * x).sum();
        // let mean_sum_square = sum_squares / (input.len() as f32);
        // let sum_squares_sqrt = mean_sum_square.sqrt() + 1e-6;
        // let inverse_sum_squares_sqrt = 1.0 / sum_squares_sqrt;
        // let normed_input: Vec<f32> = input.iter().map(|x| x * inverse_sum_squares_sqrt).collect();

        LOGICAL_EMPTY
    }
}

pub fn plan_rms_norm(graph: &mut LogicalGraph, tensor: &LogicalTensor) -> LogicalTensor {
    LogicalRmsNormOp {}.plan_forward(graph, &[tensor])
}
