use super::activations::{plan_gelu, plan_softmax, ActivationType};
use crate::logical::LogicalGraph;
use crate::logical::{LogicalOp, LogicalTensor, LogicalValueType};
use crate::opcode::OpCode;
use crate::ops::basic::inputs::plan_new_weights;
use crate::ops::basic::math::{plan_mat_mul, plan_mul};

#[derive(Debug, Clone)]
pub struct LogicalDenseOp {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    activation_type: ActivationType,
}

impl LogicalOp for LogicalDenseOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        name: String,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        let input = inputs[0];

        let ff1_activations = match &self.activation_type {
            ActivationType::Softmax => {
                let ff_w1 = plan_new_weights(
                    graph,
                    &[self.input_dim, self.hidden_dim],
                    LogicalValueType::F64,
                    format!("{}_ff_w1", name),
                );
                let ff1_logits = plan_mat_mul(graph, &input, &ff_w1);
                plan_softmax(graph, &ff1_logits)
            }
            ActivationType::Gelu => {
                let ff_w1_gate = plan_new_weights(
                    graph,
                    &[self.input_dim, self.hidden_dim],
                    LogicalValueType::F64,
                    format!("{}_ff_w1_gate", name),
                );
                let ff1_gate_logits = plan_mat_mul(graph, &input, &ff_w1_gate);
                let ff_gate = plan_gelu(graph, &ff1_gate_logits);

                let ff_w1_linear = plan_new_weights(
                    graph,
                    &[self.input_dim, self.hidden_dim],
                    LogicalValueType::F64,
                    format!("{}_ff_w1_linear", name),
                );
                let ff1_linear_logits = plan_mat_mul(graph, &input, &ff_w1_linear);

                plan_mul(graph, &ff_gate, &ff1_linear_logits)
            }
        };

        let w2 = plan_new_weights(
            graph,
            &[self.hidden_dim, self.output_dim],
            LogicalValueType::F64,
            format!("{}_ff_w2", name),
        );

        let ff1_output = plan_mat_mul(graph, &ff1_activations, &w2);

        ff1_output
    }
}

pub fn plan_dense_op(
    graph: &mut LogicalGraph,
    input: &LogicalTensor,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    activation_type: ActivationType,
) -> LogicalTensor {
    let op = LogicalDenseOp {
        input_dim: input_dim,
        hidden_dim: hidden_dim,
        output_dim: output_dim,
        activation_type: activation_type,
    };

    graph.register_call(OpCode::NnDense(op), &[input])
}
