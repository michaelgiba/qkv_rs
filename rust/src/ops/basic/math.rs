use crate::{
    logical::{LogicalGraph, LogicalOp, LogicalTensor, LogicalValueType},
    opcode::OpCode,
    ops::basic::broadcast::plan_broadcast,
    ops::basic::slice::plan_get_element,
};

fn default_logical_binary_op_inputs(
    graph: &mut LogicalGraph,
    a: LogicalTensor,
    b: LogicalTensor,
) -> (LogicalTensor, LogicalTensor) {
    if a.value_type != b.value_type {
        panic!(
            "Cannot perform binary operation on tensors of different value types: {:?} and {:?}.",
            a.value_type, b.value_type
        );
    }

    if a.is_scalar() && b.is_scalar() {
        return (a, b);
    } else if a.is_scalar() {
        let a_broadcast = plan_broadcast(graph, &a, &b);
        return (a_broadcast, b);
    } else if b.is_scalar() {
        let b_broadcast = plan_broadcast(graph, &b, &a);
        return (a, b_broadcast);
    } else {
        assert_eq!(a.shape, b.shape);
        return (a, b);
    }
}

fn default_logical_binary_op_output(
    graph: &mut LogicalGraph,
    inputs: &[&LogicalTensor],
) -> LogicalTensor {
    assert_eq!(inputs.len(), 2);
    let a = inputs[0];
    let b = inputs[1];
    assert_eq!(a.shape, b.shape);
    return graph.new_tensor(a.shape.clone(), a.value_type);
}

#[derive(Debug, Clone)]
pub struct LogicalAddOp {}

impl LogicalOp for LogicalAddOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        name: String,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        default_logical_binary_op_output(graph, inputs)
    }
}

pub fn plan_add(graph: &mut LogicalGraph, a: &LogicalTensor, b: &LogicalTensor) -> LogicalTensor {
    let (a, b) = default_logical_binary_op_inputs(graph, a.clone(), b.clone());

    graph.register_call(OpCode::BasicAdd(LogicalAddOp {}), &[&a, &b])
}

#[derive(Debug, Clone)]
pub struct LogicalSubOp {}
impl LogicalOp for LogicalSubOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        name: String,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        default_logical_binary_op_output(graph, inputs)
    }
}

pub fn plan_sub(graph: &mut LogicalGraph, a: &LogicalTensor, b: &LogicalTensor) -> LogicalTensor {
    let (a, b) = default_logical_binary_op_inputs(graph, a.clone(), b.clone());

    graph.register_call(OpCode::BasicSub(LogicalSubOp {}), &[&a, &b])
}

#[derive(Debug, Clone)]
pub struct LogicalMulOp {}
impl LogicalOp for LogicalMulOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        name: String,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        default_logical_binary_op_output(graph, inputs)
    }
}

pub fn plan_mul(graph: &mut LogicalGraph, a: &LogicalTensor, b: &LogicalTensor) -> LogicalTensor {
    let (a, b) = default_logical_binary_op_inputs(graph, a.clone(), b.clone());

    graph.register_call(OpCode::BasicMul(LogicalMulOp {}), &[&a, &b])
}

#[derive(Debug, Clone)]
pub struct LogicalDivOp {}
impl LogicalOp for LogicalDivOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        name: String,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        default_logical_binary_op_output(graph, inputs)
    }
}

pub fn plan_divide(
    graph: &mut LogicalGraph,
    a: &LogicalTensor,
    b: &LogicalTensor,
) -> LogicalTensor {
    let (a, b) = default_logical_binary_op_inputs(graph, a.clone(), b.clone());

    graph.register_call(OpCode::BasicDiv(LogicalDivOp {}), &[&a, &b])
}

pub fn plan_square(graph: &mut LogicalGraph, tensor: &LogicalTensor) -> LogicalTensor {
    plan_mul(graph, tensor, tensor)
}

#[derive(Debug, Clone)]
pub struct LogicalSumOp {}

impl LogicalOp for LogicalSumOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        name: String,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        assert_eq!(inputs.len(), 1);
        let input: &LogicalTensor = inputs[0];

        if input.is_scalar() {
            return input.clone();
        }

        let mut sum = graph.scalar_f64(0.0);

        for i in 0..input.num_elements() {
            let elem = plan_get_element(graph, input, i);
            sum = plan_add(graph, &sum, &elem);
        }

        sum
    }
}

pub fn plan_sum(graph: &mut LogicalGraph, tensor: &LogicalTensor) -> LogicalTensor {
    graph.register_call(OpCode::BasicSum(LogicalSumOp {}), &[tensor])
}

#[derive(Debug, Clone)]
pub struct LogicalSqrtOp {}

impl LogicalOp for LogicalSqrtOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        name: String,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        assert_eq!(inputs.len(), 1);
        let input: &LogicalTensor = inputs[0];

        graph.new_tensor(input.shape.clone(), input.value_type)
    }
}

pub fn plan_sqrt(graph: &mut LogicalGraph, tensor: &LogicalTensor) -> LogicalTensor {
    graph.register_call(OpCode::BasicSqrt(LogicalSqrtOp {}), &[tensor])
}

#[derive(Debug, Clone)]
pub struct LogicalMatMulOp {}

impl LogicalOp for LogicalMatMulOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        name: String,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        assert_eq!(inputs.len(), 2);
        let a = inputs[0];
        let b = inputs[1];

        assert_eq!(a.shape.len(), 2);
        assert_eq!(b.shape.len(), 2);
        assert_eq!(a.shape[1], b.shape[0]);

        let mut shape = vec![];
        shape.push(a.shape[0]);
        shape.push(b.shape[1]);

        graph.new_tensor(shape, a.value_type)
    }
}

pub fn plan_mat_mul(
    graph: &mut LogicalGraph,
    a: &LogicalTensor,
    b: &LogicalTensor,
) -> LogicalTensor {
    graph.register_call(OpCode::BasicMatMul(LogicalMatMulOp {}), &[a, b])
}

#[derive(Debug, Clone)]
pub struct LogicalDotProductOp {}

impl LogicalOp for LogicalDotProductOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        name: String,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        assert_eq!(inputs.len(), 2);
        let a = inputs[0]; // [seq_len, head_dim]
        let b = inputs[1]; // [seq_len, head_dim]

        assert_eq!(a.shape.len(), 2);
        assert_eq!(b.shape.len(), 2);
        assert_eq!(a.shape[0], b.shape[0]);

        graph.new_tensor([a.shape[0], a.shape[0]].to_vec(), LogicalValueType::F64)
    }
}

pub fn plan_dot_product(
    graph: &mut LogicalGraph,
    a: &LogicalTensor,
    b: &LogicalTensor,
) -> LogicalTensor {
    graph.register_call(OpCode::BasicDotProduct(LogicalDotProductOp {}), &[a, b])
}
