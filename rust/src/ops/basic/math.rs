
use crate::{base_types::{LogicalGraph, LogicalOp, LogicalTensor, LogicalValueType}, ops::basic::slice::plan_get_element};

fn default_binary_tensor_op(graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
    assert_eq!(inputs.len(), 2);
    let a = inputs[0];
    let b = inputs[1];
    // check for same shape or scalars 
    if a.is_scalar() && b.is_scalar() {
        return graph.scalar_tensor(a.value_type)    
    } else if a.is_scalar() {
        return graph.new_tensor(b.shape.clone(), b.value_type)            
    } else if b.is_scalar() {
        return graph.new_tensor(a.shape.clone(), a.value_type)            
    } else {
        assert_eq!(a.shape, b.shape);
        return graph.new_tensor(a.shape.clone(), a.value_type)    
    }
}

#[derive(Debug, Clone)]
struct LogicalAddOp {}

impl LogicalOp for LogicalAddOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        default_binary_tensor_op(graph, inputs)
    }
}

pub fn plan_add(graph: &mut LogicalGraph, a: &LogicalTensor, b: &LogicalTensor) -> LogicalTensor {
    graph.register_computation(Box::new(LogicalAddOp {}), &[a, b])
}

#[derive(Debug, Clone)]
struct LogicalSubOp {}
impl LogicalOp for LogicalSubOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        default_binary_tensor_op(graph, inputs)
    }
}

pub fn plan_sub(graph: &mut LogicalGraph, a: &LogicalTensor, b: &LogicalTensor) -> LogicalTensor {
    graph.register_computation(Box::new(LogicalSubOp {}), &[a, b])
}


#[derive(Debug, Clone)]
struct LogicalMulOp {}
impl LogicalOp for LogicalMulOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        default_binary_tensor_op(graph, inputs)
    }
}


pub fn plan_mul(graph: &mut LogicalGraph, a: &LogicalTensor, b: &LogicalTensor) -> LogicalTensor {
    graph.register_computation(Box::new(LogicalMulOp {}), &[a, b])
}


#[derive(Debug, Clone)]
struct LogicalDivOp {}
impl LogicalOp for LogicalDivOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        default_binary_tensor_op(graph, inputs)
    }
}


pub fn plan_divide(
    graph: &mut LogicalGraph,
    a: &LogicalTensor,
    b: &LogicalTensor,
) -> LogicalTensor {
    graph.register_computation(Box::new(LogicalDivOp {}), &[a, b])

}



pub fn plan_square(graph: &mut LogicalGraph, tensor: &LogicalTensor) -> LogicalTensor {
    plan_mul(graph, tensor, tensor)
}

#[derive(Debug, Clone)]
struct LogicalSumOp {}

impl LogicalOp for LogicalSumOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        assert_eq!(inputs.len(), 1);
        let input: &LogicalTensor = inputs[0];
        
        if input.is_scalar() {
            return input.clone()
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
    graph.register_computation(Box::new(LogicalSumOp {}), &[tensor])
}


#[derive(Debug, Clone)]
struct LogicalSqrtOp {}

impl LogicalOp for LogicalSqrtOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        assert_eq!(inputs.len(), 1);
        let input: &LogicalTensor = inputs[0];

        graph.new_tensor(input.shape.clone(), input.value_type)
    }
}


pub fn plan_sqrt(graph: &mut LogicalGraph, tensor: &LogicalTensor) -> LogicalTensor {
    graph.register_computation(Box::new(LogicalSqrtOp{}), &[tensor])
}

#[derive(Debug, Clone)]
struct LogicalMatMulOp {}

impl LogicalOp for LogicalMatMulOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
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
    graph.register_computation(Box::new(LogicalMatMulOp{}), &[a, b])
}

pub fn plan_dot_product(
    graph: &mut LogicalGraph,
    a: &LogicalTensor,
    b: &LogicalTensor,
) -> LogicalTensor {
    unimplemented!()
}

pub fn plan_concat(graph: &mut LogicalGraph, tensors: &[&LogicalTensor]) -> LogicalTensor {
    unimplemented!()
}
