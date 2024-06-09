use crate::base_types::{LogicalGraph, LogicalOp, LogicalTensor, LogicalValueType};

#[derive(Debug, Clone)]
struct SliceInterval {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone)]
struct LogicalSliceOp {
    slices: Vec<SliceInterval>,
}

impl LogicalOp for LogicalSliceOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        assert_eq!(inputs.len(), 1);
        let tensor = inputs[0];
        assert_eq!(tensor.shape.len(), self.slices.len());
        // Ensure all of the bounds are correct of the slices and within tensor shape
        for (i, slice) in self.slices.iter().enumerate() {
            assert!(slice.start < slice.end);
            assert!(slice.end <= tensor.shape[i]);
        }

        // Create new tensor of same type but sliced shape
        let output = graph.new_tensor(
            self.slices.iter().map(|s| s.end - s.start).collect(),
            tensor.value_type,
        );

        output
    }
}

pub fn plan_slice(
    graph: &mut LogicalGraph,
    tensor: &LogicalTensor,
    slice: &[SliceInterval],
) -> LogicalTensor {
    graph.register_computation(
        Box::new(LogicalSliceOp {
            slices: slice.to_vec(),
        }),
        &[tensor],
    )
}

#[derive(Debug, Clone)]
struct LogicalAddOp {}

impl LogicalOp for LogicalAddOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        // check for two inputs
        assert_eq!(inputs.len(), 2);
        let a = inputs[0];
        let b = inputs[1];
        // check for same shape
        assert_eq!(a.shape, b.shape);

        graph.new_tensor(a.shape.clone(), a.value_type)
    }
}

pub fn plan_element_wise_mul(
    graph: &mut LogicalGraph,
    a: &LogicalTensor,
    b: &LogicalTensor,
) -> LogicalTensor {
    unimplemented!()
}

pub fn plan_square(graph: &mut LogicalGraph, tensor: &LogicalTensor) -> LogicalTensor {
    unimplemented!()
}

pub fn plan_sum(graph: &mut LogicalGraph, tensor: &LogicalTensor) -> LogicalTensor {
    unimplemented!()
}

pub fn plan_divide(
    graph: &mut LogicalGraph,
    a: &LogicalTensor,
    b: &LogicalTensor,
) -> LogicalTensor {
    unimplemented!()
}

pub fn plan_sqrt(graph: &mut LogicalGraph, tensor: &LogicalTensor) -> LogicalTensor {
    unimplemented!()
}

pub fn plan_mul(graph: &mut LogicalGraph, a: &LogicalTensor, b: &LogicalTensor) -> LogicalTensor {
    unimplemented!()
}

pub fn plan_new_weights(
    graph: &mut LogicalGraph,
    shape: &[usize],
    value_type: LogicalValueType,
) -> LogicalTensor {
    unimplemented!()
}

pub fn plan_input_placeholder(
    graph: &mut LogicalGraph,
    shape: &[usize],
    value_type: LogicalValueType,
) -> LogicalTensor {
    unimplemented!()
}

pub fn plan_no_op(graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
    unimplemented!()
}

pub fn plan_mat_mul(
    graph: &mut LogicalGraph,
    a: &LogicalTensor,
    b: &LogicalTensor,
) -> LogicalTensor {
    unimplemented!()
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

pub fn plan_add(graph: &mut LogicalGraph, a: &LogicalTensor, b: &LogicalTensor) -> LogicalTensor {
    unimplemented!()
}
