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
struct LogicalGetIndexOp {
    index: usize,
}

impl LogicalOp for LogicalGetIndexOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor {
        assert_eq!(inputs.len(), 1);
        let tensor = inputs[0];

        assert!(self.index < tensor.num_elements());

        graph.scalar_tensor(tensor.value_type)        
    }
}



pub fn plan_get_element(
    graph: &mut LogicalGraph,
    tensor: &LogicalTensor,
    index: usize,
) -> LogicalTensor {

    graph.register_computation(Box::new(LogicalGetIndexOp{ index }), &[tensor])

}