use crate::logical::{LogicalGraph, LogicalOp, LogicalTensor};
use crate::opcode::OpCode;

#[derive(Debug, Clone)]
pub struct SliceInterval {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone)]
pub struct LogicalSliceOp {
    slices: Vec<SliceInterval>,
}

impl LogicalOp for LogicalSliceOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        _name: String,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
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
    graph.register_call(
        OpCode::BasicSlice(LogicalSliceOp {
            slices: slice.to_vec(),
        }),
        &[tensor],
    )
}

#[derive(Debug, Clone)]
pub struct LogicalGetIndexOp {
    pub index: usize,
}

impl LogicalOp for LogicalGetIndexOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        _name: String,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        assert_eq!(inputs.len(), 1);
        let tensor = inputs[0];

        assert!(self.index < tensor.num_elements());

        graph.new_scalar_tensor(tensor.value_type)
    }
}

pub fn plan_get_element(
    graph: &mut LogicalGraph,
    tensor: &LogicalTensor,
    index: usize,
) -> LogicalTensor {
    graph.register_call(
        OpCode::BasicGetIndex(LogicalGetIndexOp { index }),
        &[tensor],
    )
}

#[derive(Debug, Clone)]
pub struct LogicalConcatOp {
    pub axis: usize,
}

impl LogicalOp for LogicalConcatOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        _name: String,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        assert!(!inputs.is_empty());
        assert!(self.axis < 2, "Only 2D tensors are supported for now");
        assert!(self.axis < inputs[0].shape.len(), "Axis out of bounds");

        let first_input = inputs[0];
        let dims = first_input.shape.len();

        for i in 1..inputs.len() {
            assert_eq!(inputs[i].shape.len(), dims);
            assert_eq!(inputs[i].value_type, first_input.value_type);

            for j in 0..dims {
                if j == self.axis {
                    continue;
                }
                assert_eq!(inputs[i].shape[j], first_input.shape[j]);
            }
        }

        let mut new_shape = first_input.shape.clone();
        new_shape[self.axis] = inputs.iter().map(|t| t.shape[self.axis]).sum();
        graph.new_tensor(new_shape, inputs[0].value_type)
    }
}

pub fn plan_concat(
    graph: &mut LogicalGraph,
    tensors: &[&LogicalTensor],
    axis: usize,
) -> LogicalTensor {
    graph.register_call(OpCode::BasicConcat(LogicalConcatOp { axis }), tensors)
}

#[derive(Debug, Clone)]
pub struct LogicalTransposeOp {}

impl LogicalOp for LogicalTransposeOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        _name: String,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        assert_eq!(inputs.len(), 1);
        let a = inputs[0];
        assert_eq!(a.shape.len(), 2);

        graph.new_tensor(vec![a.shape[1], a.shape[0]], a.value_type)
    }
}

pub fn plan_transpose(graph: &mut LogicalGraph, tensors: &[&LogicalTensor]) -> LogicalTensor {
    graph.register_call(OpCode::BasicTranspose(LogicalTransposeOp {}), tensors)
}
