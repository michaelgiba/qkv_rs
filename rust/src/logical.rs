use crate::opcode::OpCodes;
use crate::ops::basic::inputs;
use std::{collections::HashMap, collections::HashSet, collections::VecDeque, fmt::Debug};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogicalValueType {
    F64,
    U32,
}

impl LogicalValueType {
    pub fn zero(&self) -> Vec<u8> {
        match self {
            LogicalValueType::F64 => f64_value_to_bytes(0.0),
            LogicalValueType::U32 => 0u32.to_le_bytes().to_vec(),
        }
    }

    pub fn as_f64(&self, bytes: &[u8]) -> f64 {
        match self {
            LogicalValueType::F64 => f64_value_from_bytes(bytes),
            LogicalValueType::U32 => {
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
            }
        }
    }

    pub fn from_f64(&self, value: f64) -> Vec<u8> {
        match self {
            LogicalValueType::F64 => f64_value_to_bytes(value),
            LogicalValueType::U32 => (value as u32).to_le_bytes().to_vec(),
        }
    }
}

impl LogicalValueType {
    pub fn size(&self) -> usize {
        match self {
            LogicalValueType::F64 => std::mem::size_of::<f64>(),
            LogicalValueType::U32 => std::mem::size_of::<u32>(),
        }
    }
}

pub fn f64_value_from_bytes(bytes: &[u8]) -> f64 {
    let mut array = [0; 8];
    array.copy_from_slice(bytes);
    f64::from_le_bytes(array)
}

pub fn f64_value_to_bytes(value: f64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalTensor {
    pub id: usize,
    pub shape: Vec<usize>,
    pub value_type: LogicalValueType,
}

impl LogicalTensor {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.len() == 1 && self.shape[0] == 1
    }
}

pub trait LogicalOp: Debug {
    fn logical_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor])
        -> LogicalTensor;
    fn opcode(&self) -> OpCodes;
}
#[derive(Debug)]
pub struct LiteralOp<T> {
    pub shape: Vec<usize>,
    pub value: T,
}

impl LogicalOp for LiteralOp<f64> {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        _inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        graph.new_tensor(self.shape.clone(), LogicalValueType::F64)
    }
    fn opcode(&self) -> OpCodes {
        OpCodes::Literal
    }
}

impl LogicalOp for LiteralOp<u64> {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        _inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        graph.new_tensor(self.shape.clone(), LogicalValueType::U32)
    }

    fn opcode(&self) -> OpCodes {
        OpCodes::Literal
    }
}

#[derive(Debug)]
pub struct LogicalReturnOp {
    returned_from_op: Box<dyn LogicalOp>,
}
impl LogicalOp for LogicalReturnOp {
    fn logical_forward(
        &self,
        graph: &mut LogicalGraph,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        assert_eq!(inputs.len(), 1);
        graph.new_tensor(inputs[0].shape.clone(), inputs[0].value_type)
    }

    fn opcode(&self) -> OpCodes {
        OpCodes::Return
    }
}

#[derive(Debug)]
pub struct LogicalGraphCall {
    pub op: Box<dyn LogicalOp>,
    pub input_tensor_ids: Vec<usize>,
    pub output_tensor_id: usize,
}

#[derive(Debug)]
pub struct LogicalGraph {
    tensor_id_to_tensor: HashMap<usize, LogicalTensor>,
    output_tensor_id_to_call: HashMap<usize, LogicalGraphCall>,
}

impl LogicalGraph {
    pub fn new() -> LogicalGraph {
        let mut graph = LogicalGraph {
            tensor_id_to_tensor: HashMap::new(),
            output_tensor_id_to_call: HashMap::new(),
        };
        graph.empty_tensor();
        graph
    }

    pub fn register_call(
        &mut self,
        op: Box<dyn LogicalOp>,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        let output = op.logical_forward(self, inputs);

        let input_tensor_ids: Vec<usize> = inputs.iter().map(|t| t.id).collect();

        if self.output_tensor_id_to_call.contains_key(&output.id) {
            return self.register_call(
                Box::new(LogicalReturnOp {
                    returned_from_op: op,
                }),
                &[&output],
            );
        }

        let call = LogicalGraphCall {
            op: op,
            input_tensor_ids,
            output_tensor_id: output.id,
        };
        self.output_tensor_id_to_call.insert(output.id, call);
        output
    }

    pub fn call_for_tensor_id(&self, id: usize) -> &LogicalGraphCall {
        self.output_tensor_id_to_call.get(&id).unwrap()
    }

    pub fn new_tensor(&mut self, shape: Vec<usize>, value_type: LogicalValueType) -> LogicalTensor {
        let id = self.tensor_id_to_tensor.len();
        let tensor = LogicalTensor {
            id,
            shape,
            value_type,
        };
        self.tensor_id_to_tensor.insert(id, tensor);
        self.tensor_id_to_tensor.get(&id).unwrap().clone()
    }

    pub fn empty_tensor(&mut self) -> LogicalTensor {
        self.new_tensor(vec![], LogicalValueType::F64)
    }

    pub fn scalar_tensor(&mut self, value_type: LogicalValueType) -> LogicalTensor {
        self.new_tensor(vec![1], value_type)
    }

    pub fn scalar_f64(&mut self, value: f64) -> LogicalTensor {
        self.register_call(
            Box::new(LiteralOp {
                value: value,
                shape: vec![1],
            }),
            &[],
        )
    }

    pub fn scalar_u64(&mut self, value: u64) -> LogicalTensor {
        self.register_call(
            Box::new(LiteralOp {
                value: value,
                shape: vec![1],
            }),
            &[],
        )
    }

    pub fn get_tensor(&self, id: usize) -> &LogicalTensor {
        self.tensor_id_to_tensor.get(&id).unwrap()
    }
}
