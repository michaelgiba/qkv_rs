use crate::{
    logical::{LogicalGraph, LogicalGraphCall, LogicalOp, LogicalTensor, LogicalValueType},
    opcode::OpCode,
    ops::basic::inputs,
};
use core::{f64, panic};
use std::{collections::HashMap, collections::HashSet, fmt::Debug};

#[derive(Debug, Clone)]
pub struct PhysicalValue {
    pub value_type: LogicalValueType,
    pub value: Vec<u8>,
}
impl PhysicalValue {
    pub fn zero(value_type: LogicalValueType) -> PhysicalValue {
        PhysicalValue {
            value_type,
            value: value_type.zero(),
        }
    }

    pub fn as_f64(&self) -> f64 {
        self.value_type.as_f64(&self.value)
    }

    pub fn from_f64(value: f64) -> PhysicalValue {
        PhysicalValue {
            value_type: LogicalValueType::F64,
            value: LogicalValueType::F64.from_f64(value),
        }
    }

    pub fn as_u32(&self) -> u32 {
        self.value_type.as_u32(&self.value)
    }

    pub fn from_u32(value: u32) -> PhysicalValue {
        PhysicalValue {
            value_type: LogicalValueType::U32,
            value: LogicalValueType::U32.from_u32(value),
        }
    }

    pub fn add(&self, other: PhysicalValue) -> PhysicalValue {
        let f64_value = self.as_f64() + other.as_f64();
        PhysicalValue {
            value_type: self.value_type,
            value: self.value_type.from_f64(f64_value),
        }
    }

    pub fn sub(&self, other: PhysicalValue) -> PhysicalValue {
        let f64_value = self.as_f64() - other.as_f64();
        PhysicalValue {
            value_type: self.value_type,
            value: self.value_type.from_f64(f64_value),
        }
    }

    pub fn mul(&self, other: PhysicalValue) -> PhysicalValue {
        let f64_value = self.as_f64() * other.as_f64();
        PhysicalValue {
            value_type: self.value_type,
            value: self.value_type.from_f64(f64_value),
        }
    }

    pub fn div(&self, other: PhysicalValue) -> PhysicalValue {
        let f64_value = self.as_f64() / other.as_f64();
        PhysicalValue {
            value_type: self.value_type,
            value: self.value_type.from_f64(f64_value),
        }
    }

    pub fn sqrt(&self) -> PhysicalValue {
        let f64_value = self.as_f64().sqrt();
        PhysicalValue {
            value_type: self.value_type,
            value: self.value_type.from_f64(f64_value),
        }
    }
}

#[derive(Clone)]
pub struct PhysicalTensor {
    spec: LogicalTensor,
    values: Vec<u8>,
}

impl Debug for PhysicalTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let values_in_logical_type = self
            .values
            .chunks_exact(self.spec.value_type.size())
            .map(|x| self.spec.value_type.as_f64(x))
            .collect::<Vec<f64>>();

        f.debug_struct("PhysicalTensor")
            .field("spec", &self.spec)
            .field("values", &values_in_logical_type)
            .finish()
    }
}

impl PhysicalTensor {
    pub fn alloc(spec: LogicalTensor) -> PhysicalTensor {
        if spec.is_scalar() {
            PhysicalTensor {
                spec: spec.clone(),
                // We put the actual value of scalar in values even though the shape is scalar [0]
                values: vec![0; spec.value_type.size()],
            }
        } else {
            PhysicalTensor {
                spec: spec.clone(),
                values: vec![0; spec.num_elements() * spec.value_type.size()],
            }
        }
    }

    pub fn num_elements(&self) -> usize {
        self.spec.num_elements()
    }

    fn get_element_raw(&self, index: usize) -> &[u8] {
        let start = index * self.spec.value_type.size();
        let end = start + self.spec.value_type.size();
        &self.values[start..end]
    }

    pub fn get_element(&self, index: usize) -> PhysicalValue {
        let raw_value = self.get_element_raw(index);
        PhysicalValue {
            value_type: self.spec.value_type,
            value: raw_value.to_vec(),
        }
    }

    pub fn set_element(&mut self, index: usize, value: PhysicalValue) {
        let raw_bytes = value.value;
        let start = index * self.spec.value_type.size();
        let end = start + self.spec.value_type.size();
        self.values[start..end].copy_from_slice(&raw_bytes);
    }

    pub fn is_scalar(&self) -> bool {
        self.spec.is_scalar()
    }

    pub fn get_scalar_value(&self) -> PhysicalValue {
        assert!(self.is_scalar());
        self.get_element(0)
    }

    pub fn set_scalar_value(&mut self, value: PhysicalValue) {
        assert!(self.is_scalar());
        self.set_element(0, value);
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalGraphCall {
    pub op_type: PhysicalOpType,
    pub input_tensor_ids: Vec<usize>,
    pub output_tensor_id: usize,
}

pub struct PhysicalGraph {
    tensor_id_to_tensor: HashMap<usize, PhysicalTensor>,
    tensor_id_to_calls: HashMap<usize, Vec<PhysicalGraphCall>>,
    outputs: Vec<usize>,
}

impl PhysicalGraph {
    pub fn compile(logical_graph: LogicalGraph, outputs: &[&LogicalTensor]) -> PhysicalGraph {
        let mut tensor_id_to_tensor: HashMap<usize, PhysicalTensor> = HashMap::new();
        let mut tensor_id_to_calls: HashMap<usize, Vec<PhysicalGraphCall>> = HashMap::new();
        let output_ids: Vec<usize> = outputs.iter().map(|x| x.id).collect();
        let mut stack = output_ids.clone();

        while let Some(tensor_id) = stack.pop() {
            // If we've already computed this value there is nothing to do
            if tensor_id_to_tensor.contains_key(&tensor_id) {
                continue;
            }

            // Otherwise we push this back on the stack and compute the dependencies
            let logical_call = logical_graph.call_for_tensor_id(tensor_id);

            let need_allocation_input_ids: Vec<usize> = logical_call
                .input_tensor_ids
                .iter()
                .filter(|x| !tensor_id_to_tensor.contains_key(x))
                .map(|x| *x)
                .collect();

            if need_allocation_input_ids.is_empty() {
                //  We have all the dependencies, we can allocate the value
                let logical_output = logical_graph.get_tensor(tensor_id);

                if !tensor_id_to_tensor.contains_key(&tensor_id) {
                    tensor_id_to_tensor
                        .insert(tensor_id, PhysicalTensor::alloc(logical_output.clone()));
                }

                let physical_call = PhysicalGraphCall {
                    op_type: logical_call_to_physical_op(logical_call),
                    input_tensor_ids: logical_call.input_tensor_ids.clone(),
                    output_tensor_id: tensor_id,
                };

                tensor_id_to_calls
                    .entry(tensor_id)
                    .or_insert(vec![])
                    .push(physical_call);
            } else {
                stack.push(tensor_id);
                stack.extend(need_allocation_input_ids.iter())
            }
        }

        PhysicalGraph {
            tensor_id_to_tensor,
            tensor_id_to_calls: tensor_id_to_calls,
            outputs: output_ids,
        }
    }

    fn determine_call_sequence(&self, tensor_id: usize) -> Vec<PhysicalGraphCall> {
        let mut stack: Vec<usize> = vec![];
        let mut call_sequence: Vec<PhysicalGraphCall> = vec![];
        let mut queued_tensor_ids: HashSet<usize> = HashSet::new();

        stack.push(tensor_id);

        while let Some(tensor_id) = stack.pop() {
            if queued_tensor_ids.contains(&tensor_id) {
                continue;
            }

            let calls = self.tensor_id_to_calls.get(&tensor_id).unwrap();

            let need_computation_tensors: Vec<usize> = calls
                .iter()
                .flat_map(|x| x.input_tensor_ids.clone())
                .filter(|x| !queued_tensor_ids.contains(x))
                .collect();

            if need_computation_tensors.is_empty() {
                call_sequence.extend(calls.clone());
                queued_tensor_ids.insert(tensor_id);
            } else {
                stack.push(tensor_id);
                stack.extend(need_computation_tensors);
            }
        }

        call_sequence.into()
    }

    pub fn set_value_for_tensor(&mut self, tensor: &LogicalTensor, values: Vec<f64>) {
        assert_eq!(values.len(), tensor.num_elements());

        let mut physical_tensor = PhysicalTensor::alloc(tensor.clone());

        for (i, &value) in values.iter().enumerate() {
            let physical_value = match tensor.value_type {
                LogicalValueType::F64 => PhysicalValue::from_f64(value),
                LogicalValueType::U32 => PhysicalValue::from_u32(value as u32),
            };
            physical_tensor.set_element(i, physical_value);
        }

        self.tensor_id_to_tensor.insert(tensor.id, physical_tensor);
    }

    pub fn compute(&mut self, target: &LogicalTensor) -> &PhysicalTensor {
        assert!(self.tensor_id_to_tensor.contains_key(&target.id));

        for call in self.determine_call_sequence(target.id) {
            let input_tensors: Vec<&PhysicalTensor> = call
                .input_tensor_ids
                .iter()
                .map(|id| self.tensor_id_to_tensor.get(id).unwrap())
                .collect();

            let output = self
                .tensor_id_to_tensor
                .get(&call.output_tensor_id)
                .unwrap();

            // TODO: Optimize this to avoid copy
            let mut output_copy = output.clone();

            call.op_type
                .get_physical()
                .physical_forward(&input_tensors, &mut output_copy);

            self.tensor_id_to_tensor
                .insert(call.output_tensor_id, output_copy);
        }

        self.tensor_id_to_tensor.get(&target.id).unwrap()
    }
}

pub trait PhysicalOp {
    fn physical_forward(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor);
}

#[derive(Debug, Clone)]
pub struct PhysicalLiteralF64Op {
    pub value: f64,
}
impl PhysicalOp for PhysicalLiteralF64Op {
    fn physical_forward(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        for i in 0..output.num_elements() {
            output.set_element(i, PhysicalValue::from_f64(self.value));
        }
        println!("{:?} {:?} {:?}", self, _inputs, output);
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalLiteralU32Op {
    pub value: u32,
}
impl PhysicalOp for PhysicalLiteralU32Op {
    fn physical_forward(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        println!("{:?} {:?} {:?}", self, _inputs, output);

        for i in 0..output.num_elements() {
            output.set_element(i, PhysicalValue::from_u32(self.value));
        }
    }
}

#[derive(Debug, Clone)]

pub struct PhysicalBroadcastOp {}
impl PhysicalOp for PhysicalBroadcastOp {
    fn physical_forward(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        println!("{:?} {:?} {:?}", self, _inputs, output);
        let x = _inputs[0].get_scalar_value();
        for i in 0..output.num_elements() {
            output.set_element(i, x.clone());
        }
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalPassthroughOp {}
impl PhysicalOp for PhysicalPassthroughOp {
    fn physical_forward(&self, inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        let input = inputs[0];
        for i in 0..input.num_elements() {
            output.set_element(i, input.get_element(i));
        }
        println!("{:?} {:?} {:?}", self, input, output);
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalAddOp {}
impl PhysicalOp for PhysicalAddOp {
    fn physical_forward(&self, inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        for i in 0..output.num_elements() {
            let a = inputs[0].get_element(i);
            let b = inputs[1].get_element(i);
            output.set_element(i, a.add(b));
        }
        println!("{:?} {:?} {:?}", self, inputs, output);
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalSubOp {}
impl PhysicalOp for PhysicalSubOp {
    fn physical_forward(&self, inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        println!("{:?} {:?} {:?}", self, inputs, output);
        for i in 0..output.num_elements() {
            let a = inputs[0].get_element(i);
            let b = inputs[1].get_element(i);
            output.set_element(i, a.sub(b));
        }
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalMulOp {}
impl PhysicalOp for PhysicalMulOp {
    fn physical_forward(&self, inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        for i in 0..output.num_elements() {
            let a = inputs[0].get_element(i);
            let b = inputs[1].get_element(i);
            output.set_element(i, a.mul(b));
        }
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalDivOp {}
impl PhysicalOp for PhysicalDivOp {
    fn physical_forward(&self, inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        println!("{:?} {:?} {:?}", self, inputs, output);
        for i in 0..output.num_elements() {
            let a = inputs[0].get_element(i);
            let b = inputs[1].get_element(i);
            output.set_element(i, a.div(b));
        }
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalGetIndexOp {
    index: usize,
}
impl PhysicalOp for PhysicalGetIndexOp {
    fn physical_forward(&self, inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        output.set_scalar_value(inputs[0].get_element(self.index));
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalConcatOp {}
impl PhysicalOp for PhysicalConcatOp {
    fn physical_forward(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        println!("{:?} {:?} {:?}", self, _inputs, output);
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalPlaceholderOp {}
impl PhysicalOp for PhysicalPlaceholderOp {
    fn physical_forward(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        println!("{:?} {:?} {:?}", self, _inputs, output);
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalDotProductOp {}
impl PhysicalOp for PhysicalDotProductOp {
    fn physical_forward(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        println!("{:?} {:?} {:?}", self, _inputs, output);
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalSumOp {}
impl PhysicalOp for PhysicalSumOp {
    fn physical_forward(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        println!("{:?} {:?} {:?}", self, _inputs, output);
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalSqrtOp {}
impl PhysicalOp for PhysicalSqrtOp {
    fn physical_forward(&self, inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        let input = inputs[0];
        for i in 0..output.num_elements() {
            output.set_element(i, input.get_element(i).sqrt());
        }
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalMatMulOp {}
impl PhysicalOp for PhysicalMatMulOp {
    fn physical_forward(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        println!("{:?} {:?} {:?}", self, _inputs, output);
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalSliceOp {}
impl PhysicalOp for PhysicalSliceOp {
    fn physical_forward(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        println!("{:?} {:?} {:?}", self, _inputs, output);
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalNnSoftmaxOp {}
impl PhysicalOp for PhysicalNnSoftmaxOp {
    fn physical_forward(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        println!("{:?} {:?} {:?}", self, _inputs, output);
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalNnAttentionOp {}
impl PhysicalOp for PhysicalNnAttentionOp {
    fn physical_forward(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        println!("{:?} {:?} {:?}", self, _inputs, output);
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalNnDenseOp {}
impl PhysicalOp for PhysicalNnDenseOp {
    fn physical_forward(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        println!("{:?} {:?} {:?}", self, _inputs, output);
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalNnRmsNormOp {}
impl PhysicalOp for PhysicalNnRmsNormOp {
    fn physical_forward(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        println!("{:?} {:?} {:?}", self, _inputs, output);
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalNnRopeOp {}
impl PhysicalOp for PhysicalNnRopeOp {
    fn physical_forward(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        println!("{:?} {:?} {:?}", self, _inputs, output);
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalNnTransformerOp {}
impl PhysicalOp for PhysicalNnTransformerOp {
    fn physical_forward(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {
        println!("{:?} {:?} {:?}", self, _inputs, output);
    }
}

#[derive(Debug, Clone)]
pub enum PhysicalOpType {
    Broadcast(PhysicalBroadcastOp),
    LiteralF64(PhysicalLiteralF64Op),
    LiteralU32(PhysicalLiteralU32Op),
    Placeholder(PhysicalPlaceholderOp),
    Concat(PhysicalConcatOp),
    Passthrough(PhysicalPassthroughOp),
    GetIndex(PhysicalGetIndexOp),
    Add(PhysicalAddOp),
    Sub(PhysicalSubOp),
    Div(PhysicalDivOp),
    Mul(PhysicalMulOp),
    DotProduct(PhysicalDotProductOp),
    Sum(PhysicalSumOp),
    Sqrt(PhysicalSqrtOp),
    MatMul(PhysicalMatMulOp),
    Slice(PhysicalSliceOp),
    NnSoftmax(PhysicalNnSoftmaxOp),
    NnAttention(PhysicalNnAttentionOp),
    NnDense(PhysicalNnDenseOp),
    NnRmsNorm(PhysicalNnRmsNormOp),
    NnRope(PhysicalNnRopeOp),
    NnTransformer(PhysicalNnTransformerOp),
}

impl PhysicalOpType {
    pub fn get_physical(&self) -> Box<dyn PhysicalOp> {
        match &self {
            PhysicalOpType::Broadcast(op) => Box::new(op.clone()),
            PhysicalOpType::LiteralF64(op) => Box::new(op.clone()),
            PhysicalOpType::LiteralU32(op) => Box::new(op.clone()),
            PhysicalOpType::Placeholder(op) => Box::new(op.clone()),
            PhysicalOpType::Concat(op) => Box::new(op.clone()),
            PhysicalOpType::Passthrough(op) => Box::new(op.clone()),
            PhysicalOpType::GetIndex(op) => Box::new(op.clone()),
            PhysicalOpType::Add(op) => Box::new(op.clone()),
            PhysicalOpType::Sub(op) => Box::new(op.clone()),
            PhysicalOpType::Div(op) => Box::new(op.clone()),
            PhysicalOpType::Mul(op) => Box::new(op.clone()),
            PhysicalOpType::DotProduct(op) => Box::new(op.clone()),
            PhysicalOpType::Sum(op) => Box::new(op.clone()),
            PhysicalOpType::Sqrt(op) => Box::new(op.clone()),
            PhysicalOpType::MatMul(op) => Box::new(op.clone()),
            PhysicalOpType::Slice(op) => Box::new(op.clone()),
            PhysicalOpType::NnSoftmax(op) => Box::new(op.clone()),
            PhysicalOpType::NnAttention(op) => Box::new(op.clone()),
            PhysicalOpType::NnDense(op) => Box::new(op.clone()),
            PhysicalOpType::NnRmsNorm(op) => Box::new(op.clone()),
            PhysicalOpType::NnRope(op) => Box::new(op.clone()),
            PhysicalOpType::NnTransformer(op) => Box::new(op.clone()),
        }
    }
}

fn logical_call_to_physical_op(logical_call: &LogicalGraphCall) -> PhysicalOpType {
    match &logical_call.op {
        OpCode::Broadcast(_) => PhysicalOpType::Broadcast(PhysicalBroadcastOp {}),
        OpCode::Return(_) => PhysicalOpType::Passthrough(PhysicalPassthroughOp {}),
        OpCode::LiteralU32(op) => {
            PhysicalOpType::LiteralU32(PhysicalLiteralU32Op { value: op.value })
        }
        OpCode::LiteralF64(op) => {
            PhysicalOpType::LiteralF64(PhysicalLiteralF64Op { value: op.value })
        }
        OpCode::BasicGetIndex(op) => {
            PhysicalOpType::GetIndex(PhysicalGetIndexOp { index: op.index })
        }
        OpCode::BasicConcat(_) => PhysicalOpType::Concat(PhysicalConcatOp {}),
        OpCode::BasicPlaceholder(_) => PhysicalOpType::Placeholder(PhysicalPlaceholderOp {}),
        OpCode::BasicAdd(_) => PhysicalOpType::Add(PhysicalAddOp {}),
        OpCode::BasicMul(_) => PhysicalOpType::Mul(PhysicalMulOp {}),
        OpCode::BasicDiv(_) => PhysicalOpType::Div(PhysicalDivOp {}),
        OpCode::BasicSub(_) => PhysicalOpType::Sub(PhysicalSubOp {}),
        OpCode::BasicDotProduct(_) => PhysicalOpType::DotProduct(PhysicalDotProductOp {}),
        OpCode::BasicSum(_) => PhysicalOpType::Sum(PhysicalSumOp {}),
        OpCode::BasicSqrt(_) => PhysicalOpType::Sqrt(PhysicalSqrtOp {}),
        OpCode::BasicMatMul(_) => PhysicalOpType::MatMul(PhysicalMatMulOp {}),
        OpCode::BasicSlice(_) => PhysicalOpType::Slice(PhysicalSliceOp {}),
        OpCode::NnSoftmax(_) => PhysicalOpType::NnSoftmax(PhysicalNnSoftmaxOp {}),
        OpCode::NnAttention(_) => PhysicalOpType::NnAttention(PhysicalNnAttentionOp {}),
        OpCode::NnDense(_) => PhysicalOpType::NnDense(PhysicalNnDenseOp {}),
        OpCode::NnRmsNorm(_) => PhysicalOpType::NnRmsNorm(PhysicalNnRmsNormOp {}),
        OpCode::NnRope(_) => PhysicalOpType::NnRope(PhysicalNnRopeOp {}),
        OpCode::NnTransformer(_) => PhysicalOpType::NnTransformer(PhysicalNnTransformerOp {}),
    }
}
