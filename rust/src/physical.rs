use crate::logical::{LogicalGraph, LogicalOp, LogicalTensor, LogicalValueType};
use core::f64;
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

    pub fn add(&self, other: PhysicalValue) -> PhysicalValue {
        let f64_value = self.as_f64() + other.as_f64();
        PhysicalValue {
            value_type: self.value_type,
            value: self.value_type.from_f64(f64_value),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalTensor {
    spec: LogicalTensor,
    values: Vec<u8>,
}

impl PhysicalTensor {
    pub fn alloc(spec: LogicalTensor) -> PhysicalTensor {
        PhysicalTensor {
            spec: spec.clone(),
            values: vec![0; spec.num_elements() * spec.value_type.size()],
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
                    op_type: find_physical_op(&logical_call.op),
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

            match call.op_type {
                PhysicalOpType::Literal(op) => {
                    op.physical_forward(&input_tensors, &mut output_copy);
                }
            }

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
pub struct PhysicalLiteralOp {}
impl PhysicalOp for PhysicalLiteralOp {
    fn physical_forward(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {}
}

#[derive(Debug, Clone)]
pub enum PhysicalOpType {
    Literal(PhysicalLiteralOp),
}

fn find_physical_op(logical_op: &Box<dyn LogicalOp>) -> PhysicalOpType {
    match logical_op {
        _ => PhysicalOpType::Literal(PhysicalLiteralOp {}),
    }
}
