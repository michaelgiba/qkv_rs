use std::{collections::HashMap, collections::HashSet, collections::VecDeque, fmt::Debug};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogicalValueType {
    F64,
    U64,
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
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor;
}
#[derive(Debug)]
pub struct LiteralOp<T> {
    pub shape: Vec<usize>,
    pub value: T,
}

impl LogicalOp for LiteralOp<f64> {
    fn plan_forward(&self, graph: &mut LogicalGraph, _inputs: &[&LogicalTensor]) -> LogicalTensor {
        graph.new_tensor(self.shape.clone(), LogicalValueType::F64)
    }
}

impl LogicalOp for LiteralOp<u64> {
    fn plan_forward(&self, graph: &mut LogicalGraph, _inputs: &[&LogicalTensor]) -> LogicalTensor {
        graph.new_tensor(self.shape.clone(), LogicalValueType::U64)
    }
}

#[derive(Debug)]
struct LogicalGraphCall {
    pub op: Box<dyn LogicalOp>,
    pub input_tensor_ids: Vec<usize>,
    pub output_tensor_id: usize,
}

#[derive(Debug)]
pub struct LogicalGraph {
    tensor_id_to_tensor: HashMap<usize, LogicalTensor>,
    output_tensor_id_to_calls: HashMap<usize, Vec<LogicalGraphCall>>,
}

impl LogicalGraph {
    pub fn new() -> LogicalGraph {
        let mut graph = LogicalGraph {
            tensor_id_to_tensor: HashMap::new(),
            output_tensor_id_to_calls: HashMap::new(),
        };
        graph.empty_tensor();
        graph
    }

    pub fn compile(self, target: &LogicalTensor) -> PhysicalGraph {
        PhysicalGraph::new(self, &target)
    }

    pub fn register_call(
        &mut self,
        op: Box<dyn LogicalOp>,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        let output = op.plan_forward(self, inputs);

        let input_tensor_ids: Vec<usize> = inputs.iter().map(|t| t.id).collect();
        let call = LogicalGraphCall {
            op: op,
            input_tensor_ids,
            output_tensor_id: output.id,
        };

        self.output_tensor_id_to_calls
            .entry(output.id)
            .or_insert(vec![])
            .push(call);

        output
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

#[derive(Debug, Clone)]
pub struct PhysicalTensor {
    spec: LogicalTensor,
    values: Vec<u8>,
}

impl PhysicalTensor {
    pub fn alloc(spec: LogicalTensor) -> PhysicalTensor {
        // TODO: proper number of bytes
        let num_bytes = spec.num_elements() * std::mem::size_of::<f64>();
        PhysicalTensor {
            spec,
            values: vec![0; num_bytes],
        }
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
}

impl PhysicalGraph {
    pub fn new(logical_graph: LogicalGraph, target: &LogicalTensor) -> PhysicalGraph {
        let mut tensor_id_to_tensor: HashMap<usize, PhysicalTensor> = HashMap::new();
        let mut tensor_id_to_calls: HashMap<usize, Vec<PhysicalGraphCall>> = HashMap::new();
        let mut stack = vec![target.id];

        while let Some(tensor_id) = stack.pop() {
            // If we've already computed this value there is nothing to do
            if tensor_id_to_tensor.contains_key(&tensor_id) {
                continue;
            }

            // Otherwise we push this back on the stack and compute the dependencies

            let logical_calls = logical_graph
                .output_tensor_id_to_calls
                .get(&tensor_id)
                .unwrap();

            for logical_call in logical_calls {
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

                    println!("logical call: {:?}", logical_call);
                } else {
                    stack.push(tensor_id);
                    stack.extend(need_allocation_input_ids.iter())
                }
            }
        }

        PhysicalGraph {
            tensor_id_to_tensor,
            tensor_id_to_calls: tensor_id_to_calls,
        }
    }

    fn determine_call_sequence(&self, tensor_id: usize) -> Vec<PhysicalGraphCall> {
        let mut stack = vec![];
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
                println!("physical call: {:?}, stack: {:?}", calls, stack);
            } else {
                stack.push(tensor_id);
                stack.extend(need_computation_tensors);
            }
        }

        call_sequence.into()
    }

    pub fn compute(&mut self, target: &LogicalTensor) -> &PhysicalTensor {
        let call_sequence = self.determine_call_sequence(target.id);

        for call in call_sequence {
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
                    op.compute(&input_tensors, &mut output_copy);
                }
            }

            self.tensor_id_to_tensor
                .insert(call.output_tensor_id, output_copy);
        }

        self.tensor_id_to_tensor.get(&target.id).unwrap()
    }
}

pub trait PhysicalOp {
    fn compute(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor);
}

#[derive(Debug, Clone)]
pub struct PhysicalLiteralOp {}
impl PhysicalOp for PhysicalLiteralOp {
    fn compute(&self, _inputs: &[&PhysicalTensor], output: &mut PhysicalTensor) {}
}

#[derive(Debug, Clone)]
pub enum PhysicalOpType {
    Literal(PhysicalLiteralOp),
}

fn find_physical_op(logical_op: &Box<dyn LogicalOp>) -> PhysicalOpType {
    match logical_op.as_ref() {
        _ => PhysicalOpType::Literal(PhysicalLiteralOp {}),
    }
}
