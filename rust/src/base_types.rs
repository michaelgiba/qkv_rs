use std::{collections::HashMap, fmt::Debug};

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
struct LogicalGraphRegisteredOp {
    op: Box<dyn LogicalOp>,
    input_tensor_ids: Vec<usize>,
}

#[derive(Debug)]
pub struct LogicalGraph {
    tensor_id_to_tensor: HashMap<usize, LogicalTensor>,
    output_tensor_id_to_callee: HashMap<usize, LogicalGraphRegisteredOp>,
}

impl LogicalGraph {
    pub fn new() -> LogicalGraph {
        let mut graph = LogicalGraph {
            tensor_id_to_tensor: HashMap::new(),
            output_tensor_id_to_callee: HashMap::new(),
        };
        graph.empty_tensor();
        graph
    }

    pub fn compile(self) -> PhysicalGraph {
        // TODO only get the connected component to outputs before creating
        // the physical graph
        PhysicalGraph::new(self)
    }

    pub fn register_computation(
        &mut self,
        op: Box<dyn LogicalOp>,
        inputs: &[&LogicalTensor],
    ) -> LogicalTensor {
        let output = op.plan_forward(self, inputs);

        let input_tensor_ids: Vec<usize> = inputs.iter().map(|t| t.id).collect();
        let registered_op = LogicalGraphRegisteredOp {
            op: op,
            input_tensor_ids,
        };
        self.output_tensor_id_to_callee
            .insert(output.id, registered_op);

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
        self.register_computation(
            Box::new(LiteralOp {
                value: value,
                shape: vec![1],
            }),
            &[],
        )
    }

    pub fn scalar_u64(&mut self, value: u64) -> LogicalTensor {
        self.register_computation(
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

fn assert_physical_matches_logical(physical: &PhysicalTensor, logical: &LogicalTensor) {
    assert_eq!(physical.spec.shape, logical.shape);
    assert_eq!(physical.spec.value_type, logical.value_type);
}

pub struct PhysicalGraph {
    logical_graph: LogicalGraph,
}

impl PhysicalGraph {
    pub fn new(logical_graph: LogicalGraph) -> PhysicalGraph {
        PhysicalGraph { logical_graph }
    }

    pub fn compute(&self, target: LogicalTensor) -> PhysicalTensor {
        let mut tensor_id_to_values: HashMap<usize, PhysicalTensor> = HashMap::new();
        let mut stack = vec![target.id];

        while let Some(tensor_id) = stack.pop() {
            // If we've already computed this value there is nothing to do
            if tensor_id_to_values.contains_key(&tensor_id) {
                continue;
            }

            // Otherwise we push this back on the stack and compute the dependencies
            stack.push(tensor_id);
            println!("{:?} ", self.logical_graph.get_tensor(tensor_id));

            let call_details = self
                .logical_graph
                .output_tensor_id_to_callee
                .get(&tensor_id)
                .unwrap();

            println!("{:?} ", call_details);

            let uncomputed_dependency_ids: Vec<usize> = call_details
                .input_tensor_ids
                .iter()
                .filter(|x| !tensor_id_to_values.contains_key(x))
                .map(|x| *x)
                .collect();

            if uncomputed_dependency_ids.is_empty() {
                //  We have all the dependencies, we can compute the value
                let inputs: Vec<&PhysicalTensor> = call_details
                    .input_tensor_ids
                    .iter()
                    .map(|x| tensor_id_to_values.get(x).unwrap())
                    .collect();

                let logical_output = self.logical_graph.get_tensor(tensor_id);

                let physical_output =
                    find_physical_op(&call_details.op).compute(&inputs, logical_output);

                assert_physical_matches_logical(&physical_output, logical_output);

                tensor_id_to_values.insert(tensor_id, physical_output);
            } else {
                // We need to first compute dependencies
                uncomputed_dependency_ids
                    .iter()
                    .for_each(|x| stack.push(*x));
            }
        }

        let x = tensor_id_to_values.get(&target.id);

        x.unwrap().clone()
    }
}

pub trait PhysicalOp {
    fn compute(&self, inputs: &[&PhysicalTensor], output_spec: &LogicalTensor) -> PhysicalTensor;
}

#[derive(Debug)]
pub struct PhysicalLiteralOp {}
impl PhysicalOp for PhysicalLiteralOp {
    fn compute(&self, _inputs: &[&PhysicalTensor], output_spec: &LogicalTensor) -> PhysicalTensor {
        PhysicalTensor {
            spec: output_spec.clone(),
            values: vec![1],
        }
    }
}

fn find_physical_op(logical_op: &Box<dyn LogicalOp>) -> Box<dyn PhysicalOp> {
    match logical_op.as_ref() {
        _ => Box::new(PhysicalLiteralOp {}),
    }
}
