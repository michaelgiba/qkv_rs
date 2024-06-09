use std::{collections::HashMap, fmt::Debug};

#[derive(Debug, Clone, Copy)]
pub enum LogicalValueType {
    F64,
    U64,
}

#[derive(Debug, Clone)]
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

    pub fn compile(self, outputs: &[LogicalTensor]) -> PhysicalGraph {
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
}

#[derive(Debug)]
pub struct PhysicalTensor {
    spec: LogicalTensor,
    values: Vec<u8>,
}

pub struct PhysicalGraph {
    logical_graph: LogicalGraph,
}

impl PhysicalGraph {
    pub fn new(logical_graph: LogicalGraph) -> PhysicalGraph {
        PhysicalGraph { logical_graph }
    }

    pub fn forward(&self) -> PhysicalTensor {
        // TODO
        unimplemented!()
    }
}

// pub trait PhysicalOp {
//     fn compute(&self, inputs: &[&PhysicalTensor]) -> PhysicalTensor;
// }
