#[derive(Debug)]
pub enum LogicalValueType {
    F64,
}

#[derive(Debug)]
pub struct LogicalTensor {
    shape: Vec<usize>,
    value_type: LogicalValueType,
}

#[derive(Debug)]
pub struct PhysicalTensor {
    spec: LogicalTensor,
    values: Vec<u8>,
}

pub const LOGICAL_EMPTY: LogicalTensor = LogicalTensor {
    shape: vec![],
    value_type: LogicalValueType::F64,
};
pub const PHYSICAL_EMPTY: PhysicalTensor = PhysicalTensor {
    spec: LOGICAL_EMPTY,
    values: vec![],
};

pub struct PhysicalGraph {}
impl PhysicalGraph {
    pub fn new() -> PhysicalGraph {
        PhysicalGraph {}
    }

    pub fn forward(&self) -> PhysicalTensor {
        // TODO
        PHYSICAL_EMPTY
    }
}

pub struct LogicalGraph {}

impl LogicalGraph {
    pub fn new() -> LogicalGraph {
        LogicalGraph {}
    }

    pub fn compile(&self, outputs: &[LogicalTensor]) -> PhysicalGraph {
        PhysicalGraph::new()
    }
}

pub trait LogicalOp {
    fn plan_forward(&self, graph: &mut LogicalGraph, inputs: &[&LogicalTensor]) -> LogicalTensor;
}

// pub trait PhysicalOp {
//     fn compute(&self, inputs: &[&PhysicalTensor]) -> PhysicalTensor;
// }
