use qkv_rs::base_types::{LogicalGraph, LogicalValueType};
use qkv_rs::ops::basic::plan_input_placeholder;
use qkv_rs::ops::transformer::plan_transformer_block;

const CONTEXT_LENGTH: usize = 10;
const EMBEDDING_DIM: usize = 64;
fn main() {
    let mut graph = LogicalGraph::new();

    let input_sequence_placeholder = plan_input_placeholder(
        &mut graph,
        &[CONTEXT_LENGTH, EMBEDDING_DIM],
        LogicalValueType::F64,
    );

    let transformer_output = plan_transformer_block(
        &mut graph,
        &[&input_sequence_placeholder],
        EMBEDDING_DIM,
        10,
        10,
        10,
    );

    let physical_graph = graph.compile(&[transformer_output]);

    let outputs = physical_graph.forward();

    println!("{:?}", outputs);
}
