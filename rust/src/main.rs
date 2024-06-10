use std::collections::HashMap;

use clap::Parser;
use qkv_rs::logical::{LogicalGraph, LogicalValueType};
use qkv_rs::ops::basic::inputs::plan_input_placeholder;
use qkv_rs::ops::nn::transformer::plan_transformer_block;
use qkv_rs::physical::PhysicalGraph;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Number of input batches
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    // Size of input sequence length, required
    #[arg(long)]
    input_sequence_length: usize,

    // Size of input embedding dimension, required
    #[arg(long)]
    input_sequence_embed_dim: usize,

    // Size of multihead attention head dimension, required
    #[arg(long)]
    mha_head_dim: usize,

    // Size of multihead attention number of heads, required
    #[arg(long)]
    mha_num_heads: usize,

    // Size of feed forward hidden dimension, required
    #[arg(long)]
    ff_hidden_dim: usize,

    // Size of feed forward output dimension, required
    #[arg(long)]
    ff_output_dim: usize,

    // Bool if the binary should output in json mode
    #[arg(long)]
    json: bool,

    // JSON file containing weights
    #[arg(long)]
    weights: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
struct WeightsContent {
    tensors: HashMap<String, Vec<f64>>,
}

fn fill_parameters(
    logical_graph: &mut LogicalGraph,
    physical_graph: &mut PhysicalGraph,
    weights: WeightsContent,
) {
    for (name, value) in weights.tensors.iter() {
        let tensor = logical_graph.get_tensor_by_name(name);
        physical_graph.set_value_for_tensor(&tensor, value.clone());
    }
}

fn load_weights_from_file(file_name: Option<String>) -> WeightsContent {
    match file_name {
        Some(file_name) => {
            let file = std::fs::File::open(file_name).unwrap();
            serde_json::from_reader(file).unwrap()
        }
        None => WeightsContent {
            tensors: HashMap::new(),
        },
    }
}

fn main() {
    let args = Args::parse();

    let mut graph = LogicalGraph::new();

    let input_sequence_placeholder = plan_input_placeholder(
        &mut graph,
        &[args.input_sequence_length, args.input_sequence_embed_dim],
        LogicalValueType::F64,
        "input_sequence".to_string(),
    );

    let transformer_output = plan_transformer_block(
        &mut graph,
        &input_sequence_placeholder,
        args.input_sequence_embed_dim,
        args.mha_head_dim,
        args.mha_head_dim,
        args.ff_hidden_dim,
        args.ff_output_dim,
    );

    let mut physical_graph = PhysicalGraph::compile(&graph, &[&transformer_output]);

    // Load weights from file and deserialize into WeightsContent
    let weights_content: WeightsContent = load_weights_from_file(args.weights);
    fill_parameters(&mut graph, &mut physical_graph, weights_content);

    // Provide input values
    physical_graph.set_value_for_tensor(
        &input_sequence_placeholder,
        vec![7.0; input_sequence_placeholder.num_elements()],
    );

    let outputs = physical_graph.compute(&transformer_output);

    if args.json {
        println!("{}", serde_json::to_string(&outputs).unwrap());
    }
}
