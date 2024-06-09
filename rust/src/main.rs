use clap::Parser;
use qkv_rs::base_types::{LogicalGraph, LogicalValueType};
use qkv_rs::ops::basic::inputs::plan_input_placeholder;
use qkv_rs::ops::nn::transformer::plan_transformer_block;

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
}

fn main() {
    let args = Args::parse();

    let mut graph = LogicalGraph::new();

    let input_sequence_placeholder = plan_input_placeholder(
        &mut graph,
        &[
            args.input_sequence_length,
            args.input_sequence_embed_dim,
        ],
        LogicalValueType::F64,
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

    let physical_graph = graph.compile(&[transformer_output]);

    let outputs = physical_graph.forward();

    println!("{:?}", outputs);
}
