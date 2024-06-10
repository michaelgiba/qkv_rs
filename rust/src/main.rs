use clap::Parser;
use qkv_rs::logical::{LogicalGraph, LogicalValueType};
use qkv_rs::ops::basic::inputs::plan_input_placeholder;
use qkv_rs::ops::nn::transformer::plan_transformer_block;
use qkv_rs::physical::PhysicalGraph;

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

fn fill_parameters(logical_graph: &mut LogicalGraph, physical_graph: &mut PhysicalGraph) {
    let w_mha_0q0 = logical_graph.get_tensor_by_name("NnAttention_0_q_weights");
    let w_mha_0k0 = logical_graph.get_tensor_by_name("NnAttention_0_k_weights");
    let w_mha_0v0 = logical_graph.get_tensor_by_name("NnAttention_0_v_weights");
    let w_mha_0pos0 = logical_graph.get_tensor_by_name("NnAttention_0_positions");
    let w_mha_0q1 = logical_graph.get_tensor_by_name("NnAttention_1_q_weights");
    let w_mha_0k1 = logical_graph.get_tensor_by_name("NnAttention_1_k_weights");
    let w_mha_0v1 = logical_graph.get_tensor_by_name("NnAttention_1_v_weights");
    let w_mha_0pos1 = logical_graph.get_tensor_by_name("NnAttention_1_positions");
    let w_mha_out = logical_graph.get_tensor_by_name("NnMha_0_out_weights");

    let ff_w1_gate = logical_graph.get_tensor_by_name("NnDense_0_ff_w1_gate");
    let ff_w1_linear = logical_graph.get_tensor_by_name("NnDense_0_ff_w1_linear");
    let ff_w2 = logical_graph.get_tensor_by_name("NnDense_0_ff_w2");

    physical_graph.set_value_for_tensor(&w_mha_0q0, vec![7.0; w_mha_0q0.num_elements()]);
    physical_graph.set_value_for_tensor(&w_mha_0k0, vec![11.0; w_mha_0k0.num_elements()]);
    physical_graph.set_value_for_tensor(&w_mha_0v0, vec![13.0; w_mha_0v0.num_elements()]);
    physical_graph.set_value_for_tensor(&w_mha_0pos0, vec![17.0; w_mha_0pos0.num_elements()]);

    physical_graph.set_value_for_tensor(&w_mha_0q1, vec![3.0; w_mha_0q1.num_elements()]);
    physical_graph.set_value_for_tensor(&w_mha_0k1, vec![5.0; w_mha_0k1.num_elements()]);
    physical_graph.set_value_for_tensor(&w_mha_0v1, vec![19.0; w_mha_0v1.num_elements()]);
    physical_graph.set_value_for_tensor(&w_mha_0pos1, vec![23.0; w_mha_0pos1.num_elements()]);

    physical_graph.set_value_for_tensor(&w_mha_out, vec![1.0; w_mha_out.num_elements()]);

    physical_graph.set_value_for_tensor(&ff_w1_gate, vec![1.0; ff_w1_gate.num_elements()]);
    physical_graph.set_value_for_tensor(&ff_w1_linear, vec![1.0; ff_w1_linear.num_elements()]);
    physical_graph.set_value_for_tensor(&ff_w2, vec![1.0; ff_w2.num_elements()]);
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

    fill_parameters(&mut graph, &mut physical_graph);

    // Provide input values
    physical_graph.set_value_for_tensor(
        &input_sequence_placeholder,
        vec![7.0; input_sequence_placeholder.num_elements()],
    );

    let outputs = physical_graph.compute(&transformer_output);

    println!("{:?}", outputs);
}
