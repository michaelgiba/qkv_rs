


struct TransformerConfig {
    embed_dim: usize,
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
    q_weights: Vec<f32>,
    k_weights: Vec<f32>,
    v_weights: Vec<f32>,
    dense_weights: Vec<f32>,
}

fn rms_norm(input: &Vec<Vec<f32>>) -> &Vec<Vec<f32>> {

    let sum_squares: f32 = input.iter().map(|x| x * x).sum();
    let mean_sum_square = sum_squares / (input.len() as f32);
    let sum_squares_sqrt = mean_sum_square.sqrt() + 1e-6;
    let inverse_sum_squares_sqrt = 1.0 / sum_squares_sqrt;
    let normed_input: Vec<f32> = input.iter().map(|x| x * inverse_sum_squares_sqrt).collect();

    normed_input
}


fn matmul(input: &Vec<Vec<f32>>, weights: &Vec<Vec<f32>>, hidden_dim: usize) -> Vec<Vec<f32>> {
    // Input is a 2D array of shape [sequence_length, hidden_dim]
    // Weights is a 2D array of shape [hidden_dim, hidden_dim]
    // Output is a 2D array of shape [sequence_length, hidden_dim]
    // TODO

    return vec![vec![0.0; hidden_dim]; input.len()];
}

fn position_embedding(input: &Vec<Vec<f32>>, hidden_dim: usize) -> Vec<Vec<f32>> {
    // Input is a 2D array of shape [sequence_length, hidden_dim]
    // Output is a 2D array of shape [sequence_length, hidden_dim]
    // TODO

    return vec![vec![0.0; hidden_dim]; input.len()];

}

fn softmax(input: &Vec<Vec<f32>>, hidden_dim: usize) -> Vec<Vec<f32>> {
    // Input is a 1d array of shape (1, sequence_length, sequence_length)
    // Output is a 1d array of shape (1, sequence_length, sequence_length)
    // Compute softmax 
    // TODO

    return vec![vec![0.0; hidden_dim]; input.len()];
}


fn attention(input: &Vec<Vec<f32>>, q_weights: &Vec<Vec<f32>>, k_weights: &Vec<Vec<f32>>, v_weights: &Vec<Vec<f32>>, num_heads: usize, hidden_dim: usize) -> Vec<Vec<f32>> {

    // 1. Project input into query, key, and value vectors
    let query_proj = matmul(input, &q_weights, hidden_dim);
    let key_proj = matmul(input, &k_weights, hidden_dim);
    let value_proj = matmul(input, &v_weights, hidden_dim);

    // 2. Position embed query and key 
    let emb_query_proj = position_embedding(&query_proj, hidden_dim);
    let emb_key_proj = position_embedding(&key_proj, hidden_dim);

    // 3. Compute attention weights
    let qk_logits = matmul(&emb_query_proj, &emb_key_proj, hidden_dim);
    let qk_activations = softmax(&qk_logits, hidden_dim);

    // 4. Apply attention weights to value vectors
    let encoded = matmul(&qk_activations, &value_proj, hidden_dim);

    encoded
}

fn transformer_forward(input: &Vec<Vec<f32>>, config: &TransformerConfig) -> Vec<f32> {
    // 1. Apply layer normalization
    let normed_input = rms_norm(input);

    // 2. Apply multi-head attention
    let multi_head_attention_output = attention(&normed_input, &config.q_weights, &config.k_weights, &config.v_weights, config.num_heads);

    let ff1_output = matmul(&multi_head_attention_output, &config.dense_weights);

    ff1_output
}

fn main() {
    // Example usage (similar to before, but with flattened weight vectors)
    let config = TransformerConfig { 
        embed_dim: 512,
        hidden_dim: 2048,
        num_heads: 8,
        head_dim: 64,
        q_weights: vec![0.0; 512 * 512],
        k_weights: vec![0.0; 512 * 512],
        v_weights: vec![0.0; 512 * 512],
        dense_weights: vec![0.0; 512 * 2048],
    };
    
    let input = vec![0.0; 32 * 128 * 512]; // Flattened input
        
    let output = transformer_forward(&input, &config);

    // Print the output
    println!("{:?}", output);

}
