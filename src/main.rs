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

struct LogicalTensor {
    shape: &[usize],
}

struct PhysicalOp {}

trait LogicalOp: Into<LogicalTensor> {
    fn to_physical() -> PhysicalOp;
}

struct LogicalNoOp;
impl LogicalOp for LogicalNoOp {
    fn to_physical() -> PhysicalOp {
        PhysicalOp {}
    }
}

impl LogicalOp for LogicalTensor {
    fn to_physical() -> PhysicalOp {
        PhysicalOp {}
    }
}

fn logical_mat_mul(a: LogicalTensor, b: LogicalTensor) -> impl LogicalOp {
    // TODO
    LogicalNoOp {}
}

fn logical_weights(shape: &[usize]) -> LogicalTensor {
    // TODO
    LogicalTensor { shape: shape }
}

fn logical_dot_product(a: LogicalTensor, b: LogicalTensor) -> LogicalTensor {
    // TODO
    LogicalNoOp {}
}

fn logical_concat(tensors: &[LogicalTensor]) -> LogicalTensor {
    // TODO
    LogicalNoOp {}
}

struct LogicalAttentionHeadOp {
    input_sequence_len: usize,
    input_embed_dim: usize,
    output_head_dim: usize,
}

impl LogicalAttentionHeadOp {
    fn plan_forward(self, x: LogicalTensor) -> impl LogicalOp {
        let q_weights = logical_weights(&[self.input_sequence_len, self.output_head_dim]);
        let k_weights = logical_weights(&[self.input_sequence_len, self.output_head_dim]);
        let v_weights = logical_weights(&[self.input_sequence_len, self.output_head_dim]);

        let q_proj = logical_mat_mul(x, q_weights); // [N, output_head_dim]
        let k_proj = logical_mat_mul(x, k_weights); // [N, output_head_dim]
        let v_proj = logical_mat_mul(x, v_weights); // [N, output_head_dim]

        let attention_scores = logical_dot_product(q_proj, k_proj); // [N, N]

        let attended_v_proj = logical_mat_mul(attention_scores, v_proj); // [N, output_head_dim]

        attended_v_proj
    }
}

struct LogicalMultiHeadAttentionOp {
    input_sequence_len: usize,
    input_embed_dim: usize,
    output_head_dim: usize,
    num_heads: usize,
}

impl LogicalMultiHeadAttentionOp {
    fn plan_forward(self, x: LogicalTensor) -> impl LogicalOp {
        let head = LogicalAttentionHeadOp {
            input_sequence_len: 0,
            input_embed_dim: 0,
            output_head_dim: 0,
        };

        let head_outputs = head.plan_forward(x);

        // TODO support multiple heads
        let concatted_head_outputs = logical_concat(&[head]);

        concatted_head_outputs
    }
}

fn transformer_forward(input: &Vec<Vec<f32>>, config: &TransformerConfig) -> Vec<f32> {
    // 1. An input is provided as an N item array of embeddings (N, D_emb)
    // 2. We compute Q, K, V for each head. Each of them have shape (N, D_head)
    // 3. We compute the dot product of Q and K. The output of the dot product is shape (N, N)
    // 4. The attention scores are multiplied by the V outputs giving a new attention scaled output of (N, D_head)
    // 5. The attention scaled values are passed to a feed forward layer and converted first to (N, D_hidden) then back to (N, D_emb)

    // 1. Apply layer normalization
    let normed_input = rms_norm(input);

    // 2. Apply multi-head attention
    let multi_head_attention_output = LogicalMultiHeadAttentionOp {
        input_sequence_len: 0,
        input_embed_dim: 0,
        output_head_dim: 0,
        num_heads: 0,
    };

    let logical_desnse_w1 = logical_weights(&[0]); // (D_head, D_hidden)
    let logical_desnse_w2 = logical_weights(&[0]); // (D_hidden, D_emb)

    let ff1_hidden = logical_mat_mul(multi_head_attention_output, logical_desnse_w1);
    let ff1_output = logical_mat_mul(multi_head_attention_output, logical_desnse_w2);

    ff1_output
}

fn main() {
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
