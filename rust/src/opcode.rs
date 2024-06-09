use crate::logical::{LiteralOp, LogicalOp, LogicalReturnOp};
use crate::ops::basic::broadcast::LogicalBroadcastOp;
use crate::ops::basic::inputs::LogicalPlaceholderOp;
use crate::ops::basic::math::{
    LogicalAddOp, LogicalDivOp, LogicalDotProductOp, LogicalMatMulOp, LogicalMulOp, LogicalSqrtOp,
    LogicalSubOp, LogicalSumOp,
};
use crate::ops::basic::slice::{LogicalConcatOp, LogicalGetIndexOp, LogicalSliceOp};
use crate::ops::nn::activations::LogicalSoftmaxOp;
use crate::ops::nn::attention::LogicalAttentionHeadOp;
use crate::ops::nn::dense::LogicalDenseOp;
use crate::ops::nn::norm::LogicalRmsNormOp;
use crate::ops::nn::position_embed::RotaryPositionEmbeddingOp;
use crate::ops::nn::transformer::LogicalTransformerBlockOp;

#[derive(Debug, Clone)]
pub enum OpCode {
    // -- Basic --
    Broadcast(LogicalBroadcastOp),
    Return(LogicalReturnOp),
    LiteralU32(LiteralOp<u32>),
    LiteralF64(LiteralOp<f64>),
    BasicGetIndex(LogicalGetIndexOp),
    BasicConcat(LogicalConcatOp),
    // Inputs
    BasicPlaceholder(LogicalPlaceholderOp),
    // Math
    BasicAdd(LogicalAddOp),
    BasicMul(LogicalMulOp),
    BasicDiv(LogicalDivOp),
    BasicSub(LogicalSubOp),
    BasicDotProduct(LogicalDotProductOp),
    BasicSum(LogicalSumOp),
    BasicSqrt(LogicalSqrtOp),
    BasicMatMul(LogicalMatMulOp),
    // Slice
    BasicSlice(LogicalSliceOp),
    // -- Neural Network --
    // Activation
    NnSoftmax(LogicalSoftmaxOp),
    // Attention
    NnAttention(LogicalAttentionHeadOp),
    // Dense
    NnDense(LogicalDenseOp),
    // Norm
    NnRmsNorm(LogicalRmsNormOp),
    // Position Embed
    NnRope(RotaryPositionEmbeddingOp),
    // Transformer
    NnTransformer(LogicalTransformerBlockOp),
}

impl OpCode {
    pub fn get_logical(&self) -> Box<dyn LogicalOp> {
        match &self {
            OpCode::Broadcast(op) => Box::new(op.clone()),
            OpCode::Return(op) => Box::new(op.clone()),
            OpCode::LiteralU32(op) => Box::new(op.clone()),
            OpCode::LiteralF64(op) => Box::new(op.clone()),
            OpCode::BasicGetIndex(op) => Box::new(op.clone()),
            OpCode::BasicConcat(op) => Box::new(op.clone()),
            OpCode::BasicPlaceholder(op) => Box::new(op.clone()),
            OpCode::BasicAdd(op) => Box::new(op.clone()),
            OpCode::BasicMul(op) => Box::new(op.clone()),
            OpCode::BasicDiv(op) => Box::new(op.clone()),
            OpCode::BasicSub(op) => Box::new(op.clone()),
            OpCode::BasicDotProduct(op) => Box::new(op.clone()),
            OpCode::BasicSum(op) => Box::new(op.clone()),
            OpCode::BasicSqrt(op) => Box::new(op.clone()),
            OpCode::BasicMatMul(op) => Box::new(op.clone()),
            OpCode::BasicSlice(op) => Box::new(op.clone()),
            OpCode::NnSoftmax(op) => Box::new(op.clone()),
            OpCode::NnAttention(op) => Box::new(op.clone()),
            OpCode::NnDense(op) => Box::new(op.clone()),
            OpCode::NnRmsNorm(op) => Box::new(op.clone()),
            OpCode::NnRope(op) => Box::new(op.clone()),
            OpCode::NnTransformer(op) => Box::new(op.clone()),
        }
    }

    pub fn name(&self) -> &'static str {
        match &self {
            OpCode::Broadcast(_) => "Broadcast",
            OpCode::Return(_) => "Return",
            OpCode::LiteralU32(_) => "LiteralU32",
            OpCode::LiteralF64(_) => "LiteralF64",
            OpCode::BasicGetIndex(_) => "BasicGetIndex",
            OpCode::BasicConcat(_) => "BasicConcat",
            OpCode::BasicPlaceholder(_) => "BasicPlaceholder",
            OpCode::BasicAdd(_) => "BasicAdd",
            OpCode::BasicMul(_) => "BasicMul",
            OpCode::BasicDiv(_) => "BasicDiv",
            OpCode::BasicSub(_) => "BasicSub",
            OpCode::BasicDotProduct(_) => "BasicDotProduct",
            OpCode::BasicSum(_) => "BasicSum",
            OpCode::BasicSqrt(_) => "BasicSqrt",
            OpCode::BasicMatMul(_) => "BasicMatMul",
            OpCode::BasicSlice(_) => "BasicSlice",
            OpCode::NnSoftmax(_) => "NnSoftmax",
            OpCode::NnAttention(_) => "NnAttention",
            OpCode::NnDense(_) => "NnDense",
            OpCode::NnRmsNorm(_) => "NnRmsNorm",
            OpCode::NnRope(_) => "NnRope",
            OpCode::NnTransformer(_) => "NnTransformer",
        }
    }
}
