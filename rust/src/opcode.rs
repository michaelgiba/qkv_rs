pub enum OpCodes {
    // -- Basic --
    Return,
    Literal,
    BasicGetIndex,
    BasicConcat,
    // Inputs
    BasicPlaceholder,
    BasicWeights,
    // Math
    BasicAdd,
    BasicMul,
    BasicDiv,
    BasicSub,
    BasicDotProduct,
    BasicSum,
    BasicSqrt,
    BasicMatMul,
    // Slice
    BasicSlice,
    // -- Neural Network --
    // Activation
    NnSoftmax,
    // Attention
    NnAttention,
    // Dense
    NnDense,
    // Norm
    NnRmsNorm,
    // Position Embed
    NnRope,
    // Transformer
    NnTransformer,
}
