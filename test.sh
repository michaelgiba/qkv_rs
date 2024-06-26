#!/bin/bash

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

set -euo pipefail

# echo "Building and testing Rust project..."
(cd ${PROJECT_ROOT}/rust && cargo build)


# Gemma-like defaults
RUST_BACKTRACE=1 ./rust/target/debug/qkv_rs \
    --batch-size 1 \
    --input-sequence-length 1 \
    --input-sequence-embed-dim 4 \
    --mha-head-dim 2 \
    --mha-num-heads 16 \
    --ff-hidden-dim 32 \
    --ff-output-dim 4 --json

# echo "Running Python tests..."
# (cd ${PROJECT_ROOT}/python && pytest -s)    
