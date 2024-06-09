#!/bin/bash

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

set -euo pipefail

echo "Building and testing Rust project..."
(cd ${PROJECT_ROOT}/rust && cargo test)

echo "Running Python tests..."
(cd ${PROJECT_ROOT}/python && pytest)