import subprocess
import numpy as np
import pytest
import os
from functools import cache
import json
import tempfile


@cache
def find_pyproject_root():
    """Finds the project root directory based on the location of pyproject.toml."""
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.isfile(os.path.join(current_dir, "pyproject.toml")):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if (
            parent_dir == current_dir
        ):  # Reached filesystem root without finding pyproject.toml
            raise FileNotFoundError(
                "Could not find project root containing pyproject.toml"
            )
        current_dir = parent_dir


@cache
def find_rust_project_root():
    pyproject_root = find_pyproject_root()
    return os.path.join(pyproject_root, "..", "rust")


def rust_binary_path():
    return os.path.join(find_rust_project_root(), "target", "debug", "qkv_rs")


@pytest.fixture(scope="session", autouse=True)
def build_rust_project():
    subprocess.run(["cargo", "build"], check=True, cwd=find_rust_project_root())


def run_rust_binary(args):
    """Runs the Rust binary with the given arguments and captures output."""
    binary_path = rust_binary_path()
    result = subprocess.run([binary_path, *args], capture_output=True, text=True)
    assert result.returncode == 0, f"Error running binary: {result.stderr}"
    return result.stdout


@pytest.fixture(
    scope="session"
)  # Share the binary runner function throughout the session
def rust_binary_runner():
    """Provides the function to run the Rust binary."""
    return run_rust_binary


def test_basic_transformer_outputs(rust_binary_runner):
    """Tests the transformer block's output for specific input."""

    weights = {
        "tensors": {
            "NnAttention_0_q_weights": [3.0] * 8,
            "NnAttention_0_k_weights": [5.0] * 8,
            "NnAttention_0_v_weights": [7.0] * 8,
            "NnAttention_0_positions": [11.0] * 16,
            "NnAttention_1_q_weights": [13.0] * 8,
            "NnAttention_1_k_weights": [17.0] * 8,
            "NnAttention_1_v_weights": [19.0] * 8,
            "NnAttention_1_positions": [23.0] * 16,
            "NnMha_0_out_weights": [29.0] * 16,
            "NnDense_0_ff_w1_gate": [31.0] * 32,
            "NnDense_0_ff_w1_linear": [37.0] * 32,
            "NnDense_0_ff_w2": [43.0] * 32,
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "weights.json")
        with open(path, "w") as f:
            json.dump(weights, f)

        output_str = rust_binary_runner(
            [
                "--batch-size",
                "1",
                "--input-sequence-length",
                "8",
                "--input-sequence-embed-dim",
                "4",
                "--mha-head-dim",
                "2",
                "--mha-num-heads",
                "2",
                "--ff-hidden-dim",
                "8",
                "--ff-output-dim",
                "4",
                "--weights",
                path,
                "--json",
            ]
        )

        print(output_str)

    try:
        output = json.loads(output_str)
    except json.JSONDecodeError:
        assert False, f"Failed to parse JSON: {output_str}"

    assert "spec" in output
    assert "values" in output

    spec = output["spec"]
    values = np.array(output["values"])

    assert list(spec["shape"]) == [8, 4]
    assert len(values) == 32
