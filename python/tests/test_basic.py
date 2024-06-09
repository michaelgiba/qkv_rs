import subprocess
import numpy as np
import pytest
import os
from functools import cache


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


def test_transformer_block_output(rust_binary_runner):
    """Tests the transformer block's output for specific input."""
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
        ]
    )
    assert len(output_str)  # Placeholder for actual assertions
