# qkv_rs

This is an experimental project which intends to make a program that can perform inference as a **single block** of a transformer. The idea is if we are able to develop a flexible and optimized transformer block we could launch many of them and have them
communicate to perform full inference of a model. This is very much a work in progress and I started it to keep up to speed on
best practices of transformer-based model inference optimizations. (KV caching generally, paged attention, flash attention, etc.)

## Project Goals

* **Single Block Focus:** The `qkv_rs` program should only launch a single block (i.e not trying to rebuild pytorch)
* **Modular Design:** In order to support future optimizations the project is separating into a logical/physical graph. The physical graph can have hardware specific optimization.
* **Scalable Communication (IPC):**  Eventually support IPC if multiple blocks are launched
* **Forward Pass Only** Nothing around backprop/gradient calculation etc. should be added to this codebase. It significantly adds complexity and doesn't match the project intention

## Current Status

* **Compute Graph in Progress:**
    * The logical and physical structures of the compute graph are being actively developed.
    * I'm exploring different ways of structuring the graphs, considering rewrite rules, etc.
    * Need to add support to read weight files from a few formats with `GGUF` being first
    * Need to add tests to validate the eventual outputs against `xformers` and alike
* **Potential Features:**
    * Quantization:
    * Sparse Attention:
    * Mixed Precision:

## Contributing

This is very much an educational side project, if you'd like to add something please don't hesistate. Please feel free to open issues for bug reports or feature requests. If you'd like to contribute code, fork the repository and submit a pull request.
