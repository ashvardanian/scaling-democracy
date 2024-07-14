# Scaling Democracy

## With GPUs & Algebraic Graph Theory

This repository implements the Schulze voting algorithm using CUDA for hardware acceleration.
It's built as a single `scaling_democracy.cu` CUDA file, wrapped with PyBind11, and compiled __without__ CMake directly from the `setup.py`.
To pull, build and benchmark locally:

```sh
git clone https://github.com/ashvardanian/scaling-democracy.git
cd scaling-democracy
git submodule update --init --recursive
pip install pybind11 numpy numba cupy-cuda12x
pip install -e .
python benchmark.py
```

It's a fun project by itself. It might be a good example of using GPUs for combinatorial problems with order-dependant traversal and also a good starting point for single-day CUDA hacking experiences with the header-only [CCCL](https://github.com/NVIDIA/cccl), containing Thrust, CUB, and `libcudacxx`.

## Throughput

A typical benchmark output comparing serial Numba code to 20 Intel Icelake cores to a 700 Watt SXM Nvidia H100 GPU would be:

```sh
Generating 128 random voter rankings with 8,192 candidates
Generated voter rankings, proceeding with 20 threads
Serial: 452.1626 seconds, 1,215,836,431.94 candidates^3/sec
Parallel: 66.8721 seconds, 8,221,007,862.91 candidates^3/sec
CUDA: 3.5999 seconds, 152,713,704,578.32 candidates^3/sec
```
