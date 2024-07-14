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

A typical benchmark output comparing serial Numba code to 20 Intel Icelake threads to SXM Nvidia H100 GPU would be:

```sh
Generating 128 random voter rankings with 8,192 candidates
Generated voter rankings, proceeding with 20 threads
Serial: 454.5124 seconds, 1,209,550,826.73 candidates^3/sec
Parallel: 68.4111 seconds, 8,036,063,293.27 candidates^3/sec
CUDA: 1.9802 seconds, 277,620,752,084.85 candidates^3/sec
```

CUDA outperforms the baseline JIT-compiled parallel kernel by a factor of __34.55x__.
40 core CPU uses ~270 Watts, so 10 cores use ~67.5 Watts.
Our SXM Nvidia H100 has a ~700 Watt TDP, so 10.37x more power-hungry, meaning the CUDA implementation is 3.33x more power-efficient. 
As the matrix grows, the GPU utilization improves and the experimentally observed throughput fits a sub-cubic curve:

- 8 K: 2.0 s
- 16 K: 10.5 s
- 32 K: 51.3 s
- 48 K: 105.3 s

Resources:

- [Schulze voting method description](https://en.wikipedia.org/wiki/Schulze_method)
- [On traversal order for Floyd Warshall algorithm](https://moorejs.github.io/APSP-in-parallel/)
