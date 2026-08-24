# Scaling Elections with GPUs and Mojo 🔥

![Scaling Elections Thumbnail](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/ScalingElections.jpg?raw=true)

This repository implements tiled parallel adaptations of the Schulze voting method, Split Cycle, and Kemeny-Young, with hardware acceleration across CPUs and GPUs in Mojo and CUDA C++ wrapped into Python.
The Schulze method is often used by Pirate Parties and open-source foundations, and it's a good example of a combinatorial problem that can be parallelized by changing evaluation order.

Every method here is a __C2 rule__ in Fishburn's classification, depending on the electorate only through the `N × N` matrix of pairwise counts and on nothing else about the ballots.
Ballots are folded into that matrix once and every method reads it, so the same tiled max-min kernel computes Schulze on winning votes and Split Cycle on margins, and the Kemeny solver's runtime is flat in the size of the electorate — ten thousand ranked lists and ten million cost the same once ingested.
The answers still differ: Schulze yields a ranking, Kemeny a strict ordering and the disagreement it achieves, and Split Cycle a set of undefeated candidates, irresolute by theorem rather than by omission.

The boundary is the __STV family__.
STV, CPO-STV, and Schulze STV eliminate candidates and re-transfer their votes, which the pairwise summary cannot reconstruct, so they are not C2 and nothing here implements them.

Each method is one module per language — `ballots`, `schulze`, and `kemeny`, as `.cuh`, `.mojo`, and `.py` — with `types.cuh` holding the scalar types, the HIP shim, and the compile-time tile edge.
`scalingelections.cu` and `scalingelections.mojo` are the binding layers, `scalingelections.py` and `cli.mojo` the benchmark drivers, and `test.py` cross-checks all three languages against each other.

Not a single line of CMake is used in this repository!
The entire native library build is packed into `setup.py` for Python, and `pixi` takes care of the Mojo build.

## Usage

Both Python and Mojo implementations are included.
Both support the same CLI arguments:

- `-k REGEX`, `--filter REGEX`: Select which backends run, matched against their names
- `--num-candidates N`: Number of candidates, 128 by default
- `--num-voters N`: Number of voters, 2000 by default, 0 to generate a random matrix directly
- `--warmup N`: Warm-up iterations to discard, 1 by default
- `--repeat N`: Timed iterations to average, 1 by default
- `--seed N`: Seed for the preference generator, 42 by default
- `--help`, `-h`: Show help message

Every backend is selected by name, so `-k GPU` runs both GPU rows and `-k Hopper` runs only the bulk-tensor one.
Python matches the pattern as a regular expression, Mojo as a case-insensitive substring, and the default `.` selects everything.
The backend names are:

| Driver | Names                                                                                          |
| :----- | :--------------------------------------------------------------------------------------------- |
| Python | `Serial (Numba)`, `Tiled CPU (Numba)`, `Tiled CPU (OpenMP)`, `Tiled GPU`, `Tiled GPU (Hopper)` |
| Mojo   | `Serial (Mojo)`, `Tiled CPU (Mojo)`, `Tiled CPU+SIMD (Mojo)`, `Tiled GPU (Mojo)`               |

The first backend that runs becomes the baseline the rest are validated against.
Both drivers reject unrecognized flags and refuse fewer than four candidates.

The tile edge is not a runtime flag.
It is fixed at compile time to 32, which fits a CPU L2 slice and matches an NVIDIA warp.
Override it with `-DSCALING_ELECTIONS_TILE=<n>` for the C++ build, or with `TILE_SIZE` in `schulze.mojo` and `schulze.py`.
A tile wider than the electorate is not an error: the tail is zero-filled, and zero is the identity of the max-min semiring, so padding can never win a comparison.

The extension itself takes the backend by name rather than by flags, and raises instead of quietly falling back when a device or a build cannot serve it:

```py
compute_strongest_paths(preferences, backend="cpu_openmp")
compute_strongest_paths(preferences, backend="gpu_serial")
compute_strongest_paths(preferences, backend="gpu_hopper")  # needs sm_90 or newer
```

### Python

Pull:

```sh
git clone https://github.com/ashvardanian/ScalingElections.git
cd ScalingElections
```

Build the environment and run with `uv`:

```sh
uv venv -p python3.12               # Pick a recent Python version
uv sync --extra cpu                 # Build the CUDA extension and install dependencies
uv run scalingelections.py          # Run the default problem size
uv run scalingelections.py --num-candidates 4096 --num-voters 4096
```

Alternatively, with your local environment:

```sh
pip install -e . --force-reinstall  # Build locally and install dependencies
python scalingelections.py          # Run the default problem size
```

### Mojo

The Mojo implementation runs through `pixi`:

```sh
pixi install
pixi run mojo cli.mojo --help
pixi run mojo cli.mojo --num-candidates 4096 --num-voters 4096
```

Or compile it once and run the native binary:

```sh
pixi run build-cli # Writes `build/scalingelections`
./build/scalingelections --num-candidates 4096
```

`pixi run build` compiles `scalingelections.mojo` into `build/scalingelections_mojo.so`, the extension the test suite imports to check Mojo against the other two languages.

### Testing

`pixi run test` builds both Mojo artifacts and runs the suite:

```sh
pixi run test
```

## The Participation Paradox

Schulze fails __positive involvement__, so a ballot can hurt the very candidate it ranks first.
Eleven voters ranking Python, Rust, Go, and Java are enough to show it.
Ten of them elect Java, and adding one more ballot that puts Java first elects Python instead.
Split Cycle, closing the same kernel over margins rather than winning votes, keeps Java in its winning set either way.

| Electorate                         | Schulze | Split Cycle      |
| :--------------------------------- | :------ | :--------------- |
| 10 voters                          | Java    | Python, Go, Java |
| Plus one ballot ranking Java first | Python  | Python, Java     |

Split Cycle pays for that with irresolution, naming a set where Schulze names a single winner, because no rule over four or more candidates can be anonymous, neutral, stable for winners, and resolute at once.
To run just that check:

```sh
pytest test.py -k paradox
```

## Links

- [Blogpost](https://ashvardanian.com/posts/scaling-elections)
- [Schulze voting method description](https://en.wikipedia.org/wiki/Schulze_method)
- [On traversal order for Floyd Warshall algorithm](https://moorejs.github.io/APSP-in-parallel/)
- [CUDA + Python project template](https://github.com/ashvardanian/cuda-python-starter-kit)

## Throughput

Similar to measuring matrix multiplications in FLOPS, we can measure the throughput of the Schulze algorithm in cells per second.
Or in our case, in Giga- or TeraCells per Second (gcs/tcs), where a cell is a single pairwise comparison between two candidates.

| Candidates | Numba `384c` | Mojo 🔥 `384c` | Mojo 🔥 SIMD `384c` | CUDA `h100` | Mojo 🔥 `h100` | Mojo 🔥 `mi355x` |
| :--------- | -----------: | ------------: | -----------------: | ----------: | ------------: | --------------: |
| 2'048      |     34.4 gcs |      37.9 gcs |           62.1 gcs |   182.7 gcs |     153.4 gcs |       830.8 gcs |
| 4'096      |     86.8 gcs |      59.8 gcs |          171.5 gcs |   264.1 gcs |     232.6 gcs |         1.5 tcs |
| 8'192      |     74.6 gcs |      76.6 gcs |          357.3 gcs |   495.3 gcs |     408.0 gcs |         2.4 tcs |
| 16'384     |     76.7 gcs |      80.7 gcs |          369.0 gcs |   600.7 gcs |     635.3 gcs |         2.9 tcs |
| 32'768     |    101.4 gcs |      82.3 gcs |          293.1 gcs |   921.4 gcs |     893.7 gcs |         3.5 tcs |

> The `384c` columns refer to benchmarks obtained on AWS `m8i` instances with dual-socket Xeon 6 CPUs, totalling 384 cores.
> The `h100` columns refer to benchmarks obtained on Nebius GPU instances with NVIDIA H100 GPUs.
> The `mi355x` column refers to benchmarks obtained on AMD's MI355X GPUs.

With NVIDIA Nsight Compute CLI we can dissect the kernels and see that there is more room for improvement:

```sh
ncu uv run scalingelections.py --num-candidates 4096 --num-voters 4096 -k GPU
>  void _cuda_independent<32>(unsigned int, unsigned int, unsigned int *) (128, 128, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 9.0
>    Section: GPU Speed Of Light Throughput
>    ----------------------- ----------- ------------
>    Metric Name             Metric Unit Metric Value
>    ----------------------- ----------- ------------
>    DRAM Frequency                  Ghz         3.20
>    SM Frequency                    Ghz         1.50
>    Elapsed Cycles                cycle       307595
>    Memory Throughput                 %        78.65
>    DRAM Throughput                   %        10.69
>    Duration                         us       205.22
>    L1/TEX Cache Throughput           %        79.90
>    L2 Cache Throughput               %        15.54
>    SM Active Cycles              cycle    302305.17
>    Compute (SM) Throughput           %        66.51
>    ----------------------- ----------- ------------
>
>    OPT   Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the L1 
>          bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes      
>          transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or        
>          whether there are values you can (re)compute.                                                                 
>
>    Section: Launch Statistics
>    -------------------------------- --------------- ---------------
>    Metric Name                          Metric Unit    Metric Value
>    -------------------------------- --------------- ---------------
>    Block Size                                                  1024
>    Cluster Scheduling Policy                           PolicySpread
>    Cluster Size                                                   0
>    Function Cache Configuration                     CachePreferNone
>    Grid Size                                                  16384
>    Registers Per Thread             register/thread              31
>    Shared Memory Configuration Size           Kbyte           32.77
>    Driver Shared Memory Per Block       Kbyte/block            1.02
>    Dynamic Shared Memory Per Block       byte/block               0
>    Static Shared Memory Per Block       Kbyte/block           12.29
>    # SMs                                         SM             132
>    Stack Size                                                  1024
>    Threads                                   thread        16777216
>    # TPCs                                                        66
>    Enabled TPC IDs                                              all
>    Uses Green Context                                             0
>    Waves Per SM                                               62.06
>    -------------------------------- --------------- ---------------
>
>    Section: Occupancy
>    ------------------------------- ----------- ------------
>    Metric Name                     Metric Unit Metric Value
>    ------------------------------- ----------- ------------
>    Max Active Clusters                 cluster            0
>    Max Cluster Size                      block            8
>    Overall GPU Occupancy                     %            0
>    Cluster Occupancy                         %            0
>    Block Limit Barriers                  block           32
>    Block Limit SM                        block           32
>    Block Limit Registers                 block            2
>    Block Limit Shared Mem                block            2
>    Block Limit Warps                     block            2
>    Theoretical Active Warps per SM        warp           64
>    Theoretical Occupancy                     %          100
>    Achieved Occupancy                        %        92.97
>    Achieved Active Warps Per SM           warp        59.50
>    ------------------------------- ----------- ------------
```

So feel free to fork and suggest improvements 🤗

## Citation

If ScalingElections helps your research or product, please cite it:

```bibtex
@software{Vardanian_ScalingElections,
  author = {Vardanian, Ash},
  title = {{ScalingElections: Condorcet Voting at GPU Speed — Schulze as semiring matrix multiplication, Kemeny-Young as exact NP-hard search}},
  doi = {10.5281/zenodo.22073377},
  url = {https://github.com/ashvardanian/ScalingElections},
  license = {Apache-2.0}
}
```

That is the concept DOI, so it resolves to whichever release is newest.
[`CITATION.cff`](CITATION.cff) carries it alongside the DOI minted for the specific version, for when a paper needs to name the exact code it ran.
