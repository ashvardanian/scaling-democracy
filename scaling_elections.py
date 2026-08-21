"""Benchmark suite for comparing Schulze voting algorithm implementations.

This module provides benchmarking tools to compare different implementations of the
Schulze voting method across various backends: serial Numba, parallel Numba with tiling,
OpenMP (CPU), and CUDA (GPU). It includes utilities for building pairwise preference
matrices from voter rankings and computing strongest paths using the Floyd-Warshall
algorithm with block-parallel optimizations.

Usage:
    uv run scaling_elections.py
    uv run scaling_elections.py --help
    uv run scaling_elections.py --num-candidates 4096 --num-voters 4096

Backends are selected by regex against their names, so `-k GPU` runs both GPU rows and
`-k Hopper` runs only the bulk-tensor one:
    uv run scaling_elections.py --num-candidates 2048 --num-voters 0 -k GPU --warmup 1 --repeat 20
    uv run scaling_elections.py --num-candidates 4096 --num-voters 0 -k GPU --warmup 1 --repeat 10
    uv run scaling_elections.py --num-candidates 8192 --num-voters 0 -k GPU --warmup 1 --repeat 5
    uv run scaling_elections.py --num-candidates 16384 --num-voters 0 -k GPU --warmup 1 --repeat 3
    uv run scaling_elections.py --num-candidates 32768 --num-voters 0 -k GPU --warmup 1 --repeat 1

See: https://github.com/ashvardanian/ScalingElections
"""

import re
import time
import warnings
from collections.abc import Sequence

import numpy as np
from numba import get_num_threads, njit, prange

from scaling_elections import (
    compute_strongest_paths,  # type: ignore
    log_gpus,  # type: ignore
)

# Suppress Numba TBB threading layer warnings
warnings.filterwarnings("ignore", message=".*TBB threading layer.*")


@njit
def populate_preferences_from_ranking(preferences: np.ndarray, ranking: np.ndarray):
    """
    Populates the preference matrix based on a ranking of candidates.
    The candidate must be represented as monotonic integers starting from 0.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^2), where n is the number of candidates.
    """
    for i, preferred in enumerate(ranking):
        for opponent in ranking[i + 1 :]:
            preferences[preferred, opponent] += 1


def build_pairwise_preferences(voter_rankings: Sequence[np.ndarray]) -> np.ndarray:
    """
    For every voter in the population, receives a (potentially incomplete) ranking of candidates,
    and builds a square preference matrix based on the rankings. Every cell (i, j) in the matrix
    contains the number of voters who prefer candidate i to candidate j.
    The candidate must be represented as monotonic integers starting from 0.
    If some candidates aren't included in a specific ranking, to break ties between them, random
    ballots are generated.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(m * n^2), where n is the number of candidates and m is the number of voters.
    """
    # The number of candidates is the maximum candidate index in the rankings plus one.
    count_candidates = 1
    for ranking in voter_rankings:
        count_candidates = max(count_candidates, np.max(ranking) + 1)

    # Initialize the preference matrix
    preferences = np.zeros((count_candidates, count_candidates), dtype=np.uint32)

    # Process each voter's ranking
    for ranking in voter_rankings:
        # We may be dealing with incomplete rankings
        if len(ranking) != count_candidates:
            # Create a mask for integers from 0 to N
            full_mask = np.ones(count_candidates, dtype=bool)
            # Mark the integers present in the incomplete array
            full_mask[ranking] = False
            # Find the missing integers
            missing_integers = np.nonzero(full_mask)[0]
            # Append the missing integers to the incomplete array
            ranking = np.append(ranking, missing_integers)

        # By now the ranking should be complete
        populate_preferences_from_ranking(preferences, ranking)

    return preferences


@njit
def compute_strongest_paths_numba_serial(preferences: np.ndarray) -> np.ndarray:
    """
    Computes the widest path strengths using the Schulze method.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^3), where n is the number of candidates.
    """
    num_candidates = preferences.shape[0]

    # Initialize the strongest paths matrix
    # assert preferences.dtype == np.uint32, f"Wrong type: {preferences.dtype}"
    strongest_paths = np.zeros((num_candidates, num_candidates), dtype=np.uint32)

    # Step 1: Populate the strongest paths matrix based on direct comparisons
    for i in range(num_candidates):
        for j in range(num_candidates):
            if i != j:
                if preferences[i, j] > preferences[j, i]:
                    strongest_paths[i, j] = preferences[i, j]
                else:
                    strongest_paths[i, j] = 0

    # Step 2: Compute the strongest paths using Floyd-Warshall-like algorithm
    for i in range(num_candidates):
        for j in range(num_candidates):
            if i != j:
                for k in range(num_candidates):
                    if i != k and j != k:
                        strongest_paths[j, k] = max(
                            strongest_paths[j, k],
                            min(strongest_paths[j, i], strongest_paths[i, k]),
                        )

    return strongest_paths


@njit
def compute_strongest_paths_tile_numba(
    c: np.ndarray,
    c_row: int,
    c_col: int,
    a: np.ndarray,
    a_row: int,
    a_col: int,
    b: np.ndarray,
    b_row: int,
    b_col: int,
    tile_size: int = 16,
):
    """
    In-place computation of the widest path path using the Schulze method with tiling for better cache utilization.
    For input of size (n x n), would perform (n) iterations of quadratic complexity each.

    Time complexity: O(n^3), where n is the tile size.
    Space complexity: O(n^2), where n is the tile size.
    """

    # `njit` compiles with `boundscheck=False`, so the tail of a non-divisible matrix has
    # to be clamped here rather than trapped on access.
    num_candidates = c.shape[0]
    k_extent = min(tile_size, num_candidates - max(a_col, b_row))
    i_extent = min(tile_size, num_candidates - max(c_row, a_row))
    j_extent = min(tile_size, num_candidates - max(c_col, b_col))

    for k in range(k_extent):
        for i in range(i_extent):
            for j in range(j_extent):
                if (c_row + i != c_col + j) and (a_row + i != a_col + k) and (b_row + k != b_col + j):
                    replacement = min(a[a_row + i, a_col + k], b[b_row + k, b_col + j])
                    if replacement > c[c_row + i, c_col + j]:
                        c[c_row + i, c_col + j] = replacement


@njit(parallel=True)
def compute_strongest_paths_numba_parallel(
    preferences: np.ndarray,
    tile_size: int = 16,
) -> np.ndarray:
    """
    Computes the widest path strengths using the Schulze method with tiling for better cache utilization.
    This implementation not only parallelizes the outer loop but also tiles the computation, to maximize
    the utilization of CPU caches.

    Space complexity:
    Time complexity:
    """
    num_candidates = preferences.shape[0]

    # Initialize the strongest paths matrix
    # assert preferences.dtype == np.uint32, f"Wrong type: {preferences.dtype}"
    strongest_paths = np.zeros((num_candidates, num_candidates), dtype=np.uint32)

    # Step 1: Populate the strongest paths matrix based on direct comparisons
    for i in range(num_candidates):
        for j in range(num_candidates):
            if i != j:
                if preferences[i, j] > preferences[j, i]:
                    strongest_paths[i, j] = preferences[i, j]
                else:
                    strongest_paths[i, j] = 0

    # Step 2: Compute the strongest paths using Floyd-Warshall-like algorithm with tiling
    tiles_count = (num_candidates + tile_size - 1) // tile_size
    for k in range(tiles_count):
        # Dependent phase
        k_start = k * tile_size

        # f(S_kk, S_kk, S_kk)
        compute_strongest_paths_tile_numba(
            strongest_paths,
            k_start,
            k_start,
            strongest_paths,
            k_start,
            k_start,
            strongest_paths,
            k_start,
            k_start,
            tile_size,
        )

        # Partially dependent phase (first of two)
        for i in prange(tiles_count):
            if i == k:
                continue
            i_start = i * tile_size
            # f(S_ik, S_ik, S_kk)
            compute_strongest_paths_tile_numba(
                strongest_paths,
                i_start,
                k_start,
                strongest_paths,
                i_start,
                k_start,
                strongest_paths,
                k_start,
                k_start,
                tile_size,
            )

        # Partially dependent phase (second of two)
        for j in prange(tiles_count):
            if j == k:
                continue
            j_start = j * tile_size
            # f(S_kj, S_kk, S_kj)
            compute_strongest_paths_tile_numba(
                strongest_paths,
                k_start,
                j_start,
                strongest_paths,
                k_start,
                k_start,
                strongest_paths,
                k_start,
                j_start,
                tile_size,
            )

        # Independent phase
        for i in prange(tiles_count):
            if i == k:
                continue
            i_start = i * tile_size
            for j in range(tiles_count):
                if j == k:
                    continue
                j_start = j * tile_size
                # f(S_ij, S_ik, S_kj)
                compute_strongest_paths_tile_numba(
                    strongest_paths,
                    i_start,
                    j_start,
                    strongest_paths,
                    i_start,
                    k_start,
                    strongest_paths,
                    k_start,
                    j_start,
                    tile_size,
                )

    return strongest_paths


def get_winner_and_ranking(
    candidates: list,
    strongest_paths: np.ndarray,
) -> tuple[int, list[int]]:
    """
    Determines the winner and the overall ranking of candidates based on the strongest paths matrix.

    Space complexity: O(n), where n is the number of candidates.
    Time complexity: O(n^2), where n is the number of candidates.
    """
    num_candidates = len(candidates)
    wins = np.zeros(num_candidates, dtype=int)

    for i in range(num_candidates):
        for j in range(num_candidates):
            if i != j and strongest_paths[i, j] > strongest_paths[j, i]:
                wins[i] += 1

    ranking_indices = sorted(range(num_candidates), key=lambda x: wins[x], reverse=True)
    winner = candidates[ranking_indices[0]]
    ranked_candidates = [candidates[i] for i in ranking_indices]

    return winner, ranked_candidates


def format_time(elapsed_sec: float) -> str:
    """Format time with appropriate unit (ms or s)."""
    elapsed_ms = elapsed_sec * 1000
    if elapsed_ms < 1000:
        return f"{int(elapsed_ms)} ms"
    else:
        return f"{elapsed_sec:.2f} s"


def format_throughput(cells_per_sec: float) -> str:
    """Format throughput with appropriate unit (T/G/M cells/s)."""
    if cells_per_sec >= 1e12:
        return f"{cells_per_sec / 1e12:.1f} Tcells/s"
    elif cells_per_sec >= 1e9:
        return f"{cells_per_sec / 1e9:.1f} Gcells/s"
    elif cells_per_sec >= 1e6:
        return f"{cells_per_sec / 1e6:.1f} Mcells/s"
    else:
        return f"{cells_per_sec / 1e3:.1f} Kcells/s"


def benchmark_implementation(
    callback,
    preferences: np.ndarray,
    warmup: int,
    repeat: int,
):
    """Run warmup and benchmark iterations, returning the mean seconds and the last result.

    Args:
        callback: Function to benchmark
        preferences: Input preference matrix
        warmup: Number of warmup iterations
        repeat: Number of benchmark iterations

    Returns:
        Tuple of (avg_time, result, success) where success is True if all iterations passed
    """
    # Warmup iterations on full dataset (important for JIT/GPU tuning)
    for i in range(warmup):
        start_time = time.perf_counter()
        _ = callback(preferences)
        elapsed_time = time.perf_counter() - start_time
        if warmup > 1:
            print(f"  Warm-up {i + 1}/{warmup}: {format_time(elapsed_time)}")
        else:
            print(f"  Warm-up: {format_time(elapsed_time)}")

    # Benchmark iterations
    times = []
    result = None
    for _ in range(repeat):
        start_time = time.perf_counter()
        result = callback(preferences)
        elapsed_time = time.perf_counter() - start_time
        times.append(elapsed_time)

    avg_time = sum(times) / len(times)
    return avg_time, result


# Tile sizes the C++ extension actually instantiates, in `scaling_elections.cu`.
CPU_TILE_SIZES = (4, 8, 16, 32, 64, 128)
GPU_TILE_SIZES = (4, 8, 16, 32)


# Benchmark and comparison code remains the same
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the Schulze method")
    parser.add_argument(
        "--num-voters",
        type=int,
        default=2000,
        help="Number of voters in the population, 0 for random preference matrix",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=128,
        help="Number of candidates in the election",
    )
    parser.add_argument(
        "-k",
        "--filter",
        metavar="REGEX",
        default=".",
        help="Regex selecting which backends to run, matched against their names",
    )
    parser.add_argument(
        "--cpu-tile-size",
        type=int,
        default=32,
        choices=CPU_TILE_SIZES,
        help="CPU tile size for tiling optimization (default: 32)",
    )
    parser.add_argument(
        "--gpu-tile-size",
        type=int,
        default=32,
        choices=GPU_TILE_SIZES,
        help="GPU tile size for tiling optimization (default: 32)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the preference generator (default: 42)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of benchmark iterations (default: 1)",
    )
    args = parser.parse_args()
    if args.num_candidates < 4:
        parser.error("--num-candidates must be at least 4")
    if args.num_voters < 0:
        parser.error("--num-voters cannot be negative")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    cpu_tile_size = args.cpu_tile_size
    gpu_tile_size = args.gpu_tile_size
    num_voters = args.num_voters
    num_candidates = args.num_candidates
    selector = re.compile(args.filter)

    # The first selected backend that succeeds becomes the baseline every later one validates against.
    backends = [
        ("Serial (Numba)", lambda p: compute_strongest_paths_numba_serial(p)),
        ("Tiled CPU (Numba)", lambda p: compute_strongest_paths_numba_parallel(p, tile_size=cpu_tile_size)),
        ("Tiled CPU (OpenMP)", lambda p: compute_strongest_paths(p, backend="cpu_openmp", tile_size=cpu_tile_size)),
        ("Tiled GPU", lambda p: compute_strongest_paths(p, backend="gpu_serial", tile_size=gpu_tile_size)),
        ("Tiled GPU (Hopper)", lambda p: compute_strongest_paths(p, backend="gpu_hopper", tile_size=gpu_tile_size)),
    ]
    selected = [(name, callback) for name, callback in backends if selector.search(name)]
    if not selected:
        parser.error(f"--filter {args.filter!r} matched no backend of: " + ", ".join(name for name, _ in backends))

    # Print header
    print("Schulze Voting Algorithm (Python)")
    print()

    # Print GPU info if available
    try:
        log_gpus()
    except Exception as e:
        print(f"✗ Could not detect GPU: {e}")

    # Print configuration
    print("Configuration:")
    voters_str = f"{num_voters:,}" if num_voters > 0 else "random"
    print(f"  Problem size: {num_candidates:,} candidates × {voters_str} voters")
    print(f"  CPU tile: {cpu_tile_size} × {cpu_tile_size}")
    print(f"  GPU tile: {gpu_tile_size} × {gpu_tile_size}")
    print(f"  CPU threads: {get_num_threads()}")
    print(f"  Warmup: {args.warmup}, Repeat: {args.repeat}")
    print(f"  Backends: {', '.join(name for name, _ in selected)}")
    print()

    # Generate random voter rankings
    print("Generating preferences...")
    generator = np.random.default_rng(args.seed)
    if num_voters == 0:
        preferences = generator.integers(0, num_candidates, (num_candidates, num_candidates), dtype=np.uint32)
    else:
        voter_rankings = [generator.permutation(num_candidates) for _ in range(num_voters)]
        preferences = build_pairwise_preferences(voter_rankings)

    # Benchmarking section
    print()
    print("Benchmarking")
    print()

    baseline = None
    baseline_callback = None
    for name, callback in selected:
        print(f"→ {name}")
        try:
            avg_time, result = benchmark_implementation(callback, preferences, args.warmup, args.repeat)
            throughput = num_candidates**3 / avg_time
            if args.repeat > 1:
                print(f"  Run:     {format_time(avg_time)} (avg of {args.repeat}) │ {format_throughput(throughput)}")
            else:
                print(f"  Run:     {format_time(avg_time)} │ {format_throughput(throughput)}")

            if baseline is None:
                baseline, baseline_callback = result, callback
            elif np.array_equal(result, baseline):
                print("  ✓ Results validated")
            else:
                print("  ✗ Results don't match baseline!")
        except Exception as e:
            print(f"  ✗ Benchmark failed: {e}")
        print()

    # Name the winner from whatever already ran, so a filtered-out kernel is never run behind
    # the caller's back.
    if baseline is None:
        print("Election Results")
        print()
        print("  No implementation was run, so there is no ranking to report.")
        print()
        raise SystemExit(0)

    candidates = list(range(baseline.shape[0]))
    winner, ranking = get_winner_and_ranking(candidates, baseline)

    # Print election results
    print("Election Results")
    print()
    print(f"  Winner: Candidate #{winner}")
    if len(ranking) >= 5:
        print(f"  Top 5:  #{ranking[0]}, #{ranking[1]}, #{ranking[2]}, #{ranking[3]}, #{ranking[4]}")
    print()
