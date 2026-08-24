"""Benchmark suite for comparing Schulze voting algorithm implementations.

This module provides benchmarking tools to compare different implementations of the
Schulze voting method across various backends: serial Numba, parallel Numba with tiling,
OpenMP (CPU), and CUDA (GPU). It includes utilities for building pairwise preference
matrices from voter rankings and computing strongest paths using the Floyd-Warshall
algorithm with block-parallel optimizations.

Usage:
    uv run scalingelections.py
    uv run scalingelections.py --help
    uv run scalingelections.py --num-candidates 4096 --num-voters 4096

Backends are selected by regex against their names, so `-k GPU` runs both GPU rows and
`-k Hopper` runs only the bulk-tensor one:
    uv run scalingelections.py --num-candidates 2048 --num-voters 0 -k GPU --warmup 1 --repeat 20
    uv run scalingelections.py --num-candidates 4096 --num-voters 0 -k GPU --warmup 1 --repeat 10
    uv run scalingelections.py --num-candidates 8192 --num-voters 0 -k GPU --warmup 1 --repeat 5
    uv run scalingelections.py --num-candidates 16384 --num-voters 0 -k GPU --warmup 1 --repeat 3
    uv run scalingelections.py --num-candidates 32768 --num-voters 0 -k GPU --warmup 1 --repeat 1

See: https://github.com/ashvardanian/ScalingElections
"""

import re
import time

import numpy as np
from numba import get_num_threads
from scalingelections_cuda import (
    compute_strongest_paths,  # type: ignore
    log_gpus,  # type: ignore
)

from ballots import (
    build_pairwise_preferences,
    generate_preferences,
    populate_preferences_from_ranking,
    positive_margins,
)
from kemeny import (
    kemeny_costs,
    kemeny_ranking,
    kemeny_subset_sums,
    kemeny_votes_against,
)
from schulze import (
    compute_strongest_paths_numba_parallel,
    compute_strongest_paths_numba_serial,
    compute_strongest_paths_tile_numba,
    get_winner_and_ranking,
    split_cycle_winners,
)

# Re-exported so the split into per-method modules stays invisible to callers.
__all__ = [
    "benchmark_implementation",
    "build_pairwise_preferences",
    "compute_strongest_paths",
    "compute_strongest_paths_numba_parallel",
    "compute_strongest_paths_numba_serial",
    "compute_strongest_paths_tile_numba",
    "format_throughput",
    "format_time",
    "generate_preferences",
    "get_winner_and_ranking",
    "kemeny_costs",
    "kemeny_ranking",
    "kemeny_subset_sums",
    "kemeny_votes_against",
    "log_gpus",
    "populate_preferences_from_ranking",
    "positive_margins",
    "split_cycle_winners",
]


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
    # Warm-up runs on the full dataset, so JIT compilation and GPU clocks settle before timing.
    for iteration in range(warmup):
        start_time = time.perf_counter()
        _ = callback(preferences)
        elapsed_time = time.perf_counter() - start_time
        iteration_note = f" {iteration + 1}/{warmup}" if warmup > 1 else ""
        print(f"  Warm-up{iteration_note}: {format_time(elapsed_time)}")

    times = []
    result = None
    for _ in range(repeat):
        start_time = time.perf_counter()
        result = callback(preferences)
        elapsed_time = time.perf_counter() - start_time
        times.append(elapsed_time)

    avg_time = sum(times) / len(times)
    return avg_time, result


CONFIGURATION = """Configuration:
  Problem size: {num_candidates:,} candidates × {voters_description} voters
  CPU threads: {cpu_threads}
  Warmup: {warmup}, Repeat: {repeat}
  Backends: {backends}
"""

ELECTION_RESULTS = """Election Results

  Schulze winner: Candidate #{winner}
  Top {top_count}:          {top_shown}
  Split Cycle:    {undefeated_shown}{undefeated_more}
"""

NO_ELECTION_RESULTS = """Election Results

  No implementation was run, so there is no ranking to report.
"""


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

    num_voters = args.num_voters
    num_candidates = args.num_candidates
    selector = re.compile(args.filter)

    # The first selected backend that succeeds becomes the baseline every later one validates against.
    backends = [
        ("Serial (Numba)", lambda p: compute_strongest_paths_numba_serial(p)),
        ("Tiled CPU (Numba)", lambda p: compute_strongest_paths_numba_parallel(p)),
        ("Tiled CPU (OpenMP)", lambda p: compute_strongest_paths(p, backend="cpu_openmp")),
        ("Tiled GPU", lambda p: compute_strongest_paths(p, backend="gpu_serial")),
        ("Tiled GPU (Hopper)", lambda p: compute_strongest_paths(p, backend="gpu_hopper")),
    ]
    selected = [(name, callback) for name, callback in backends if selector.search(name)]
    if not selected:
        parser.error(f"--filter {args.filter!r} matched no backend of: " + ", ".join(name for name, _ in backends))

    print("Schulze Voting Algorithm (Python)\n")

    try:
        log_gpus()
    except Exception as error:
        print(f"✗ Could not detect GPU: {error}")

    voters_description = f"{num_voters:,}" if num_voters > 0 else "random"
    print(
        CONFIGURATION.format(
            num_candidates=num_candidates,
            voters_description=voters_description,
            cpu_threads=get_num_threads(),
            warmup=args.warmup,
            repeat=args.repeat,
            backends=", ".join(name for name, _ in selected),
        )
    )

    print("Generating preferences...")
    generator = np.random.default_rng(args.seed)
    preferences = generate_preferences(num_candidates, num_voters, generator)

    print("\nBenchmarking\n")

    baseline = None
    baseline_callback = None
    for name, callback in selected:
        print(f"→ {name}")
        try:
            avg_time, result = benchmark_implementation(callback, preferences, args.warmup, args.repeat)
            throughput = num_candidates**3 / avg_time
            average_note = f" (avg of {args.repeat})" if args.repeat > 1 else ""
            print(f"  Run:     {format_time(avg_time)}{average_note} │ {format_throughput(throughput)}")

            if baseline is None:
                baseline, baseline_callback = result, callback
            elif np.array_equal(result, baseline):
                print("  ✓ Results validated")
            else:
                print("  ✗ Results don't match baseline!")
        except Exception as error:
            print(f"  ✗ Benchmark failed: {error}")
        print()

    # Name the winner from whatever already ran, so a filtered-out kernel is never run behind
    # the caller's back.
    if baseline is None:
        print(NO_ELECTION_RESULTS)
        raise SystemExit(0)

    candidates = list(range(baseline.shape[0]))
    winner, ranking = get_winner_and_ranking(candidates, baseline)

    # Split Cycle needs the same recurrence over margins rather than winning votes, so it runs
    # the backend that produced the baseline instead of a serial pass the filter excluded.
    margin_paths = baseline_callback(positive_margins(preferences))
    undefeated = split_cycle_winners(preferences, margin_paths)

    top_count = min(5, len(ranking))
    top_shown = ", ".join(f"#{candidate}" for candidate in ranking[:top_count])
    undefeated_shown = ", ".join(f"#{candidate}" for candidate in undefeated[:5])
    undefeated_more = f" and {len(undefeated) - 5} more" if len(undefeated) > 5 else ""
    print(
        ELECTION_RESULTS.format(
            winner=winner,
            top_count=top_count,
            top_shown=top_shown,
            undefeated_shown=undefeated_shown,
            undefeated_more=undefeated_more,
        )
    )
