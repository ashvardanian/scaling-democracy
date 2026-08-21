"""
Mojo implementation of the Schulze voting algorithm with CPU and GPU acceleration.

This implementation provides parallel computation of strongest paths for democratic voting,
ported from the original Python/CUDA implementation in scaling_elections.py and scaling_elections.cu.
The Schulze method is a Condorcet voting system that selects the candidate who would win
in head-to-head comparisons against all other candidates.

## Usage

Run directly with Mojo via Pixi:

```bash
pixi run mojo scaling_elections.mojo
pixi run mojo scaling_elections.mojo --num-candidates 4096 --num-voters 4096
```

For proper benchmarking with large random-generated preference matrices:

```bash
pixi run mojo scaling_elections.mojo --num-candidates 2048 --num-voters 0 -k GPU --warmup 1 --repeat 20
pixi run mojo scaling_elections.mojo --num-candidates 4096 --num-voters 0 -k GPU --warmup 1 --repeat 10
pixi run mojo scaling_elections.mojo --num-candidates 8192 --num-voters 0 -k GPU --warmup 1 --repeat 5
pixi run mojo scaling_elections.mojo --num-candidates 16384 --num-voters 0 -k GPU --warmup 1 --repeat 3
pixi run mojo scaling_elections.mojo --num-candidates 32768 --num-voters 0 -k GPU --warmup 1 --repeat 1
```

Or compile and run:

```bash
pixi run mojo build scaling_elections.mojo -o scaling_elections
./scaling_elections
```

See: https://ashvardanian.com/posts/scaling-elections
"""

from std.builtin.sort import sort
from std.gpu import block_idx, thread_idx
from std.math import iota
from std.memory import (
    AddressSpace,
    Layout,
    alloc,
    stack_allocation,
    unsafe_memset_zero,
)
from std.random import random_si64, seed
from std.sys import argv, exit, has_accelerator
from std.time import perf_counter_ns
from max.algorithm import parallelize
from max.gpu import barrier
from max.gpu.host import DeviceContext

# region Types

# ! Tile sizes control cache utilization and parallelism efficiency.
# ! CPU: 32x32 fits L2 cache and simplifies AVX-512.
# ! GPU: 32x32 matches NVIDIA warp size (32 threads), maximizes occupancy.
# ! Pre-compiled variants allow runtime selection without JIT overhead.
comptime DEFAULT_CPU_TILE_SIZE = 32
comptime DEFAULT_GPU_TILE_SIZE = 32
comptime ALLOWED_CPU_TILE_SIZES = (4, 8, 16, 32, 64, 128)
comptime ALLOWED_GPU_TILE_SIZES = (4, 8, 16, 32)


@fieldwise_init
struct TilePhase(Copyable, Equatable, Movable):
    """Which tile phase a processor runs, fixing aliasing and diagonal handling together.

    The three states are the only combinations the recurrence produces.
    """

    var value: UInt8
    comptime aliased = Self(0)
    comptime distinct_diagonal = Self(1)
    comptime distinct_independent = Self(2)

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value


@fieldwise_init
struct IndexedScore(Comparable, Copyable, Equatable, Movable):
    """Pairs a candidate index with their win count for sorting."""

    var idx: Int
    var score: Int

    def __lt__(self, other: Self) -> Bool:
        # Sort in descending order by score (higher scores first)
        return self.score > other.score

    def __le__(self, other: Self) -> Bool:
        return self.score >= other.score

    def __eq__(self, other: Self) -> Bool:
        return self.score == other.score

    def __ne__(self, other: Self) -> Bool:
        return self.score != other.score

    def __gt__(self, other: Self) -> Bool:
        return self.score < other.score

    def __ge__(self, other: Self) -> Bool:
        return self.score <= other.score


@fieldwise_init
struct PreferenceMatrix(Movable):
    """Represents a preference matrix for Schulze voting."""

    var data: Pointer[UInt32, MutUntrackedOrigin]
    var num_candidates: Int

    def __init__(out self, num_candidates: Int):
        self.num_candidates = num_candidates
        var size = num_candidates * num_candidates
        self.data = alloc(Layout[UInt32](count=size)).unsafe_leak()
        unsafe_memset_zero(self.data, size)

    def __getitem__(self, i: Int, j: Int) -> UInt32:
        return self.data[unsafe_offset=i * self.num_candidates + j]

    def __setitem__(mut self, i: Int, j: Int, value: UInt32):
        self.data[unsafe_offset=i * self.num_candidates + j] = value

    def __deinit__(deinit self):
        self.data.unsafe_free()


@fieldwise_init
struct StrongestPathsMatrix(Movable):
    """Represents the strongest paths matrix result."""

    var data: Pointer[UInt32, MutUntrackedOrigin]
    var num_candidates: Int

    def __init__(out self, num_candidates: Int):
        self.num_candidates = num_candidates
        var size = num_candidates * num_candidates
        self.data = alloc(Layout[UInt32](count=size)).unsafe_leak()
        unsafe_memset_zero(self.data, size)

    def __getitem__(self, i: Int, j: Int) -> UInt32:
        return self.data[unsafe_offset=i * self.num_candidates + j]

    def __setitem__(mut self, i: Int, j: Int, value: UInt32):
        self.data[unsafe_offset=i * self.num_candidates + j] = value

    def __deinit__(deinit self):
        self.data.unsafe_free()


# endregion Types

# region Preferences


def populate_preferences_from_ranking(
    mut preferences: PreferenceMatrix, ranking: List[Int]
):
    """
    Populates the preference matrix based on a ranking of candidates.

    Args:
        preferences: The preference matrix to populate.
        ranking: List of candidate indices in order of preference.
    """
    var n = len(ranking)
    for i in range(n):
        var preferred = ranking[i]
        for j in range(i + 1, n):
            var opponent = ranking[j]
            var current_count = preferences[preferred, opponent]
            preferences[preferred, opponent] = current_count + 1


def build_pairwise_preferences(
    voter_rankings: List[List[Int]],
) -> PreferenceMatrix:
    """
    Builds a pairwise preference matrix from voter rankings.

    Args:
        voter_rankings: List of voter rankings (each ranking is a list of candidate indices).

    Returns:
        PreferenceMatrix with pairwise vote counts.
    """
    # Find maximum candidate index to determine matrix size
    var max_candidate = 0
    for i in range(len(voter_rankings)):
        var ranking = voter_rankings[i].copy()
        for j in range(len(ranking)):
            var candidate = ranking[j]
            if candidate > max_candidate:
                max_candidate = candidate

    var num_candidates = max_candidate + 1
    var preferences = PreferenceMatrix(num_candidates)

    # Process each voter's ranking
    for i in range(len(voter_rankings)):
        var ranking = voter_rankings[i].copy()

        # Create complete ranking if incomplete
        var complete_ranking = List[Int]()
        var used = List[Bool]()
        used.resize(num_candidates, False)

        # Add provided candidates
        for j in range(len(ranking)):
            var candidate = ranking[j]
            complete_ranking.append(candidate)
            used[candidate] = True

        # Add missing candidates in arbitrary order
        for k in range(num_candidates):
            if not used[k]:
                complete_ranking.append(k)

        populate_preferences_from_ranking(preferences, complete_ranking)

    return preferences^


# endregion Preferences

# region Serial Reference


def compute_strongest_paths_serial(
    preferences: PreferenceMatrix,
) raises -> StrongestPathsMatrix:
    """
    Serial implementation of Schulze strongest paths computation.

    Args:
        preferences: Input preference matrix.

    Returns:
        StrongestPathsMatrix with computed strongest paths.
    """
    var num_candidates = preferences.num_candidates
    var strongest_paths = StrongestPathsMatrix(num_candidates)

    # Step 1: Initialize strongest paths based on direct comparisons
    for i in range(num_candidates):
        for j in range(num_candidates):
            if i != j:
                var pref_ij = preferences[i, j]
                var pref_ji = preferences[j, i]
                if pref_ij > pref_ji:
                    strongest_paths[i, j] = pref_ij
                else:
                    strongest_paths[i, j] = 0

    # Step 2: Floyd-Warshall-like algorithm for strongest paths
    for i in range(num_candidates):
        for j in range(num_candidates):
            if i != j:
                for k in range(num_candidates):
                    if i != k and j != k:
                        var path_j_i = strongest_paths[j, i]
                        var path_i_k = strongest_paths[i, k]
                        var path_j_k = strongest_paths[j, k]
                        var new_path = min(path_j_i, path_i_k)
                        var max_path = max(path_j_k, new_path)
                        strongest_paths[j, k] = max_path

    return strongest_paths^


# endregion Serial Reference

# region CPU Tiles


def process_tile_cpu[
    tile_size: Int
](
    c: Pointer[UInt32, MutUntrackedOrigin],
    a: Pointer[UInt32, MutUntrackedOrigin],
    b: Pointer[UInt32, MutUntrackedOrigin],
    c_row: Int,
    c_col: Int,
    a_col: Int,
    b_col: Int,
    num_candidates: Int,
    tile_stride: Int,
):
    """
    CPU-optimized tile processing for blocked Schulze algorithm.

    Args:
        c: Output tile.
        a: First input tile.
        b: Second input tile.
        c_row: Row index of output tile.
        c_col: Column index of output tile.
        a_col: Column index of first input tile.
        b_col: Column index of second input tile.
        num_candidates: Total number of candidates.
        tile_stride: Stride for accessing tiles.
    """
    for k in range(tile_size):
        for bi in range(tile_size):
            for bj in range(tile_size):
                # Check bounds
                var global_i = c_row + bi
                var global_j = c_col + bj
                var global_k = a_col + k

                if (
                    global_i >= num_candidates
                    or global_j >= num_candidates
                    or global_k >= num_candidates
                ):
                    continue

                # Skip diagonal elements
                if (
                    global_i == global_j
                    or global_i == global_k
                    or global_k == global_j
                ):
                    continue

                var a_val = a[unsafe_offset=bi * tile_stride + k]
                var b_val = b[unsafe_offset=k * tile_stride + bj]
                var c_idx = bi * tile_stride + bj
                var c_val = c[unsafe_offset=c_idx]
                var new_val = min(a_val, b_val)

                if new_val > c_val:
                    c[unsafe_offset=c_idx] = new_val


def process_tile_cpu_simd_independent[
    tile_size: Int, simd_width: Int
](
    c: Pointer[UInt32, MutUntrackedOrigin],
    a: Pointer[UInt32, MutUntrackedOrigin],
    b: Pointer[UInt32, MutUntrackedOrigin],
    tile_stride: Int,
):
    """
    SIMD-vectorized tile processor for independent tiles (no diagonal checking needed).
    Processes multiple elements along the j dimension using SIMD vectors.

    Args:
        c: Output tile.
        a: First input tile.
        b: Second input tile.
        tile_stride: Stride for accessing tiles.
    """
    # Process k loop over intermediate values
    for k in range(tile_size):
        # Process each row
        for bi in range(tile_size):
            var a_val = a[unsafe_offset=bi * tile_stride + k]

            # Vectorized processing of columns (j dimension)
            comptime num_simd_chunks = tile_size // simd_width

            # Process all elements with SIMD
            for chunk in range(num_simd_chunks):
                var bj = chunk * simd_width
                var c_base_idx = bi * tile_stride + bj
                var b_base_idx = k * tile_stride + bj

                # Load SIMD vectors
                var c_vec = c.unsafe_load[width=simd_width](c_base_idx)
                var b_vec = b.unsafe_load[width=simd_width](b_base_idx)

                # Broadcast a_val to SIMD vector
                var a_vec = SIMD[DType.uint32, simd_width](a_val)

                # Compute min(a, b) and max(c, min_val)
                var min_val = min(a_vec, b_vec)
                var new_c = max(c_vec, min_val)

                # Store result
                c.unsafe_store[width=simd_width](c_base_idx, new_c)


def process_tile_cpu_simd_diagonal[
    tile_size: Int, simd_width: Int
](
    c: Pointer[UInt32, MutUntrackedOrigin],
    a: Pointer[UInt32, MutUntrackedOrigin],
    b: Pointer[UInt32, MutUntrackedOrigin],
    c_row: Int,
    c_col: Int,
    a_col: Int,
    num_candidates: Int,
    tile_stride: Int,
):
    """
    SIMD-vectorized tile processor for diagonal tiles (requires diagonal avoidance).
    Uses masking and select operations to avoid branches.

    Args:
        c: Output tile.
        a: First input tile.
        b: Second input tile.
        c_row: Global row index of output tile.
        c_col: Global column index of output tile.
        a_col: Global column index for intermediate dimension.
        num_candidates: Total number of candidates.
        tile_stride: Stride for accessing tiles.
    """
    # Process k loop
    for k in range(tile_size):
        var global_k = a_col + k

        # Process each row
        for bi in range(tile_size):
            var global_i = c_row + bi
            var a_val = a[unsafe_offset=bi * tile_stride + k]

            # Vectorized processing with diagonal masking
            comptime num_simd_chunks = tile_size // simd_width

            # Process all elements with SIMD
            for chunk in range(num_simd_chunks):
                var bj = chunk * simd_width
                var c_base_idx = bi * tile_stride + bj
                var b_base_idx = k * tile_stride + bj

                # Load SIMD vectors
                var c_vec = c.unsafe_load[width=simd_width](c_base_idx)
                var b_vec = b.unsafe_load[width=simd_width](b_base_idx)
                var a_vec = SIMD[DType.uint32, simd_width](a_val)

                # Compute candidate new values
                var min_val = min(a_vec, b_vec)

                # Lane `lane` covers candidate `c_col + bj + lane`; skip the three diagonals.
                var global_j = iota[DType.int32, simd_width]() + Int32(
                    c_col + bj
                )
                var mask = (
                    global_j.ne(Int32(global_i))
                    & global_j.ne(Int32(global_k))
                    & min_val.gt(c_vec)
                    & SIMD[DType.bool, simd_width](fill=global_i != global_k)
                )
                c.unsafe_store[width=simd_width](
                    c_base_idx, mask.select(min_val, c_vec)
                )


def copy_tile_to_buffer(
    source: Pointer[UInt32, MutUntrackedOrigin],
    dest: Pointer[UInt32, MutUntrackedOrigin],
    start_row: Int,
    start_col: Int,
    tile_size: Int,
    num_candidates: Int,
):
    """Copy a tile from the global matrix to a local buffer."""
    for i in range(tile_size):
        for j in range(tile_size):
            var row = start_row + i
            var col = start_col + j
            if row < num_candidates and col < num_candidates:
                dest[unsafe_offset=i * tile_size + j] = source[
                    unsafe_offset=row * num_candidates + col
                ]
            else:
                dest[unsafe_offset=i * tile_size + j] = 0


def copy_buffer_to_tile(
    source: Pointer[UInt32, MutUntrackedOrigin],
    dest: Pointer[UInt32, MutUntrackedOrigin],
    start_row: Int,
    start_col: Int,
    tile_size: Int,
    num_candidates: Int,
):
    """Copy a tile from a local buffer back to the global matrix."""
    for i in range(tile_size):
        for j in range(tile_size):
            var row = start_row + i
            var col = start_col + j
            if row < num_candidates and col < num_candidates:
                dest[unsafe_offset=row * num_candidates + col] = source[
                    unsafe_offset=i * tile_size + j
                ]


@always_inline
def calculate_tile_bounds(
    tile_idx: Int, tile_size: Int, total_size: Int
) -> Tuple[Int, Int, Int]:
    """
    Calculate tile boundaries for blocking algorithms.

    Args:
        tile_idx: Index of the tile.
        tile_size: Size of each tile.
        total_size: Total problem size.

    Returns:
        Tuple of (start_index, end_index, actual_size).
    """
    var start = tile_idx * tile_size
    var end = min(start + tile_size, total_size)
    var size = end - start
    return (start, end, size)


# endregion CPU Tiles

# region CPU Drivers


def compute_strongest_paths_tiled_cpu[
    tile_size: Int = DEFAULT_CPU_TILE_SIZE
](preferences: PreferenceMatrix) raises -> StrongestPathsMatrix:
    """
    Tiled CPU implementation of Schulze strongest paths computation.
    Uses blocking for better cache utilization.

    Parameters:
        tile_size: Compile-time tile size for CPU processing (default: 16).

    Args:
        preferences: Input preference matrix.

    Returns:
        StrongestPathsMatrix with computed strongest paths.
    """
    var num_candidates = preferences.num_candidates
    var strongest_paths = StrongestPathsMatrix(num_candidates)

    # Step 1: Initialize strongest paths
    @parameter
    def init_paths(i: Int):
        for j in range(num_candidates):
            if i != j:
                var pref_ij = preferences[i, j]
                var pref_ji = preferences[j, i]
                if pref_ij > pref_ji:
                    strongest_paths[i, j] = pref_ij
                else:
                    strongest_paths[i, j] = 0

    parallelize[init_paths](num_candidates)

    # Step 2: Tiled Floyd-Warshall computation
    var num_tiles = (num_candidates + tile_size - 1) // tile_size

    for k in range(num_tiles):
        var k_index = k
        var k_bounds = calculate_tile_bounds(k, tile_size, num_candidates)
        var k_start = k_bounds[0]

        # Dependent phase: process diagonal tile
        var diagonal_tile = stack_allocation[
            tile_size * tile_size, UInt32, alignment=64
        ]()
        unsafe_memset_zero(diagonal_tile, tile_size * tile_size)

        copy_tile_to_buffer(
            strongest_paths.data,
            diagonal_tile,
            k_start,
            k_start,
            tile_size,
            num_candidates,
        )

        process_tile_cpu[tile_size](
            diagonal_tile,
            diagonal_tile,
            diagonal_tile,
            k_start,
            k_start,
            k_start,
            k_start,
            num_candidates,
            tile_size,
        )

        copy_buffer_to_tile(
            diagonal_tile,
            strongest_paths.data,
            k_start,
            k_start,
            tile_size,
            num_candidates,
        )

        # Partially dependent phases - row tiles
        @parameter
        def process_row_tiles(i: Int):
            if i == k_index:
                return

            var i_bounds = calculate_tile_bounds(i, tile_size, num_candidates)
            var i_start = i_bounds[0]

            var c_tile = stack_allocation[
                tile_size * tile_size, UInt32, alignment=64
            ]()
            var b_tile = stack_allocation[
                tile_size * tile_size, UInt32, alignment=64
            ]()
            unsafe_memset_zero(c_tile, tile_size * tile_size)
            unsafe_memset_zero(b_tile, tile_size * tile_size)

            copy_tile_to_buffer(
                strongest_paths.data,
                c_tile,
                i_start,
                k_start,
                tile_size,
                num_candidates,
            )
            copy_tile_to_buffer(
                strongest_paths.data,
                b_tile,
                k_start,
                k_start,
                tile_size,
                num_candidates,
            )

            process_tile_cpu[tile_size](
                c_tile,
                c_tile,
                b_tile,
                i_start,
                k_start,
                k_start,
                k_start,
                num_candidates,
                tile_size,
            )

            copy_buffer_to_tile(
                c_tile,
                strongest_paths.data,
                i_start,
                k_start,
                tile_size,
                num_candidates,
            )

        parallelize[process_row_tiles](num_tiles)

        # Partially dependent phases - column tiles
        @parameter
        def process_col_tiles(j: Int):
            if j == k_index:
                return

            var j_bounds = calculate_tile_bounds(j, tile_size, num_candidates)
            var j_start = j_bounds[0]

            var c_tile = stack_allocation[
                tile_size * tile_size, UInt32, alignment=64
            ]()
            var a_tile = stack_allocation[
                tile_size * tile_size, UInt32, alignment=64
            ]()
            unsafe_memset_zero(c_tile, tile_size * tile_size)
            unsafe_memset_zero(a_tile, tile_size * tile_size)

            copy_tile_to_buffer(
                strongest_paths.data,
                c_tile,
                k_start,
                j_start,
                tile_size,
                num_candidates,
            )
            copy_tile_to_buffer(
                strongest_paths.data,
                a_tile,
                k_start,
                k_start,
                tile_size,
                num_candidates,
            )

            process_tile_cpu[tile_size](
                c_tile,
                a_tile,
                c_tile,
                k_start,
                j_start,
                k_start,
                j_start,
                num_candidates,
                tile_size,
            )

            copy_buffer_to_tile(
                c_tile,
                strongest_paths.data,
                k_start,
                j_start,
                tile_size,
                num_candidates,
            )

        parallelize[process_col_tiles](num_tiles)

        # Independent phase
        @parameter
        def process_independent_tiles(idx: Int):
            var i = idx // num_tiles
            var j = idx % num_tiles

            if i == k_index or j == k_index:
                return

            var i_bounds = calculate_tile_bounds(i, tile_size, num_candidates)
            var i_start = i_bounds[0]

            var j_bounds = calculate_tile_bounds(j, tile_size, num_candidates)
            var j_start = j_bounds[0]

            var c_tile = stack_allocation[
                tile_size * tile_size, UInt32, alignment=64
            ]()
            var a_tile = stack_allocation[
                tile_size * tile_size, UInt32, alignment=64
            ]()
            var b_tile = stack_allocation[
                tile_size * tile_size, UInt32, alignment=64
            ]()
            unsafe_memset_zero(c_tile, tile_size * tile_size)
            unsafe_memset_zero(a_tile, tile_size * tile_size)
            unsafe_memset_zero(b_tile, tile_size * tile_size)

            copy_tile_to_buffer(
                strongest_paths.data,
                c_tile,
                i_start,
                j_start,
                tile_size,
                num_candidates,
            )
            copy_tile_to_buffer(
                strongest_paths.data,
                a_tile,
                i_start,
                k_start,
                tile_size,
                num_candidates,
            )
            copy_tile_to_buffer(
                strongest_paths.data,
                b_tile,
                k_start,
                j_start,
                tile_size,
                num_candidates,
            )

            process_tile_cpu[tile_size](
                c_tile,
                a_tile,
                b_tile,
                i_start,
                j_start,
                k_start,
                j_start,
                num_candidates,
                tile_size,
            )

            copy_buffer_to_tile(
                c_tile,
                strongest_paths.data,
                i_start,
                j_start,
                tile_size,
                num_candidates,
            )

        parallelize[process_independent_tiles](num_tiles * num_tiles)

    return strongest_paths^


def compute_strongest_paths_tiled_cpu_simd[
    tile_size: Int = DEFAULT_CPU_TILE_SIZE
](preferences: PreferenceMatrix) raises -> StrongestPathsMatrix:
    """
    SIMD-vectorized tiled CPU implementation of Schulze strongest paths computation.
    Uses phase-specific SIMD tile processors for optimal vectorization and minimal branching.

    Parameters:
        tile_size: Compile-time tile size for CPU processing (default: 16).

    Args:
        preferences: Input preference matrix.

    Returns:
        StrongestPathsMatrix with computed strongest paths.
    """
    # Use largest power-of-2 SIMD width (up to 16 for AVX-512) that divides tile_size
    comptime simd_width = (
        16 if tile_size % 16
        == 0 else 8 if tile_size % 8
        == 0 else 4 if tile_size % 4
        == 0 else 2 if tile_size % 2
        == 0 else 1
    )
    var num_candidates = preferences.num_candidates
    var strongest_paths = StrongestPathsMatrix(num_candidates)

    # Step 1: Initialize strongest paths (same as regular CPU version)
    @parameter
    def init_paths(i: Int):
        for j in range(num_candidates):
            if i != j:
                var pref_ij = preferences[i, j]
                var pref_ji = preferences[j, i]
                if pref_ij > pref_ji:
                    strongest_paths[i, j] = pref_ij
                else:
                    strongest_paths[i, j] = 0

    parallelize[init_paths](num_candidates)

    # Step 2: SIMD-vectorized tiled Floyd-Warshall computation
    var num_tiles = (num_candidates + tile_size - 1) // tile_size

    for k in range(num_tiles):
        var k_index = k
        var k_bounds = calculate_tile_bounds(k, tile_size, num_candidates)
        var k_start = k_bounds[0]

        # Diagonal phase: uses diagonal-aware SIMD processor
        var diagonal_tile = stack_allocation[
            tile_size * tile_size, UInt32, alignment=64
        ]()
        unsafe_memset_zero(diagonal_tile, tile_size * tile_size)

        copy_tile_to_buffer(
            strongest_paths.data,
            diagonal_tile,
            k_start,
            k_start,
            tile_size,
            num_candidates,
        )

        process_tile_cpu_simd_diagonal[tile_size, simd_width](
            diagonal_tile,
            diagonal_tile,
            diagonal_tile,
            k_start,
            k_start,
            k_start,
            num_candidates,
            tile_size,
        )

        copy_buffer_to_tile(
            diagonal_tile,
            strongest_paths.data,
            k_start,
            k_start,
            tile_size,
            num_candidates,
        )

        # Partially dependent phases - row and column tiles
        @parameter
        def process_row_col_tiles(i: Int):
            if i == k_index:
                return

            var i_bounds = calculate_tile_bounds(i, tile_size, num_candidates)
            var i_start = i_bounds[0]

            # Row tile (i, k)
            var c_tile_row = stack_allocation[
                tile_size * tile_size, UInt32, alignment=64
            ]()
            var b_tile = stack_allocation[
                tile_size * tile_size, UInt32, alignment=64
            ]()
            unsafe_memset_zero(c_tile_row, tile_size * tile_size)
            unsafe_memset_zero(b_tile, tile_size * tile_size)

            copy_tile_to_buffer(
                strongest_paths.data,
                c_tile_row,
                i_start,
                k_start,
                tile_size,
                num_candidates,
            )
            copy_tile_to_buffer(
                strongest_paths.data,
                b_tile,
                k_start,
                k_start,
                tile_size,
                num_candidates,
            )

            process_tile_cpu_simd_diagonal[tile_size, simd_width](
                c_tile_row,
                c_tile_row,
                b_tile,
                i_start,
                k_start,
                k_start,
                num_candidates,
                tile_size,
            )

            copy_buffer_to_tile(
                c_tile_row,
                strongest_paths.data,
                i_start,
                k_start,
                tile_size,
                num_candidates,
            )

            # Column tile (k, i)
            var c_tile_col = stack_allocation[
                tile_size * tile_size, UInt32, alignment=64
            ]()
            var a_tile = stack_allocation[
                tile_size * tile_size, UInt32, alignment=64
            ]()
            unsafe_memset_zero(c_tile_col, tile_size * tile_size)
            unsafe_memset_zero(a_tile, tile_size * tile_size)

            copy_tile_to_buffer(
                strongest_paths.data,
                c_tile_col,
                k_start,
                i_start,
                tile_size,
                num_candidates,
            )
            copy_tile_to_buffer(
                strongest_paths.data,
                a_tile,
                k_start,
                k_start,
                tile_size,
                num_candidates,
            )

            process_tile_cpu_simd_diagonal[tile_size, simd_width](
                c_tile_col,
                a_tile,
                c_tile_col,
                k_start,
                i_start,
                k_start,
                num_candidates,
                tile_size,
            )

            copy_buffer_to_tile(
                c_tile_col,
                strongest_paths.data,
                k_start,
                i_start,
                tile_size,
                num_candidates,
            )

        parallelize[process_row_col_tiles](num_tiles)

        # Independent phase: uses fast SIMD processor (no diagonal checks)
        @parameter
        def process_independent_tiles(idx: Int):
            var i = idx // num_tiles
            var j = idx % num_tiles

            if i == k_index or j == k_index:
                return

            var i_bounds = calculate_tile_bounds(i, tile_size, num_candidates)
            var i_start = i_bounds[0]

            var j_bounds = calculate_tile_bounds(j, tile_size, num_candidates)
            var j_start = j_bounds[0]

            var c_tile = stack_allocation[
                tile_size * tile_size, UInt32, alignment=64
            ]()
            var a_tile = stack_allocation[
                tile_size * tile_size, UInt32, alignment=64
            ]()
            var b_tile = stack_allocation[
                tile_size * tile_size, UInt32, alignment=64
            ]()
            unsafe_memset_zero(c_tile, tile_size * tile_size)
            unsafe_memset_zero(a_tile, tile_size * tile_size)
            unsafe_memset_zero(b_tile, tile_size * tile_size)

            copy_tile_to_buffer(
                strongest_paths.data,
                c_tile,
                i_start,
                j_start,
                tile_size,
                num_candidates,
            )
            copy_tile_to_buffer(
                strongest_paths.data,
                a_tile,
                i_start,
                k_start,
                tile_size,
                num_candidates,
            )
            copy_tile_to_buffer(
                strongest_paths.data,
                b_tile,
                k_start,
                j_start,
                tile_size,
                num_candidates,
            )

            # Use independent processor if not on diagonal, otherwise use diagonal processor
            if i == j:
                process_tile_cpu_simd_diagonal[tile_size, simd_width](
                    c_tile,
                    a_tile,
                    b_tile,
                    i_start,
                    j_start,
                    k_start,
                    num_candidates,
                    tile_size,
                )
            else:
                process_tile_cpu_simd_independent[tile_size, simd_width](
                    c_tile, a_tile, b_tile, tile_size
                )

            copy_buffer_to_tile(
                c_tile,
                strongest_paths.data,
                i_start,
                j_start,
                tile_size,
                num_candidates,
            )

        parallelize[process_independent_tiles](num_tiles * num_tiles)

    return strongest_paths^


# endregion CPU Drivers

# region Results


def compute_election_results(
    strongest_paths: StrongestPathsMatrix,
) -> Tuple[Int, List[Int]]:
    """
    Determines the winner and ranking based on strongest paths matrix.

    Args:
        strongest_paths: Computed strongest paths matrix.

    Returns:
        Tuple of (winner_candidate_id, ranked_candidate_ids).
    """
    var num_candidates = strongest_paths.num_candidates
    var wins = List[Int]()
    wins.resize(num_candidates, 0)

    # Count wins for each candidate
    for i in range(num_candidates):
        var win_count = 0
        for j in range(num_candidates):
            if i != j and strongest_paths[i, j] > strongest_paths[j, i]:
                win_count += 1
        wins[i] = win_count

    # Find winner (candidate with most wins)
    var winner_idx = 0
    var max_wins = wins[0]
    for i in range(1, num_candidates):
        if wins[i] > max_wins:
            max_wins = wins[i]
            winner_idx = i

    # Create ranking by sorting candidates by win count (O(n log n) using built-in sort)
    var scored_candidates = List[IndexedScore]()
    for i in range(num_candidates):
        scored_candidates.append(IndexedScore(i, wins[i]))

    # Sort by score (IndexedScore's __lt__ sorts in descending order)
    sort(scored_candidates)

    # Extract just the candidate indices
    var ranking = List[Int]()
    for i in range(len(scored_candidates)):
        ranking.append(scored_candidates[i].idx)

    return (winner_idx, ranking^)


# endregion Results

# region Benchmarking


def generate_random_preferences(
    num_candidates: Int, num_voters: Int
) -> PreferenceMatrix:
    """
    Generates random preference matrix for testing.

    Args:
        num_candidates: Number of candidates.
        num_voters: Number of voters. If 0, generates random preference matrix directly.

    Returns:
        Random preference matrix.
    """
    var preferences = PreferenceMatrix(num_candidates)

    # Fast path: directly generate random preference matrix (parallelized)
    if num_voters == 0:

        @parameter
        def fill_row(i: Int):
            for j in range(num_candidates):
                preferences[i, j] = UInt32(
                    random_si64(0, Int64(num_candidates - 1))
                )

        parallelize[fill_row](num_candidates)
        return preferences^

    # Slow path: generate from voter rankings
    # Allocate ranking once and reuse for all voters
    var ranking = List[Int]()
    ranking.resize(num_candidates, 0)

    for _ in range(num_voters):
        # Re-initialize ranking in-place
        for i in range(num_candidates):
            ranking[i] = i

        # Fisher-Yates shuffle
        for i in range(num_candidates - 1, 0, -1):
            var j = Int(random_si64(0, Int64(i)))
            var temp = ranking[i]
            ranking[i] = ranking[j]
            ranking[j] = temp

        populate_preferences_from_ranking(preferences, ranking)

    return preferences^


def run_warmup[
    implementation: def(PreferenceMatrix) raises thin -> StrongestPathsMatrix
](preferences: PreferenceMatrix, warmup: Int) raises:
    """Run warmup iterations and print timing."""
    for i in range(warmup):
        var start_time = perf_counter_ns()
        _ = implementation(preferences)
        var elapsed_ns = perf_counter_ns() - start_time
        if warmup > 1:
            print(
                "  Warm-up {}/{}: {}".format(
                    i + 1, warmup, format_time(elapsed_ns)
                )
            )
        else:
            print("  Warm-up: {}".format(format_time(elapsed_ns)))


def measure_and_average[
    implementation: def(PreferenceMatrix) raises thin -> StrongestPathsMatrix
](
    preferences: PreferenceMatrix, repeat: Int, mut result: StrongestPathsMatrix
) raises -> Int:
    """Run benchmark iterations and return avg_time_ns, storing result in mutable parameter.
    """
    var times = List[Int]()
    for _ in range(repeat):
        var start_time = perf_counter_ns()
        result = implementation(preferences)
        var elapsed_ns = perf_counter_ns() - start_time
        times.append(elapsed_ns)

    # Calculate average time
    var total_time: Int = 0
    for i in range(len(times)):
        total_time += times[i]
    var avg_time = total_time // repeat
    return avg_time


def profile_and_report[
    implementation: def(PreferenceMatrix) raises thin -> StrongestPathsMatrix
](
    label: String,
    preferences: PreferenceMatrix,
    warmup: Int,
    repeat: Int,
    num_candidates: Int,
    mut baseline: StrongestPathsMatrix,
    mut has_baseline: Bool,
) raises -> Tuple[Int, List[Int]]:
    """Times one implementation, adopting the first result as the baseline.

    Returns the winner and the full ranking.
    """
    print("→ {}".format(label))

    run_warmup[implementation](preferences, warmup)

    var result = StrongestPathsMatrix(0)
    var avg_time = measure_and_average[implementation](
        preferences, repeat, result
    )

    if repeat > 1:
        print(
            "  Run:     {} (avg of {}) │ {}".format(
                format_time(avg_time),
                repeat,
                format_throughput_from_ns(avg_time, num_candidates),
            )
        )
    else:
        print(
            "  Run:     {} │ {}".format(
                format_time(avg_time),
                format_throughput_from_ns(avg_time, num_candidates),
            )
        )

    if has_baseline:
        if validate_against_baseline(result, baseline):
            print("  ✓ Results validated")
        else:
            print("  ✗ Results don't match baseline!")

    var result_tuple = compute_election_results(result)
    var winner_idx = result_tuple[0]
    var winner_ranking = result_tuple[1].copy()

    if not has_baseline:
        baseline = result^
        has_baseline = True

    print()
    return (winner_idx, winner_ranking^)


def format_time(elapsed_ns: Int) -> String:
    """Format time with appropriate unit (ms or s)."""
    var elapsed_ms = elapsed_ns // 1_000_000
    if elapsed_ms < 1000:
        return "{} ms".format(elapsed_ms)
    else:
        var elapsed_sec = Float64(elapsed_ns) / 1_000_000_000.0
        var sec_x100 = Int(elapsed_sec * 100.0)
        return "{}.{}{} s".format(
            sec_x100 // 100, (sec_x100 % 100) // 10, sec_x100 % 10
        )


def format_throughput(cells_per_sec: Float64) -> String:
    """Format throughput with appropriate unit (T/G/M cells/s)."""
    if cells_per_sec >= 1e12:
        var t_x10 = Int(cells_per_sec / 1e11)
        return "{}.{} Tcells/s".format(t_x10 // 10, t_x10 % 10)
    elif cells_per_sec >= 1e9:
        var g_x10 = Int(cells_per_sec / 1e8)
        return "{}.{} Gcells/s".format(g_x10 // 10, g_x10 % 10)
    elif cells_per_sec >= 1e6:
        var m_x10 = Int(cells_per_sec / 1e5)
        return "{}.{} Mcells/s".format(m_x10 // 10, m_x10 % 10)
    else:
        var k_x10 = Int(cells_per_sec / 1e2)
        return "{}.{} Kcells/s".format(k_x10 // 10, k_x10 % 10)


def format_throughput_from_ns(elapsed_ns: Int, num_candidates: Int) -> String:
    """Calculate and format throughput from timing."""
    if elapsed_ns <= 0:
        return "N/A"
    var elapsed_sec = Float64(elapsed_ns) / 1_000_000_000.0
    if elapsed_sec <= 0.0:
        return "N/A"
    var candidates_f = Float64(num_candidates)
    var total_cells = candidates_f * candidates_f * candidates_f
    var cells_per_sec = total_cells / elapsed_sec
    return format_throughput(cells_per_sec)


# endregion Benchmarking

# region GPU Kernels

comptime SharedUInt32Ptr = Pointer[
    UInt32, MutUntrackedOrigin, address_space=AddressSpace.SHARED
]


@always_inline
def process_tile_gpu_device[
    tile_size: Int, phase: TilePhase
](
    c_shared: SharedUInt32Ptr,
    a_shared: SharedUInt32Ptr,
    b_shared: SharedUInt32Ptr,
    c_row: Int,
    c_col: Int,
    a_row: Int,
    a_col: Int,
    b_row: Int,
    b_col: Int,
):
    """
    Core tile processing logic for GPU - runs on each thread.
    Processes one cell (bi, bj) of the tile through all k values.

    This matches the CUDA process_tile_cuda_ template function.
    """
    var bi = Int(thread_idx.y)
    var bj = Int(thread_idx.x)
    var c_idx = bi * tile_size + bj

    # Each thread processes one cell of the output tile
    var c_val = c_shared[unsafe_offset=c_idx]

    # Floyd-Warshall inner loop over k
    for k in range(tile_size):
        var global_k = a_col + k

        var a_idx = bi * tile_size + k
        var b_idx = k * tile_size + bj

        var a_val = a_shared[unsafe_offset=a_idx]
        var b_val = b_shared[unsafe_offset=b_idx]
        var smallest = min(a_val, b_val)

        comptime if phase != TilePhase.distinct_independent:
            # Compute global indices
            var global_i = c_row + bi
            var global_j = c_col + bj

            # Diagonal avoidance using branchless bit operations
            var is_not_diagonal_c = UInt32(
                1
            ) if global_i != global_j else UInt32(0)
            var is_not_diagonal_a = UInt32(
                1
            ) if global_i != global_k else UInt32(0)
            var is_not_diagonal_b = UInt32(
                1
            ) if global_k != global_j else UInt32(0)
            var is_bigger = UInt32(1) if smallest > c_val else UInt32(0)
            var will_replace = (
                is_not_diagonal_c
                & is_not_diagonal_a
                & is_not_diagonal_b
                & is_bigger
            )

            if will_replace == 1:
                c_val = smallest
        else:
            # Non-diagonal case - simple max
            c_val = max(c_val, smallest)

        # Write back IMMEDIATELY after update - critical for correctness!
        # When a_shared/b_shared/c_shared point to the same buffer (diagonal phase),
        # threads must see updated values from previous k iterations.
        c_shared[unsafe_offset=c_idx] = c_val

        comptime if phase == TilePhase.aliased:
            barrier()


def gpu_diagonal_kernel[
    tile_size: Int
](graph: Pointer[UInt32, MutUntrackedOrigin], n_arg: Int32, k_arg: Int32):
    """
    GPU kernel for diagonal phase - processes tile (k, k).
    Matches cuda_diagonal_ from CUDA implementation.
    """
    var n = Int(n_arg)
    var k = Int(k_arg)
    var bi = Int(thread_idx.y)
    var bj = Int(thread_idx.x)

    # Allocate shared memory for one tile
    var c_shared = stack_allocation[
        tile_size * tile_size,
        UInt32,
        address_space=AddressSpace.SHARED,
    ]()

    # Load tile from global memory
    c_shared[unsafe_offset=bi * tile_size + bj] = graph[
        unsafe_offset=k * tile_size * n + k * tile_size + bi * n + bj
    ]

    # Synchronize after load
    barrier()

    # Process tile (all three inputs are the same tile, need synchronization)
    process_tile_gpu_device[tile_size, TilePhase.aliased](
        c_shared,
        c_shared,
        c_shared,
        tile_size * k,
        tile_size * k,
        tile_size * k,
        tile_size * k,
        tile_size * k,
        tile_size * k,
    )

    # Synchronize before store
    barrier()

    # Write back to global memory
    graph[
        unsafe_offset=k * tile_size * n + k * tile_size + bi * n + bj
    ] = c_shared[unsafe_offset=bi * tile_size + bj]


def gpu_partially_independent_kernel[
    tile_size: Int
](graph: Pointer[UInt32, MutUntrackedOrigin], n_arg: Int32, k_arg: Int32):
    """
    GPU kernel for partially independent phase.
    Processes row and column tiles relative to diagonal tile k.
    Matches cuda_partially_independent_ from CUDA.
    """
    var n = Int(n_arg)
    var k = Int(k_arg)
    var i = Int(block_idx.x)
    var bi = Int(thread_idx.y)
    var bj = Int(thread_idx.x)

    if i == k:
        return

    # Allocate shared memory for three tiles
    var a_shared = stack_allocation[
        tile_size * tile_size,
        UInt32,
        address_space=AddressSpace.SHARED,
    ]()
    var b_shared = stack_allocation[
        tile_size * tile_size,
        UInt32,
        address_space=AddressSpace.SHARED,
    ]()
    var c_shared = stack_allocation[
        tile_size * tile_size,
        UInt32,
        address_space=AddressSpace.SHARED,
    ]()

    # Phase 1: Process row tile (i, k) using (i, k) and (k, k)
    # Load c[unsafe_offset=i,k] and b[unsafe_offset=k,k]
    c_shared[unsafe_offset=bi * tile_size + bj] = graph[
        unsafe_offset=i * tile_size * n + k * tile_size + bi * n + bj
    ]
    b_shared[unsafe_offset=bi * tile_size + bj] = graph[
        unsafe_offset=k * tile_size * n + k * tile_size + bi * n + bj
    ]

    barrier()

    process_tile_gpu_device[tile_size, TilePhase.aliased](
        c_shared,
        c_shared,
        b_shared,
        i * tile_size,
        k * tile_size,
        i * tile_size,
        k * tile_size,
        k * tile_size,
        k * tile_size,
    )

    barrier()

    # Store phase 1 result
    graph[
        unsafe_offset=i * tile_size * n + k * tile_size + bi * n + bj
    ] = c_shared[unsafe_offset=bi * tile_size + bj]

    # Phase 2: Process column tile (k, i) using (k, k) and (k, i)
    # Load c[unsafe_offset=k,i] and a[unsafe_offset=k,k]
    c_shared[unsafe_offset=bi * tile_size + bj] = graph[
        unsafe_offset=k * tile_size * n + i * tile_size + bi * n + bj
    ]
    a_shared[unsafe_offset=bi * tile_size + bj] = graph[
        unsafe_offset=k * tile_size * n + k * tile_size + bi * n + bj
    ]

    barrier()

    process_tile_gpu_device[tile_size, TilePhase.aliased](
        c_shared,
        a_shared,
        c_shared,
        k * tile_size,
        i * tile_size,
        k * tile_size,
        k * tile_size,
        k * tile_size,
        i * tile_size,
    )

    barrier()

    # Store phase 2 result
    graph[
        unsafe_offset=k * tile_size * n + i * tile_size + bi * n + bj
    ] = c_shared[unsafe_offset=bi * tile_size + bj]


def gpu_independent_kernel[
    tile_size: Int
](graph: Pointer[UInt32, MutUntrackedOrigin], n_arg: Int32, k_arg: Int32):
    """
    GPU kernel for independent phase - processes all tiles except row/column k.
    Matches cuda_independent_ from CUDA implementation.
    """
    var n = Int(n_arg)
    var k = Int(k_arg)
    var j = Int(block_idx.x)
    var i = Int(block_idx.y)
    var bi = Int(thread_idx.y)
    var bj = Int(thread_idx.x)

    if i == k and j == k:
        return

    # Allocate shared memory for three tiles
    var a_shared = stack_allocation[
        tile_size * tile_size,
        UInt32,
        address_space=AddressSpace.SHARED,
    ]()
    var b_shared = stack_allocation[
        tile_size * tile_size,
        UInt32,
        address_space=AddressSpace.SHARED,
    ]()
    var c_shared = stack_allocation[
        tile_size * tile_size,
        UInt32,
        address_space=AddressSpace.SHARED,
    ]()

    # Load three tiles: c[unsafe_offset=i,j], a[unsafe_offset=i,k], b[unsafe_offset=k,j]
    c_shared[unsafe_offset=bi * tile_size + bj] = graph[
        unsafe_offset=i * tile_size * n + j * tile_size + bi * n + bj
    ]
    a_shared[unsafe_offset=bi * tile_size + bj] = graph[
        unsafe_offset=i * tile_size * n + k * tile_size + bi * n + bj
    ]
    b_shared[unsafe_offset=bi * tile_size + bj] = graph[
        unsafe_offset=k * tile_size * n + j * tile_size + bi * n + bj
    ]

    barrier()

    # Process tile - use diagonal check if i == j, no synchronization needed (different tiles)
    if i == j:
        process_tile_gpu_device[tile_size, TilePhase.distinct_diagonal](
            c_shared,
            a_shared,
            b_shared,
            i * tile_size,
            j * tile_size,
            i * tile_size,
            k * tile_size,
            k * tile_size,
            j * tile_size,
        )
    else:
        process_tile_gpu_device[tile_size, TilePhase.distinct_independent](
            c_shared,
            a_shared,
            b_shared,
            i * tile_size,
            j * tile_size,
            i * tile_size,
            k * tile_size,
            k * tile_size,
            j * tile_size,
        )

    # No barrier needed - independent tiles write to different locations

    # Write back result
    graph[
        unsafe_offset=i * tile_size * n + j * tile_size + bi * n + bj
    ] = c_shared[unsafe_offset=bi * tile_size + bj]


# endregion GPU Kernels

# region GPU Driver


def compute_strongest_paths_gpu[
    tile_size: Int = DEFAULT_GPU_TILE_SIZE
](preferences: PreferenceMatrix) raises -> StrongestPathsMatrix:
    """
    Pure Mojo GPU implementation of Schulze strongest paths computation.

    Implements three-phase tiled Floyd-Warshall algorithm on GPU using native
    Mojo GPU kernels. Matches the CUDA implementation in scaling_elections.cu.

    Parameters:
        tile_size: Compile-time tile size for GPU processing (default: 32).

    Args:
        preferences: Input preference matrix.

    Returns:
        StrongestPathsMatrix with computed strongest paths.
    """
    var num_candidates = preferences.num_candidates
    var result = StrongestPathsMatrix(num_candidates)

    # Rounding up to a whole number of tiles keeps the kernels free of tail checks: the
    # padding is zero, which is the identity of the max-min semiring.
    var num_tiles = (num_candidates + tile_size - 1) // tile_size
    var padded = num_tiles * tile_size

    var ctx = DeviceContext()
    var host_graph = ctx.enqueue_create_host_buffer[DType.uint32](
        padded * padded
    )
    var device_graph = ctx.enqueue_create_buffer[DType.uint32](padded * padded)
    var host_ptr = host_graph.unsafe_ptr()
    unsafe_memset_zero(host_ptr, padded * padded)

    @parameter
    def init_paths(i: Int):
        for j in range(num_candidates):
            if i != j:
                var pref_ij = preferences[i, j]
                var pref_ji = preferences[j, i]
                if pref_ij > pref_ji:
                    host_ptr[unsafe_offset=i * padded + j] = pref_ij

    parallelize[init_paths](num_candidates)

    host_graph.enqueue_copy_to(device_graph)
    ctx.synchronize()

    var graph_ptr = device_graph.unsafe_ptr()
    var block_dim_tuple = (tile_size, tile_size, 1)

    for k in range(num_tiles):
        # Phase 1: Diagonal tile (sequential, 1 block)
        ctx.enqueue_function[gpu_diagonal_kernel[tile_size]](
            graph_ptr,
            Int32(padded),
            Int32(k),
            grid_dim=(1, 1, 1),
            block_dim=block_dim_tuple,
        )

        # Phase 2: Partially independent tiles (num_tiles blocks)
        ctx.enqueue_function[gpu_partially_independent_kernel[tile_size]](
            graph_ptr,
            Int32(padded),
            Int32(k),
            grid_dim=(num_tiles, 1, 1),
            block_dim=block_dim_tuple,
        )

        # Phase 3: Independent tiles (num_tiles x num_tiles blocks)
        ctx.enqueue_function[gpu_independent_kernel[tile_size]](
            graph_ptr,
            Int32(padded),
            Int32(k),
            grid_dim=(num_tiles, num_tiles, 1),
            block_dim=block_dim_tuple,
        )

        # Synchronize after each k iteration
        ctx.synchronize()

    # Step 7: Copy results back from GPU to CPU
    ctx.synchronize()  # Ensure all GPU operations complete

    # Copy from device buffer to host buffer
    device_graph.enqueue_copy_to(host_graph)
    ctx.synchronize()

    # The answer is the leading sub-block of the padded matrix.
    for i in range(num_candidates):
        for j in range(num_candidates):
            result.data[unsafe_offset=i * num_candidates + j] = host_ptr[
                unsafe_offset=i * padded + j
            ]

    return result^


# endregion GPU Driver

# region Command Line


def run_tiled_backend[
    family: def[tile_size: Int](
        PreferenceMatrix
    ) raises thin -> StrongestPathsMatrix,
    *sizes: Int,
](
    label: String,
    tile_size: Int,
    preferences: PreferenceMatrix,
    warmup: Int,
    repeat: Int,
    num_candidates: Int,
    mut baseline: StrongestPathsMatrix,
    mut has_baseline: Bool,
) raises -> Tuple[Int, List[Int]]:
    """Lowers a runtime tile size onto the set this family instantiates."""
    comptime for tile in sizes:
        if tile_size == tile:
            return profile_and_report[family[tile]](
                label,
                preferences,
                warmup,
                repeat,
                num_candidates,
                baseline,
                has_baseline,
            )
    raise Error(String("Unsupported tile size: ", tile_size))


def validate_against_baseline(
    result: StrongestPathsMatrix, baseline: StrongestPathsMatrix
) -> Bool:
    """Check if two results match."""
    var n = result.num_candidates
    for i in range(n):
        for j in range(n):
            if result[i, j] != baseline[i, j]:
                return False
    return True


def parse_int_arg(
    args: Span[StaticString, ImmStaticOrigin], flag: String, default: Int
) raises -> Int:
    """Parse an integer command-line argument, raising if it is malformed."""
    for i in range(len(args)):
        if String(args[i]) != flag:
            continue
        if i + 1 >= len(args):
            raise Error(String(flag, " needs a value"))
        try:
            return Int(String(args[i + 1]))
        except:
            raise Error(String("Invalid value for ", flag, ": ", args[i + 1]))
    return default


def parse_text_arg(
    args: Span[StaticString, ImmStaticOrigin], flag: String, default: String
) raises -> String:
    """Parse a string argument, raising when its value is missing."""
    for i in range(len(args)):
        if String(args[i]) != flag:
            continue
        if i + 1 >= len(args):
            raise Error(String(flag, " needs a value"))
        return String(args[i + 1])
    return default


def selected_by(pattern: String, name: String) -> Bool:
    """Whether a backend name matches the selector, case-insensitively."""
    return pattern == "." or pattern.lower() in name.lower()


def has_flag(args: Span[StaticString, ImmStaticOrigin], flag: String) -> Bool:
    """Check if a flag exists in command-line arguments."""
    for i in range(len(args)):
        if String(args[i]) == flag:
            return True
    return False


def reject_unknown_flags(
    args: Span[StaticString, ImmStaticOrigin],
) raises:
    """Rejects unrecognized flags, so a typo cannot silently use defaults."""
    comptime valued = (
        "--num-candidates",
        "--num-voters",
        "--cpu-tile-size",
        "--gpu-tile-size",
        "--warmup",
        "--repeat",
        "--seed",
        "--filter",
        "-k",
    )
    comptime bare = ("--help", "-h")
    var i = 1
    while i < len(args):
        var argument = String(args[i])
        var matched = False

        comptime for slot in range(len(valued)):
            if not matched and argument == valued[slot]:
                matched = True
                i += 1

        comptime for slot in range(len(bare)):
            if argument == bare[slot]:
                matched = True

        if not matched:
            raise Error(String("Unknown option: ", argument))
        i += 1


def print_usage():
    """Print usage information."""
    var cpu_sizes = String("")
    comptime for i in range(len(ALLOWED_CPU_TILE_SIZES)):
        if i > 0:
            cpu_sizes += ", "
        cpu_sizes += String(ALLOWED_CPU_TILE_SIZES[i])
    var gpu_sizes = String("")
    comptime for i in range(len(ALLOWED_GPU_TILE_SIZES)):
        if i > 0:
            gpu_sizes += ", "
        gpu_sizes += String(ALLOWED_GPU_TILE_SIZES[i])

    print(
        """Usage: mojo scaling_elections.mojo [OPTIONS]

Options:
  --num-candidates N    Number of candidates (default: 128)
  --num-voters N        Number of voters (default: 2000)
                        Set to 0 for instant random preference matrix generation
  -k, --filter TEXT     Select backends whose name contains TEXT, case-insensitively
                        Names: Serial (Mojo), Tiled CPU (Mojo), Tiled CPU+SIMD (Mojo), Tiled GPU (Mojo)
  --cpu-tile-size N     CPU tile size: {} (default: {})
  --gpu-tile-size N     GPU tile size: {} (default: {})
  --warmup N            Number of warmup iterations (default: 1)
  --repeat N            Number of benchmark iterations (default: 1)
  --seed N              Seed for the preference generator (default: 42)
  --help, -h            Show this help message

Examples:
  pixi run mojo scaling_elections.mojo --num-candidates 256 --num-voters 4000
  pixi run mojo scaling_elections.mojo --num-candidates 4096 -k GPU
  pixi run mojo scaling_elections.mojo --num-candidates 16384 --num-voters 0 -k 'Tiled CPU'""".format(
            cpu_sizes,
            DEFAULT_CPU_TILE_SIZE,
            gpu_sizes,
            DEFAULT_GPU_TILE_SIZE,
        )
    )


def main():
    """Benchmarks the selected backends and reports the election."""
    var args = argv()

    # Check for help flag
    if has_flag(args, "--help") or has_flag(args, "-h"):
        print_usage()
        return
    var num_candidates = 128
    var num_voters = 2000
    var cpu_tile_size = DEFAULT_CPU_TILE_SIZE
    var gpu_tile_size = DEFAULT_GPU_TILE_SIZE
    var warmup = 1
    var repeat = 1
    var seed_value = 42
    var selector = String(".")
    try:
        reject_unknown_flags(args)
        num_candidates = parse_int_arg(args, "--num-candidates", num_candidates)
        num_voters = parse_int_arg(args, "--num-voters", num_voters)
        cpu_tile_size = parse_int_arg(args, "--cpu-tile-size", cpu_tile_size)
        gpu_tile_size = parse_int_arg(args, "--gpu-tile-size", gpu_tile_size)
        warmup = parse_int_arg(args, "--warmup", warmup)
        repeat = parse_int_arg(args, "--repeat", repeat)
        seed_value = parse_int_arg(args, "--seed", seed_value)
        selector = parse_text_arg(args, "--filter", selector)
        selector = parse_text_arg(args, "-k", selector)
        if cpu_tile_size not in ALLOWED_CPU_TILE_SIZES:
            raise Error("--cpu-tile-size must be 4, 8, 16, 32, 64, or 128")
        if gpu_tile_size not in ALLOWED_GPU_TILE_SIZES:
            raise Error("--gpu-tile-size must be 4, 8, 16, or 32")
        if num_candidates < 4:
            raise Error("--num-candidates must be at least 4")
        if num_voters < 0:
            raise Error("--num-voters cannot be negative")
        if warmup < 0:
            raise Error("--warmup cannot be negative")
        if repeat < 1:
            raise Error("--repeat must be at least 1")
    except error:
        print("Error:", error)
        print()
        print_usage()
        exit(2)

    print("Schulze Voting Algorithm (Mojo)")
    print()

    comptime serial_label = "Serial (Mojo)"
    comptime cpu_label = "Tiled CPU (Mojo)"
    comptime simd_label = "Tiled CPU+SIMD (Mojo)"
    comptime gpu_label = "Tiled GPU (Mojo)"
    var wants_gpu = selected_by(selector, gpu_label)
    if wants_gpu and not has_accelerator():
        print("✗ No GPU detected, so {} is skipped".format(gpu_label))
        print()
        wants_gpu = False

    print("Configuration:")
    var voters_str = "{} voters".format(
        num_voters
    ) if num_voters > 0 else String("random")
    print(
        "  Problem size: {} candidates × {}".format(num_candidates, voters_str)
    )
    print("  CPU tile: {} × {}".format(cpu_tile_size, cpu_tile_size))
    print("  GPU tile: {} × {}".format(gpu_tile_size, gpu_tile_size))
    print("  Warmup: {}, Repeat: {}".format(warmup, repeat))
    print()

    print("Generating preferences...")
    seed(seed_value)
    var preferences = generate_random_preferences(num_candidates, num_voters)

    # Benchmarking section
    print()
    print("Benchmarking")
    print()

    # The first backend that succeeds becomes the baseline the rest validate against.
    var baseline = StrongestPathsMatrix(0)
    var has_baseline = False
    var winner = 0
    var ranking = List[Int]()
    var has_winner = False

    if selected_by(selector, serial_label):
        try:
            var outcome = profile_and_report[compute_strongest_paths_serial](
                serial_label,
                preferences,
                warmup,
                repeat,
                num_candidates,
                baseline,
                has_baseline,
            )
            if not has_winner:
                winner = outcome[0]
                ranking = outcome[1].copy()
                has_winner = True
        except e:
            print("  ✗ {} failed: {}".format(serial_label, e))
            print()

    if selected_by(selector, cpu_label):
        try:
            var outcome = run_tiled_backend[
                compute_strongest_paths_tiled_cpu, 4, 8, 16, 32, 64, 128
            ](
                cpu_label,
                cpu_tile_size,
                preferences,
                warmup,
                repeat,
                num_candidates,
                baseline,
                has_baseline,
            )
            if not has_winner:
                winner = outcome[0]
                ranking = outcome[1].copy()
                has_winner = True
        except e:
            print("  ✗ {} failed: {}".format(cpu_label, e))
            print()

    if selected_by(selector, simd_label):
        try:
            var outcome = run_tiled_backend[
                compute_strongest_paths_tiled_cpu_simd, 4, 8, 16, 32, 64, 128
            ](
                simd_label,
                cpu_tile_size,
                preferences,
                warmup,
                repeat,
                num_candidates,
                baseline,
                has_baseline,
            )
            if not has_winner:
                winner = outcome[0]
                ranking = outcome[1].copy()
                has_winner = True
        except e:
            print("  ✗ {} failed: {}".format(simd_label, e))
            print()

    if wants_gpu:
        try:
            var outcome = run_tiled_backend[
                compute_strongest_paths_gpu, 4, 8, 16, 32
            ](
                gpu_label,
                gpu_tile_size,
                preferences,
                warmup,
                repeat,
                num_candidates,
                baseline,
                has_baseline,
            )
            if not has_winner:
                winner = outcome[0]
                ranking = outcome[1].copy()
                has_winner = True
        except e:
            print("  ✗ {} failed: {}".format(gpu_label, e))
            print()

    # Display election results
    print("Election Results")
    print()
    if not has_winner:
        print("  No implementation was run, so there is no ranking to report.")
        print()
        return

    print("  Winner: Candidate #{}".format(winner))
    var shown = String("")
    for i in range(min(5, len(ranking))):
        if i > 0:
            shown += ", "
        shown += "#{}".format(ranking[i])
    print("  Top {}:  {}".format(min(5, len(ranking)), shown))
    print()


# endregion Command Line
