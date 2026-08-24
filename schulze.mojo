"""
Schulze strongest paths, from a serial reference up to a three-phase tiled GPU sweep.

The Schulze method is a Condorcet system: the winner is whoever beats every rival along the
widest chain of pairwise wins, which is a max-min closure over the winning-votes graph and so a
Floyd-Warshall in the `(max, min)` semiring rather than `(min, +)`.

Every backend here computes the same closure and differs only in how it walks it. The serial
version is the parity oracle. The tiled CPU versions block the sweep into three dependency
phases so the working set fits cache, once scalar and once with the diagonal handled by SIMD
masking. The GPU version runs those same phases as three kernels per diagonal tile, mirroring
`scalingelections.cu`.
"""

from std.builtin.sort import sort
from std.gpu import block_idx, thread_idx
from std.math import iota
from std.memory import AddressSpace, stack_allocation, unsafe_memset_zero

from max.algorithm import parallelize
from max.gpu import barrier
from max.gpu.host import DeviceContext

from ballots import (
    PreferenceMatrix,
    StrongestPathsMatrix,
    winning_votes_graph,
)


# region Types

comptime TILE_SIZE = 32
"""The tile edge every backend compiles for, matching a warp; a tile wider than the electorate is zero-filled, not an error."""


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
    """Pairs a candidate index with their win count, ordered by descending score so a plain sort ranks winners first."""

    var index: Int
    var score: Int

    def __lt__(self, other: Self) -> Bool:
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


# endregion Types


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

                if global_i >= num_candidates or global_j >= num_candidates or global_k >= num_candidates:
                    continue

                # Skip diagonal elements
                if global_i == global_j or global_i == global_k or global_k == global_j:
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

                var min_val = min(a_vec, b_vec)

                # Lane `lane` covers candidate `c_col + bj + lane`; skip the three diagonals.
                var global_j = iota[DType.int32, simd_width]() + Int32(c_col + bj)
                var mask = (
                    global_j.ne(Int32(global_i))
                    & global_j.ne(Int32(global_k))
                    & min_val.gt(c_vec)
                    & SIMD[DType.bool, simd_width](fill=global_i != global_k)
                )
                c.unsafe_store[width=simd_width](c_base_idx, mask.select(min_val, c_vec))


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
                dest[unsafe_offset=i * tile_size + j] = source[unsafe_offset=row * num_candidates + col]
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
                dest[unsafe_offset=row * num_candidates + col] = source[unsafe_offset=i * tile_size + j]


@always_inline
def calculate_tile_bounds(tile_idx: Int, tile_size: Int, total_size: Int) -> Tuple[Int, Int, Int]:
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
    tile_size: Int = TILE_SIZE
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
    winning_votes_graph(preferences, strongest_paths.data, num_candidates)

    # Step 2: Tiled Floyd-Warshall computation
    var num_tiles = (num_candidates + tile_size - 1) // tile_size

    for k in range(num_tiles):
        var k_index = k
        var k_bounds = calculate_tile_bounds(k, tile_size, num_candidates)
        var k_start = k_bounds[0]

        # Dependent phase: process diagonal tile
        var diagonal_tile = stack_allocation[tile_size * tile_size, UInt32, alignment=64]()
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

            var c_tile = stack_allocation[tile_size * tile_size, UInt32, alignment=64]()
            var b_tile = stack_allocation[tile_size * tile_size, UInt32, alignment=64]()
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

            var c_tile = stack_allocation[tile_size * tile_size, UInt32, alignment=64]()
            var a_tile = stack_allocation[tile_size * tile_size, UInt32, alignment=64]()
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

            var c_tile = stack_allocation[tile_size * tile_size, UInt32, alignment=64]()
            var a_tile = stack_allocation[tile_size * tile_size, UInt32, alignment=64]()
            var b_tile = stack_allocation[tile_size * tile_size, UInt32, alignment=64]()
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
    tile_size: Int = TILE_SIZE
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
    # Largest power-of-two SIMD width dividing the tile, capped at 16 for AVX-512.
    comptime simd_width = (
        16 if tile_size % 16
        == 0 else 8 if tile_size % 8
        == 0 else 4 if tile_size % 4
        == 0 else 2 if tile_size % 2
        == 0 else 1
    )
    var num_candidates = preferences.num_candidates
    var strongest_paths = StrongestPathsMatrix(num_candidates)

    # Step 1: Initialize strongest paths
    winning_votes_graph(preferences, strongest_paths.data, num_candidates)

    # Step 2: SIMD-vectorized tiled Floyd-Warshall computation
    var num_tiles = (num_candidates + tile_size - 1) // tile_size

    for k in range(num_tiles):
        var k_index = k
        var k_bounds = calculate_tile_bounds(k, tile_size, num_candidates)
        var k_start = k_bounds[0]

        # Diagonal phase: uses diagonal-aware SIMD processor
        var diagonal_tile = stack_allocation[tile_size * tile_size, UInt32, alignment=64]()
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
            var c_tile_row = stack_allocation[tile_size * tile_size, UInt32, alignment=64]()
            var b_tile = stack_allocation[tile_size * tile_size, UInt32, alignment=64]()
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
            var c_tile_col = stack_allocation[tile_size * tile_size, UInt32, alignment=64]()
            var a_tile = stack_allocation[tile_size * tile_size, UInt32, alignment=64]()
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

            var c_tile = stack_allocation[tile_size * tile_size, UInt32, alignment=64]()
            var a_tile = stack_allocation[tile_size * tile_size, UInt32, alignment=64]()
            var b_tile = stack_allocation[tile_size * tile_size, UInt32, alignment=64]()
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
                process_tile_cpu_simd_independent[tile_size, simd_width](c_tile, a_tile, b_tile, tile_size)

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


# region GPU Kernels

comptime SharedUInt32Ptr = Pointer[UInt32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]


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
            var global_i = c_row + bi
            var global_j = c_col + bj

            # Diagonal avoidance using branchless bit operations
            var is_not_diagonal_c = UInt32(1) if global_i != global_j else UInt32(0)
            var is_not_diagonal_a = UInt32(1) if global_i != global_k else UInt32(0)
            var is_not_diagonal_b = UInt32(1) if global_k != global_j else UInt32(0)
            var is_bigger = UInt32(1) if smallest > c_val else UInt32(0)
            var will_replace = is_not_diagonal_c & is_not_diagonal_a & is_not_diagonal_b & is_bigger

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


def gpu_diagonal_kernel[tile_size: Int](graph: Pointer[UInt32, MutUntrackedOrigin], n_arg: Int32, k_arg: Int32):
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
    c_shared[unsafe_offset=bi * tile_size + bj] = graph[unsafe_offset=k * tile_size * n + k * tile_size + bi * n + bj]

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
    graph[unsafe_offset=k * tile_size * n + k * tile_size + bi * n + bj] = c_shared[unsafe_offset=bi * tile_size + bj]


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
    c_shared[unsafe_offset=bi * tile_size + bj] = graph[unsafe_offset=i * tile_size * n + k * tile_size + bi * n + bj]
    b_shared[unsafe_offset=bi * tile_size + bj] = graph[unsafe_offset=k * tile_size * n + k * tile_size + bi * n + bj]

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
    graph[unsafe_offset=i * tile_size * n + k * tile_size + bi * n + bj] = c_shared[unsafe_offset=bi * tile_size + bj]

    # Phase 2: Process column tile (k, i) using (k, k) and (k, i)
    # Load c[unsafe_offset=k,i] and a[unsafe_offset=k,k]
    c_shared[unsafe_offset=bi * tile_size + bj] = graph[unsafe_offset=k * tile_size * n + i * tile_size + bi * n + bj]
    a_shared[unsafe_offset=bi * tile_size + bj] = graph[unsafe_offset=k * tile_size * n + k * tile_size + bi * n + bj]

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
    graph[unsafe_offset=k * tile_size * n + i * tile_size + bi * n + bj] = c_shared[unsafe_offset=bi * tile_size + bj]


def gpu_independent_kernel[tile_size: Int](graph: Pointer[UInt32, MutUntrackedOrigin], n_arg: Int32, k_arg: Int32):
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
    c_shared[unsafe_offset=bi * tile_size + bj] = graph[unsafe_offset=i * tile_size * n + j * tile_size + bi * n + bj]
    a_shared[unsafe_offset=bi * tile_size + bj] = graph[unsafe_offset=i * tile_size * n + k * tile_size + bi * n + bj]
    b_shared[unsafe_offset=bi * tile_size + bj] = graph[unsafe_offset=k * tile_size * n + j * tile_size + bi * n + bj]

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
    graph[unsafe_offset=i * tile_size * n + j * tile_size + bi * n + bj] = c_shared[unsafe_offset=bi * tile_size + bj]


# endregion GPU Kernels


# region GPU Driver


def compute_strongest_paths_gpu[
    tile_size: Int = TILE_SIZE
](preferences: PreferenceMatrix) raises -> StrongestPathsMatrix:
    """
    Pure Mojo GPU implementation of Schulze strongest paths computation.

    Implements three-phase tiled Floyd-Warshall algorithm on GPU using native
    Mojo GPU kernels. Matches the CUDA implementation in scalingelections.cu.

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
    var host_graph = ctx.enqueue_create_host_buffer[DType.uint32](padded * padded)
    var device_graph = ctx.enqueue_create_buffer[DType.uint32](padded * padded)
    var host_ptr = host_graph.unsafe_ptr()
    unsafe_memset_zero(host_ptr, padded * padded)

    winning_votes_graph(preferences, host_ptr, padded)

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
            result.data[unsafe_offset=i * num_candidates + j] = host_ptr[unsafe_offset=i * padded + j]

    return result^


# endregion GPU Driver


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

    var scored_candidates = List[IndexedScore]()
    for i in range(num_candidates):
        scored_candidates.append(IndexedScore(i, wins[i]))

    # Sort by score (IndexedScore's __lt__ sorts in descending order)
    sort(scored_candidates)

    # Extract just the candidate indices
    var ranking = List[Int]()
    for i in range(len(scored_candidates)):
        ranking.append(scored_candidates[i].index)

    return (winner_idx, ranking^)


# endregion Results
