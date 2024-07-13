from typing import Sequence
import random

import numpy as np
import cupy as cp
from numba import njit, prange, get_num_threads

tile_size = 32
cuda_tile_size = 16


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
    count_candidates = max(max(ranking) for ranking in voter_rankings) + 1

    # Initialize the preference matrix
    preferences = np.zeros((count_candidates, count_candidates), dtype=np.uint32)

    # Process each voter's ranking
    for ranking in voter_rankings:
        if len(ranking) == count_candidates:
            for i, preferred in enumerate(ranking):
                for opponent in ranking[i + 1 :]:
                    preferences[preferred, opponent] += 1
        else:
            # Mark the ranked candidates
            ranked_candidates = set(ranking)

            # Update preferences based on the ranking
            for i, preferred in enumerate(ranking):
                for opponent in ranking[i + 1 :]:
                    preferences[preferred, opponent] += 1

            # Generate random ballots for non-ranked candidates
            non_ranked_candidates = [
                c for c in range(count_candidates) if c not in ranked_candidates
            ]
            random.shuffle(non_ranked_candidates)

            # Update preferences with random ballots
            for i, preferred in enumerate(non_ranked_candidates):
                for opponent in non_ranked_candidates[i + 1 :]:
                    preferences[preferred, opponent] += 1

    return preferences


@njit
def compute_strongest_paths(preferences: np.ndarray) -> np.ndarray:
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
    tile_size: int,
):
    """
    In-place computation of the widest path path using the Schulze method with tiling for better cache utilization.
    For input of size (n x n), would perform (n) iterations of quadratic complexity each.

    Time complexity: O(n^3), where n is the tile size.
    Space complexity: O(n^2), where n is the tile size.
    """

    for k in range(tile_size):
        for i in range(tile_size):
            for j in range(tile_size):
                if (
                    (c_row + i != c_col + j)
                    and (a_row + i != a_col + k)
                    and (b_row + k != b_col + j)
                ):
                    replacement = min(a[a_row + i, a_col + k], b[b_row + k, b_col + j])
                    if replacement > c[c_row + i, c_col + j]:
                        c[c_row + i, c_col + j] = replacement


@njit(parallel=True)
def compute_strongest_paths_numba(preferences: np.ndarray) -> np.ndarray:
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

        # Partially dependent phase
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


compute_strongest_paths_tile_cuda = """
__device__
void compute_strongest_paths_tile_cuda(
    uint32_t * c, uint32_t c_row, uint32_t c_col,
    uint32_t const * a, uint32_t a_row, uint32_t a_col,
    uint32_t const * b, uint32_t b_row, uint32_t b_col,
    uint32_t const num_candidates) {
        
    __shared__ uint32_t a_shared[cuda_tile_size][cuda_tile_size];
    __shared__ uint32_t b_shared[cuda_tile_size][cuda_tile_size];
    __shared__ uint32_t c_shared[cuda_tile_size][cuda_tile_size];
    
    uint32_t i = threadIdx.x; // Load the tiles into shared memory
    uint32_t j = threadIdx.y;
    a_shared[i][j] = a[(a_row + i) * num_candidates + a_col + j];
    b_shared[i][j] = b[(b_row + i) * num_candidates + b_col + j];
    c_shared[i][j] = c[(c_row + i) * num_candidates + c_col + j];
        
    __syncthreads();
    for (uint32_t k = 0; k < cuda_tile_size; k++) {
        if ((c_row + i != c_col + j) && (a_row + i != a_col + k) && (b_row + k != b_col + j))
            c_shared[i][j] = max(c_shared[i][j], min(a_shared[i][k], b_shared[k][j]));
        __syncthreads();
    }
    
    c[(c_row + i) * num_candidates + c_col + j] = c_shared[i][j];
}
""".replace(
    "cuda_tile_size", str(cuda_tile_size)
)


compute_strongest_paths_cuda_partials = (
    compute_strongest_paths_tile_cuda
    + """
__global__
void compute_strongest_paths_cuda_partials(
    uint32_t * strongest_paths, uint32_t num_candidates, uint32_t k) {
    
    uint32_t j = blockIdx.x;
    if (j == k) return;
    
    uint32_t k_start = k * cuda_tile_size;
    uint32_t j_start = j * cuda_tile_size;
    
    compute_strongest_paths_tile_cuda(
        strongest_paths, k_start, j_start,
        strongest_paths, k_start, k_start,
        strongest_paths, k_start, j_start,
        num_candidates
    );
)
""".replace(
        "cuda_tile_size", str(cuda_tile_size)
    )
)

compute_strongest_paths_cuda_independent = (
    compute_strongest_paths_tile_cuda
    + """
__global__
void compute_strongest_paths_cuda_independent(
    uint32_t * strongest_paths, uint32_t num_candidates, uint32_t k) {
    
    uint32_t i = blockIdx.x;
    if (i == k) return;
    
    uint32_t i_start = i * cuda_tile_size;
    uint32_t k_start = k * cuda_tile_size;
    
    compute_strongest_paths_tile_cuda(
        strongest_paths, i_start, k_start,
        strongest_paths, i_start, k_start,
        strongest_paths, k_start, k_start,
        num_candidates
    );
    
    for (uint32_t j = 0; j < num_candidates / cuda_tile_size; j++) {
        if (j == k) continue;
        uint32_t j_start = j * cuda_tile_size;
        
        compute_strongest_paths_tile_cuda(
            strongest_paths, i_start, j_start,
            strongest_paths, i_start, k_start,
            strongest_paths, k_start, j_start,
            num_candidates
        );
    }
}
""".replace(
        "cuda_tile_size", str(cuda_tile_size)
    )
)


compute_strongest_paths_cuda_partials_kernel = cp.RawKernel(
    compute_strongest_paths_cuda_partials,
    "compute_strongest_paths_cuda_partials",
)
compute_strongest_paths_cuda_independent_kernel = cp.RawKernel(
    compute_strongest_paths_cuda_independent,
    "compute_strongest_paths_cuda_independent",
)


def compute_strongest_paths_cuda(preferences: np.ndarray) -> np.ndarray:
    """
    Computes the widest path strengths using the Schulze method on Nvidia GPUs.

    Space complexity:
    Time complexity:
    """
    num_candidates = preferences.shape[0]
    tile_size = cuda_tile_size

    # Let's make sure we use unified memory for the GPU
    cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)

    # Initialize the strongest paths matrix
    # assert preferences.dtype == np.uint32, f"Wrong type: {preferences.dtype}"
    strongest_paths = cp.zeros((num_candidates, num_candidates), dtype=np.uint32)

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

        # Partially dependent phase
        compute_strongest_paths_cuda_partials_kernel(
            (tiles_count, 1),  # blocks grid
            (cuda_tile_size, cuda_tile_size),  # shape of each block
            (strongest_paths, num_candidates, k),  # function arguments
        )

        # Independent phase
        compute_strongest_paths_cuda_independent(
            (tiles_count, 1),  # blocks grid
            (cuda_tile_size, cuda_tile_size),  # shape of each block
            (strongest_paths, num_candidates, k),  # function arguments
        )

    return strongest_paths.get()


def get_winner_and_ranking(candidates: list, strongest_paths: np.ndarray):
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


# Benchmark and comparison code remains the same
if __name__ == "__main__":
    import time

    # Generate random voter rankings
    num_voters = 128
    num_candidates = 128
    print(
        f"Generating {num_voters:,} random voter rankings with {num_candidates:,} candidates"
    )

    # To simplify the benchmark, let's generate a simple square matrix
    # voter_rankings = [np.random.permutation(num_candidates) for _ in range(num_voters)]
    # preferences = build_pairwise_preferences(voter_rankings)
    preferences = np.random.randint(
        0, num_voters, (num_candidates, num_candidates)
    ).astype(np.uint32)
    print(f"Generated voter rankings, proceeding with {get_num_threads()} threads")

    # To avoid cold-start and aggregating JIT costs, let's run all functions on tiny inputs first
    sub_preferences = preferences[: num_candidates // 16, : num_candidates // 16]
    strongest_paths = compute_strongest_paths(sub_preferences)
    strongest_paths_numba = compute_strongest_paths_numba(sub_preferences)
    # strongest_paths_cuda = compute_strongest_paths_cuda(sub_preferences)
    assert np.array_equal(strongest_paths, strongest_paths_numba), "Results differ"
    # assert np.array_equal(strongest_paths, strongest_paths_cuda), "Results differ"

    # Serial code:
    start_time = time.time()
    strongest_paths = compute_strongest_paths(preferences)
    elapsed_time = time.time() - start_time
    throughput = num_candidates**3 / elapsed_time
    print(f"Serial: {elapsed_time:.4f} seconds, {throughput:,.2f} candidates^3/sec")

    # Parallel CPU code:
    start_time = time.time()
    strongest_paths_numba = compute_strongest_paths_numba(preferences)
    elapsed_time = time.time() - start_time
    throughput = num_candidates**3 / elapsed_time
    print(f"Parallel: {elapsed_time:.4f} seconds, {throughput:,.2f} candidates^3/sec")
    assert np.array_equal(strongest_paths, strongest_paths_numba), "Results differ"

    # Parallel GPU code:
    # start_time = time.time()
    # strongest_paths_cuda = compute_strongest_paths_cuda(preferences)
    # elapsed_time = time.time() - start_time
    # throughput = num_candidates**3 / elapsed_time
    # print(f"GPU: {elapsed_time:.4f} seconds, {throughput:,.2f} candidates^3/sec")
    # assert np.array_equal(strongest_paths, strongest_paths_cuda), "Results differ"

    # Determine the winner and ranking for the final method (they should be the same for all methods)
    candidates = list(range(preferences.shape[0]))
    winner, ranking = get_winner_and_ranking(candidates, strongest_paths)

    # Print the results
    print(f"Winner is {winner}")
    print(f"Ranked {len(ranking)} candidates")
