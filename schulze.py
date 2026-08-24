"""Widest-path recurrences behind the Schulze method and Split Cycle.

Both methods close the pairwise graph under a max-min recurrence; they differ only in the
edge weights fed to it, winning votes for Schulze and positive margins for Split Cycle.
"""

import warnings

import numpy as np
from numba import njit, prange

from ballots import positive_margins

# Suppress Numba TBB threading layer warnings
warnings.filterwarnings("ignore", message=".*TBB threading layer.*")


TILE_SIZE = 32
"""The tile edge every backend is compiled for, matching `SCALING_ELECTIONS_TILE` in `types.cuh`."""


@njit
def compute_strongest_paths_numba_serial(preferences: np.ndarray) -> np.ndarray:
    """
    Computes the widest path strengths using the Schulze method.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^3), where n is the number of candidates.
    """
    num_candidates = preferences.shape[0]

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
    tile_size: int = TILE_SIZE,
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
    tile_size: int = TILE_SIZE,
) -> np.ndarray:
    """
    Computes the widest path strengths using the Schulze method with tiling for better cache utilization.
    This implementation not only parallelizes the outer loop but also tiles the computation, to maximize
    the utilization of CPU caches.

    Space complexity:
    Time complexity:
    """
    num_candidates = preferences.shape[0]

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


def split_cycle_winners(preferences: np.ndarray, margin_paths: np.ndarray) -> list[int]:
    """
    Determines the Split Cycle winners, which are the candidates nobody defeats.

    Holliday and Pacuit's Lemma 3.17: `a` defeats `b` when the margin of `a` over `b` is
    positive and exceeds the strength of the widest path from `b` back to `a`. Unlike
    Schulze, which this repository runs on winning votes, Split Cycle is defined on margins.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^2), where n is the number of candidates.
    """
    margins = positive_margins(preferences)
    defeats = (margins > 0) & (margins > margin_paths.astype(np.int64).T)
    return [i for i in range(preferences.shape[0]) if not defeats[:, i].any()]


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
