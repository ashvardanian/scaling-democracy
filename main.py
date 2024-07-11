from typing import Sequence
import random

import numpy as np
from numba import njit, prange


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
    preferences = np.zeros((count_candidates, count_candidates), dtype=np.uint64)

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


def compute_strongest_paths(preferences: np.ndarray) -> np.ndarray:
    """
    Computes the widest path strengths using the Schulze method.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^3), where n is the number of candidates.
    """
    num_candidates = preferences.shape[0]

    # Initialize the strongest paths matrix
    strongest_paths = np.zeros((num_candidates, num_candidates), dtype=np.uint64)

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


@njit(parallel=True)
def compute_strongest_paths_numba(preferences: np.ndarray) -> np.ndarray:
    """
    Computes the widest path strengths using the Schulze method and NumBa's JIT compiler.
    This baseline implementation only parallelizes the outer loop.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^3), where n is the number of candidates.
    """
    num_candidates = preferences.shape[0]

    # Initialize the strongest paths matrix
    strongest_paths = np.zeros((num_candidates, num_candidates), dtype=np.uint64)

    # Step 1: Populate the strongest paths matrix based on direct comparisons
    for i in prange(num_candidates):
        for j in range(num_candidates):
            if i != j:
                if preferences[i, j] > preferences[j, i]:
                    strongest_paths[i, j] = preferences[i, j]
                else:
                    strongest_paths[i, j] = 0

    # Step 2: Compute the strongest paths using Floyd-Warshall-like algorithm
    for i in prange(num_candidates):
        for j in range(num_candidates):
            if i != j:
                for k in range(num_candidates):
                    if i != k and j != k:
                        strongest_paths[j, k] = max(
                            strongest_paths[j, k],
                            min(strongest_paths[j, i], strongest_paths[i, k]),
                        )

    return strongest_paths


@njit(parallel=True)
def compute_strongest_paths_numba_tiled(
    preferences: np.ndarray, tile_size: int = 16
) -> np.ndarray:
    """
    Computes the widest path strengths using the Schulze method with tiling for better cache utilization.
    This implementation not only parallelizes the outer loop but also tiles the computation, to maximize
    the utilization of CPU caches.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^3), where n is the number of candidates.
    """
    num_candidates = preferences.shape[0]

    # Initialize the strongest paths matrix
    strongest_paths = np.zeros((num_candidates, num_candidates), dtype=np.int64)

    # Step 1: Populate the strongest paths matrix based on direct comparisons
    for i in prange(0, num_candidates, tile_size):
        for j in range(0, num_candidates, tile_size):
            for ii in range(i, min(i + tile_size, num_candidates)):
                for jj in range(j, min(j + tile_size, num_candidates)):
                    if ii != jj:
                        if preferences[ii, jj] > preferences[jj, ii]:
                            strongest_paths[ii, jj] = preferences[ii, jj]
                        else:
                            strongest_paths[ii, jj] = 0

    # Step 2: Compute the strongest paths using Floyd-Warshall-like algorithm with tiling
    for i in prange(0, num_candidates, tile_size):
        for j in range(0, num_candidates, tile_size):
            for k in range(0, num_candidates, tile_size):
                for ii in range(i, min(i + tile_size, num_candidates)):
                    for jj in range(j, min(j + tile_size, num_candidates)):
                        for kk in range(k, min(k + tile_size, num_candidates)):
                            if ii != kk and jj != kk and ii != jj:
                                strongest_paths[jj, kk] = max(
                                    strongest_paths[jj, kk],
                                    min(
                                        strongest_paths[jj, ii], strongest_paths[ii, kk]
                                    ),
                                )

    return strongest_paths


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


if __name__ == "__main__":

    # Example usage:
    voter_rankings = [
        np.array([0, 2, 1, 3]),
        np.array([1, 0, 3, 2]),
        np.array([2, 1, 0]),  # Incomplete ranking
        np.array([3, 2, 1, 0]),
        np.array([0, 1, 2, 3]),
        np.array([1, 2, 0]),  # Incomplete ranking
    ]

    # Let's instead generate some random voter rankings
    num_voters = 1000
    num_candidates = 4
    voter_rankings = [np.random.permutation(num_candidates) for _ in range(num_voters)]

    # Build the pairwise preference matrix
    preferences = build_pairwise_preferences(voter_rankings)

    # Compute the strongest paths using the Schulze method
    strongest_paths = compute_strongest_paths(preferences)

    # Determine the winner and ranking
    candidates = list(range(preferences.shape[0]))
    winner, ranking = get_winner_and_ranking(candidates, strongest_paths)

    # Print the results
    print("Pairwise preference matrix:")
    print(preferences)
    print("\nStrongest paths matrix:")
    print(strongest_paths)
    print("\nWinner:", winner)
    print("Ranking:", ranking)
