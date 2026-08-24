"""Pairwise preference matrices, built from voter ballots or drawn at random.

Every downstream method in this repository consumes the same square matrix, where cell
(i, j) counts the voters preferring candidate i to candidate j.
"""

from collections.abc import Sequence

import numpy as np
from numba import njit


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

    preferences = np.zeros((count_candidates, count_candidates), dtype=np.uint32)

    for ranking in voter_rankings:
        # A ballot that omits candidates ranks every one of them last, in index order.
        if len(ranking) != count_candidates:
            unranked = np.ones(count_candidates, dtype=bool)
            unranked[ranking] = False
            ranking = np.append(ranking, np.nonzero(unranked)[0])
        populate_preferences_from_ranking(preferences, ranking)

    return preferences


def generate_preferences(num_candidates: int, num_voters: int, generator: np.random.Generator) -> np.ndarray:
    """
    Draws a preference matrix for a synthetic election of the requested shape.

    Zero voters asks for the counts themselves to be random, which keeps the largest benchmarks
    from materializing ballots nobody reads.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(m * n^2), where n is the number of candidates and m is the number of voters.
    """
    if num_voters == 0:
        return generator.integers(0, num_candidates, (num_candidates, num_candidates), dtype=np.uint32)
    voter_rankings = [generator.permutation(num_candidates) for _ in range(num_voters)]
    return build_pairwise_preferences(voter_rankings)


def positive_margins(preferences: np.ndarray) -> np.ndarray:
    """
    Rewrites pairwise counts so the strongest-paths kernel closes over margins.

    The kernel keeps `preferences[i, j]` when it exceeds `preferences[j, i]` and zeroes it
    otherwise, so feeding it the clipped margin makes it compute the widest paths of the
    positive-margin graph without any change to the kernel itself.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^2), where n is the number of candidates.
    """
    signed = preferences.astype(np.int64)
    return np.maximum(signed - signed.T, 0).astype(np.uint32)
