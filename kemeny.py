"""Exact Kemeny-Young consensus ranking over a pairwise preference matrix.

The optimum is found by subset dynamic programming, so the cost is exponential in the
number of candidates rather than approximate.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def kemeny_subset_sums(preferences: np.ndarray, low_bits: int):
    """
    Tabulates, for each candidate, the votes it loses to every subset of the others.

    One table would be `n * 2^n` wide. Splitting the subset into a low and a high half makes
    two of `n * 2^(n/2)`, small enough to stay in cache while the score table streams past.

    Space complexity: O(n * 2^(n/2)), where n is the number of candidates.
    Time complexity: O(n * 2^(n/2)), where n is the number of candidates.
    """
    num_candidates = preferences.shape[0]
    high_bits = num_candidates - low_bits
    low = np.zeros((num_candidates, 1 << low_bits), dtype=np.int64)
    high = np.zeros((num_candidates, 1 << high_bits), dtype=np.int64)
    for candidate in range(num_candidates):
        for offset in range(low_bits):
            bit = 1 << offset
            votes = preferences[candidate, offset]
            for subset in range(bit, 1 << low_bits):
                if subset & bit:
                    low[candidate, subset] = low[candidate, subset ^ bit] + votes
        for offset in range(high_bits):
            bit = 1 << offset
            votes = preferences[candidate, low_bits + offset]
            for subset in range(bit, 1 << high_bits):
                if subset & bit:
                    high[candidate, subset] = high[candidate, subset ^ bit] + votes
    return low, high


@njit(cache=True)
def kemeny_votes_against(low, high, low_bits: int, candidate: int, subset: int) -> int:
    """Votes that preferred this candidate to every member of the subset."""
    return low[candidate, subset & ((1 << low_bits) - 1)] + high[candidate, subset >> low_bits]


@njit(cache=True)
def kemeny_costs(preferences: np.ndarray, low_bits: int, low, high):
    """
    Computes the least disagreement achievable for every subset of candidates.

    Entry `subset` is the score of the best ordering of those candidates in the leading
    seats, counting only the pairs inside it. Clearing a bit only ever lowers the index, so
    plain increasing order is already a valid topological order.

    Space complexity: O(2^n), where n is the number of candidates.
    Time complexity: O(n * 2^n), where n is the number of candidates.
    """
    num_candidates = preferences.shape[0]
    states = 1 << num_candidates
    costs = np.empty(states, dtype=np.int64)
    costs[0] = 0
    for subset in range(1, states):
        best = np.int64(np.iinfo(np.int64).max)
        for candidate in range(num_candidates):
            bit = 1 << candidate
            if not subset & bit:
                continue
            # Seating this candidate last within the subset costs the votes that preferred
            # it to each of the others.
            rest = subset ^ bit
            score = costs[rest] + kemeny_votes_against(low, high, low_bits, candidate, rest)
            if score < best:
                best = score
        costs[subset] = best
    return costs


def kemeny_ranking(preferences: np.ndarray) -> tuple[list[int], int]:
    """
    Determines the exact Kemeny-Young consensus ranking and its disagreement score.

    The ranking minimises the summed Kendall-tau distance to the ballots, so no ordering
    disagrees with the electorate less. This is the exact optimum rather than an
    approximation, at O(n * 2^n) time against O(2^n) memory.

    Space complexity: O(2^n), where n is the number of candidates.
    Time complexity: O(n * 2^n), where n is the number of candidates.
    """
    num_candidates = preferences.shape[0]
    counts = preferences.astype(np.int64)
    # The worst ordering pays the larger side of every pair. The C++ and Mojo ports carry a
    # 32-bit score, so refuse here too rather than disagree with them.
    worst_case = int(np.triu(np.maximum(counts, counts.T), 1).sum())
    if worst_case > np.iinfo(np.uint32).max:
        raise ValueError("Ballot counts exceed what a 32-bit Kemeny score can hold")
    low_bits = num_candidates // 2
    low, high = kemeny_subset_sums(counts, low_bits)
    costs = kemeny_costs(counts, low_bits, low, high)

    # Walk the choices back out, which recovers the ranking from its last seat upwards.
    ranking = []
    subset = (1 << num_candidates) - 1
    while subset:
        for candidate in range(num_candidates):
            bit = 1 << candidate
            if not subset & bit:
                continue
            rest = subset ^ bit
            if costs[subset] != costs[rest] + kemeny_votes_against(low, high, low_bits, candidate, rest):
                continue
            ranking.append(candidate)
            subset = rest
            break
    ranking.reverse()
    return ranking, int(costs[(1 << num_candidates) - 1])
