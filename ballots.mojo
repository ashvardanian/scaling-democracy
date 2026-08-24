"""
Ballot storage shared by the Schulze and Kemeny-Young solvers.

Both stages pass around one square `UInt32` matrix. Read as preferences, entry `(i, j)` counts
the voters who ranked candidate `i` above candidate `j`; read as strongest paths, it holds the
widest bottleneck from `i` to `j`. The two meanings share a layout, an allocation and an
accessor, so `VoteMatrix` is the storage and the aliases below name the reading.

`winning_votes_graph` turns the first into the seed of the second, keeping only the winning side
of each pairwise contest, which is where every Schulze backend starts.
"""

from std.memory import Layout, alloc, unsafe_memset_zero
from std.random import random_si64

from max.algorithm import parallelize

# region Matrix


@fieldwise_init
struct VoteMatrix(Movable):
    """Dense square matrix of pairwise vote counts, indexed by a pair of candidates."""

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


comptime PreferenceMatrix = VoteMatrix
"""The two names read differently at a call site and denote the same storage."""
comptime StrongestPathsMatrix = VoteMatrix

# endregion Matrix

# region Preferences


def populate_preferences_from_ranking(mut preferences: PreferenceMatrix, ranking: List[Int]):
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


def generate_random_preferences(num_candidates: Int, num_voters: Int) -> PreferenceMatrix:
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
                preferences[i, j] = UInt32(random_si64(0, Int64(num_candidates - 1)))

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


# endregion Preferences

# region Graph


def winning_votes_graph(
    preferences: PreferenceMatrix,
    graph: Pointer[UInt32, MutUntrackedOrigin],
    row_stride: Int,
):
    """
    Seeds a strongest-paths graph with the winning side of each pairwise contest.

    Entry `(i, j)` keeps the votes for `i` over `j` when that side won and zero otherwise, which
    is the direct-comparison step every Schulze backend runs before its Floyd-Warshall sweep.
    Only the leading `num_candidates` columns of each row are written, so a padded destination
    keeps whatever its tail already held.

    Args:
        preferences: Input preference matrix.
        graph: Destination graph, at least `num_candidates` rows of `row_stride` entries.
        row_stride: Distance in entries between consecutive rows of the destination.
    """
    var num_candidates = preferences.num_candidates

    @parameter
    def init_paths(i: Int):
        for j in range(num_candidates):
            if i != j:
                var pref_ij = preferences[i, j]
                var pref_ji = preferences[j, i]
                if pref_ij > pref_ji:
                    graph[unsafe_offset=i * row_stride + j] = pref_ij
                else:
                    graph[unsafe_offset=i * row_stride + j] = 0

    parallelize[init_paths](num_candidates)


# endregion Graph
