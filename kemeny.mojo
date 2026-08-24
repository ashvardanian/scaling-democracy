"""
Exact Kemeny-Young consensus ranking over a pairwise preference matrix.

Schulze answers who wins; Kemeny-Young answers what the whole ordering should be, choosing the
ranking that disagrees with the fewest ballot pairs. That optimum is NP-hard to reach by search,
so this is the Held-Karp subset dynamic program instead: `O(n * 2^n)` time against `O(2^n)`
memory, exact rather than approximate, and practical to roughly two dozen candidates.
"""

from ballots import PreferenceMatrix

# region Kemeny


comptime KemenyScore = UInt32
"""Pairwise disagreements summed, bounded by `voters * n * (n - 1) / 2`, which 32 bits hold."""


def accumulate_subset_sums(
    mut table: List[KemenyScore],
    base: Int,
    states: Int,
    preferences: PreferenceMatrix,
    candidate: Int,
    first_opponent: Int,
):
    """Folds one candidate's votes into every subset that contains each opponent."""
    var offset = 0
    var bit = 1
    while bit < states:
        var votes = preferences[candidate, first_opponent + offset]
        for subset in range(bit, states):
            if subset & bit:
                table[base + subset] = table[base + (subset ^ bit)] + votes
        offset += 1
        bit <<= 1


struct KemenySums(Movable):
    """Votes each candidate loses to every subset of the others.

    One table would be `n * 2^n` wide. Splitting the subset into a low and a high half makes
    two of `n * 2^(n/2)`, small enough to stay in cache while the score table streams past.
    """

    var low: List[KemenyScore]
    var high: List[KemenyScore]
    var low_bits: Int
    var low_states: Int
    var high_states: Int

    def __init__(out self, preferences: PreferenceMatrix):
        var num_candidates = preferences.num_candidates
        self.low_bits = num_candidates // 2
        self.low_states = 1 << self.low_bits
        self.high_states = 1 << (num_candidates - self.low_bits)
        self.low = List[KemenyScore]()
        self.low.resize(num_candidates * self.low_states, 0)
        self.high = List[KemenyScore]()
        self.high.resize(num_candidates * self.high_states, 0)

        for candidate in range(num_candidates):
            accumulate_subset_sums(
                self.low,
                candidate * self.low_states,
                self.low_states,
                preferences,
                candidate,
                0,
            )
            accumulate_subset_sums(
                self.high,
                candidate * self.high_states,
                self.high_states,
                preferences,
                candidate,
                self.low_bits,
            )

    def against(self, candidate: Int, subset: Int) -> KemenyScore:
        """Votes that preferred this candidate to every member of the subset."""
        return (
            self.low[candidate * self.low_states + (subset & (self.low_states - 1))]
            + self.high[candidate * self.high_states + (subset >> self.low_bits)]
        )


def kemeny_ranking(
    preferences: PreferenceMatrix,
) raises -> Tuple[List[Int], Int]:
    """
    Determines the exact Kemeny-Young consensus ranking and its disagreement score.

    The ranking minimises the summed Kendall-tau distance to the ballots, so no ordering
    disagrees with the electorate less. This is the exact optimum rather than an
    approximation, at `O(n * 2^n)` time against `O(2^n)` memory.

    Args:
        preferences: Input preference matrix.

    Returns:
        Tuple of (ranked_candidate_ids, disagreement score).
    """
    var num_candidates = preferences.num_candidates

    # The worst ordering pays the larger side of every pair, so that sum is what the score
    # type has to hold. Wrapping here would answer confidently and wrongly.
    var worst_case = UInt64(0)
    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            worst_case += UInt64(max(preferences[i, j], preferences[j, i]))
    if worst_case > UInt64(KemenyScore.MAX):
        raise Error("Ballot counts exceed what a 32-bit Kemeny score can hold")

    var sums = KemenySums(preferences)
    var states = 1 << num_candidates

    # Entry `subset` is the least disagreement achievable seating those candidates in the
    # leading places, counting only the pairs inside it. Clearing a bit only ever lowers the
    # index, so plain increasing order is already a valid topological order.
    var costs = List[KemenyScore]()
    costs.resize(states, 0)
    for subset in range(1, states):
        var best = KemenyScore.MAX
        for candidate in range(num_candidates):
            var bit = 1 << candidate
            if not subset & bit:
                continue
            # Seating this candidate last within the subset costs the votes that preferred
            # it to each of the others.
            var rest = subset ^ bit
            var score = costs[rest] + sums.against(candidate, rest)
            if score < best:
                best = score
        costs[subset] = best

    # Walk the choices back out, which recovers the ranking from its last place upwards.
    var reversed_ranking = List[Int]()
    var subset = states - 1
    while subset:
        for candidate in range(num_candidates):
            var bit = 1 << candidate
            if not subset & bit:
                continue
            var rest = subset ^ bit
            if costs[subset] != costs[rest] + sums.against(candidate, rest):
                continue
            reversed_ranking.append(candidate)
            subset = rest
            break

    var ranking = List[Int]()
    for i in range(len(reversed_ranking)):
        ranking.append(reversed_ranking[len(reversed_ranking) - 1 - i])
    return (ranking^, Int(costs[states - 1]))


# endregion Kemeny
