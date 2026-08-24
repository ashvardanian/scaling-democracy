"""Cross-validates the Python, CUDA and Mojo implementations of the C2 voting rules.

Every expected answer comes from a brute-force oracle written straight from the definition of
the rule, never from the code under test, so a bug shared by all three ports cannot hide behind
their agreement. That bounds the sizes: the oracles enumerate permutations, so correctness is
checked at a handful of candidates and the backends are checked for agreement beyond that.

Run with `uv run --no-sync pytest test.py`.
"""

import itertools
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import scalingelections_cuda as cuda

import ballots
import kemeny
import schulze

sys.path.insert(0, str(Path(__file__).resolve().parent / "build"))

CUDA_BACKENDS = ("cpu_openmp", "gpu_serial", "gpu_hopper")

# Ten ballots over Python, Rust, Go and Java, plus the voter who joins them ranking Java first.
PYTHON, RUST, GO, JAVA = range(4)
CANDIDATE_NAMES = ("Python", "Rust", "Go", "Java")
INVOLVEMENT_BALLOTS = [[0, 3, 1, 2]] + [[0, 3, 2, 1]] * 2 + [[1, 0, 2, 3]] * 2 + [[2, 3, 1, 0]] * 4 + [[3, 0, 1, 2]]
JAVA_FIRST_BALLOT = [3, 0, 2, 1]


# region Oracles


def oracle_pairwise_preferences(rankings: Sequence[Sequence[int]]) -> np.ndarray:
    """Counts the ballots placing each candidate ahead of each other one, by definition."""
    num_candidates = max(max(ranking) for ranking in rankings) + 1
    preferences = np.zeros((num_candidates, num_candidates), dtype=np.uint32)
    for ranking in rankings:
        for position, preferred in enumerate(ranking):
            for opponent in ranking[position + 1 :]:
                preferences[preferred, opponent] += 1
    return preferences


def oracle_strongest_paths(preferences: np.ndarray) -> np.ndarray:
    """Widest paths over winning votes, by the textbook triple loop."""
    num_candidates = len(preferences)
    strengths = [[0] * num_candidates for _ in range(num_candidates)]
    for source in range(num_candidates):
        for target in range(num_candidates):
            if source != target and preferences[source][target] > preferences[target][source]:
                strengths[source][target] = int(preferences[source][target])
    for pivot in range(num_candidates):
        for source in range(num_candidates):
            for target in range(num_candidates):
                if pivot in (source, target) or source == target:
                    continue
                through_pivot = min(strengths[source][pivot], strengths[pivot][target])
                strengths[source][target] = max(strengths[source][target], through_pivot)
    return np.array(strengths, dtype=np.uint32).reshape(num_candidates, num_candidates)


def oracle_schulze_wins(preferences: np.ndarray) -> list[int]:
    """How many rivals each candidate beats on strongest paths, which is what Schulze ranks by."""
    strengths = oracle_strongest_paths(preferences)
    num_candidates = len(preferences)
    return [
        sum(1 for target in range(num_candidates) if strengths[source][target] > strengths[target][source])
        for source in range(num_candidates)
    ]


def kendall_score(preferences: np.ndarray, ranking: Sequence[int]) -> int:
    """Votes contradicting the ranking, summed over every pair it orders."""
    return sum(
        int(preferences[ranking[later]][ranking[earlier]])
        for earlier in range(len(ranking))
        for later in range(earlier + 1, len(ranking))
    )


def oracle_kemeny(preferences: np.ndarray) -> tuple[tuple[int, ...], int]:
    """The exact consensus ranking, by scoring every ordering there is."""
    orderings = itertools.permutations(range(len(preferences)))
    return min(((order, kendall_score(preferences, order)) for order in orderings), key=lambda scored: scored[1])


def oracle_split_cycle_winners(preferences: np.ndarray) -> list[int]:
    """
    The undefeated candidates under Holliday and Pacuit's Definition 3.3.

    A candidate defeats another when its margin is positive and strictly exceeds the smallest
    margin of every majority cycle carrying that edge, with the cycles enumerated rather than
    summarised by a widest path.
    """
    num_candidates = len(preferences)
    margins = [
        [int(preferences[source][target]) - int(preferences[target][source]) for target in range(num_candidates)]
        for source in range(num_candidates)
    ]
    defeated = set()
    for winner in range(num_candidates):
        for loser in range(num_candidates):
            if winner == loser or margins[winner][loser] <= 0:
                continue
            intermediates = [other for other in range(num_candidates) if other not in (winner, loser)]
            defeats = True
            for length in range(len(intermediates) + 1):
                for middle in itertools.permutations(intermediates, length):
                    cycle = (winner, loser, *middle, winner)
                    edges = [margins[cycle[step]][cycle[step + 1]] for step in range(len(cycle) - 1)]
                    if min(edges) <= 0:
                        continue  # Not a majority cycle, so it constrains nothing.
                    if margins[winner][loser] <= min(edges):
                        defeats = False
                        break
                if not defeats:
                    break
            if defeats:
                defeated.add(loser)
    return [candidate for candidate in range(num_candidates) if candidate not in defeated]


def tied_preferences(num_candidates: int, votes: int = 4) -> np.ndarray:
    """A profile where every pair is an exact tie, so no edge survives the winning-votes filter."""
    preferences = np.full((num_candidates, num_candidates), votes, dtype=np.uint32)
    np.fill_diagonal(preferences, 0)
    return preferences


def random_preferences(num_candidates: int, num_voters: int, seed: int) -> np.ndarray:
    """A reproducible profile drawn from random ballots."""
    return ballots.generate_preferences(num_candidates, num_voters, np.random.default_rng(seed))


BALLOT_SETS = {
    "involvement_10": INVOLVEMENT_BALLOTS,
    "involvement_11": [*INVOLVEMENT_BALLOTS, JAVA_FIRST_BALLOT],
    "condorcet_cycle": [[0, 1, 2]] * 3 + [[1, 2, 0]] * 3 + [[2, 0, 1]] * 3,
    "unanimous_four": [[0, 1, 2, 3]] * 5,
    "mirrored_four": [[0, 1, 2, 3]] * 3 + [[3, 2, 1, 0]] * 3,
}

PROFILES = {name: oracle_pairwise_preferences(rankings) for name, rankings in BALLOT_SETS.items()} | {
    "single_candidate": np.zeros((1, 1), dtype=np.uint32),
    "two_tied": tied_preferences(2),
    "all_tied_five": tied_preferences(5),
    "random_5": random_preferences(5, 9, 11),
    "random_6": random_preferences(6, 12, 12),
    "random_7": random_preferences(7, 15, 13),
    "random_8": random_preferences(8, 21, 14),
}

PROFILE_CASES = pytest.mark.parametrize("preferences", list(PROFILES.values()), ids=list(PROFILES))


@pytest.fixture(scope="session")
def mojo():
    """The Mojo extension from `build/`, skipping the case when it has not been compiled."""
    return pytest.importorskip("scalingelections_mojo", reason="Build it with `pixi run build-python`")


@pytest.fixture(scope="session")
def gpu_ready() -> bool:
    """Whether a device backend can actually run, as opposed to a CPU-only build or an empty box."""
    try:
        cuda.compute_strongest_paths(np.zeros((2, 2), dtype=np.uint32), backend="gpu_serial")
    except RuntimeError:
        return False
    return True


@pytest.fixture(params=CUDA_BACKENDS)
def cuda_backend(request, gpu_ready: bool) -> str:
    """Each backend the C++/CUDA extension exposes, skipping the device ones without a device."""
    if request.param.startswith("gpu") and not gpu_ready:
        pytest.skip("No CUDA device visible")
    return request.param


@pytest.mark.parametrize("rankings", list(BALLOT_SETS.values()), ids=list(BALLOT_SETS))
def test_pairwise_preferences_match_oracle(rankings: Sequence[Sequence[int]]):
    built = ballots.build_pairwise_preferences([np.asarray(ranking, dtype=np.int64) for ranking in rankings])
    assert np.array_equal(built, oracle_pairwise_preferences(rankings))


@PROFILE_CASES
def test_strongest_paths_match_oracle(preferences: np.ndarray):
    assert np.array_equal(
        schulze.compute_strongest_paths_numba_serial(preferences), oracle_strongest_paths(preferences)
    )


@PROFILE_CASES
def test_kemeny_matches_oracle(preferences: np.ndarray):
    ranking, score = kemeny.kemeny_ranking(preferences)
    assert sorted(ranking) == list(range(len(preferences))), "The ranking must seat every candidate once"
    assert kendall_score(preferences, ranking) == score, "The reported ranking must achieve the reported score"
    assert score == oracle_kemeny(preferences)[1]


@PROFILE_CASES
def test_split_cycle_matches_oracle(preferences: np.ndarray):
    margin_paths = schulze.compute_strongest_paths_numba_serial(ballots.positive_margins(preferences))
    assert schulze.split_cycle_winners(preferences, margin_paths) == oracle_split_cycle_winners(preferences)


# endregion Oracles


# region Cross Language


@PROFILE_CASES
def test_strongest_paths_agree_across_languages(preferences: np.ndarray, mojo):
    from_python = schulze.compute_strongest_paths_numba_serial(preferences)
    from_cuda = cuda.compute_strongest_paths(preferences)
    from_mojo = np.asarray(mojo.strongest_paths(preferences.tolist()), dtype=np.uint32)
    assert np.array_equal(from_cuda, from_python)
    assert np.array_equal(from_mojo, from_python)


@PROFILE_CASES
def test_kemeny_agrees_across_languages(preferences: np.ndarray, mojo):
    python_ranking, python_score = kemeny.kemeny_ranking(preferences)
    cuda_ranking, cuda_score = cuda.compute_kemeny_ranking(preferences)
    mojo_ranking, mojo_score = mojo.kemeny_consensus(preferences.tolist())
    assert kendall_score(preferences, cuda_ranking) == cuda_score, "CUDA must achieve the score it reports"
    assert kendall_score(preferences, mojo_ranking) == mojo_score, "Mojo must achieve the score it reports"
    assert (cuda_score, mojo_score) == (python_score, python_score)
    assert list(cuda_ranking) == python_ranking
    assert list(mojo_ranking) == python_ranking


@PROFILE_CASES
def test_split_cycle_agrees_across_languages(preferences: np.ndarray, mojo):
    margins = ballots.positive_margins(preferences)
    expected = oracle_split_cycle_winners(preferences)
    from_cuda = cuda.compute_strongest_paths(margins)
    from_mojo = np.asarray(mojo.strongest_paths(margins.tolist()), dtype=np.uint32)
    assert schulze.split_cycle_winners(preferences, from_cuda) == expected
    assert schulze.split_cycle_winners(preferences, from_mojo) == expected


# endregion Cross Language


# region Schulze Backends

BACKEND_SIZES = (4, 8, 32, 64)


@pytest.mark.parametrize("num_candidates", BACKEND_SIZES)
def test_cuda_backend_matches_numba_serial(num_candidates: int, cuda_backend: str):
    preferences = random_preferences(num_candidates, 2 * num_candidates, num_candidates)
    expected = schulze.compute_strongest_paths_numba_serial(preferences)
    assert np.array_equal(cuda.compute_strongest_paths(preferences, backend=cuda_backend), expected)


@pytest.mark.parametrize("num_candidates", BACKEND_SIZES)
def test_numba_parallel_matches_numba_serial(num_candidates: int):
    preferences = random_preferences(num_candidates, 2 * num_candidates, num_candidates)
    expected = schulze.compute_strongest_paths_numba_serial(preferences)
    assert np.array_equal(schulze.compute_strongest_paths_numba_parallel(preferences), expected)


@PROFILE_CASES
def test_cuda_backends_match_on_the_margin_graph(preferences: np.ndarray, cuda_backend: str):
    margins = ballots.positive_margins(preferences)
    expected = schulze.compute_strongest_paths_numba_serial(margins)
    assert np.array_equal(cuda.compute_strongest_paths(margins, backend=cuda_backend), expected)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="cpu_openmp"):
        cuda.compute_strongest_paths(tied_preferences(4), backend="gpu_blackwell")


# endregion Schulze Backends


# region Edge Cases


@pytest.mark.parametrize("num_candidates", (1, 2, 3, 8, schulze.TILE_SIZE - 1))
def test_sizes_below_the_tile_edge(num_candidates: int, cuda_backend: str):
    preferences = random_preferences(num_candidates, 2 * num_candidates + 1, num_candidates)
    expected = schulze.compute_strongest_paths_numba_serial(preferences)
    assert np.array_equal(cuda.compute_strongest_paths(preferences, backend=cuda_backend), expected)
    assert np.array_equal(schulze.compute_strongest_paths_numba_parallel(preferences), expected)


@pytest.mark.parametrize("num_candidates", (schulze.TILE_SIZE + 1, schulze.TILE_SIZE + 15, 2 * schulze.TILE_SIZE + 1))
def test_sizes_not_divisible_by_the_tile(num_candidates: int, cuda_backend: str):
    preferences = random_preferences(num_candidates, 2 * num_candidates + 1, num_candidates)
    expected = schulze.compute_strongest_paths_numba_serial(preferences)
    assert np.array_equal(cuda.compute_strongest_paths(preferences, backend=cuda_backend), expected)
    assert np.array_equal(schulze.compute_strongest_paths_numba_parallel(preferences), expected)


@pytest.mark.parametrize("num_candidates", (2, 5, schulze.TILE_SIZE + 1))
def test_tie_heavy_profiles_leave_no_edges(num_candidates: int, cuda_backend: str):
    preferences = tied_preferences(num_candidates)
    strengths = cuda.compute_strongest_paths(preferences, backend=cuda_backend)
    assert not strengths.any(), "A tie beats nobody, so no path can carry strength"
    assert np.array_equal(strengths, oracle_strongest_paths(preferences))


def test_tie_heavy_profiles_keep_every_candidate():
    preferences = PROFILES["mirrored_four"]
    margin_paths = schulze.compute_strongest_paths_numba_serial(ballots.positive_margins(preferences))
    assert schulze.split_cycle_winners(preferences, margin_paths) == list(range(len(preferences)))


def test_kemeny_score_guard_rejects_wider_than_thirty_two_bits(mojo):
    # Six pairs of just over two billion votes each overflow a 32-bit score.
    preferences = np.full((4, 4), 2**31, dtype=np.uint32)
    np.fill_diagonal(preferences, 0)
    with pytest.raises(ValueError, match="32-bit Kemeny score"):
        kemeny.kemeny_ranking(preferences)
    with pytest.raises(RuntimeError, match="32-bit Kemeny score"):
        cuda.compute_kemeny_ranking(preferences)
    with pytest.raises(Exception, match="32-bit Kemeny score"):
        mojo.kemeny_consensus(preferences.tolist())


def test_kemeny_score_guard_admits_the_widest_that_fits(mojo):
    # Three pairs summing to exactly `UINT32_MAX`, which is the last profile the score can hold.
    preferences = np.zeros((3, 3), dtype=np.uint32)
    preferences[0, 1] = preferences[1, 2] = preferences[0, 2] = np.iinfo(np.uint32).max // 3
    python_ranking, python_score = kemeny.kemeny_ranking(preferences)
    assert kendall_score(preferences, python_ranking) == python_score
    assert (python_ranking, python_score) == ([0, 1, 2], 0)
    assert tuple(cuda.compute_kemeny_ranking(preferences)) == (python_ranking, python_score)
    assert list(mojo.kemeny_consensus(preferences.tolist())) == [python_ranking, python_score]


# endregion Edge Cases


# region Paradoxes


def test_participation_paradox_makes_schulze_punish_its_own_winner():
    """Schulze violates positive involvement: a voter ranking Java first costs Java the election."""
    before, after = PROFILES["involvement_10"], PROFILES["involvement_11"]
    wins_before, wins_after = oracle_schulze_wins(before), oracle_schulze_wins(after)
    assert wins_before.count(max(wins_before)) == 1, "The winner has to be unique for the paradox to bite"
    assert wins_after.count(max(wins_after)) == 1
    assert wins_before.index(max(wins_before)) == JAVA, f"Ten voters elect {CANDIDATE_NAMES[JAVA]}"
    assert wins_after.index(max(wins_after)) == PYTHON, f"The Java-first ballot elects {CANDIDATE_NAMES[PYTHON]}"

    candidates = list(range(len(CANDIDATE_NAMES)))
    winner_before, _ = schulze.get_winner_and_ranking(candidates, schulze.compute_strongest_paths_numba_serial(before))
    winner_after, _ = schulze.get_winner_and_ranking(candidates, schulze.compute_strongest_paths_numba_serial(after))
    assert (winner_before, winner_after) == (JAVA, PYTHON)


def test_participation_paradox_leaves_split_cycle_unmoved():
    """Split Cycle satisfies positive involvement on the same profile: Java stays in the winner set."""
    before, after = PROFILES["involvement_10"], PROFILES["involvement_11"]
    winners = []
    for preferences in (before, after):
        margin_paths = schulze.compute_strongest_paths_numba_serial(ballots.positive_margins(preferences))
        computed = schulze.split_cycle_winners(preferences, margin_paths)
        assert computed == oracle_split_cycle_winners(preferences)
        winners.append(computed)
    assert winners[0] == [PYTHON, GO, JAVA]
    assert winners[1] == [PYTHON, JAVA]
    assert JAVA in winners[0] and JAVA in winners[1], "The extra Java-first ballot must not unseat Java"


# endregion Paradoxes


# region Differential


def random_profile(seed: int, num_candidates: int, num_voters: int, tied: bool = False) -> np.ndarray:
    """Tallies random ballots, so the matrix is always realizable rather than an arbitrary grid."""
    generator = np.random.default_rng(seed)
    preferences = np.zeros((num_candidates, num_candidates), dtype=np.uint32)
    for ballot in range(num_voters):
        # Replaying one canonical order for half the electorate manufactures the ties that
        # separate implementations agreeing on a score from implementations agreeing on a ranking.
        ranking = np.arange(num_candidates) if tied and ballot % 2 else generator.permutation(num_candidates)
        ballots.populate_preferences_from_ranking(preferences, ranking)
    return preferences


# Sizes straddling the compile-time tile, where a mishandled tail is invisible on round numbers.
TILE_BOUNDARY_SIZES = (1, 2, 31, 32, 33, 47, 63, 64, 65, 96, 97, 129)


@pytest.mark.parametrize("num_candidates", TILE_BOUNDARY_SIZES)
def test_backends_agree_across_tile_boundaries(num_candidates: int, mojo):
    """Every backend must match the serial baseline whether or not the tile divides the electorate."""
    preferences = random_profile(seed=7 + num_candidates, num_candidates=num_candidates, num_voters=25)
    expected = schulze.compute_strongest_paths_numba_serial(preferences)
    assert np.array_equal(schulze.compute_strongest_paths_numba_parallel(preferences), expected)
    for backend in CUDA_BACKENDS:
        assert np.array_equal(cuda.compute_strongest_paths(preferences, backend=backend), expected), backend
    from_mojo = np.asarray(mojo.strongest_paths(preferences.tolist()), dtype=np.uint32)
    assert np.array_equal(from_mojo, expected)


@pytest.mark.parametrize("seed", range(12))
def test_languages_agree_on_random_profiles(seed: int, mojo):
    """Schulze and Kemeny must agree across all three ports on profiles nobody chose by hand."""
    num_candidates = 2 + seed % 9
    preferences = random_profile(seed=1000 + seed, num_candidates=num_candidates, num_voters=1 + seed * 3, tied=seed % 3 == 0)

    expected = schulze.compute_strongest_paths_numba_serial(preferences)
    for backend in CUDA_BACKENDS:
        assert np.array_equal(cuda.compute_strongest_paths(preferences, backend=backend), expected), backend
    assert np.array_equal(np.asarray(mojo.strongest_paths(preferences.tolist()), dtype=np.uint32), expected)

    # Kemeny is exact, so the ranking has to match and not merely the score it achieves.
    python_ranking, python_score = kemeny.kemeny_ranking(preferences)
    cuda_ranking, cuda_score = cuda.compute_kemeny_ranking(preferences)
    mojo_ranking, mojo_score = mojo.kemeny_consensus(preferences.tolist())
    assert list(python_ranking) == list(cuda_ranking) == list(mojo_ranking)
    assert int(python_score) == int(cuda_score) == int(mojo_score)
    assert int(python_score) == kendall_score(preferences, python_ranking)


@pytest.mark.parametrize("seed", range(12))
def test_schulze_winner_always_inside_split_cycle(seed: int):
    """The direct edge is itself a path, so the Schulze winner can never fall outside Split Cycle."""
    num_candidates = 3 + seed % 8
    preferences = random_profile(seed=2000 + seed, num_candidates=num_candidates, num_voters=2 + seed * 2)
    strongest = schulze.compute_strongest_paths_numba_serial(preferences)
    winner, _ = schulze.get_winner_and_ranking(list(range(num_candidates)), strongest)
    margin_paths = schulze.compute_strongest_paths_numba_serial(ballots.positive_margins(preferences))
    assert winner in schulze.split_cycle_winners(preferences, margin_paths)


# endregion Differential


# region Kemeny Backends


KEMENY_BACKENDS = ("auto", "cpu_serial", "gpu_layered")

# The dispatcher keeps anything below its crossover on the host, so only these sizes reach the kernel.
ABOVE_KEMENY_CROSSOVER = (14, 16)


@PROFILE_CASES
def test_kemeny_backends_agree_below_the_crossover(preferences: np.ndarray, gpu_ready: bool):
    """Forcing the device on a profile the dispatcher would keep on the host must not change the answer."""
    if not gpu_ready:
        pytest.skip("No usable CUDA device")
    host_ranking, host_score = cuda.compute_kemeny_ranking(preferences, backend="cpu_serial")
    device_ranking, device_score = cuda.compute_kemeny_ranking(preferences, backend="gpu_layered")
    assert list(device_ranking) == list(host_ranking)
    assert int(device_score) == int(host_score)


@pytest.mark.parametrize("num_candidates", ABOVE_KEMENY_CROSSOVER)
def test_kemeny_backends_agree_above_the_crossover(num_candidates: int, gpu_ready: bool):
    """Above the crossover `auto` picks the device, so all three spellings must land on one ranking."""
    if not gpu_ready:
        pytest.skip("No usable CUDA device")
    preferences = random_profile(seed=3000 + num_candidates, num_candidates=num_candidates, num_voters=30)
    rankings, scores = zip(*(cuda.compute_kemeny_ranking(preferences, backend=name) for name in KEMENY_BACKENDS))
    assert all(list(ranking) == list(rankings[0]) for ranking in rankings)
    assert len(set(int(score) for score in scores)) == 1
    # The score a backend reports has to be the score its own ranking achieves.
    assert int(scores[0]) == kendall_score(preferences, rankings[0])


@pytest.mark.parametrize("backend", KEMENY_BACKENDS)
def test_kemeny_refuses_more_candidates_than_the_table_can_hold(backend: str):
    """The cost table is exponential, so every backend is refused before it allocates anything."""
    with pytest.raises(RuntimeError, match="34 candidates"):
        cuda.compute_kemeny_ranking(np.zeros((36, 36), dtype=np.uint32), backend=backend)


def test_kemeny_rejects_an_unknown_backend():
    """A misspelled backend names the alternatives rather than silently falling back."""
    with pytest.raises(ValueError, match="cpu_serial"):
        cuda.compute_kemeny_ranking(np.zeros((4, 4), dtype=np.uint32), backend="gpu_layerd")


# endregion Kemeny Backends


# region Narrow Sweep


# The packed sweep holds two candidates per word, so it is only reachable below this ceiling.
SIXTEEN_BIT_CEILING = 65535


def scaled_profile(num_candidates: int, peak: int, seed: int) -> np.ndarray:
    """Tallies real ballots, then scales them so the largest count lands exactly on `peak`."""
    preferences = random_profile(seed=seed, num_candidates=num_candidates, num_voters=40)
    largest = int(preferences.max())
    scaled = (preferences.astype(np.uint64) * (peak // largest)).astype(np.uint32)
    scaled[preferences == largest] = peak
    return scaled


@pytest.mark.parametrize("peak", (SIXTEEN_BIT_CEILING, SIXTEEN_BIT_CEILING + 1, 4_200_000_000))
@pytest.mark.parametrize("num_candidates", (33, 64))
def test_backends_agree_across_the_sixteen_bit_boundary(num_candidates: int, peak: int, cuda_backend: str):
    """One vote count above the ceiling has to move the whole sweep to the wide path, not truncate."""
    preferences = scaled_profile(num_candidates, peak, seed=4000 + num_candidates)
    expected = schulze.compute_strongest_paths_numba_serial(preferences)
    assert np.array_equal(cuda.compute_strongest_paths(preferences, backend=cuda_backend), expected)


# endregion Narrow Sweep
