"""Every method in one namespace, so the split into per-method modules stays invisible.

Ballots fold into one `N x N` matrix of pairwise counts, and Schulze, Split Cycle and
Kemeny-Young all read that matrix and nothing else. `bench.py` drives the benchmarks.
"""

from scalingelections_cuda import (
    compute_kemeny_ranking,  # type: ignore
    compute_split_cycle_winners,  # type: ignore
    compute_strongest_paths,  # type: ignore
    log_gpus,  # type: ignore
    tally_ballots,  # type: ignore
)

from ballots import (
    build_pairwise_preferences,
    complete_rankings,
    generate_preferences,
    populate_preferences_from_ranking,
    positive_margins,
    tally_chunks,
)
from kemeny import kemeny_ranking
from schulze import (
    compute_strongest_paths_numba_parallel,
    compute_strongest_paths_numba_serial,
    get_winner_and_ranking,
    split_cycle_winners,
)

__all__ = [
    # Ballots into the pairwise matrix every method reads.
    "build_pairwise_preferences",
    "complete_rankings",
    "generate_preferences",
    "populate_preferences_from_ranking",
    "positive_margins",
    "tally_ballots",
    "tally_chunks",
    # Schulze and Split Cycle, which share one max-min kernel over different graphs.
    "compute_split_cycle_winners",
    "compute_strongest_paths",
    "compute_strongest_paths_numba_parallel",
    "compute_strongest_paths_numba_serial",
    "get_winner_and_ranking",
    "split_cycle_winners",
    # Exact Kemeny-Young.
    "compute_kemeny_ranking",
    "kemeny_ranking",
    # Devices this build can reach.
    "log_gpus",
]
