"""Python bindings for the Mojo implementations, so one test suite can cross-check every backend.

This module holds nothing but argument marshalling and the module table. `main` lives in
`cli.mojo`, because Mojo refuses to emit a shared library from a module that defines one.

Matrices cross the boundary element by element rather than as a buffer, which is fine for the
cross-language checks this exists to serve and wrong for benchmarking. Time the native binary
`cli.mojo` builds instead.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from ballots import PreferenceMatrix
from kemeny import kemeny_ranking
from schulze import TILE_SIZE, compute_strongest_paths_tiled_cpu

# region Python Bindings


def matrix_from(preferences: PythonObject) raises -> PreferenceMatrix:
    """Reads a square two-dimensional integer array into an owned matrix."""
    var num_candidates = Int(String(preferences.__len__()))
    if num_candidates < 1:
        raise Error("Preferences must have at least one candidate")

    var matrix = PreferenceMatrix(num_candidates)
    for i in range(num_candidates):
        var row = preferences[i]
        if Int(String(row.__len__())) != num_candidates:
            raise Error("Preferences must be a square matrix")
        for j in range(num_candidates):
            matrix[i, j] = UInt32(Int(String(row[j])))
    return matrix^


def strongest_paths(preferences: PythonObject) raises -> PythonObject:
    """Widest paths over winning votes, as a list of rows."""
    var matrix = matrix_from(preferences)
    var strengths = compute_strongest_paths_tiled_cpu[TILE_SIZE](matrix)

    var rows = Python().list()
    for i in range(strengths.num_candidates):
        var row = Python().list()
        for j in range(strengths.num_candidates):
            row.append(PythonObject(Int(strengths[i, j])))
        rows.append(row)
    return rows


def kemeny_consensus(preferences: PythonObject) raises -> PythonObject:
    """The exact Kemeny-Young ranking and the disagreement it achieves."""
    var matrix = matrix_from(preferences)
    var outcome = kemeny_ranking(matrix)

    var ranking = Python().list()
    for i in range(len(outcome[0])):
        ranking.append(PythonObject(outcome[0][i]))
    var pair = Python().list()
    pair.append(ranking)
    pair.append(PythonObject(outcome[1]))
    return pair


@export
def PyInit_scalingelections_mojo() abi("C") -> PythonObject:
    try:
        var builder = PythonModuleBuilder("scalingelections_mojo")
        builder.def_function[strongest_paths]("strongest_paths")
        builder.def_function[kemeny_consensus]("kemeny_consensus")
        return builder.finalize()
    except error:
        abort(String("Failed to initialize scalingelections_mojo: ", error))


# endregion Python Bindings
