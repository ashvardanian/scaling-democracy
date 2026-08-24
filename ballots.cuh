/**
 *  @brief Turns a pairwise preference matrix into the winning-votes graph the Schulze method walks.
 *  @file ballots.cuh
 *  @author Ash Vardanian
 *  @date July 12, 2024
 *  @see https://ashvardanian.com/posts/scaling-elections
 */
#pragma once
#include "types.cuh"

/**
 *  @brief Seeds the strongest-paths matrix with the direct pairwise wins.
 *
 *  @param[in] preferences The preferences matrix.
 *  @param[in] num_candidates The number of candidates.
 *  @param[in] row_stride The stride between rows in the preferences matrix.
 *  @param[out] graph The matrix of strongest paths.
 *  @param[in] graph_stride The row stride of the output, which may exceed @p num_candidates when padded.
 *
 *  A cell keeps its vote count only where it strictly beats the opposite direction, so ties, losses,
 *  and the diagonal all read as zero, the identity of the max-min semiring.
 */
inline void winning_votes_graph(                                                                      //
    votes_count_t const* preferences, candidate_index_t num_candidates, candidate_index_t row_stride, //
    votes_count_t* graph, candidate_index_t graph_stride) {

#pragma omp parallel for collapse(2)
    for (candidate_index_t i = 0; i < num_candidates; i++)
        for (candidate_index_t j = 0; j < num_candidates; j++)
            graph[i * graph_stride + j] = i != j && preferences[i * row_stride + j] > preferences[j * row_stride + i]
                                              ? preferences[i * row_stride + j]
                                              : 0;
}
