/**
 * @brief  CUDA-accelerated Schulze voting alrogithm implementation.
 * @author Ash Vardanian
 * @date   July 12, 2024
 *
 *
 */
#include <cstdint>

#include <cuda_runtime.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using votes_count_t = uint32_t;
using candidate_idx_t = uint32_t;

template <uint32_t cuda_tile_size>
__device__ void _process_tile(                                            //
    votes_count_t* c, candidate_idx_t c_row, candidate_idx_t c_col,       //
    votes_count_t const* a, candidate_idx_t a_row, candidate_idx_t a_col, //
    votes_count_t const* b, candidate_idx_t b_row, candidate_idx_t b_col, //
    candidate_idx_t const num_candidates) {

    __shared__ votes_count_t a_shared[cuda_tile_size][cuda_tile_size];
    __shared__ votes_count_t b_shared[cuda_tile_size][cuda_tile_size];
    __shared__ votes_count_t c_shared[cuda_tile_size][cuda_tile_size];

    // Load the tiles into shared memory
    candidate_idx_t i = threadIdx.x;
    candidate_idx_t j = threadIdx.y;
    a_shared[i][j] = a[(a_row + i) * num_candidates + a_col + j];
    b_shared[i][j] = b[(b_row + i) * num_candidates + b_col + j];
    c_shared[i][j] = c[(c_row + i) * num_candidates + c_col + j];

    __syncthreads();
    for (candidate_idx_t k = 0; k < cuda_tile_size; k++) {
        if ((c_row + i != c_col + j) && (a_row + i != a_col + k) && (b_row + k != b_col + j))
            c_shared[i][j] = max(c_shared[i][j], min(a_shared[i][k], b_shared[k][j]));
        __syncthreads();
    }

    c[(c_row + i) * num_candidates + c_col + j] = c_shared[i][j];
}

template <uint32_t cuda_tile_size>
__global__ void _cuda_step_partially_dependent( //
    votes_count_t* strongest_paths, candidate_idx_t num_candidates, candidate_idx_t k) {
    candidate_idx_t j = blockIdx.x;
    if (j == k)
        return;

    candidate_idx_t k_start = k * cuda_tile_size;
    candidate_idx_t j_start = j * cuda_tile_size;

    _process_tile<cuda_tile_size>(         //
        strongest_paths, k_start, j_start, //
        strongest_paths, k_start, k_start, //
        strongest_paths, k_start, j_start, num_candidates);
}

template <uint32_t cuda_tile_size>
__global__ void _cuda_step_independent( //
    votes_count_t* strongest_paths, candidate_idx_t num_candidates, candidate_idx_t k) {
    candidate_idx_t i = blockIdx.x;
    if (i == k)
        return;

    candidate_idx_t i_start = i * cuda_tile_size;
    candidate_idx_t k_start = k * cuda_tile_size;

    _process_tile<cuda_tile_size>(         //
        strongest_paths, i_start, k_start, //
        strongest_paths, i_start, k_start, //
        strongest_paths, k_start, k_start, num_candidates);

    for (candidate_idx_t j = 0; j < num_candidates / cuda_tile_size; j++) {
        if (j == k)
            continue;
        candidate_idx_t j_start = j * cuda_tile_size;

        _process_tile<cuda_tile_size>(         //
            strongest_paths, i_start, j_start, //
            strongest_paths, i_start, k_start, //
            strongest_paths, k_start, j_start, num_candidates);
    }
}

template <uint32_t cuda_tile_size> //
void compute_strongest_paths_cuda( //
    votes_count_t* preferences, candidate_idx_t num_candidates, votes_count_t* strongest_paths) {

    for (candidate_idx_t i = 0; i < num_candidates; i++) {
        for (candidate_idx_t j = 0; j < num_candidates; j++) {
            if (i != j) {
                if (preferences[i * num_candidates + j] > preferences[j * num_candidates + i]) {
                    strongest_paths[i * num_candidates + j] = preferences[i * num_candidates + j];
                } else {
                    strongest_paths[i * num_candidates + j] = 0;
                }
            }
        }
    }

    candidate_idx_t tiles_count = (num_candidates + cuda_tile_size - 1) / cuda_tile_size;
    for (candidate_idx_t k = 0; k < tiles_count; k++) {
        candidate_idx_t k_start = k * cuda_tile_size;

        _process_tile<cuda_tile_size>(         //
            strongest_paths, k_start, k_start, //
            strongest_paths, k_start, k_start, //
            strongest_paths, k_start, k_start, num_candidates);
        _cuda_step_partially_dependent<cuda_tile_size><<<tiles_count, 1>>>(strongest_paths, num_candidates, k);
        _cuda_step_independent<cuda_tile_size><<<tiles_count, 1>>>(strongest_paths, num_candidates, k);
    }
}

static py::array_t<votes_count_t>
compute_strongest_paths(py::array_t<votes_count_t, py::array::c_style | py::array::forcecast> preferences) {
    auto buf = preferences.request();
    auto preferences_ptr = reinterpret_cast<votes_count_t*>(buf.ptr);
    auto num_candidates = static_cast<candidate_idx_t>(buf.shape[0]);

    votes_count_t* strongest_paths_ptr = nullptr;
    cudaMallocManaged(&strongest_paths_ptr, num_candidates * num_candidates * sizeof(votes_count_t));
    cudaMemset(strongest_paths_ptr, 0, num_candidates * num_candidates * sizeof(votes_count_t));

    compute_strongest_paths_cuda<32>(preferences_ptr, num_candidates, strongest_paths_ptr);
    auto result = py::array_t<votes_count_t>({num_candidates, num_candidates}, strongest_paths_ptr);
    cudaFree(strongest_paths_ptr);
    return result;
}

PYBIND11_MODULE(scaling_democracy, m) { m.def("compute_strongest_paths", &compute_strongest_paths); }
