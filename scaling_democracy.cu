/**
 * @brief  CUDA-accelerated Schulze voting alrogithm implementation.
 * @author Ash Vardanian
 * @date   July 12, 2024
 */
#include <cstdint>

#include <cuda_runtime.h>

#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using votes_count_t = uint32_t;
using candidate_idx_t = uint32_t;

/**
 * @brief Processes a tile of the preferences matrix for the block-parallel Schulze voting algorithm in CUDA.
 *
 * @tparam tile_size The size of the tile to be processed.
 * @tparam synchronize Whether to synchronize threads within the tile processing.
 * @tparam may_be_diagonal Whether the tile may contain diagonal elements.
 * @param c The output tile.
 * @param a The first input tile.
 * @param b The second input tile.
 * @param bi Row index within the tile.
 * @param bj Column index within the tile.
 * @param c_row Row index of the output tile in the global matrix.
 * @param c_col Column index of the output tile in the global matrix.
 * @param a_row Row index of the first input tile in the global matrix.
 * @param a_col Column index of the first input tile in the global matrix.
 * @param b_row Row index of the second input tile in the global matrix.
 * @param b_col Column index of the second input tile in the global matrix.
 */
template <uint32_t tile_size, bool synchronize = true, bool may_be_diagonal = true>
__forceinline__ __device__ void _process_tile_cuda(       //
    votes_count_t* c, votes_count_t* a, votes_count_t* b, //
    candidate_idx_t bi, candidate_idx_t bj,               //
    candidate_idx_t c_row, candidate_idx_t c_col,         //
    candidate_idx_t a_row, candidate_idx_t a_col,         //
    candidate_idx_t b_row, candidate_idx_t b_col) {

#pragma unroll
    for (candidate_idx_t k = 0; k < tile_size; k++) {
        votes_count_t smallest = umin(a[bi * tile_size + k], b[k * tile_size + bj]);
        if constexpr (may_be_diagonal) {
            uint32_t is_not_diagonal_c = (c_row + bi) != (c_col + bj);
            uint32_t is_not_diagonal_a = (a_row + bi) != (a_col + k);
            uint32_t is_not_diagonal_b = (b_row + k) != (b_col + bj);
            uint32_t is_bigger = smallest > c[bi * tile_size + bj];
            if (is_not_diagonal_c + is_not_diagonal_a + is_not_diagonal_b + is_bigger == 4)
                c[bi * tile_size + bj] = smallest;
        } else
            c[bi * tile_size + bj] = umax(c[bi * tile_size + bj], smallest);
        if constexpr (synchronize)
            __syncthreads();
    }
}

/**
 * @brief Performs the diagonal step of the block-parallel Schulze voting algorithm in CUDA.
 *
 * @tparam tile_size The size of the tile to be processed.
 * @param n The number of candidates.
 * @param k The index of the current tile being processed.
 * @param graph The graph of strongest paths.
 */
template <uint32_t tile_size>
__global__ void _step_diagonal(candidate_idx_t n, candidate_idx_t k, votes_count_t* graph) {
    candidate_idx_t const bi = threadIdx.y;
    candidate_idx_t const bj = threadIdx.x;

    __shared__ votes_count_t c[tile_size * tile_size];
    c[bi * tile_size + bj] = graph[k * tile_size * n + k * tile_size + bi * n + bj];

    __syncthreads();
    _process_tile_cuda<tile_size>(    //
        c, c, c, bi, bj,              //
        tile_size * k, tile_size * k, //
        tile_size * k, tile_size * k, //
        tile_size * k, tile_size * k  //
    );

    graph[k * tile_size * n + k * tile_size + bi * n + bj] = c[bi * tile_size + bj];
}

/**
 * @brief Performs the partially independent step of the block-parallel Schulze voting algorithm in CUDA.
 *
 * @tparam tile_size The size of the tile to be processed.
 * @param n The number of candidates.
 * @param k The index of the current tile being processed.
 * @param graph The graph of strongest paths.
 */
template <uint32_t tile_size>
__global__ void _step_partially_independent(candidate_idx_t n, candidate_idx_t k, votes_count_t* graph) {
    candidate_idx_t const i = blockIdx.x;
    candidate_idx_t const bi = threadIdx.y;
    candidate_idx_t const bj = threadIdx.x;

    if (i == k)
        return;

    __shared__ votes_count_t a[tile_size * tile_size];
    __shared__ votes_count_t b[tile_size * tile_size];
    __shared__ votes_count_t c[tile_size * tile_size];

    // Walking down within a group of adjacent columns
    c[bi * tile_size + bj] = graph[i * tile_size * n + k * tile_size + bi * n + bj];
    b[bi * tile_size + bj] = graph[k * tile_size * n + k * tile_size + bi * n + bj];

    __syncthreads();
    _process_tile_cuda<tile_size>(    //
        c, c, b, bi, bj,              //
        i * tile_size, k * tile_size, //
        i * tile_size, k * tile_size, //
        k * tile_size, k * tile_size);

    // Walking right within a group of adjacent rows
    graph[i * tile_size * n + k * tile_size + bi * n + bj] = c[bi * tile_size + bj];
    c[bi * tile_size + bj] = graph[k * tile_size * n + i * tile_size + bi * n + bj];
    a[bi * tile_size + bj] = graph[k * tile_size * n + k * tile_size + bi * n + bj];

    __syncthreads();
    _process_tile_cuda<tile_size>(    //
        c, a, c, bi, bj,              //
        k * tile_size, i * tile_size, //
        k * tile_size, k * tile_size, //
        k * tile_size, i * tile_size  //
    );

    graph[k * tile_size * n + i * tile_size + bi * n + bj] = c[bi * tile_size + bj];
}

/**
 * @brief Performs then independent step of the block-parallel Schulze voting algorithm in CUDA.
 *
 * @tparam tile_size The size of the tile to be processed.
 * @param n The number of candidates.
 * @param k The index of the current tile being processed.
 * @param graph The graph of strongest paths.
 */
template <uint32_t tile_size>
__global__ void _step_independent(candidate_idx_t n, candidate_idx_t k, votes_count_t* graph) {
    candidate_idx_t const j = blockIdx.x;
    candidate_idx_t const i = blockIdx.y;
    candidate_idx_t const bi = threadIdx.y;
    candidate_idx_t const bj = threadIdx.x;

    if (i == k && j == k)
        return;

    __shared__ votes_count_t a[tile_size * tile_size];
    __shared__ votes_count_t b[tile_size * tile_size];
    __shared__ votes_count_t c[tile_size * tile_size];

    c[bi * tile_size + bj] = graph[i * tile_size * n + j * tile_size + bi * n + bj];
    a[bi * tile_size + bj] = graph[i * tile_size * n + k * tile_size + bi * n + bj];
    b[bi * tile_size + bj] = graph[k * tile_size * n + j * tile_size + bi * n + bj];

    __syncthreads();
    if (i == j)
        // We don't need to "synchronize", because A, C, and B tile arguments
        // are different in the independent state and will address different shared buffers.
        _process_tile_cuda<tile_size, false, true>( //
            c, a, b, bi, bj,                        //
            i * tile_size, j * tile_size,           //
            i * tile_size, k * tile_size,           //
            k * tile_size, j * tile_size            //
        );
    else
        // We don't need to "synchronize", because A, C, and B tile arguments
        // are different in the independent state and will address different shared buffers.
        // We also mark as "non diagonal", because the `i != j`, and in that case
        // we can avoid some branches.
        _process_tile_cuda<tile_size, false, false>( //
            c, a, b, bi, bj,                         //
            i * tile_size, j * tile_size,            //
            i * tile_size, k * tile_size,            //
            k * tile_size, j * tile_size             //
        );

    graph[i * tile_size * n + j * tile_size + bi * n + bj] = c[bi * tile_size + bj];
}

/**
 * @brief Computes the strongest paths for the block-parallel Schulze voting algorithm in CUDA.
 *
 * @tparam tile_size The size of the tile to be processed.
 * @param preferences The preferences matrix.
 * @param num_candidates The number of candidates.
 * @param row_stride The stride between rows in the preferences matrix.
 * @param strongest_paths The output matrix of strongest paths.
 */
template <uint32_t tile_size>      //
void compute_strongest_paths_cuda( //
    votes_count_t* preferences, candidate_idx_t num_candidates, candidate_idx_t row_stride,
    votes_count_t* strongest_paths) {

#pragma omp parallel for collapse(2)
    for (candidate_idx_t i = 0; i < num_candidates; i++)
        for (candidate_idx_t j = 0; j < num_candidates; j++)
            if (i != j)
                strongest_paths[i * num_candidates + j] =
                    preferences[i * row_stride + j] > preferences[j * row_stride + i] //
                        ? preferences[i * row_stride + j]
                        : 0;

    candidate_idx_t tiles_count = (num_candidates + tile_size - 1) / tile_size;
    dim3 tile_shape(tile_size, tile_size, 1);
    dim3 independent_grid(tiles_count, tiles_count, 1);
    for (candidate_idx_t k = 0; k < tiles_count; k++) {
        _step_diagonal<tile_size><<<1, tile_shape>>>(num_candidates, k, strongest_paths);
        _step_partially_independent<tile_size><<<tiles_count, tile_shape>>>(num_candidates, k, strongest_paths);
        _step_independent<tile_size><<<independent_grid, tile_shape>>>(num_candidates, k, strongest_paths);
    }
}

/**
 * @brief Computes the strongest paths for the block-parallel Schulze voting algorithm.
 *
 * @param preferences The preferences matrix.
 * @return A NumPy array containing the strongest paths matrix.
 */
static py::array_t<votes_count_t> compute_strongest_paths(py::array_t<votes_count_t, py::array::c_style> preferences) {
    auto buf = preferences.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");
    if (buf.shape[0] != buf.shape[1])
        throw std::runtime_error("Preferences matrix must be square");
    auto preferences_ptr = reinterpret_cast<votes_count_t*>(buf.ptr);
    auto num_candidates = static_cast<candidate_idx_t>(buf.shape[0]);
    auto row_stride = static_cast<candidate_idx_t>(buf.strides[0] / sizeof(votes_count_t));

    votes_count_t* strongest_paths_ptr = nullptr;
    cudaError_t error;
    error = cudaMallocManaged(&strongest_paths_ptr, num_candidates * num_candidates * sizeof(votes_count_t));
    if (error != cudaSuccess)
        throw std::runtime_error("Failed to allocate memory on device");

    cudaMemset(strongest_paths_ptr, 0, num_candidates * num_candidates * sizeof(votes_count_t));
    compute_strongest_paths_cuda<16>(preferences_ptr, num_candidates, row_stride, strongest_paths_ptr);

    // Synchronize to ensure all CUDA operations are complete
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        cudaFree(strongest_paths_ptr);
        throw std::runtime_error("CUDA operations did not complete successfully");
    }

    // Allocate NumPy array for the result
    auto result = py::array_t<votes_count_t>({num_candidates, num_candidates});
    auto result_buf = result.request();
    auto result_ptr = reinterpret_cast<votes_count_t*>(result_buf.ptr);

    // Copy data from the GPU to the NumPy array
    error = cudaMemcpy(result_ptr, strongest_paths_ptr, num_candidates * num_candidates * sizeof(votes_count_t),
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        cudaFree(strongest_paths_ptr);
        throw std::runtime_error("Failed to copy data from device to host");
    }

    // Synchronize to ensure all CUDA transfers are complete
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        cudaFree(strongest_paths_ptr);
        throw std::runtime_error("CUDA transfers did not complete successfully");
    }

    // Free the GPU memory
    error = cudaFree(strongest_paths_ptr);
    if (error != cudaSuccess)
        throw std::runtime_error("Failed to free memory on device");

    return result;
}

PYBIND11_MODULE(scaling_democracy, m) {

    // Let's show how to wrap `void` functions for basic logging
    m.def("log_devices", []() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp deviceProps;
            cudaGetDeviceProperties(&deviceProps, i);
            printf("Device %d: %s\n", i, deviceProps.name);
            printf("\tSMs: %d\n", deviceProps.multiProcessorCount);
            printf("\tGlobal mem: %.2fGB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024 * 1024));
            printf("\tCUDA Cap: %d.%d\n", deviceProps.major, deviceProps.minor);
        }
    });

    // This is how we could have used `thrust::` for higher-level operations
    m.def("reduce", [](py::array_t<float> const& data) -> float {
        py::buffer_info buf = data.request();
        if (buf.ndim != 1 || buf.strides[0] != sizeof(float))
            throw std::runtime_error("Input should be a contiguous 1D float array");
        float* ptr = static_cast<float*>(buf.ptr);
        thrust::device_vector<float> d_data(ptr, ptr + buf.size);
        return thrust::reduce(thrust::device, d_data.begin(), d_data.end(), 0.0f);
    });

    m.def("compute_strongest_paths", &compute_strongest_paths);
}
