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
__forceinline__ __device__ void _process_tile_cuda( //
    votes_count_t* c, votes_count_t* a, votes_count_t* b, candidate_idx_t bj, candidate_idx_t bi) {
    for (candidate_idx_t k = 0; k < cuda_tile_size; k++) {
        votes_count_t smallest = min(a[bi * cuda_tile_size + k], b[k * cuda_tile_size + bj]);
        c[bi * cuda_tile_size + bj] = max(c[bi * cuda_tile_size + bj], smallest);
        __syncthreads();
    }
}

template <uint32_t cuda_tile_size>
__global__ void _step_diagonal(candidate_idx_t n, candidate_idx_t k, votes_count_t* graph) {
    candidate_idx_t const bi = threadIdx.y;
    candidate_idx_t const bj = threadIdx.x;

    __shared__ votes_count_t c[cuda_tile_size * cuda_tile_size];
    __syncthreads();

    c[bi * cuda_tile_size + bj] = graph[k * cuda_tile_size * n + k * cuda_tile_size + bi * n + bj];

    __syncthreads();
    _process_tile_cuda<cuda_tile_size>(c, c, c, bi, bj);
    __syncthreads();

    graph[k * cuda_tile_size * n + k * cuda_tile_size + bi * n + bj] = c[bi * cuda_tile_size + bj];
}

template <uint32_t cuda_tile_size>
__global__ void _step_partially_independent(candidate_idx_t n, candidate_idx_t k, votes_count_t* graph) {
    candidate_idx_t const i = blockIdx.x;
    candidate_idx_t const bi = threadIdx.y;
    candidate_idx_t const bj = threadIdx.x;

    if (i == k)
        return;

    __shared__ votes_count_t a[cuda_tile_size * cuda_tile_size];
    __shared__ votes_count_t b[cuda_tile_size * cuda_tile_size];
    __shared__ votes_count_t c[cuda_tile_size * cuda_tile_size];
    __syncthreads();

    c[bi * cuda_tile_size + bj] = graph[i * cuda_tile_size * n + k * cuda_tile_size + bi * n + bj];
    b[bi * cuda_tile_size + bj] = graph[k * cuda_tile_size * n + k * cuda_tile_size + bi * n + bj];

    __syncthreads();
    _process_tile_cuda<cuda_tile_size>(c, c, b, bi, bj);
    __syncthreads();

    graph[i * cuda_tile_size * n + k * cuda_tile_size + bi * n + bj] = c[bi * cuda_tile_size + bj];
    c[bi * cuda_tile_size + bj] = graph[k * cuda_tile_size * n + i * cuda_tile_size + bi * n + bj];
    a[bi * cuda_tile_size + bj] = graph[k * cuda_tile_size * n + k * cuda_tile_size + bi * n + bj];

    __syncthreads();
    _process_tile_cuda<cuda_tile_size>(c, a, c, bi, bj);
    __syncthreads();

    graph[k * cuda_tile_size * n + i * cuda_tile_size + bi * n + bj] = c[bi * cuda_tile_size + bj];
}

template <uint32_t cuda_tile_size>
__global__ void _step_independent(candidate_idx_t n, candidate_idx_t k, votes_count_t* graph) {
    candidate_idx_t const j = blockIdx.x;
    candidate_idx_t const i = blockIdx.y;
    candidate_idx_t const bi = threadIdx.y;
    candidate_idx_t const bj = threadIdx.x;

    if (i == k && j == k)
        return;

    __shared__ votes_count_t a[cuda_tile_size * cuda_tile_size];
    __shared__ votes_count_t b[cuda_tile_size * cuda_tile_size];
    __shared__ votes_count_t c[cuda_tile_size * cuda_tile_size];
    __syncthreads();

    c[bi * cuda_tile_size + bj] = graph[i * cuda_tile_size * n + j * cuda_tile_size + bi * n + bj];
    a[bi * cuda_tile_size + bj] = graph[i * cuda_tile_size * n + k * cuda_tile_size + bi * n + bj];
    b[bi * cuda_tile_size + bj] = graph[k * cuda_tile_size * n + j * cuda_tile_size + bi * n + bj];

    __syncthreads();
    _process_tile_cuda<cuda_tile_size>(c, a, b, bi, bj);
    __syncthreads();

    graph[i * cuda_tile_size * n + j * cuda_tile_size + bi * n + bj] = c[bi * cuda_tile_size + bj];
}

template <uint32_t cuda_tile_size> //
void compute_strongest_paths_cuda( //
    votes_count_t* preferences, candidate_idx_t num_candidates, candidate_idx_t row_stride,
    votes_count_t* strongest_paths) {

    for (candidate_idx_t i = 0; i < num_candidates; i++)
        for (candidate_idx_t j = 0; j < num_candidates; j++)
            if (i != j)
                strongest_paths[i * num_candidates + j] =
                    preferences[i * row_stride + j] > preferences[j * row_stride + i] //
                        ? preferences[i * row_stride + j]
                        : 0;

    candidate_idx_t tiles_count = (num_candidates + cuda_tile_size - 1) / cuda_tile_size;
    dim3 tile_shape(cuda_tile_size, cuda_tile_size, 1);
    dim3 independent_grid(tiles_count, tiles_count, 1);
    for (candidate_idx_t k = 0; k < tiles_count; k++) {
        _step_diagonal<cuda_tile_size><<<1, tile_shape>>>(num_candidates, k, strongest_paths);
        _step_partially_independent<cuda_tile_size><<<tiles_count, tile_shape>>>(num_candidates, k, strongest_paths);
        _step_independent<cuda_tile_size><<<independent_grid, tile_shape>>>(num_candidates, k, strongest_paths);
    }
}

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
    }); // Test, make sure this works ;)

    m.def("compute_strongest_paths", &compute_strongest_paths);
}
