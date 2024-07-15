/**
 * @brief  CUDA-accelerated Schulze voting alrogithm implementation.
 * @author Ash Vardanian
 * @date   July 12, 2024
 */
#include <cstdint>

#include <cuda_runtime.h>

#include <cuda.h> // `CUtensorMap`
#include <cuda/barrier>
#include <cudaTypedefs.h> // `PFN_cuTensorMapEncodeTiled`
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <pybind11/numpy.h> // `array_t`
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
#define SCALING_DEMOCRACY_KEPLER 1
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define SCALING_DEMOCRACY_HOPPER 1
#endif

namespace py = pybind11;
namespace cde = cuda::device::experimental;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;

using votes_count_t = uint32_t;
using candidate_idx_t = uint32_t;

#if defined(SCALING_DEMOCRACY_KEPLER)

/**
 * @brief   Processes a tile of the preferences matrix for the block-parallel Schulze voting algorithm
 *          in CUDA on Nvidia @b Kepler GPUs and newer (sm_30).
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
__forceinline__ __device__ void _process_tile_cuda(                   //
    votes_count_t* c, votes_count_t const* a, votes_count_t const* b, //
    candidate_idx_t bi, candidate_idx_t bj,                           //
    candidate_idx_t c_row, candidate_idx_t c_col,                     //
    candidate_idx_t a_row, candidate_idx_t a_col,                     //
    candidate_idx_t b_row, candidate_idx_t b_col) {

#pragma unroll(tile_size)
    for (candidate_idx_t k = 0; k < tile_size; k++) {
        votes_count_t smallest = umin(a[bi * tile_size + k], b[k * tile_size + bj]);
        if constexpr (may_be_diagonal) {
            uint32_t is_not_diagonal_c = (c_row + bi) != (c_col + bj);
            uint32_t is_not_diagonal_a = (a_row + bi) != (a_col + k);
            uint32_t is_not_diagonal_b = (b_row + k) != (b_col + bj);
            uint32_t is_bigger = smallest > c[bi * tile_size + bj];
            uint32_t will_replace = is_not_diagonal_c & is_not_diagonal_a & is_not_diagonal_b & is_bigger;
            // On Kepler an newer we can use `__funnelshift_lc` to avoid branches
            c[bi * tile_size + bj] = __funnelshift_lc(c[bi * tile_size + bj], smallest, will_replace - 1);
        } else
            c[bi * tile_size + bj] = umax(c[bi * tile_size + bj], smallest);
        if constexpr (synchronize)
            __syncthreads();
    }
}

#else

/**
 * @brief   Processes a tile of the preferences matrix for the block-parallel Schulze voting algorithm
 *          in CUDA or @b HIP.
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
__forceinline__ __device__ void _process_tile_cuda(                   //
    votes_count_t* c, votes_count_t const* a, votes_count_t const* b, //
    candidate_idx_t bi, candidate_idx_t bj,                           //
    candidate_idx_t c_row, candidate_idx_t c_col,                     //
    candidate_idx_t a_row, candidate_idx_t a_col,                     //
    candidate_idx_t b_row, candidate_idx_t b_col) {

#pragma unroll(tile_size)
    for (candidate_idx_t k = 0; k < tile_size; k++) {
        votes_count_t smallest = min(a[bi * tile_size + k], b[k * tile_size + bj]);
        if constexpr (may_be_diagonal) {
            uint32_t is_not_diagonal_c = (c_row + bi) != (c_col + bj);
            uint32_t is_not_diagonal_a = (a_row + bi) != (a_col + k);
            uint32_t is_not_diagonal_b = (b_row + k) != (b_col + bj);
            uint32_t is_bigger = smallest > c[bi * tile_size + bj];
            uint32_t will_replace = is_not_diagonal_c & is_not_diagonal_a & is_not_diagonal_b & is_bigger;
            if (will_replace)
                c[bi * tile_size + bj] = smallest;
        } else
            c[bi * tile_size + bj] = max(c[bi * tile_size + bj], min(a[bi * tile_size + k], b[k * tile_size + bj]));
        if constexpr (synchronize)
            __syncthreads();
    }
}

#endif

/**
 * @brief Performs the diagonal step of the block-parallel Schulze voting algorithm in CUDA or @b HIP.
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

    __shared__ alignas(16) votes_count_t c[tile_size * tile_size];
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
 * @brief Performs the partially independent step of the block-parallel Schulze voting algorithm in CUDA or @b HIP.
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

    __shared__ alignas(16) votes_count_t a[tile_size * tile_size];
    __shared__ alignas(16) votes_count_t b[tile_size * tile_size];
    __shared__ alignas(16) votes_count_t c[tile_size * tile_size];

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
    __syncthreads();
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
 * @brief Performs then independent step of the block-parallel Schulze voting algorithm in CUDA or @b HIP.
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

    __shared__ alignas(16) votes_count_t a[tile_size * tile_size];
    __shared__ alignas(16) votes_count_t b[tile_size * tile_size];
    __shared__ alignas(16) votes_count_t c[tile_size * tile_size];

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
 * @brief Performs then independent step of the block-parallel Schulze voting algorithm in CUDA or @b HIP.
 *
 * @tparam tile_size The size of the tile to be processed.
 * @param n The number of candidates.
 * @param k The index of the current tile being processed.
 * @param graph The graph of strongest paths represented as a `CUtensorMap`.
 */
template <uint32_t tile_size>
__global__ void _step_independent_hopper(candidate_idx_t n, candidate_idx_t k,
                                         __grid_constant__ CUtensorMap const graph) {
    candidate_idx_t const j = blockIdx.x;
    candidate_idx_t const i = blockIdx.y;
    candidate_idx_t const bi = threadIdx.y;
    candidate_idx_t const bj = threadIdx.x;

#if defined(SCALING_DEMOCRACY_HOPPER)

    if (i == k && j == k)
        return;

    __shared__ alignas(128) votes_count_t a[tile_size][tile_size];
    __shared__ alignas(128) votes_count_t b[tile_size][tile_size];
    __shared__ alignas(128) votes_count_t c[tile_size][tile_size];

#pragma nv_diag_suppress static_var_with_dynamic_init
    // Initialize shared memory barrier with the number of threads participating in the barrier.
    __shared__ barrier_t bar;
    if (threadIdx.x == 0) {
        // We have one thread per tile cell.
        init(&bar, tile_size * tile_size);
        // Make initialized barrier visible in async proxy.
        cde::fence_proxy_async_shared_cta();
    }
    // Syncthreads so initialized barrier is visible to all threads.
    __syncthreads();

    // Only the first thread in the tile invokes the bulk transfers.
    barrier_t::arrival_token token;
    if (threadIdx.x == 0) {
        // Initiate three bulk tensor copies for different part of the graph.
        cde::cp_async_bulk_tensor_2d_global_to_shared(&c, &graph, i * tile_size, j * tile_size, bar);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&a, &graph, i * tile_size, k * tile_size, bar);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&b, &graph, k * tile_size, j * tile_size, bar);
        // Arrive on the barrier and tell how many bytes are expected to come in.
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(c) + sizeof(a) + sizeof(b));
    } else {
        // Other threads just arrive.
        token = bar.arrive(1);
    }

    // Wait for the data to have arrived.
    // After this point we expect shared memory to contain the following data:
    //
    //  c[bi * tile_size + bj] = graph[i * tile_size * n + j * tile_size + bi * n + bj];
    //  a[bi * tile_size + bj] = graph[i * tile_size * n + k * tile_size + bi * n + bj];
    //  b[bi * tile_size + bj] = graph[k * tile_size * n + j * tile_size + bi * n + bj];
    bar.wait(std::move(token));

    if (i == j)
        // We don't need to "synchronize", because A, C, and B tile arguments
        // are different in the independent state and will address different shared buffers.
        _process_tile_cuda<tile_size, false, true>( //
            &c[0][0], &a[0][0], &b[0][0], bi, bj,   //
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
            &c[0][0], &a[0][0], &b[0][0], bi, bj,    //
            i * tile_size, j * tile_size,            //
            i * tile_size, k * tile_size,            //
            k * tile_size, j * tile_size             //
        );

    // Wait for shared memory writes to be visible to TMA engine.
    cde::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_2d_shared_to_global(&graph, i * tile_size, j * tile_size, &c);
        // Wait for TMA transfer to have finished reading shared memory.
        // Create a "bulk async-group" out of the previous bulk copy operation.
        cde::cp_async_bulk_commit_group();
        // Wait for the group to have completed reading from shared memory.
        cde::cp_async_bulk_wait_group_read<0>();

        // Destroy barrier. This invalidates the memory region of the barrier. If
        // further computations were to take place in the kernel, this allows the
        // memory location of the shared memory barrier to be reused.
        (&bar)->~barrier();
    }
#else
    // This is a trap :)
    if (i == 0 && j == 0 && bi == 0 && bj == 0)
        printf("This kernel is only supported on Hopper and newer GPUs\n");
#endif
}

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    // Get pointer to cuGetProcAddress
    cudaDriverEntryPointQueryResult driver_status;
    void* cuGetProcAddress_ptr = nullptr;
    cudaError_t error =
        cudaGetDriverEntryPoint("cuGetProcAddress", &cuGetProcAddress_ptr, cudaEnableDefault, &driver_status);
    if (error != cudaSuccess)
        throw std::runtime_error("Failed to get cuGetProcAddress");
    if (driver_status != cudaDriverEntryPointSuccess)
        throw std::runtime_error("Failed to get cuGetProcAddress entry point");
    PFN_cuGetProcAddress_v12000 cuGetProcAddress = reinterpret_cast<PFN_cuGetProcAddress_v12000>(cuGetProcAddress_ptr);

    // Use cuGetProcAddress to get a pointer to the CTK 12.0 version of cuTensorMapEncodeTiled
    CUdriverProcAddressQueryResult symbol_status;
    void* cuTensorMapEncodeTiled_ptr = nullptr;
    CUresult res = cuGetProcAddress("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000,
                                    CU_GET_PROC_ADDRESS_DEFAULT, &symbol_status);
    if (res != CUDA_SUCCESS || symbol_status != CU_GET_PROC_ADDRESS_SUCCESS)
        throw std::runtime_error("Failed to get cuTensorMapEncodeTiled");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

/**
 * @brief Computes the strongest paths for the block-parallel Schulze voting algorithm in CUDA or @b HIP.
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

    // Check if we can use newer CUDA features.
    cudaError_t error;
    int current_device;
    cudaDeviceProp device_props;
    error = cudaGetDevice(&current_device);
    if (error != cudaSuccess)
        throw std::runtime_error("Failed to get current device");
    error = cudaGetDeviceProperties(&device_props, current_device);
    if (error != cudaSuccess)
        throw std::runtime_error("Failed to get device properties");
    bool supports_tma = device_props.major >= 9;
    printf("Device %d supports TMA: %s\n", current_device, supports_tma ? "yes" : "no");

    CUtensorMap strongest_paths_tensor_map{};
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {num_candidates, num_candidates};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride[rank - 1] = {num_candidates * sizeof(votes_count_t)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size[rank] = {tile_size, tile_size};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    uint32_t elem_stride[rank] = {1, 1};

    // Create the tensor descriptor.
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
    PFN_cuTensorMapEncodeTiled_v12000 cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    CUresult res = cuTensorMapEncodeTiled( //
        &strongest_paths_tensor_map,       // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
        rank,            // cuuint32_t tensorRank,
        strongest_paths, // void *globalAddress,
        size,            // const cuuint64_t *globalDim,
        stride,          // const cuuint64_t *globalStrides,
        box_size,        // const cuuint32_t *boxDim,
        elem_stride,     // const cuuint32_t *elementStrides,
        // Interleave patterns can be used to accelerate loading of values that
        // are less than 4 bytes long.
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        // Swizzling can be used to avoid shared memory bank conflicts.
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        // L2 Promotion can be used to widen the effect of a cache-policy to a wider
        // set of L2 cache lines. Can be 64b, 128b, 256b, or none.
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        // Any element that is outside of bounds will be set to zero by the TMA transfer.
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    candidate_idx_t tiles_count = (num_candidates + tile_size - 1) / tile_size;
    dim3 tile_shape(tile_size, tile_size, 1);
    dim3 independent_grid(tiles_count, tiles_count, 1);
    for (candidate_idx_t k = 0; k < tiles_count; k++) {
        _step_diagonal<tile_size><<<1, tile_shape>>>(num_candidates, k, strongest_paths);
        _step_partially_independent<tile_size><<<tiles_count, tile_shape>>>(num_candidates, k, strongest_paths);
        if (supports_tma)
            _step_independent_hopper<tile_size>
                <<<independent_grid, tile_shape>>>(num_candidates, k, strongest_paths_tensor_map);
        else
            _step_independent<tile_size><<<independent_grid, tile_shape>>>(num_candidates, k, strongest_paths);

        error = cudaGetLastError();
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
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
        int device_count;
        cudaDeviceProp device_props;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess)
            throw std::runtime_error("Failed to get device count");
        for (int i = 0; i < device_count; i++) {
            error = cudaGetDeviceProperties(&device_props, i);
            if (error != cudaSuccess)
                throw std::runtime_error("Failed to get device properties");
            printf("Device %d: %s\n", i, device_props.name);
            printf("\tSMs: %d\n", device_props.multiProcessorCount);
            printf("\tGlobal mem: %.2fGB\n", static_cast<float>(device_props.totalGlobalMem) / (1024 * 1024 * 1024));
            printf("\tCUDA Cap: %d.%d\n", device_props.major, device_props.minor);
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
