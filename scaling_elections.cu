/**
 *  @brief CUDA-accelerated Schulze voting algorithm implementation.
 *  @file scaling_elections.cu
 *  @author Ash Vardanian
 *  @date July 12, 2024
 *  @see https://ashvardanian.com/posts/scaling-elections
 */
#include <csignal> // `std::signal`
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `std::printf`
#include <cstdlib> // `std::rand`

#include <algorithm>   // `std::min`, `std::max`
#include <numeric>     // `std::accumulate`
#include <optional>    // `std::optional`, `std::nullopt`
#include <stdexcept>   // `std::runtime_error`
#include <string>      // `std::to_string`
#include <string_view> // `std::string_view`
#include <thread>      // `std::thread::hardware_concurrency()`
#include <type_traits> // `std::integral_constant`, `std::is_same`
#include <vector>      // `std::vector`

// OpenMP support detection
#if defined(_OPENMP)
#include <omp.h> // `omp_set_num_threads`
#define SCALING_ELECTIONS_WITH_OPENMP (1)
#endif

#if (defined(__ARM_NEON) || defined(__aarch64__))
#define SCALING_ELECTIONS_WITH_NEON (1)
#endif
#if defined(__NVCC__)
#define SCALING_ELECTIONS_WITH_CUDA (1)
#endif
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#define SCALING_ELECTIONS_WITH_HIP  (1)
#define SCALING_ELECTIONS_WITH_CUDA (1) // HIP is CUDA-compatible
#endif

#if defined(SCALING_ELECTIONS_WITH_NEON)
#include <arm_neon.h>
#endif

#if defined(SCALING_ELECTIONS_WITH_CUDA) && !defined(SCALING_ELECTIONS_WITH_HIP)
// NVIDIA CUDA headers
#include <cuda.h> // `CUtensorMap`
#include <cuda/barrier>
#include <cudaTypedefs.h> // `PFN_cuTensorMapEncodeTiled`
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

// AMD HIP headers (CUDA-compatible)
#elif defined(SCALING_ELECTIONS_WITH_HIP)
#include <hip/hip_runtime.h>

// HIP compatibility layer: map CUDA types/functions to HIP equivalents
#if defined(__HIP_PLATFORM_AMD__)
#define cudaError_t             hipError_t
#define cudaSuccess             hipSuccess
#define cudaGetDevice           hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp          hipDeviceProp_t
#define cudaMallocManaged       hipMallocManaged
#define cudaFree                hipFree
#define cudaMemcpy              hipMemcpy
#define cudaMemcpyDeviceToHost  hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice  hipMemcpyHostToDevice
#define cudaMemset              hipMemset
#define cudaDeviceSynchronize   hipDeviceSynchronize
#define cudaGetLastError        hipGetLastError
#define cudaGetErrorString      hipGetErrorString
#define cudaGetDeviceCount      hipGetDeviceCount

#endif
#endif

// Raw-kernel test builds link no Python.
#if !defined(SCALING_ELECTIONS_TEST)
#include <pybind11/numpy.h> // `array_t`
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300 && !defined(SCALING_ELECTIONS_WITH_HIP)
#define SCALING_ELECTIONS_KEPLER (1)
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !defined(SCALING_ELECTIONS_WITH_HIP)
#define SCALING_ELECTIONS_HOPPER (1)
#endif

using votes_count_t = std::uint32_t;
using candidate_index_t = std::uint32_t;

template <std::uint32_t tile_size_>
using votes_count_tile = votes_count_t[tile_size_][tile_size_];

/**
 *  @brief Which tile phase a processor runs, fixing both buffer aliasing and diagonal handling.
 *
 *  The three enumerators are the only combinations the recurrence produces, so the fourth
 *  pairing of the underlying flags cannot be spelled.
 */
enum class tile_phase_t : std::uint8_t {
    /** @brief Output and inputs share one buffer, so every step needs a barrier. */
    aliased_k,
    /** @brief Separate buffers, but the tile may straddle the matrix diagonal. */
    distinct_diagonal_k,
    /** @brief Separate buffers, provably off the diagonal, so the update is a plain maximum. */
    distinct_independent_k,
};

/** @brief Whether a tile copy bounds-checks its edges; `checked_k` also disables the NEON path. */
enum class tile_march_t : bool { fast_k = true, checked_k = false };

/** @brief Which family computes the strongest paths. */
enum class backend_t : std::uint8_t {
    /** @brief Tiled CPU kernels across an OpenMP team. */
    cpu_openmp_k,
    /** @brief Tiled GPU kernels staging tiles through per-thread loads, on CUDA or HIP. */
    gpu_serial_k,
    /** @brief Tiled GPU kernels staging tiles through the bulk-tensor engine, NVIDIA sm_90 and newer. */
    gpu_hopper_k,
};

/** @brief Resolves the name the Python layer passes, throwing on anything unrecognized. */
inline backend_t backend_from_name(std::string_view name) {
    if (name == "cpu_openmp") return backend_t::cpu_openmp_k;
    if (name == "gpu_serial") return backend_t::gpu_serial_k;
    if (name == "gpu_hopper") return backend_t::gpu_hopper_k;
    throw std::invalid_argument("Backend must be one of: cpu_openmp, gpu_serial, gpu_hopper");
}

/**
 *  @brief The closed set of tile sizes a backend instantiates, lowered onto compile-time constants.
 *
 *  The sizes must be listed in ascending order, which @c largest_fitting relies on.
 */
template <std::uint32_t... tile_sizes_>
struct tile_ladder {

    static constexpr bool ascending() noexcept {
        constexpr std::uint32_t sizes[] = {tile_sizes_...};
        for (std::size_t index = 1; index < sizeof...(tile_sizes_); ++index)
            if (sizes[index] <= sizes[index - 1]) return false;
        return true;
    }
    static_assert(ascending(), "Tile sizes must be listed in ascending order");

    /** @brief Whether @p tile_size is one of the instantiated sizes. */
    static constexpr bool contains(std::size_t tile_size) noexcept { return ((tile_size == tile_sizes_) || ...); }

    /** @brief The largest size not exceeding @p num_candidates, or zero when none fits. */
    static constexpr std::uint32_t largest_fitting(candidate_index_t num_candidates) noexcept {
        std::uint32_t best = 0;
        (((tile_sizes_ <= num_candidates) && (best = tile_sizes_, true)), ...);
        return best;
    }

    /** @brief Invokes @p callback with the matching size as an integral constant, or returns false. */
    template <typename callback_type_>
    static bool dispatch(std::size_t tile_size, callback_type_&& callback) {
        return ((tile_size == tile_sizes_ ? (callback(std::integral_constant<std::uint32_t, tile_sizes_> {}), true)
                                          : false) ||
                ...);
    }
};

using cpu_tiles_t = tile_ladder<4, 8, 16, 32, 64, 128>;
using gpu_tiles_t = tile_ladder<4, 8, 16, 32>;

/**
 *  @brief Stores the interrupt signal status.
 */
volatile std::sig_atomic_t global_signal_status = 0;

void signal_handler(int signal) { global_signal_status = signal; }

#pragma region CUDA

#if defined(SCALING_ELECTIONS_WITH_CUDA)

#if !defined(SCALING_ELECTIONS_WITH_HIP)
namespace cde = cuda::device::experimental;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;
#endif

/** @brief Owns one managed reservation for the padded strongest-paths matrix. */
struct managed_graph_t {
    votes_count_t* pointer = nullptr;

    explicit managed_graph_t(std::size_t bytes) {
        if (cudaMallocManaged(&pointer, bytes) != cudaSuccess)
            throw std::runtime_error("Failed to allocate memory on device");
    }
    ~managed_graph_t() noexcept {
        if (pointer) cudaFree(pointer);
    }
    managed_graph_t(managed_graph_t const&) = delete;
    managed_graph_t& operator=(managed_graph_t const&) = delete;
};

#if defined(SCALING_ELECTIONS_KEPLER)

/**
 *  @brief Processes a tile of the preferences matrix for the block-parallel Schulze voting algorithm
 *      in CUDA on Nvidia @b Kepler GPUs and newer (sm_30).
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @tparam phase_ Whether the tiles alias and whether the tile may straddle the diagonal.
 *
 *  Tile @p c is the output, @p a and @p b the inputs; @p bi and @p bj address a cell within a
 *  tile, and each @p *_row and @p *_col pair the tile's origin in the global matrix.
 */
template <std::uint32_t tile_size_, tile_phase_t phase_>
__forceinline__ __device__ void process_tile_cuda_(   //
    votes_count_tile<tile_size_>& c,                  //
    votes_count_tile<tile_size_> const& a,            //
    votes_count_tile<tile_size_> const& b,            //
    candidate_index_t bi, candidate_index_t bj,       //
    candidate_index_t c_row, candidate_index_t c_col, //
    candidate_index_t a_row, candidate_index_t a_col, //
    candidate_index_t b_row, candidate_index_t b_col) {

    votes_count_t& c_cell = c[bi][bj];

#pragma unroll tile_size_
    for (candidate_index_t k = 0; k < tile_size_; k++) {
        votes_count_t smallest = umin(a[bi][k], b[k][bj]);
        if constexpr (phase_ != tile_phase_t::distinct_independent_k) {
            std::uint32_t is_not_diagonal_c = (c_row + bi) != (c_col + bj);
            std::uint32_t is_not_diagonal_a = (a_row + bi) != (a_col + k);
            std::uint32_t is_not_diagonal_b = (b_row + k) != (b_col + bj);
            std::uint32_t is_bigger = smallest > c_cell;
            std::uint32_t will_replace = is_not_diagonal_c & is_not_diagonal_a & is_not_diagonal_b & is_bigger;
            // On Kepler an newer we can use `__funnelshift_lc` to avoid branches
            c_cell = __funnelshift_lc(c_cell, smallest, will_replace - 1);
        }
        else c_cell = umax(c_cell, smallest);
        if constexpr (phase_ == tile_phase_t::aliased_k) __syncthreads();
    }
}

#else

/**
 *  @brief Processes a tile of the preferences matrix for the block-parallel Schulze voting algorithm
 *      in CUDA or @b HIP.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @tparam phase_ Whether the tiles alias and whether the tile may straddle the diagonal.
 *
 *  Tile @p c is the output, @p a and @p b the inputs; @p bi and @p bj address a cell within a
 *  tile, and each @p *_row and @p *_col pair the tile's origin in the global matrix.
 */
template <std::uint32_t tile_size_, tile_phase_t phase_>
__forceinline__ __device__ void process_tile_cuda_(   //
    votes_count_tile<tile_size_>& c,                  //
    votes_count_tile<tile_size_> const& a,            //
    votes_count_tile<tile_size_> const& b,            //
    candidate_index_t bi, candidate_index_t bj,       //
    candidate_index_t c_row, candidate_index_t c_col, //
    candidate_index_t a_row, candidate_index_t a_col, //
    candidate_index_t b_row, candidate_index_t b_col) {

    votes_count_t& c_cell = c[bi][bj];

#pragma unroll tile_size_
    for (candidate_index_t k = 0; k < tile_size_; k++) {
        votes_count_t smallest = min(a[bi][k], b[k][bj]);
        if constexpr (phase_ != tile_phase_t::distinct_independent_k) {
            std::uint32_t is_not_diagonal_c = (c_row + bi) != (c_col + bj);
            std::uint32_t is_not_diagonal_a = (a_row + bi) != (a_col + k);
            std::uint32_t is_not_diagonal_b = (b_row + k) != (b_col + bj);
            std::uint32_t is_bigger = smallest > c_cell;
            std::uint32_t will_replace = is_not_diagonal_c & is_not_diagonal_a & is_not_diagonal_b & is_bigger;
            if (will_replace) c_cell = smallest;
        }
        else c_cell = max(c_cell, smallest);
        if constexpr (phase_ == tile_phase_t::aliased_k) __syncthreads();
    }
}

#endif

/**
 *  @brief Performs the diagonal step of the block-parallel Schulze voting algorithm in CUDA or @b HIP.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @param[in] n The number of candidates.
 *  @param[in] k The index of the current tile being processed.
 *  @param[inout] graph The graph of strongest paths.
 */
template <std::uint32_t tile_size_>
__global__ void cuda_diagonal_(candidate_index_t n, candidate_index_t k, votes_count_t* graph) {
    candidate_index_t const bi = threadIdx.y;
    candidate_index_t const bj = threadIdx.x;

    alignas(16) __shared__ votes_count_t c[tile_size_][tile_size_];
    c[bi][bj] = graph[k * tile_size_ * n + k * tile_size_ + bi * n + bj];

    __syncthreads();
    process_tile_cuda_<tile_size_, tile_phase_t::aliased_k>( //
        c, c, c, bi, bj,                                     //
        tile_size_ * k, tile_size_ * k,                      //
        tile_size_ * k, tile_size_ * k,                      //
        tile_size_ * k, tile_size_ * k                       //
    );

    graph[k * tile_size_ * n + k * tile_size_ + bi * n + bj] = c[bi][bj];
}

/**
 *  @brief Performs the partially independent step of the block-parallel Schulze voting algorithm in CUDA or @b HIP.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @param[in] n The number of candidates.
 *  @param[in] k The index of the current tile being processed.
 *  @param[inout] graph The graph of strongest paths.
 */
template <std::uint32_t tile_size_>
__global__ void cuda_partially_independent_(candidate_index_t n, candidate_index_t k, votes_count_t* graph) {
    candidate_index_t const i = blockIdx.x;
    candidate_index_t const bi = threadIdx.y;
    candidate_index_t const bj = threadIdx.x;

    if (i == k) return;

    alignas(16) __shared__ votes_count_tile<tile_size_> a;
    alignas(16) __shared__ votes_count_tile<tile_size_> b;
    alignas(16) __shared__ votes_count_tile<tile_size_> c;

    // Partially dependent phase (first of two)
    // Walking down within a group of adjacent columns
    c[bi][bj] = graph[i * tile_size_ * n + k * tile_size_ + bi * n + bj];
    b[bi][bj] = graph[k * tile_size_ * n + k * tile_size_ + bi * n + bj];

    __syncthreads();
    process_tile_cuda_<tile_size_, tile_phase_t::aliased_k>( //
        c, c, b, bi, bj,                                     //
        i * tile_size_, k * tile_size_,                      //
        i * tile_size_, k * tile_size_,                      //
        k * tile_size_, k * tile_size_);

    // Partially dependent phase (second of two)
    // Walking right within a group of adjacent rows
    __syncthreads();
    graph[i * tile_size_ * n + k * tile_size_ + bi * n + bj] = c[bi][bj];
    c[bi][bj] = graph[k * tile_size_ * n + i * tile_size_ + bi * n + bj];
    a[bi][bj] = graph[k * tile_size_ * n + k * tile_size_ + bi * n + bj];

    __syncthreads();
    process_tile_cuda_<tile_size_, tile_phase_t::aliased_k>( //
        c, a, c, bi, bj,                                     //
        k * tile_size_, i * tile_size_,                      //
        k * tile_size_, k * tile_size_,                      //
        k * tile_size_, i * tile_size_                       //
    );

    graph[k * tile_size_ * n + i * tile_size_ + bi * n + bj] = c[bi][bj];
}

/**
 *  @brief Performs then independent step of the block-parallel Schulze voting algorithm in CUDA or @b HIP.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @param[in] n The number of candidates.
 *  @param[in] k The index of the current tile being processed.
 *  @param[inout] graph The graph of strongest paths.
 */
template <std::uint32_t tile_size_>
__global__ void cuda_independent_(candidate_index_t n, candidate_index_t k, votes_count_t* graph) {
    candidate_index_t const j = blockIdx.x;
    candidate_index_t const i = blockIdx.y;
    candidate_index_t const bi = threadIdx.y;
    candidate_index_t const bj = threadIdx.x;

    if (i == k && j == k) return;

    alignas(16) __shared__ votes_count_tile<tile_size_> a;
    alignas(16) __shared__ votes_count_tile<tile_size_> b;
    alignas(16) __shared__ votes_count_tile<tile_size_> c;

    c[bi][bj] = graph[i * tile_size_ * n + j * tile_size_ + bi * n + bj];
    a[bi][bj] = graph[i * tile_size_ * n + k * tile_size_ + bi * n + bj];
    b[bi][bj] = graph[k * tile_size_ * n + j * tile_size_ + bi * n + bj];

    __syncthreads();
    if (i == j)
        process_tile_cuda_<tile_size_, tile_phase_t::distinct_diagonal_k>( //
            c, a, b, bi, bj,                                               //
            i * tile_size_, j * tile_size_,                                //
            i * tile_size_, k * tile_size_,                                //
            k * tile_size_, j * tile_size_                                 //
        );
    else
        process_tile_cuda_<tile_size_, tile_phase_t::distinct_independent_k>( //
            c, a, b, bi, bj,                                                  //
            i * tile_size_, j * tile_size_,                                   //
            i * tile_size_, k * tile_size_,                                   //
            k * tile_size_, j * tile_size_                                    //
        );

    graph[i * tile_size_ * n + j * tile_size_ + bi * n + bj] = c[bi][bj];
}

/**
 *  @brief Performs then independent step of the block-parallel Schulze voting algorithm in CUDA (NVIDIA Hopper only).
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @param[in] n The number of candidates.
 *  @param[in] k The index of the current tile being processed.
 *  @param[inout] graph The graph of strongest paths, as a @c CUtensorMap .
 *
 *  @note This kernel uses NVIDIA-specific Tensor Memory Access (TMA) and is not available on AMD GPUs.
 */
#if !defined(SCALING_ELECTIONS_WITH_HIP)
template <std::uint32_t tile_size_>
__global__ void cuda_independent_hopper_(candidate_index_t n, candidate_index_t k,
                                         __grid_constant__ CUtensorMap const graph) {
    candidate_index_t const j = blockIdx.x;
    candidate_index_t const i = blockIdx.y;
    candidate_index_t const bi = threadIdx.y;
    candidate_index_t const bj = threadIdx.x;

#if defined(SCALING_ELECTIONS_HOPPER)

    if (i == k && j == k) return;

    alignas(128) __shared__ votes_count_tile<tile_size_> a;
    alignas(128) __shared__ votes_count_tile<tile_size_> b;
    alignas(128) __shared__ votes_count_tile<tile_size_> c;

#pragma nv_diag_suppress static_var_with_dynamic_init
    // Initialize shared memory barrier with the number of threads participating in the barrier.
    __shared__ barrier_t bar;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // We have one thread per tile cell.
        init(&bar, tile_size_ * tile_size_);
        // Make initialized barrier visible in async proxy.
        cde::fence_proxy_async_shared_cta();
    }
    // Sync threads so initialized barrier is visible to all threads.
    __syncthreads();

    // Only the first thread in the tile invokes the bulk transfers.
    barrier_t::arrival_token token;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // Initiate three bulk tensor copies for different part of the graph.
        // The first coordinate is the column, as dimension 0 is the contiguous one.
        cde::cp_async_bulk_tensor_2d_global_to_shared(&c, &graph, j * tile_size_, i * tile_size_, bar);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&a, &graph, k * tile_size_, i * tile_size_, bar);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&b, &graph, j * tile_size_, k * tile_size_, bar);
        // Arrive on the barrier and tell how many bytes are expected to come in.
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(c) + sizeof(a) + sizeof(b));
    }
    else {
        // Other threads just arrive.
        token = bar.arrive(1);
    }

    // Past this point the three tiles are resident in shared memory.
    bar.wait(std::move(token));

    if (i == j)
        process_tile_cuda_<tile_size_, tile_phase_t::distinct_diagonal_k>( //
            c, a, b, bi, bj,                                               //
            i * tile_size_, j * tile_size_,                                //
            i * tile_size_, k * tile_size_,                                //
            k * tile_size_, j * tile_size_                                 //
        );
    else
        process_tile_cuda_<tile_size_, tile_phase_t::distinct_independent_k>( //
            c, a, b, bi, bj,                                                  //
            i * tile_size_, j * tile_size_,                                   //
            i * tile_size_, k * tile_size_,                                   //
            k * tile_size_, j * tile_size_                                    //
        );

    // Wait for shared memory writes to be visible to TMA engine.
    cde::fence_proxy_async_shared_cta();
    __syncthreads();
    // After `syncthreads`, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        cde::cp_async_bulk_tensor_2d_shared_to_global(&graph, j * tile_size_, i * tile_size_, &c);
        // Wait for TMA transfer to have finished reading shared memory.
        // Create a "bulk async-group" out of the previous bulk copy operation.
        cde::cp_async_bulk_commit_group();
        // Wait for the group to have completed reading from shared memory.
        cde::cp_async_bulk_wait_group_read<0>();

        // The barrier needs no explicit destruction at the end of the kernel.
    }
#else
    // This is a trap :)
    if (i == 0 && j == 0 && bi == 0 && bj == 0) printf("This kernel is only supported on Hopper and newer GPUs\n");
#endif
}
#endif // !defined(SCALING_ELECTIONS_WITH_HIP)

#if !defined(SCALING_ELECTIONS_WITH_HIP)
PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    // Get pointer to cuGetProcAddress
    cudaDriverEntryPointQueryResult driver_status;
    void* cuGetProcAddress_ptr = nullptr;
    cudaError_t error = cudaGetDriverEntryPoint("cuGetProcAddress", &cuGetProcAddress_ptr, cudaEnableDefault,
                                                &driver_status);
    if (error != cudaSuccess) throw std::runtime_error("Failed to get cuGetProcAddress");
    if (driver_status != cudaDriverEntryPointSuccess)
        throw std::runtime_error("Failed to get cuGetProcAddress entry point");
    PFN_cuGetProcAddress_v12000 cuGetProcAddress = reinterpret_cast<PFN_cuGetProcAddress_v12000>(cuGetProcAddress_ptr);

    // Use cuGetProcAddress to get a pointer to the CTK 12.0 version of cuTensorMapEncodeTiled
    CUdriverProcAddressQueryResult symbol_status;
    void* cuTensorMapEncodeTiled_ptr = nullptr;
    CUresult encode_status = cuGetProcAddress("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000,
                                              CU_GET_PROC_ADDRESS_DEFAULT, &symbol_status);
    if (encode_status != CUDA_SUCCESS || symbol_status != CU_GET_PROC_ADDRESS_SUCCESS)
        throw std::runtime_error("Failed to get cuTensorMapEncodeTiled");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}
#endif // !defined(SCALING_ELECTIONS_WITH_HIP)

#if !defined(SCALING_ELECTIONS_WITH_HIP)

/** @brief Tensor map for the Hopper bulk-tensor path, absent when the device or the layout rules it out. */
using tma_descriptor_t = std::optional<CUtensorMap>;

/**
 *  @brief Builds the tensor map describing the padded strongest-paths matrix.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @param[in] graph The padded matrix of strongest paths.
 *  @param[in] graph_stride The row stride of the padded matrix.
 *  @param[in] device_properties Properties of the device the kernels will run on.
 */
template <std::uint32_t tile_size_>
tma_descriptor_t describe_tma_(votes_count_t* graph, candidate_index_t graph_stride,
                               cudaDeviceProp const& device_properties) {
    if (device_properties.major < 9) return std::nullopt;

    CUtensorMap descriptor_map {};

    // rank is the number of dimensions of the array.
    constexpr std::uint32_t rank = 2;
    uint64_t size[rank] = {graph_stride, graph_stride};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride[rank - 1] = {graph_stride * sizeof(votes_count_t)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    std::uint32_t box_size[rank] = {tile_size_, tile_size_};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    std::uint32_t elem_stride[rank] = {1, 1};

    // Create the tensor descriptor.
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
    PFN_cuTensorMapEncodeTiled_v12000 cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    CUresult encode_status = cuTensorMapEncodeTiled( //
        &descriptor_map,                             // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
        rank,        // cuuint32_t tensorRank,
        graph,       // void *globalAddress,
        size,        // const cuuint64_t *globalDim,
        stride,      // const cuuint64_t *globalStrides,
        box_size,    // const cuuint32_t *boxDim,
        elem_stride, // const cuuint32_t *elementStrides,
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
    if (encode_status != CUDA_SUCCESS) return std::nullopt;
    return descriptor_map;
}

/**
 *  @brief Launches the independent phase, preferring the bulk-tensor kernel where it is available.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @param[in] grid The grid shape covering every tile pair.
 *  @param[in] block The block shape, one thread per tile cell.
 *  @param[in] graph_stride The row stride of the padded matrix.
 *  @param[in] k The index of the current pivot tile.
 *  @param[inout] graph The padded matrix of strongest paths.
 *  @param[in] tma The tensor-map descriptor produced by @c require_tma_ .
 *  @param[in] backend Which family the caller asked for.
 */
template <std::uint32_t tile_size_>
void launch_independent_(dim3 grid, dim3 block, candidate_index_t graph_stride, candidate_index_t k,
                         votes_count_t* graph, tma_descriptor_t const& tma, backend_t backend) {
    if (backend == backend_t::gpu_hopper_k)
        cuda_independent_hopper_<tile_size_><<<grid, block>>>(graph_stride, k, *tma);
    else cuda_independent_<tile_size_><<<grid, block>>>(graph_stride, k, graph);
}

/** @brief Builds the descriptor the bulk-tensor path needs, throwing when the device cannot supply one. */
template <std::uint32_t tile_size_>
tma_descriptor_t require_tma_(votes_count_t* graph, candidate_index_t graph_stride,
                              cudaDeviceProp const& device_properties) {
    tma_descriptor_t descriptor = describe_tma_<tile_size_>(graph, graph_stride, device_properties);
    if (!descriptor.has_value())
        throw std::runtime_error("The `gpu_hopper` backend needs compute capability 9.0, found " +
                                 std::to_string(device_properties.major) + "." +
                                 std::to_string(device_properties.minor));
    return descriptor;
}

#else

/** @brief HIP has no bulk-tensor engine, so the descriptor type can never hold one. */
using tma_descriptor_t = std::nullopt_t;

template <std::uint32_t tile_size_>
tma_descriptor_t describe_tma_(votes_count_t*, candidate_index_t, cudaDeviceProp const&) {
    return std::nullopt;
}

template <std::uint32_t tile_size_>
void launch_independent_(dim3 grid, dim3 block, candidate_index_t graph_stride, candidate_index_t k,
                         votes_count_t* graph, tma_descriptor_t const&, backend_t) {
    cuda_independent_<tile_size_><<<grid, block>>>(graph_stride, k, graph);
}

/** @brief HIP has no bulk-tensor engine, so the descriptor can never be built. */
template <std::uint32_t tile_size_>
tma_descriptor_t require_tma_(votes_count_t*, candidate_index_t, cudaDeviceProp const&) {
    throw std::runtime_error("The `gpu_hopper` backend is unavailable in a HIP build");
}

#endif

/**
 *  @brief Computes the strongest paths for the block-parallel Schulze voting algorithm in CUDA or @b HIP.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @param[in] preferences The preferences matrix.
 *  @param[in] num_candidates The number of candidates.
 *  @param[in] row_stride The stride between rows in the preferences matrix.
 *  @param[out] graph The output matrix of strongest paths.
 *  @param[in] graph_stride The row stride of the padded output.
 *  @param[in] backend Which GPU family to launch.
 */
template <std::uint32_t tile_size_> //
void compute_strongest_paths_cuda(  //
    votes_count_t* preferences, candidate_index_t num_candidates, candidate_index_t row_stride, votes_count_t* graph,
    candidate_index_t graph_stride, backend_t backend) {

#if defined(SCALING_ELECTIONS_WITH_OPENMP)
#pragma omp parallel for collapse(2)
#endif
    for (candidate_index_t i = 0; i < num_candidates; i++)
        for (candidate_index_t j = 0; j < num_candidates; j++)
            graph[i * graph_stride + j] = i != j && preferences[i * row_stride + j] > preferences[j * row_stride + i]
                                              ? preferences[i * row_stride + j]
                                              : 0;

    // Check if we can use newer CUDA features.
    cudaError_t error;
    int current_device;
    cudaDeviceProp device_properties;
    error = cudaGetDevice(&current_device);
    if (error != cudaSuccess) throw std::runtime_error("Failed to get current device");
    error = cudaGetDeviceProperties(&device_properties, current_device);
    if (error != cudaSuccess) throw std::runtime_error("Failed to get device properties");

    tma_descriptor_t const tma = backend == backend_t::gpu_hopper_k
                                     ? require_tma_<tile_size_>(graph, graph_stride, device_properties)
                                     : tma_descriptor_t {std::nullopt};

    candidate_index_t tiles_count = graph_stride / tile_size_;
    dim3 tile_shape(tile_size_, tile_size_, 1);
    dim3 independent_grid(tiles_count, tiles_count, 1);
    for (candidate_index_t k = 0; k < tiles_count; k++) {
        cuda_diagonal_<tile_size_><<<1, tile_shape>>>(graph_stride, k, graph);
        cuda_partially_independent_<tile_size_><<<tiles_count, tile_shape>>>(graph_stride, k, graph);
        launch_independent_<tile_size_>(independent_grid, tile_shape, graph_stride, k, graph, tma, backend);

        error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }
}

#endif // defined(SCALING_ELECTIONS_WITH_CUDA)

#pragma endregion CUDA

#pragma region OpenMP

#if defined(SCALING_ELECTIONS_WITH_OPENMP)

/** @brief Fixes the team size for one call, restoring the runtime's prior settings on scope exit. */
struct openmp_team_t {
    int const previous_threads;
    int const previous_dynamic;

    explicit openmp_team_t(unsigned threads)
        : previous_threads(omp_get_max_threads()), previous_dynamic(omp_get_dynamic()) {
        omp_set_dynamic(0);
        // `hardware_concurrency` is permitted to answer zero, which OpenMP rejects.
        if (threads > 0) omp_set_num_threads(static_cast<int>(threads));
    }
    ~openmp_team_t() noexcept {
        omp_set_num_threads(previous_threads);
        omp_set_dynamic(previous_dynamic);
    }
    openmp_team_t(openmp_team_t const&) = delete;
    openmp_team_t& operator=(openmp_team_t const&) = delete;
};

#endif
/**
 *  @brief Processes a tile of the preferences matrix for the block-parallel Schulze
 *      voting algorithm on CPU using @b OpenMP.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @tparam phase_ Whether the tile may straddle the matrix diagonal.
 *
 *  Tile @p c is the output, @p a and @p b the inputs, and each @p *_row and @p *_col pair the
 *  tile's origin in the global matrix. Every cell is walked serially, so an aliased phase needs
 *  no barrier here.
 */
template <std::uint32_t tile_size_, tile_phase_t phase_>
inline void process_tile_openmp_(                     //
    votes_count_tile<tile_size_>& c,                  //
    votes_count_tile<tile_size_> const& a,            //
    votes_count_tile<tile_size_> const& b,            //
    candidate_index_t c_row, candidate_index_t c_col, //
    candidate_index_t a_row, candidate_index_t a_col, //
    candidate_index_t b_row, candidate_index_t b_col) {

#if defined(SCALING_ELECTIONS_WITH_NEON)
    if constexpr (std::is_same<votes_count_t, std::uint32_t>() && tile_size_ % 4 == 0) {
        uint32x4_t bj_step = {0, 1, 2, 3};
        for (candidate_index_t k = 0; k < tile_size_; k++) {
            uint32x4_t b_row_plus_k_vec = vdupq_n_u32(b_row + k);
            for (candidate_index_t bi = 0; bi < tile_size_; bi++) {
                uint32x4_t a_vec = vdupq_n_u32(a[bi][k]);
                uint32x4_t is_not_diagonal_a = vmvnq_u32(vceqq_u32(vdupq_n_u32(a_row + bi), vdupq_n_u32(a_col + k)));
                uint32x4_t c_row_plus_bi_vec = vdupq_n_u32(c_row + bi);
#if defined(__clang__) // Apple's Clang can't handle `#pragma unroll`
#pragma clang loop unroll(full)
#else
#pragma unroll full
#endif
                for (candidate_index_t bj = 0; bj < tile_size_; bj += 4) {
                    votes_count_t* c_ptr = &c[bi][bj];
                    uint32x4_t c_vec = vld1q_u32(c_ptr);
                    uint32x4_t b_vec = vld1q_u32(&b[k][bj]);
                    uint32x4_t smallest = vminq_u32(a_vec, b_vec);

                    if constexpr (phase_ != tile_phase_t::distinct_independent_k) {
                        uint32x4_t is_diagonal_c =       //
                            vceqq_u32(c_row_plus_bi_vec, //
                                      vaddq_u32(vdupq_n_u32(c_col + bj), bj_step));
                        uint32x4_t is_diagonal_b =      //
                            vceqq_u32(b_row_plus_k_vec, //
                                      vaddq_u32(vdupq_n_u32(b_col + bj), bj_step));
                        uint32x4_t is_bigger = vcgtq_u32(smallest, c_vec);
                        uint32x4_t will_replace =                                   //
                            vandq_u32(                                              //
                                vmvnq_u32(vorrq_u32(is_diagonal_c, is_diagonal_b)), //
                                vandq_u32(is_not_diagonal_a, is_bigger));
                        c_vec = vbslq_u32(will_replace, smallest, c_vec);
                    }
                    else { c_vec = vmaxq_u32(c_vec, smallest); }
                    vst1q_u32(c_ptr, c_vec);
                }
            }
        }
        return;
    }
#endif
    for (candidate_index_t k = 0; k < tile_size_; k++) {
        for (candidate_index_t bi = 0; bi < tile_size_; bi++) {
            votes_count_t* const c_cells = &c[bi][0];
#pragma omp simd
            for (candidate_index_t bj = 0; bj < tile_size_; bj++) {
                votes_count_t c_cell = c_cells[bj];
                votes_count_t smallest = std::min(a[bi][k], b[k][bj]);
                if constexpr (phase_ != tile_phase_t::distinct_independent_k) {
                    std::uint32_t is_not_diagonal_c = (c_row + bi) != (c_col + bj);
                    std::uint32_t is_not_diagonal_a = (a_row + bi) != (a_col + k);
                    std::uint32_t is_not_diagonal_b = (b_row + k) != (b_col + bj);
                    std::uint32_t is_bigger = smallest > c_cell;
                    std::uint32_t will_replace = is_not_diagonal_c & is_not_diagonal_a & is_not_diagonal_b & is_bigger;
                    c_cells[bj] = will_replace ? smallest : c_cell;
                }
                else { c_cells[bj] = std::max(c_cell, smallest); }
            }
        }
    }
}

template <std::uint32_t tile_size_, tile_march_t march_ = tile_march_t::fast_k>
void memcpy2d(votes_count_t const* source, candidate_index_t stride, votes_count_tile<tile_size_>& target,
              candidate_index_t remaining_rows, candidate_index_t remaining_cols) {

#if defined(SCALING_ELECTIONS_WITH_NEON)
    if constexpr (std::is_same<votes_count_t, std::uint32_t>() && tile_size_ % 4 == 0 &&
                  march_ == tile_march_t::fast_k) {
        for (candidate_index_t i = 0; i < tile_size_; i++) {
#if defined(__clang__) // Apple's Clang can't handle `#pragma unroll`
#pragma clang loop unroll(full)
#else
#pragma unroll full
#endif
            for (candidate_index_t j = 0; j < tile_size_; j += 4) {
                vst1q_u32(&target[i][j], vld1q_u32(&source[i * stride + j]));
            }
        }
        return;
    }
#endif

    for (candidate_index_t i = 0; i < tile_size_; i++)
        for (candidate_index_t j = 0; j < tile_size_; j++)
            if constexpr (march_ == tile_march_t::checked_k)
                target[i][j] = i < remaining_rows && j < remaining_cols ? source[i * stride + j] : 0;
            else target[i][j] = source[i * stride + j];
}

template <std::uint32_t tile_size_, tile_march_t march_ = tile_march_t::fast_k>
void memcpy2d(votes_count_tile<tile_size_> const& source, candidate_index_t stride, votes_count_t* target,
              candidate_index_t remaining_rows, candidate_index_t remaining_cols) {

#if defined(SCALING_ELECTIONS_WITH_NEON)
    if constexpr (std::is_same<votes_count_t, std::uint32_t>() && tile_size_ % 4 == 0 &&
                  march_ == tile_march_t::fast_k) {
        for (candidate_index_t i = 0; i < tile_size_; i++) {
#if defined(__clang__) // Apple's Clang can't handle `#pragma unroll`
#pragma clang loop unroll(full)
#else
#pragma unroll full
#endif
            for (candidate_index_t j = 0; j < tile_size_; j += 4) {
                vst1q_u32(&target[i * stride + j], vld1q_u32(&source[i][j]));
            }
        }
        return;
    }
#endif
    for (candidate_index_t i = 0; i < tile_size_; i++)
        for (candidate_index_t j = 0; j < tile_size_; j++)
            if constexpr (march_ == tile_march_t::checked_k) {
                if (i < remaining_rows && j < remaining_cols) target[i * stride + j] = source[i][j];
            }
            else target[i * stride + j] = source[i][j];
}

template <std::uint32_t tile_size_, tile_march_t march_ = tile_march_t::fast_k> //
void compute_strongest_paths_openmp(                                            //
    votes_count_t* preferences, candidate_index_t num_candidates, candidate_index_t row_stride, votes_count_t* graph) {

#pragma omp parallel for schedule(dynamic) collapse(2)
    // Populate the strongest paths matrix based on direct comparisons
    for (candidate_index_t i = 0; i < num_candidates; i++)
        for (candidate_index_t j = 0; j < num_candidates; j++)
            if (i != j)
                graph[i * num_candidates + j] =                                       //
                    preferences[i * row_stride + j] > preferences[j * row_stride + i] //
                        ? preferences[i * row_stride + j]
                        : 0;
            else graph[i * num_candidates + j] = 0;

    // Time for the actual core implementation
    candidate_index_t const tiles_count = (num_candidates + tile_size_ - 1) / tile_size_;
    for (candidate_index_t k = 0; k < tiles_count; k++) {

        if (global_signal_status != 0) throw std::runtime_error("Stopped by signal");

        // Dependent phase
        {
            alignas(64) votes_count_t c[tile_size_][tile_size_];
            memcpy2d<tile_size_, march_>(graph + k * tile_size_ * num_candidates + k * tile_size_, num_candidates, c,
                                         num_candidates - k * tile_size_, num_candidates - k * tile_size_);
            process_tile_openmp_<tile_size_, tile_phase_t::aliased_k>( //
                c, c, c,                                               //
                tile_size_ * k, tile_size_ * k,                        //
                tile_size_ * k, tile_size_ * k,                        //
                tile_size_ * k, tile_size_ * k                         //
            );
            memcpy2d<tile_size_, march_>(c, num_candidates, graph + k * tile_size_ * num_candidates + k * tile_size_,
                                         num_candidates - k * tile_size_, num_candidates - k * tile_size_);
        }
        // Partially dependent phase (first of two)
#pragma omp parallel for schedule(dynamic)
        for (candidate_index_t i = 0; i < tiles_count; i++) {
            if (i == k) continue;
            alignas(64) votes_count_tile<tile_size_> b;
            alignas(64) votes_count_tile<tile_size_> c;
            memcpy2d<tile_size_, march_>(graph + i * tile_size_ * num_candidates + k * tile_size_, num_candidates, c,
                                         num_candidates - i * tile_size_, num_candidates - k * tile_size_);
            memcpy2d<tile_size_, march_>(graph + k * tile_size_ * num_candidates + k * tile_size_, num_candidates, b,
                                         num_candidates - k * tile_size_, num_candidates - k * tile_size_);
            process_tile_openmp_<tile_size_, tile_phase_t::aliased_k>( //
                c, c, b,                                               //
                i * tile_size_, k * tile_size_,                        //
                i * tile_size_, k * tile_size_,                        //
                k * tile_size_, k * tile_size_);
            memcpy2d<tile_size_, march_>(c, num_candidates, graph + i * tile_size_ * num_candidates + k * tile_size_,
                                         num_candidates - i * tile_size_, num_candidates - k * tile_size_);
        }
        // Partially dependent phase (second of two)
#pragma omp parallel for schedule(dynamic)
        for (candidate_index_t j = 0; j < tiles_count; j++) {
            if (j == k) continue;
            alignas(64) votes_count_tile<tile_size_> a;
            alignas(64) votes_count_tile<tile_size_> c;
            memcpy2d<tile_size_, march_>(graph + k * tile_size_ * num_candidates + j * tile_size_, num_candidates, c,
                                         num_candidates - k * tile_size_, num_candidates - j * tile_size_);
            memcpy2d<tile_size_, march_>(graph + k * tile_size_ * num_candidates + k * tile_size_, num_candidates, a,
                                         num_candidates - k * tile_size_, num_candidates - k * tile_size_);
            process_tile_openmp_<tile_size_, tile_phase_t::aliased_k>( //
                c, a, c,                                               //
                k * tile_size_, j * tile_size_,                        //
                k * tile_size_, k * tile_size_,                        //
                k * tile_size_, j * tile_size_                         //
            );
            memcpy2d<tile_size_, march_>(c, num_candidates, graph + k * tile_size_ * num_candidates + j * tile_size_,
                                         num_candidates - k * tile_size_, num_candidates - j * tile_size_);
        }
        // Independent phase
#pragma omp parallel for schedule(dynamic) collapse(2)
        for (candidate_index_t i = 0; i < tiles_count; i++) {
            for (candidate_index_t j = 0; j < tiles_count; j++) {
                if (i == k || j == k) continue;
                alignas(64) votes_count_tile<tile_size_> a;
                alignas(64) votes_count_tile<tile_size_> b;
                alignas(64) votes_count_tile<tile_size_> c;
                memcpy2d<tile_size_, march_>(graph + i * tile_size_ * num_candidates + j * tile_size_, num_candidates,
                                             c, num_candidates - i * tile_size_, num_candidates - j * tile_size_);
                memcpy2d<tile_size_, march_>(graph + i * tile_size_ * num_candidates + k * tile_size_, num_candidates,
                                             a, num_candidates - i * tile_size_, num_candidates - k * tile_size_);
                memcpy2d<tile_size_, march_>(graph + k * tile_size_ * num_candidates + j * tile_size_, num_candidates,
                                             b, num_candidates - k * tile_size_, num_candidates - j * tile_size_);
                if (i != j)
                    process_tile_openmp_<tile_size_, tile_phase_t::distinct_independent_k>( //
                        c, a, b,                                                            //
                        i * tile_size_, j * tile_size_,                                     //
                        i * tile_size_, k * tile_size_,                                     //
                        k * tile_size_, j * tile_size_                                      //
                    );
                else
                    process_tile_openmp_<tile_size_, tile_phase_t::distinct_diagonal_k>( //
                        c, a, b,                                                         //
                        i * tile_size_, j * tile_size_,                                  //
                        i * tile_size_, k * tile_size_,                                  //
                        k * tile_size_, j * tile_size_                                   //
                    );
                memcpy2d<tile_size_, march_>(c, num_candidates,
                                             graph + i * tile_size_ * num_candidates + j * tile_size_,
                                             num_candidates - i * tile_size_, num_candidates - j * tile_size_);
            }
        }
    }
}

#pragma endregion OpenMP

#pragma region Python bindings
#if !defined(SCALING_ELECTIONS_TEST)

/**
 *  @brief Computes the strongest paths for the block-parallel Schulze voting algorithm.
 *
 *  @param[in] preferences The preferences matrix.
 *  @param[in] backend_name One of `cpu_openmp`, `gpu_serial`, or `gpu_hopper`.
 *  @param[in] tile_size_ The tile edge, or zero to pick the largest that fits.
 *  @return A NumPy array containing the strongest paths matrix.
 *
 *  @note A backend the build or the device cannot serve raises rather than downgrading.
 */
static py::array_t<votes_count_t> compute_strongest_paths(      //
    py::array_t<votes_count_t, py::array::c_style> preferences, //
    std::string_view backend_name, std::size_t tile_size = 0) {

    backend_t const backend = backend_from_name(backend_name);

    auto buffer = preferences.request();
    if (buffer.ndim != 2) throw std::runtime_error("Number of dimensions must be two");
    if (buffer.shape[0] != buffer.shape[1]) throw std::runtime_error("Preferences matrix must be square");
    auto preferences_ptr = reinterpret_cast<votes_count_t*>(buffer.ptr);
    auto num_candidates = static_cast<candidate_index_t>(buffer.shape[0]);
    auto row_stride = static_cast<candidate_index_t>(buffer.strides[0] / sizeof(votes_count_t));

    // Allocate NumPy array for the result
    auto result = py::array_t<votes_count_t>({num_candidates, num_candidates});
    auto result_buf = result.request();
    auto result_ptr = reinterpret_cast<votes_count_t*>(result_buf.ptr);
    auto result_row_stride = static_cast<candidate_index_t>(result_buf.strides[0] / sizeof(votes_count_t));
    if (result_row_stride != num_candidates) throw std::runtime_error("Result matrix must be contiguous");

#if defined(SCALING_ELECTIONS_WITH_CUDA)

    if (backend != backend_t::cpu_openmp_k) {
        if (tile_size == 0) tile_size = 32;
        // Validate before reserving, so an unsupported size never allocates.
        if (!gpu_tiles_t::contains(tile_size)) throw std::runtime_error("Unsupported tile size");

        // Rounding the matrix up to a whole number of tiles keeps the kernels free of tail
        // checks: the padding is zero, which is the identity of the max-min semiring.
        candidate_index_t const graph_stride = (num_candidates + tile_size - 1) / tile_size * tile_size;
        std::size_t const graph_bytes = static_cast<std::size_t>(graph_stride) * graph_stride * sizeof(votes_count_t);

        managed_graph_t const graph(graph_bytes);
        cudaError_t error = cudaMemset(graph.pointer, 0, graph_bytes);
        if (error != cudaSuccess) throw std::runtime_error("Failed to clear device memory");
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) throw std::runtime_error("Failed to clear device memory");

        gpu_tiles_t::dispatch(tile_size, [&](auto tile) {
            compute_strongest_paths_cuda<tile.value>(preferences_ptr, num_candidates, row_stride, graph.pointer,
                                                     graph_stride, backend);
        });

        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) throw std::runtime_error("CUDA operations did not complete successfully");

        // Copy the leading sub-block back, dropping the padding.
        error = cudaMemcpy2D(result_ptr, num_candidates * sizeof(votes_count_t), graph.pointer,
                             graph_stride * sizeof(votes_count_t), num_candidates * sizeof(votes_count_t),
                             num_candidates, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) throw std::runtime_error("Failed to copy data from device to host");

        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) throw std::runtime_error("CUDA transfers did not complete successfully");
        return result;
    }

#else

    if (backend != backend_t::cpu_openmp_k) throw std::runtime_error("This build has no GPU support compiled in");

#endif // defined(SCALING_ELECTIONS_WITH_CUDA)

#if defined(SCALING_ELECTIONS_WITH_OPENMP)
    openmp_team_t const team(std::thread::hardware_concurrency());
#endif

    // Probe for the largest possible tile size, if not previously specified
    if (tile_size == 0) tile_size = cpu_tiles_t::largest_fitting(num_candidates);
    if (tile_size == 0) throw std::runtime_error("Number of candidates should be at least 4, ideally divisible by 4");
    if (tile_size > num_candidates)
        throw std::runtime_error("Tile size should be less than or equal to the number of candidates");

    bool const dispatched = cpu_tiles_t::dispatch(tile_size, [&](auto tile) {
        if (num_candidates % tile.value == 0)
            compute_strongest_paths_openmp<tile.value, tile_march_t::fast_k>( //
                preferences_ptr, num_candidates, row_stride, result_ptr);
        else
            compute_strongest_paths_openmp<tile.value, tile_march_t::checked_k>( //
                preferences_ptr, num_candidates, row_stride, result_ptr);
    });
    if (!dispatched) throw std::runtime_error("Unsupported tile size");
    return result;
}

PYBIND11_MODULE(scaling_elections, m) {

    std::signal(SIGINT, signal_handler);

    // Let's show how to wrap `void` functions for basic logging
    m.def("log_gpus", []() {
#if defined(SCALING_ELECTIONS_WITH_CUDA)
        int device_count;
        cudaDeviceProp device_properties;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess) throw std::runtime_error("Failed to get device count");
        for (int i = 0; i < device_count; i++) {
            error = cudaGetDeviceProperties(&device_properties, i);
            if (error != cudaSuccess) throw std::runtime_error("Failed to get device properties");
            std::printf("Device %d: %s\n", i, device_properties.name);
            std::printf("\tSMs: %d\n", device_properties.multiProcessorCount);
            std::printf("\tGlobal mem: %.2fGB\n",
                        static_cast<float>(device_properties.totalGlobalMem) / (1024 * 1024 * 1024));
            std::printf("\tCUDA Cap: %d.%d\n", device_properties.major, device_properties.minor);
        }
#else
        throw std::runtime_error("No CUDA devices available\n");
#endif
    });

    // This is how we could have used `thrust::` for higher-level operations
    m.def("reduce", [](py::array_t<float> const& data) -> float {
#if defined(SCALING_ELECTIONS_WITH_CUDA) && !defined(SCALING_ELECTIONS_WITH_HIP)
        // Thrust support - CUDA only (rocThrust not guaranteed to be available)
        py::buffer_info buffer = data.request();
        if (buffer.ndim != 1 || buffer.strides[0] != sizeof(float))
            throw std::runtime_error("Input should be a contiguous 1D float array");
        float* ptr = static_cast<float*>(buffer.ptr);
        thrust::device_vector<float> d_data(ptr, ptr + buffer.size);
        return thrust::reduce(thrust::device, d_data.begin(), d_data.end(), 0.0f);
#else
        // CPU fallback for HIP and non-CUDA builds
        return std::accumulate(data.data(), data.data() + data.size(), 0.0f);
#endif
    });

    m.def("compute_strongest_paths", &compute_strongest_paths, //
          py::arg("preferences"), py::kw_only(),               //
          py::arg("backend") = "cpu_openmp",                   //
          py::arg("tile_size") = 0);
}

#endif // !defined(SCALING_ELECTIONS_TEST)
#pragma endregion Python bindings

#if defined(SCALING_ELECTIONS_TEST)

int main() {

    std::size_t num_candidates = 256;
    std::vector<votes_count_t> preferences(num_candidates * num_candidates);
    std::generate(preferences.begin(), preferences.end(),
                  [=]() { return static_cast<votes_count_t>(std::rand() % num_candidates); });

    std::vector<votes_count_t> graph(num_candidates * num_candidates);
    compute_strongest_paths_openmp<64, tile_march_t::fast_k>( //
        preferences.data(), num_candidates, num_candidates, graph.data());

    return 0;
}

#endif // defined(SCALING_ELECTIONS_TEST)
