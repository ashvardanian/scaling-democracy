/**
 *  @brief Block-parallel Schulze strongest-paths kernels for CUDA, HIP, and OpenMP.
 *  @file schulze.cuh
 *  @author Ash Vardanian
 *  @date July 12, 2024
 *  @see https://ashvardanian.com/posts/scaling-elections
 */
#pragma once
#include "ballots.cuh"
#include "types.cuh"

template <std::uint32_t tile_size_, typename element_type_ = votes_count_t>
using votes_count_tile = element_type_[tile_size_][tile_size_];

/**
 *  @brief Which tile phase a processor runs, fixing both buffer aliasing and diagonal handling.
 *
 *  The three enumerators are the only combinations the recurrence produces, so the fourth
 *  pairing of the underlying flags cannot be spelled.
 */
enum class tile_phase_t : std::uint8_t {
    /** Output and inputs share one buffer, so every step needs a barrier. */
    aliased_k,
    /** Separate buffers, but the tile may straddle the matrix diagonal. */
    distinct_diagonal_k,
    /** Separate buffers, provably off the diagonal, so the update is a plain maximum. */
    distinct_independent_k,
};

/** Whether a tile copy bounds-checks its edges; `checked_k` also disables the NEON path. */
enum class tile_march_t : bool { fast_k = true, checked_k = false };

/** Which family computes the strongest paths. */
enum class backend_t : std::uint8_t {
    /** Tiled CPU kernels across an OpenMP team. */
    cpu_openmp_k,
    /** Tiled GPU kernels staging tiles through per-thread loads, on CUDA or HIP. */
    gpu_serial_k,
    /** Tiled GPU kernels staging tiles through the bulk-tensor engine, NVIDIA sm_90 and newer. */
    gpu_hopper_k,
};

#pragma region CUDA

#if defined(SCALING_ELECTIONS_WITH_CUDA)

#if !defined(SCALING_ELECTIONS_WITH_HIP)
namespace cde = cuda::device::experimental;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;
#endif

/** Owns one managed reservation for the padded strongest-paths matrix. */
template <typename element_type_>
struct managed_graph {
    element_type_* pointer = nullptr;

    explicit managed_graph(std::size_t bytes) {
        if (cudaMallocManaged(&pointer, bytes) != cudaSuccess)
            throw std::runtime_error("Failed to allocate memory on device");
    }
    ~managed_graph() noexcept {
        if (pointer) cudaFree(pointer);
    }
    managed_graph(managed_graph const&) = delete;
    managed_graph& operator=(managed_graph const&) = delete;
};

using managed_graph_t = managed_graph<votes_count_t>;

#if defined(SCALING_ELECTIONS_KEPLER)

/**
 *  @brief Processes a tile of the preferences matrix for the block-parallel Schulze voting algorithm
 *      in CUDA on Nvidia @b Kepler GPUs and newer (sm_30).
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @tparam phase_ Whether the tiles alias and whether the tile may straddle the diagonal.
 *  @tparam element_type_ The width one vote count occupies, deduced from the tiles.
 *
 *  Tile @p c is the output, @p a and @p b the inputs; @p bi and @p bj address a cell within a
 *  tile, and each @p *_row and @p *_col pair the tile's origin in the global matrix.
 */
template <std::uint32_t tile_size_, tile_phase_t phase_, typename element_type_>
__forceinline__ __device__ void process_tile_cuda_(       //
    votes_count_tile<tile_size_, element_type_>& c,       //
    votes_count_tile<tile_size_, element_type_> const& a, //
    votes_count_tile<tile_size_, element_type_> const& b, //
    candidate_index_t bi, candidate_index_t bj,           //
    candidate_index_t c_row, candidate_index_t c_col,     //
    candidate_index_t a_row, candidate_index_t a_col,     //
    candidate_index_t b_row, candidate_index_t b_col) {

    element_type_& c_cell = c[bi][bj];

#pragma unroll tile_size_
    for (candidate_index_t k = 0; k < tile_size_; k++) {
        element_type_ smallest = umin(a[bi][k], b[k][bj]);
        if constexpr (phase_ != tile_phase_t::distinct_independent_k) {
            std::uint32_t is_not_diagonal_c = (c_row + bi) != (c_col + bj);
            std::uint32_t is_not_diagonal_a = (a_row + bi) != (a_col + k);
            std::uint32_t is_not_diagonal_b = (b_row + k) != (b_col + bj);
            std::uint32_t is_bigger = smallest > c_cell;
            std::uint32_t will_replace = is_not_diagonal_c & is_not_diagonal_a & is_not_diagonal_b & is_bigger;
            // On Kepler an newer we can use `__funnelshift_lc` to avoid branches
            c_cell = static_cast<element_type_>(__funnelshift_lc(c_cell, smallest, will_replace - 1));
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
 *  @tparam element_type_ The width one vote count occupies, deduced from the tiles.
 *
 *  Tile @p c is the output, @p a and @p b the inputs; @p bi and @p bj address a cell within a
 *  tile, and each @p *_row and @p *_col pair the tile's origin in the global matrix.
 */
template <std::uint32_t tile_size_, tile_phase_t phase_, typename element_type_>
__forceinline__ __device__ void process_tile_cuda_(       //
    votes_count_tile<tile_size_, element_type_>& c,       //
    votes_count_tile<tile_size_, element_type_> const& a, //
    votes_count_tile<tile_size_, element_type_> const& b, //
    candidate_index_t bi, candidate_index_t bj,           //
    candidate_index_t c_row, candidate_index_t c_col,     //
    candidate_index_t a_row, candidate_index_t a_col,     //
    candidate_index_t b_row, candidate_index_t b_col) {

    element_type_& c_cell = c[bi][bj];

#pragma unroll tile_size_
    for (candidate_index_t k = 0; k < tile_size_; k++) {
        element_type_ smallest = min(a[bi][k], b[k][bj]);
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
 *  @tparam element_type_ The width one vote count occupies.
 *  @param[in] n The number of candidates.
 *  @param[in] k The index of the current tile being processed.
 *  @param[inout] graph The graph of strongest paths.
 */
template <std::uint32_t tile_size_, typename element_type_ = votes_count_t>
__global__ void cuda_diagonal_(candidate_index_t n, candidate_index_t k, element_type_* graph) {
    candidate_index_t const bi = threadIdx.y;
    candidate_index_t const bj = threadIdx.x;

    alignas(16) __shared__ votes_count_tile<tile_size_, element_type_> c;
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
 *  @tparam element_type_ The width one vote count occupies.
 *  @param[in] n The number of candidates.
 *  @param[in] k The index of the current tile being processed.
 *  @param[inout] graph The graph of strongest paths.
 */
template <std::uint32_t tile_size_, typename element_type_ = votes_count_t>
__global__ void cuda_partially_independent_(candidate_index_t n, candidate_index_t k, element_type_* graph) {
    candidate_index_t const i = blockIdx.x;
    candidate_index_t const bi = threadIdx.y;
    candidate_index_t const bj = threadIdx.x;

    if (i == k) return;

    alignas(16) __shared__ votes_count_tile<tile_size_, element_type_> a;
    alignas(16) __shared__ votes_count_tile<tile_size_, element_type_> b;
    alignas(16) __shared__ votes_count_tile<tile_size_, element_type_> c;

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
 *  @tparam element_type_ The width one vote count occupies.
 *  @param[in] n The number of candidates.
 *  @param[in] k The index of the current tile being processed.
 *  @param[inout] graph The graph of strongest paths.
 */
template <std::uint32_t tile_size_, typename element_type_ = votes_count_t>
__global__ void cuda_independent_(candidate_index_t n, candidate_index_t k, element_type_* graph) {
    candidate_index_t const j = blockIdx.x;
    candidate_index_t const i = blockIdx.y;
    candidate_index_t const bi = threadIdx.y;
    candidate_index_t const bj = threadIdx.x;

    if (i == k && j == k) return;

    alignas(16) __shared__ votes_count_tile<tile_size_, element_type_> a;
    alignas(16) __shared__ votes_count_tile<tile_size_, element_type_> b;
    alignas(16) __shared__ votes_count_tile<tile_size_, element_type_> c;

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

#pragma region Packed Sixteen Bit

/** Which element width the pivot sweep runs on. */
enum class graph_width_t : std::uint8_t {
    /** The 32-bit path, which every device and every backend supports. */
    wide_32_k,
    /** The 16-bit path, two candidates per word, needing every vote count below 65536. */
    narrow_16_k,
};

/** Copies the graph into a narrower shadow, raising @p overflowed for any cell that will not fit. */
template <typename wide_type_, typename narrow_type_>
__global__ void cuda_narrow_(std::size_t cells, wide_type_ const* wide, narrow_type_* narrow, wide_type_* overflowed) {
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    std::size_t const first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    constexpr wide_type_ widest = static_cast<narrow_type_>(~static_cast<narrow_type_>(0));
    for (std::size_t cell = first; cell < cells; cell += stride) {
        wide_type_ const value = wide[cell];
        if (value > widest) *overflowed = 1;
        narrow[cell] = static_cast<narrow_type_>(value);
    }
}

/** Copies the narrow shadow back over the graph the caller owns. */
template <typename wide_type_, typename narrow_type_>
__global__ void cuda_widen_(std::size_t cells, narrow_type_ const* narrow, wide_type_* wide) {
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    std::size_t const first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (std::size_t cell = first; cell < cells; cell += stride) wide[cell] = narrow[cell];
}

#if !defined(SCALING_ELECTIONS_WITH_HIP)

/**
 *  @brief Performs the independent step on a 16-bit graph, two candidates per 32-bit word.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @param[in] words_per_row The row stride of the packed matrix, counted in 32-bit words.
 *  @param[in] k The index of the current tile being processed.
 *  @param[inout] graph The graph of strongest paths, viewed as pairs of adjacent vote counts.
 *
 *  Each thread owns two words, so one block covers a tile with a quarter of the threads the
 *  32-bit kernel needs. The max-min semiring has no packed primitive, so the pair is spelled as
 *  a three-way minimum with a repeated argument, then a three-way maximum.
 */
template <std::uint32_t tile_size_>
__global__ void cuda_independent_packed_(candidate_index_t words_per_row, candidate_index_t k, std::uint32_t* graph) {
    static_assert(tile_size_ % 4 == 0, "A packed tile row must divide evenly across the lanes");
    constexpr std::uint32_t tile_words_ = tile_size_ / 2;
    constexpr std::uint32_t words_per_thread_ = 2;
    constexpr std::uint32_t lanes_ = tile_words_ / words_per_thread_;

    candidate_index_t const j = blockIdx.x;
    candidate_index_t const i = blockIdx.y;
    candidate_index_t const bi = threadIdx.y;
    candidate_index_t const lane = threadIdx.x;

    if (i == k && j == k) return;

    alignas(16) __shared__ std::uint32_t a[tile_size_][tile_words_];
    alignas(16) __shared__ std::uint32_t b[tile_size_][tile_words_];
    alignas(16) __shared__ std::uint32_t c[tile_size_][tile_words_];

    // Staging strides by the lane count so each warp reads one contiguous run.
#pragma unroll
    for (std::uint32_t slice = 0; slice < words_per_thread_; slice++) {
        candidate_index_t const word = lane + slice * lanes_;
        c[bi][word] = graph[(i * tile_size_ + bi) * words_per_row + j * tile_words_ + word];
        a[bi][word] = graph[(i * tile_size_ + bi) * words_per_row + k * tile_words_ + word];
        b[bi][word] = graph[(k * tile_size_ + bi) * words_per_row + j * tile_words_ + word];
    }
    __syncthreads();

    candidate_index_t const first_word = lane * words_per_thread_;
    candidate_index_t const diagonal_word = bi / 2;
    uint2 c_pair = *reinterpret_cast<uint2 const*>(&c[bi][first_word]);
    std::uint32_t const diagonal_before = diagonal_word == first_word ? c_pair.x : c_pair.y;

#pragma unroll tile_size_
    for (candidate_index_t step = 0; step < tile_size_; step++) {
        std::uint32_t const a_word = a[bi][step / 2];
        std::uint32_t const a_pair = __byte_perm(a_word, 0u, (step % 2) ? 0x3232u : 0x1010u);
        uint2 const b_pair = *reinterpret_cast<uint2 const*>(&b[step][first_word]);
        std::uint32_t const smallest_low = __vimin3_u16x2(a_pair, b_pair.x, b_pair.x);
        std::uint32_t const smallest_high = __vimin3_u16x2(a_pair, b_pair.y, b_pair.y);
        c_pair.x = __vimax3_u16x2(c_pair.x, smallest_low, smallest_low);
        c_pair.y = __vimax3_u16x2(c_pair.y, smallest_high, smallest_high);
    }

    // A tile straddling the matrix diagonal leaves those cells at the semiring identity.
    if (i == j) {
        std::uint32_t const keep_diagonal = (bi % 2) ? 0x7610u : 0x3254u;
        if (diagonal_word == first_word) c_pair.x = __byte_perm(c_pair.x, diagonal_before, keep_diagonal);
        else if (diagonal_word == first_word + 1) c_pair.y = __byte_perm(c_pair.y, diagonal_before, keep_diagonal);
    }

    *reinterpret_cast<uint2*>(&c[bi][first_word]) = c_pair;
    __syncthreads();

#pragma unroll
    for (std::uint32_t slice = 0; slice < words_per_thread_; slice++) {
        candidate_index_t const word = lane + slice * lanes_;
        graph[(i * tile_size_ + bi) * words_per_row + j * tile_words_ + word] = c[bi][word];
    }
}

#endif // !defined(SCALING_ELECTIONS_WITH_HIP)

#pragma endregion Packed Sixteen Bit

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

/** Tensor map for the Hopper bulk-tensor path, absent when the device or the layout rules it out. */
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

/** Builds the descriptor the bulk-tensor path needs, throwing when the device cannot supply one. */
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

/** HIP has no bulk-tensor engine, so the descriptor type can never hold one. */
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

/** HIP has no bulk-tensor engine, so the descriptor can never be built. */
template <std::uint32_t tile_size_>
tma_descriptor_t require_tma_(votes_count_t*, candidate_index_t, cudaDeviceProp const&) {
    throw std::runtime_error("The `gpu_hopper` backend is unavailable in a HIP build");
}

#endif

#if !defined(SCALING_ELECTIONS_WITH_HIP)

#if defined(__CUDA_ARCH_LIST__)
/** Whether this build carries native sm_90 code, where packed min-max is a single instruction. */
constexpr bool carries_native_packed_min_max() {
    for (int compiled : {__CUDA_ARCH_LIST__})
        if (compiled >= 900) return true;
    return false;
}
#else
/** Without an architecture list the build cannot promise a native packed min-max. */
constexpr bool carries_native_packed_min_max() { return false; }
#endif

/**
 *  @brief Picks the width the sweep would prefer, before any vote count has been looked at.
 *
 *  Below sm_90 the packed intrinsics are emulated and lose to the 32-bit path, so both the
 *  compiled architectures and the running device have to offer the real instruction.
 */
template <std::uint32_t tile_size_>
graph_width_t choose_width_(backend_t backend, cudaDeviceProp const& device_properties) {
    if constexpr (tile_size_ % 4 != 0) return graph_width_t::wide_32_k;
    else {
        if (backend != backend_t::gpu_serial_k) return graph_width_t::wide_32_k;
        if (!carries_native_packed_min_max()) return graph_width_t::wide_32_k;
        if (device_properties.major < 9) return graph_width_t::wide_32_k;
        return graph_width_t::narrow_16_k;
    }
}

/** Runs every pivot step on the 16-bit shadow, with the independent phase packed two per word. */
template <std::uint32_t tile_size_>
void sweep_narrow_(candidate_index_t graph_stride, std::uint16_t* graph) {
    candidate_index_t const tiles_count = graph_stride / tile_size_;
    dim3 const tile_shape(tile_size_, tile_size_, 1);
    dim3 const packed_shape(tile_size_ / 4, tile_size_, 1);
    dim3 const independent_grid(tiles_count, tiles_count, 1);
    for (candidate_index_t k = 0; k < tiles_count; k++) {
        cuda_diagonal_<tile_size_, std::uint16_t><<<1, tile_shape>>>(graph_stride, k, graph);
        cuda_partially_independent_<tile_size_, std::uint16_t><<<tiles_count, tile_shape>>>(graph_stride, k, graph);
        cuda_independent_packed_<tile_size_>
            <<<independent_grid, packed_shape>>>(graph_stride / 2, k, reinterpret_cast<std::uint32_t*>(graph));

        cudaError_t const error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }
}

#else

/** HIP has no packed 16-bit min-max, so the sweep never narrows. */
template <std::uint32_t tile_size_>
graph_width_t choose_width_(backend_t, cudaDeviceProp const&) {
    return graph_width_t::wide_32_k;
}

/** HIP has no packed 16-bit min-max, so the narrow sweep can never run. */
template <std::uint32_t tile_size_>
void sweep_narrow_(candidate_index_t, std::uint16_t*) {
    throw std::runtime_error("The packed 16-bit path is unavailable in a HIP build");
}

#endif

/** Runs every pivot step on the 32-bit graph, through whichever independent kernel the backend names. */
template <std::uint32_t tile_size_>
void sweep_wide_(candidate_index_t graph_stride, votes_count_t* graph, tma_descriptor_t const& tma, backend_t backend) {
    candidate_index_t const tiles_count = graph_stride / tile_size_;
    dim3 const tile_shape(tile_size_, tile_size_, 1);
    dim3 const independent_grid(tiles_count, tiles_count, 1);
    for (candidate_index_t k = 0; k < tiles_count; k++) {
        cuda_diagonal_<tile_size_><<<1, tile_shape>>>(graph_stride, k, graph);
        cuda_partially_independent_<tile_size_><<<tiles_count, tile_shape>>>(graph_stride, k, graph);
        launch_independent_<tile_size_>(independent_grid, tile_shape, graph_stride, k, graph, tma, backend);

        cudaError_t const error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }
}

/** How many blocks a flat pass over @p cells needs, capped so the grid stays resident. */
inline std::uint32_t flat_blocks_(std::size_t cells, std::uint32_t threads_per_block) {
    std::size_t const needed = (cells + threads_per_block - 1) / threads_per_block;
    return static_cast<std::uint32_t>(std::min<std::size_t>(needed, 4096));
}

/** Fills the 16-bit shadow, reporting the width the sweep can actually use. */
inline graph_width_t narrow_graph_(std::size_t cells, votes_count_t const* graph, std::uint16_t* narrow,
                                   votes_count_t* overflowed) {
    constexpr std::uint32_t threads_per_block = 256;
    *overflowed = 0;
    cuda_narrow_<<<flat_blocks_(cells, threads_per_block), threads_per_block>>>(cells, graph, narrow, overflowed);
    cudaError_t const error = cudaDeviceSynchronize();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    return *overflowed != 0 ? graph_width_t::wide_32_k : graph_width_t::narrow_16_k;
}

/** Copies the 16-bit shadow back over the graph the caller owns. */
inline void widen_graph_(std::size_t cells, std::uint16_t const* narrow, votes_count_t* graph) {
    constexpr std::uint32_t threads_per_block = 256;
    cuda_widen_<<<flat_blocks_(cells, threads_per_block), threads_per_block>>>(cells, narrow, graph);
    cudaError_t const error = cudaDeviceSynchronize();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

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
 *
 *  On sm_90 the per-thread backend narrows the graph to 16 bits and runs the independent phase two
 *  candidates per word, falling back to the 32-bit sweep when any vote count reaches 65536.
 */
template <std::uint32_t tile_size_> //
void compute_strongest_paths_cuda(  //
    votes_count_t* preferences, candidate_index_t num_candidates, candidate_index_t row_stride, votes_count_t* graph,
    candidate_index_t graph_stride, backend_t backend) {

    winning_votes_graph(preferences, num_candidates, row_stride, graph, graph_stride);

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

    std::size_t const cells = static_cast<std::size_t>(graph_stride) * graph_stride;
    if (choose_width_<tile_size_>(backend, device_properties) == graph_width_t::narrow_16_k) {
        managed_graph<std::uint16_t> narrow(cells * sizeof(std::uint16_t));
        managed_graph<votes_count_t> overflowed(sizeof(votes_count_t));
        if (narrow_graph_(cells, graph, narrow.pointer, overflowed.pointer) == graph_width_t::narrow_16_k) {
            sweep_narrow_<tile_size_>(graph_stride, narrow.pointer);
            widen_graph_(cells, narrow.pointer, graph);
            return;
        }
    }

    sweep_wide_<tile_size_>(graph_stride, graph, tma, backend);
}

#endif // defined(SCALING_ELECTIONS_WITH_CUDA)

#pragma endregion CUDA

#pragma region OpenMP

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
                SCALING_ELECTIONS_UNROLL
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
            SCALING_ELECTIONS_UNROLL
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
            SCALING_ELECTIONS_UNROLL
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

/**
 *  @brief Computes the strongest paths for the block-parallel Schulze voting algorithm using @b OpenMP.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @tparam march_ Whether tile copies bounds-check their edges.
 *  @param[in] preferences The preferences matrix.
 *  @param[in] num_candidates The number of candidates.
 *  @param[in] row_stride The stride between rows in the preferences matrix.
 *  @param[out] graph The output matrix of strongest paths, packed to @p num_candidates per row.
 *  @param[in] cancelled Polled between pivots, aborting the run once it reads non-zero.
 */
template <std::uint32_t tile_size_, tile_march_t march_ = tile_march_t::fast_k>                 //
void compute_strongest_paths_openmp(                                                            //
    votes_count_t* preferences, candidate_index_t num_candidates, candidate_index_t row_stride, //
    votes_count_t* graph, volatile std::sig_atomic_t const* cancelled = nullptr) {

    winning_votes_graph(preferences, num_candidates, row_stride, graph, num_candidates);

    // Time for the actual core implementation
    candidate_index_t const tiles_count = (num_candidates + tile_size_ - 1) / tile_size_;
    for (candidate_index_t k = 0; k < tiles_count; k++) {

        if (cancelled && *cancelled) throw std::runtime_error("Stopped by signal");

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
