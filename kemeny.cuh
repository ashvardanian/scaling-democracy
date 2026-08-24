/**
 *  @brief Exact Kemeny-Young consensus ranking over a pairwise preference matrix.
 *  @file kemeny.cuh
 *  @author Ash Vardanian
 *  @date July 12, 2024
 *  @see https://ashvardanian.com/posts/scaling-elections
 */
#pragma once
#include "types.cuh"

#pragma region Kemeny

/**
 *  A Kemeny score sums pairwise disagreements, bounded by `voters * n * (n - 1) / 2`.
 *  Thirty-two bits cover any real electorate and halve the table against a 64-bit score,
 *  which is the difference between fitting one more candidate and not.
 */
using kemeny_score_t = std::uint32_t;

/**
 *  @brief Votes each candidate loses to every subset of the others.
 *
 *  One table would be `n * 2^n` wide. Splitting the subset into a low and a high half makes
 *  two of `n * 2^(n/2)`, small enough to stay in cache while the score table streams past.
 */
struct kemeny_sums_t {
    candidate_index_t const low_bits;
    std::size_t const low_states;
    std::size_t const high_states;
    std::vector<kemeny_score_t> low;
    std::vector<kemeny_score_t> high;

    kemeny_sums_t(votes_count_t const* preferences, candidate_index_t num_candidates)
        : low_bits(num_candidates / 2), low_states(std::size_t {1} << low_bits),
          high_states(std::size_t {1} << (num_candidates - low_bits)), low(num_candidates * low_states, 0),
          high(num_candidates * high_states, 0) {

        for (candidate_index_t candidate = 0; candidate < num_candidates; candidate++) {
            votes_count_t const* row = preferences + candidate * num_candidates;
            accumulate(low.data() + candidate * low_states, low_states, row);
            accumulate(high.data() + candidate * high_states, high_states, row + low_bits);
        }
    }

    /** Votes that preferred @p candidate to every member of @p subset. */
    kemeny_score_t against(candidate_index_t candidate, std::size_t subset) const noexcept {
        return low[candidate * low_states + (subset & (low_states - 1))] +
               high[candidate * high_states + (subset >> low_bits)];
    }

  private:
    /** Folds one candidate's votes into every subset that contains each opponent. */
    static void accumulate(kemeny_score_t* row, std::size_t states, votes_count_t const* votes) noexcept {
        for (std::size_t bit = 1; bit < states; bit <<= 1, votes++)
            for (std::size_t subset = bit; subset < states; subset++)
                if (subset & bit) row[subset] = row[subset ^ bit] + *votes;
    }
};

/** An exact Kemeny-Young consensus ranking and the disagreement it achieves. */
struct kemeny_solution_t {
    std::vector<candidate_index_t> ranking;
    kemeny_score_t score = 0;
};

/**
 *  @brief Rejects an electorate whose worst ordering would wrap a 32-bit score.
 *
 *  The worst ordering pays the larger side of every pair, so that sum is what the score type
 *  has to hold. Wrapping here would answer confidently and wrongly.
 */
inline void kemeny_require_score_fits_(votes_count_t const* preferences, candidate_index_t num_candidates) {
    std::uint64_t worst_case = 0;
    for (candidate_index_t first = 0; first < num_candidates; first++)
        for (candidate_index_t second = first + 1; second < num_candidates; second++)
            worst_case += std::max(preferences[first * num_candidates + second],
                                   preferences[second * num_candidates + first]);
    if (worst_case > std::numeric_limits<kemeny_score_t>::max())
        throw std::runtime_error("Ballot counts exceed what a 32-bit Kemeny score can hold");
}

/**
 *  @brief Walks a completed cost table back out into a ranking, best candidate first.
 *
 *  Every backend fills the same table, so recovering the ranking in one place is also what
 *  keeps their tie-breaking identical.
 */
inline kemeny_solution_t kemeny_trace_(kemeny_sums_t const& sums, kemeny_score_t const* costs,
                                       candidate_index_t num_candidates) {
    std::size_t const states = std::size_t {1} << num_candidates;

    kemeny_solution_t solution;
    solution.score = costs[states - 1];
    solution.ranking.reserve(num_candidates);

    for (std::size_t subset = states - 1; subset;)
        for (candidate_index_t candidate = 0; candidate < num_candidates; candidate++) {
            std::size_t const bit = std::size_t {1} << candidate;
            if (!(subset & bit)) continue;
            std::size_t const rest = subset ^ bit;
            if (costs[subset] != costs[rest] + sums.against(candidate, rest)) continue;
            solution.ranking.push_back(candidate);
            subset = rest;
            break;
        }
    std::reverse(solution.ranking.begin(), solution.ranking.end());
    return solution;
}

/**
 *  @brief Determines the exact Kemeny-Young consensus ranking on one host thread.
 *
 *  @param preferences The pairwise preference matrix.
 *  @param num_candidates The number of candidates, which the score table bounds to 34.
 */
inline kemeny_solution_t kemeny_solve_host_(votes_count_t const* preferences, candidate_index_t num_candidates) {

    kemeny_require_score_fits_(preferences, num_candidates);

    kemeny_sums_t const sums(preferences, num_candidates);
    std::size_t const states = std::size_t {1} << num_candidates;

    // Entry `subset` is the least disagreement achievable seating those candidates in the
    // leading places, counting only the pairs inside it. Clearing a bit only ever lowers the
    // index, so plain increasing order is already a valid topological order.
    std::vector<kemeny_score_t> costs(states);
    costs[0] = 0;
    for (std::size_t subset = 1; subset < states; subset++) {
        kemeny_score_t best = std::numeric_limits<kemeny_score_t>::max();
        for (candidate_index_t candidate = 0; candidate < num_candidates; candidate++) {
            std::size_t const bit = std::size_t {1} << candidate;
            if (!(subset & bit)) continue;
            // Seating this candidate last within the subset costs the votes that preferred
            // it to each of the others.
            std::size_t const rest = subset ^ bit;
            best = std::min<kemeny_score_t>(best, costs[rest] + sums.against(candidate, rest));
        }
        costs[subset] = best;
    }

    return kemeny_trace_(sums, costs.data(), num_candidates);
}

#pragma endregion Kemeny

#pragma region CUDA

#if defined(SCALING_ELECTIONS_WITH_CUDA)

/** Threads per block for one layer of the subset dynamic program. */
constexpr std::uint32_t kemeny_block_size_k = 256;

/** Device memory the tables must leave behind for the driver and everyone else. */
constexpr std::size_t kemeny_device_headroom_k = std::size_t {1} << 30;

/** Owns one managed reservation, releasing it on scope exit. */
template <typename value_type_>
struct managed_buffer {
    value_type_* pointer = nullptr;

    explicit managed_buffer(std::size_t count) {
        if (cudaMallocManaged(&pointer, count * sizeof(value_type_)) != cudaSuccess)
            throw std::runtime_error("Failed to allocate memory on device");
    }
    ~managed_buffer() noexcept {
        if (pointer) cudaFree(pointer);
    }
    managed_buffer(managed_buffer const&) = delete;
    managed_buffer& operator=(managed_buffer const&) = delete;
};

/** The split-mask tables of `kemeny_sums_t` as a kernel addresses them. */
struct kemeny_device_sums_t {
    /** Votes against each candidate, indexed by the mask's low half. */
    kemeny_score_t const* low;
    /** Votes against each candidate, indexed by the mask's high half. */
    kemeny_score_t const* high;
    /** How many of the mask's low bits the low table covers. */
    candidate_index_t low_bits;
    /** The low table's per-candidate stride. */
    std::uint32_t low_states;
    /** The high table's per-candidate stride. */
    std::uint32_t high_states;

    /** Votes that preferred @p candidate to every member of @p subset. */
    __forceinline__ __device__ kemeny_score_t against(candidate_index_t candidate, std::size_t subset) const noexcept {
        return low[candidate * low_states + (subset & (low_states - 1))] +
               high[candidate * high_states + (subset >> low_bits)];
    }
};

#if defined(SCALING_ELECTIONS_HOPPER)

/** Scores seating one candidate last, spelling the add-min explicitly for Hopper. */
__forceinline__ __device__ kemeny_score_t kemeny_relax_(kemeny_score_t rest, kemeny_score_t against,
                                                        kemeny_score_t best) noexcept {
    return __viaddmin_u32(rest, against, best);
}

#else

/** Scores seating one candidate last, keeping whichever of the two orderings costs less. */
__forceinline__ __device__ kemeny_score_t kemeny_relax_(kemeny_score_t rest, kemeny_score_t against,
                                                        kemeny_score_t best) noexcept {
    return min(rest + against, best);
}

#endif // defined(SCALING_ELECTIONS_HOPPER)

/**
 *  @brief Fills every subset that seats @p seated candidates, one thread to a subset.
 *
 *  Threads take colex ranks that @p binomials unranks into a mask, so a layer costs exactly as
 *  many threads as it has subsets. Clearing a bit drops the population count by one, and that
 *  layer is complete before this one launches.
 */
__global__ void kemeny_layer_cuda_(                                   //
    kemeny_device_sums_t sums, kemeny_score_t* costs,                 //
    std::uint32_t const* binomials, candidate_index_t num_candidates, //
    candidate_index_t seated, std::uint64_t layer_states) {

    std::uint64_t rank = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (rank >= layer_states) return;

    // Colex unranking, one binomial per candidate: the mask this rank names among its layer.
    std::size_t subset = 0;
    candidate_index_t remaining = seated;
    for (candidate_index_t candidate = num_candidates; remaining != 0 && candidate != 0;) {
        candidate--;
        std::uint32_t const below = binomials[candidate * (num_candidates + 1) + remaining];
        if (rank < below) continue;
        rank -= below;
        subset |= std::size_t {1} << candidate;
        remaining--;
    }

    kemeny_score_t best = ~kemeny_score_t {0};
    for (candidate_index_t candidate = 0; candidate < num_candidates; candidate++) {
        std::size_t const bit = std::size_t {1} << candidate;
        if (!(subset & bit)) continue;
        std::size_t const rest = subset ^ bit;
        best = kemeny_relax_(costs[rest], sums.against(candidate, rest), best);
    }
    costs[subset] = best;
}

/**
 *  @brief Determines the exact Kemeny-Young consensus ranking on a CUDA or @b HIP device.
 *
 *  One launch per population count, because subsets of one width depend only on the width below
 *  and never on each other. The answer is the host's, entry for entry: the same integer sums
 *  reach the same minimum whatever order the threads take them in.
 *
 *  @param preferences The pairwise preference matrix.
 *  @param num_candidates The number of candidates, which free device memory bounds further.
 */
inline kemeny_solution_t kemeny_solve_cuda_(votes_count_t const* preferences, candidate_index_t num_candidates) {

    kemeny_require_score_fits_(preferences, num_candidates);

    kemeny_sums_t const sums(preferences, num_candidates);
    std::size_t const states = std::size_t {1} << num_candidates;
    std::size_t const sums_states = sums.low.size() + sums.high.size();
    candidate_index_t const binomials_stride = num_candidates + 1;
    std::size_t const binomials_states = static_cast<std::size_t>(binomials_stride) * binomials_stride;
    std::size_t const wanted_bytes = (states + sums_states) * sizeof(kemeny_score_t) +
                                     binomials_states * sizeof(std::uint32_t);

    // Managed memory oversubscribes rather than failing, so an unaffordable table is refused here.
    std::size_t free_bytes = 0;
    [[maybe_unused]] std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
        throw std::runtime_error("Failed to query device memory");
    if (wanted_bytes + kemeny_device_headroom_k > free_bytes)
        throw std::runtime_error("Kemeny over " + std::to_string(num_candidates) + " candidates wants " +
                                 std::to_string(wanted_bytes >> 20) + " MiB of device memory, of which " +
                                 std::to_string(free_bytes >> 20) + " MiB is free");

    managed_buffer<kemeny_score_t> const costs(states);
    managed_buffer<kemeny_score_t> const device_sums(sums_states);
    managed_buffer<std::uint32_t> const binomials(binomials_states);

    std::copy(sums.low.begin(), sums.low.end(), device_sums.pointer);
    std::copy(sums.high.begin(), sums.high.end(), device_sums.pointer + sums.low.size());

    // Pascal's rule, and `C(34, 17)` is the widest entry a 32-bit slot has to hold.
    std::fill(binomials.pointer, binomials.pointer + binomials_states, 0u);
    for (candidate_index_t upper = 0; upper <= num_candidates; upper++) {
        binomials.pointer[upper * binomials_stride] = 1;
        for (candidate_index_t lower = 1; lower <= upper; lower++)
            binomials.pointer[upper * binomials_stride + lower] =
                binomials.pointer[(upper - 1) * binomials_stride + lower] +
                binomials.pointer[(upper - 1) * binomials_stride + lower - 1];
    }

    kemeny_device_sums_t const device_view {
        device_sums.pointer,
        device_sums.pointer + sums.low.size(),
        sums.low_bits,
        static_cast<std::uint32_t>(sums.low_states),
        static_cast<std::uint32_t>(sums.high_states),
    };

    // Zeroes the empty subset the first layer reads, and faults the table's pages onto the device.
    if (cudaMemset(costs.pointer, 0, states * sizeof(kemeny_score_t)) != cudaSuccess)
        throw std::runtime_error("Failed to clear device memory");

    for (candidate_index_t seated = 1; seated <= num_candidates; seated++) {
        std::uint64_t const layer_states = binomials.pointer[num_candidates * binomials_stride + seated];
        std::uint64_t const blocks = (layer_states + kemeny_block_size_k - 1) / kemeny_block_size_k;
        kemeny_layer_cuda_<<<static_cast<unsigned int>(blocks), kemeny_block_size_k>>>( //
            device_view, costs.pointer, binomials.pointer, num_candidates, seated, layer_states);

        cudaError_t const error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }

    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("CUDA operations did not complete successfully");

    return kemeny_trace_(sums, costs.pointer, num_candidates);
}

#endif // defined(SCALING_ELECTIONS_WITH_CUDA)

#pragma endregion CUDA

#pragma region Dispatch

/** Which processor runs the subset dynamic program. */
enum class kemeny_backend_t : std::uint8_t {
    /** The host below the crossover and the device above it. */
    automatic_k,
    /** The single-threaded host loop over every subset in turn. */
    cpu_serial_k,
    /** One kernel launch per population count, on CUDA or HIP. */
    gpu_layered_k,
};

/**
 *  The candidate count from which a layer is wide enough to repay the launches, the managed
 *  reservation, and the tables that travel with it. Below it the device spends longer starting
 *  up than the host spends solving.
 */
constexpr candidate_index_t kemeny_device_crossover_k = 14;

#if defined(SCALING_ELECTIONS_WITH_CUDA)

/**
 *  @brief Determines the exact Kemeny-Young consensus ranking.
 *
 *  The ranking minimises the summed Kendall-tau distance to the ballots, so no ordering
 *  disagrees with the electorate less. This is the exact optimum rather than an
 *  approximation, at `O(n * 2^n)` time against `O(2^n)` memory.
 *
 *  @param preferences The pairwise preference matrix.
 *  @param num_candidates The number of candidates, which the score table bounds to 34.
 *  @param backend Which processor to run on, or `automatic_k` to pick by candidate count.
 *
 *  @note A named backend the device cannot serve raises, where `automatic_k` falls back.
 */
inline kemeny_solution_t kemeny_solve(votes_count_t const* preferences, candidate_index_t num_candidates,
                                      kemeny_backend_t backend = kemeny_backend_t::automatic_k) {
    if (backend == kemeny_backend_t::cpu_serial_k) return kemeny_solve_host_(preferences, num_candidates);
    if (backend == kemeny_backend_t::gpu_layered_k) return kemeny_solve_cuda_(preferences, num_candidates);
    if (num_candidates < kemeny_device_crossover_k) return kemeny_solve_host_(preferences, num_candidates);

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        return kemeny_solve_host_(preferences, num_candidates);
    return kemeny_solve_cuda_(preferences, num_candidates);
}

#else

/**
 *  @brief Determines the exact Kemeny-Young consensus ranking.
 *
 *  The ranking minimises the summed Kendall-tau distance to the ballots, so no ordering
 *  disagrees with the electorate less. This is the exact optimum rather than an
 *  approximation, at `O(n * 2^n)` time against `O(2^n)` memory.
 *
 *  @param preferences The pairwise preference matrix.
 *  @param num_candidates The number of candidates, which the score table bounds to 34.
 *  @param backend Which processor to run on, of which this build has only the host.
 */
inline kemeny_solution_t kemeny_solve(votes_count_t const* preferences, candidate_index_t num_candidates,
                                      kemeny_backend_t backend = kemeny_backend_t::automatic_k) {
    if (backend == kemeny_backend_t::gpu_layered_k)
        throw std::runtime_error("This build has no GPU support compiled in");
    return kemeny_solve_host_(preferences, num_candidates);
}

#endif // defined(SCALING_ELECTIONS_WITH_CUDA)

#pragma endregion Dispatch
