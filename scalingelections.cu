/**
 *  @brief CUDA-accelerated Schulze voting algorithm implementation.
 *  @file scalingelections.cu
 *  @author Ash Vardanian
 *  @date July 12, 2024
 *  @see https://ashvardanian.com/posts/scaling-elections
 */
#if !defined(SCALING_ELECTIONS_TEST)
#include <pybind11/numpy.h> // `array_t`
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
#endif

#include "types.cuh"
#include "ballots.cuh"
#include "schulze.cuh"
#include "kemeny.cuh"

/** Resolves the name the Python layer passes, throwing on anything unrecognized. */
inline backend_t backend_from_name(std::string_view name) {
    if (name == "cpu_openmp") return backend_t::cpu_openmp_k;
    if (name == "gpu_serial") return backend_t::gpu_serial_k;
    if (name == "gpu_hopper") return backend_t::gpu_hopper_k;
    throw std::invalid_argument("Backend must be one of: cpu_openmp, gpu_serial, gpu_hopper");
}

/** Resolves the Kemeny backend name the Python layer passes, throwing on anything unrecognized. */
inline kemeny_backend_t kemeny_backend_from_name(std::string_view name) {
    if (name == "auto") return kemeny_backend_t::automatic_k;
    if (name == "cpu_serial") return kemeny_backend_t::cpu_serial_k;
    if (name == "gpu_layered") return kemeny_backend_t::gpu_layered_k;
    throw std::invalid_argument("Kemeny backend must be one of: auto, cpu_serial, gpu_layered");
}

/** Stores the interrupt signal status. */
volatile std::sig_atomic_t global_signal_status = 0;

void signal_handler(int signal) { global_signal_status = signal; }

#if defined(SCALING_ELECTIONS_WITH_OPENMP)

/** Fixes the team size for one call, restoring the runtime's prior settings on scope exit. */
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

#pragma region Python bindings
#if !defined(SCALING_ELECTIONS_TEST)

/**
 *  @brief Computes the strongest paths for the block-parallel Schulze voting algorithm.
 *
 *  @param[in] preferences The preferences matrix.
 *  @param[in] backend_name One of `cpu_openmp`, `gpu_serial`, or `gpu_hopper`.
 *  @return A NumPy array containing the strongest paths matrix.
 *
 *  @note A backend the build or the device cannot serve raises rather than downgrading.
 */
static py::array_t<votes_count_t> compute_strongest_paths(      //
    py::array_t<votes_count_t, py::array::c_style> preferences, //
    std::string_view backend_name) {

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

        // Rounding the matrix up to a whole number of tiles keeps the kernels free of tail
        // checks: the padding is zero, which is the identity of the max-min semiring.
        candidate_index_t const graph_stride = (num_candidates + tile_size_k - 1) / tile_size_k * tile_size_k;
        std::size_t const graph_bytes = static_cast<std::size_t>(graph_stride) * graph_stride * sizeof(votes_count_t);

        managed_graph_t const graph(graph_bytes);
        cudaError_t error = cudaMemset(graph.pointer, 0, graph_bytes);
        if (error != cudaSuccess) throw std::runtime_error("Failed to clear device memory");
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) throw std::runtime_error("Failed to clear device memory");

        compute_strongest_paths_cuda<tile_size_k>(preferences_ptr, num_candidates, row_stride, graph.pointer,
                                                  graph_stride, backend);

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

    // A tile wider than the electorate is not an error: `checked_k` zero-fills the tail, and
    // zero is the identity of the max-min semiring, so the padding can never win a comparison.
    if (num_candidates % tile_size_k == 0)
        compute_strongest_paths_openmp<tile_size_k, tile_march_t::fast_k>( //
            preferences_ptr, num_candidates, row_stride, result_ptr, &global_signal_status);
    else
        compute_strongest_paths_openmp<tile_size_k, tile_march_t::checked_k>( //
            preferences_ptr, num_candidates, row_stride, result_ptr, &global_signal_status);
    return result;
}

/**
 *  @brief Prints what each visible CUDA device is, through Python's own stdout.
 *
 *  @note Raises on a build without CUDA, rather than reporting an empty device list.
 */
#if defined(SCALING_ELECTIONS_WITH_CUDA)
static void log_gpus() {
    int device_count;
    cudaDeviceProp device_properties;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) throw std::runtime_error("Failed to get device count");
    for (int index = 0; index < device_count; index++) {
        error = cudaGetDeviceProperties(&device_properties, index);
        if (error != cudaSuccess) throw std::runtime_error("Failed to get device properties");
        py::print(py::str("Device {}: {}").format(index, device_properties.name));
        py::print(py::str("\tSMs: {}").format(device_properties.multiProcessorCount));
        py::print(py::str("\tGlobal mem: {:.2f}GB")
                      .format(static_cast<float>(device_properties.totalGlobalMem) / (1024 * 1024 * 1024)));
        py::print(py::str("\tCUDA Cap: {}.{}").format(device_properties.major, device_properties.minor));
    }
}
#else
static void log_gpus() { throw std::runtime_error("No CUDA devices available"); }
#endif

/**
 *  @brief Sums a contiguous float array, showing how `thrust::` reaches the same device.
 *
 *  @param[in] data A contiguous one-dimensional float array.
 *  @return The sum of its elements.
 *
 *  @note `rocThrust` is not guaranteed to be present, so HIP and CPU-only builds sum on the host.
 */
#if defined(SCALING_ELECTIONS_WITH_CUDA) && !defined(SCALING_ELECTIONS_WITH_HIP)
static float reduce(py::array_t<float> const& data) {
    py::buffer_info buffer = data.request();
    if (buffer.ndim != 1 || buffer.strides[0] != sizeof(float))
        throw std::runtime_error("Input should be a contiguous 1D float array");
    float* pointer = static_cast<float*>(buffer.ptr);
    thrust::device_vector<float> on_device(pointer, pointer + buffer.size);
    return thrust::reduce(thrust::device, on_device.begin(), on_device.end(), 0.0f);
}
#else
static float reduce(py::array_t<float> const& data) {
    return std::accumulate(data.data(), data.data() + data.size(), 0.0f);
}
#endif

/**
 *  @brief Computes the exact Kemeny-Young consensus ranking and its disagreement score.
 *
 *  @param[in] preferences The preferences matrix.
 *  @param[in] backend_name One of `auto`, `cpu_serial`, or `gpu_layered`.
 *  @return A tuple of the ranking, best first, and the disagreement it achieves.
 */
static py::tuple compute_kemeny_ranking(py::array_t<votes_count_t, py::array::c_style> const& preferences,
                                        std::string_view backend_name) {
    kemeny_backend_t const backend = kemeny_backend_from_name(backend_name);

    py::buffer_info buffer = preferences.request();
    if (buffer.ndim != 2 || buffer.shape[0] != buffer.shape[1])
        throw std::runtime_error("Preferences must be a square matrix");
    candidate_index_t const num_candidates = static_cast<candidate_index_t>(buffer.shape[0]);
    if (num_candidates < 1 || num_candidates > 34)
        throw std::runtime_error("Kemeny is exact to 34 candidates, beyond which the table exceeds 64 GiB");

    kemeny_solution_t const solution = kemeny_solve(static_cast<votes_count_t const*>(buffer.ptr), num_candidates,
                                                    backend);
    return py::make_tuple(solution.ranking, solution.score);
}

PYBIND11_MODULE(scalingelections_cuda, m) {

    std::signal(SIGINT, signal_handler);

    m.def("log_gpus", &log_gpus);
    m.def("reduce", &reduce, py::arg("data"));
    m.def("compute_kemeny_ranking", &compute_kemeny_ranking, //
          py::arg("preferences"), py::kw_only(),             //
          py::arg("backend") = "auto");
    m.def("compute_strongest_paths", &compute_strongest_paths, //
          py::arg("preferences"), py::kw_only(),               //
          py::arg("backend") = "cpu_openmp");
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
