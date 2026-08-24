/**
 *  @brief Shared scalar types, feature macros, and the HIP-to-CUDA shim.
 *  @file types.cuh
 *  @author Ash Vardanian
 *  @date July 12, 2024
 *  @see https://ashvardanian.com/posts/scaling-elections
 */
#pragma once
#include <csignal> // `std::signal`
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `std::printf`
#include <cstdlib> // `std::rand`

#include <algorithm>   // `std::min`, `std::max`, `std::reverse`
#include <limits>      // `std::numeric_limits`
#include <numeric>     // `std::accumulate`
#include <optional>    // `std::optional`, `std::nullopt`
#include <stdexcept>   // `std::runtime_error`
#include <string>      // `std::to_string`
#include <string_view> // `std::string_view`
#include <thread>      // `std::thread::hardware_concurrency()`
#include <type_traits> // `std::integral_constant`, `std::is_same`
#include <vector>      // `std::vector`

#if defined(_OPENMP)
#include <omp.h> // `omp_set_num_threads`
#define SCALING_ELECTIONS_WITH_OPENMP (1)
#endif

#if (defined(__ARM_NEON) || defined(__aarch64__))
#define SCALING_ELECTIONS_WITH_NEON (1)
#endif

#if defined(__clang__) // Apple's Clang can't handle `#pragma unroll`
#define SCALING_ELECTIONS_UNROLL _Pragma("clang loop unroll(full)")
#else
#define SCALING_ELECTIONS_UNROLL _Pragma("unroll full")
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
#include <cuda.h> // `CUtensorMap`
#include <cuda/barrier>
#include <cudaTypedefs.h> // `PFN_cuTensorMapEncodeTiled`
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#elif defined(SCALING_ELECTIONS_WITH_HIP)
#include <hip/hip_runtime.h>

#if defined(__HIP_PLATFORM_AMD__)
#define cudaError_t             hipError_t
#define cudaSuccess             hipSuccess
#define cudaGetDevice           hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp          hipDeviceProp_t
#define cudaMallocManaged       hipMallocManaged
#define cudaFree                hipFree
#define cudaMemGetInfo          hipMemGetInfo
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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300 && !defined(SCALING_ELECTIONS_WITH_HIP)
#define SCALING_ELECTIONS_KEPLER (1)
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !defined(SCALING_ELECTIONS_WITH_HIP)
#define SCALING_ELECTIONS_HOPPER (1)
#endif

/**
 *  The tile edge every backend is compiled for. Thirty-two fits a CPU L2 slice and matches an
 *  NVIDIA warp; override at build time with `-DSCALING_ELECTIONS_TILE=<n>` to compare.
 */
#ifndef SCALING_ELECTIONS_TILE
#define SCALING_ELECTIONS_TILE 32
#endif

using votes_count_t = std::uint32_t;
using candidate_index_t = std::uint32_t;

constexpr std::uint32_t tile_size_k = SCALING_ELECTIONS_TILE;
