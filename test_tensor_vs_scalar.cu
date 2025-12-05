// ============================================================================
// test_tensor_vs_scalar.cu
// Compare performance of double vs Tensor<double,1> in a CUDA kernel
// ============================================================================
// Compile (example):
//   nvcc -O3 -use_fast_math test_tensor_vs_scalar.cu -o test_tensor_vs_scalar
//
// Run:
//   ./test_tensor_vs_scalar
// ============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>

// Adjust this include to your actual header file name/path:
#include "data_struct_shared.cuh"   // must contain the optimized Tensor<ValueT,Dim>

// Type alias for convenience
using Tensor1d = Tensor<double, 1>;

// ----------------------------------------------------------------------------
// CUDA error check helper
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                         cudaGetErrorString(err__), __FILE__, __LINE__);      \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// ----------------------------------------------------------------------------
// Scalar kernel: pure double math
// ----------------------------------------------------------------------------
__global__ void scalar_kernel(const double* __restrict__ a,
                              const double* __restrict__ b,
                              double* __restrict__ out,
                              int n,
                              int iters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double x = a[idx];
    double y = b[idx];
    double acc = 0.0;

    // Simple but non-trivial arithmetic to stress ALU and registers
    for (int i = 0; i < iters; ++i) {
        acc = acc * x + y;
        acc = acc + x * y;
        acc = acc / (x + 1.0);
    }

    out[idx] = acc;
}

// ----------------------------------------------------------------------------
// Tensor kernel: use Tensor<double,1> inside the kernel
// (Inputs/outputs stored as doubles to keep global memory similar)
// ----------------------------------------------------------------------------
__global__ void tensor_kernel(const double* __restrict__ a,
                              const double* __restrict__ b,
                              double* __restrict__ out,
                              int n,
                              int iters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Wrap scalars into Tensor<double,1>
    Tensor1d x(a[idx]);   // broadcast constructor
    Tensor1d y(b[idx]);
    Tensor1d acc(0.0);

    for (int i = 0; i < iters; ++i) {
        acc = acc * x + y;
        acc = acc + x * y;
        acc = acc / (x + Tensor1d(1.0));
    }

    // Extract back to scalar
    out[idx] = acc[0];
}

// ----------------------------------------------------------------------------
// Timing helper using cudaEvent
// ----------------------------------------------------------------------------
float time_kernel(void (*kernel)(void), int /*dummy*/) { return 0.0f; } // unused

template <typename KernelFunc, typename... Args>
float run_and_time_kernel(dim3 grid, dim3 block,
                          int warmup_iters,
                          int timed_iters,
                          KernelFunc kernel,
                          Args... args)
{
    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        kernel<<<grid, block>>>(args...);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < timed_iters; ++i) {
        kernel<<<grid, block>>>(args...);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Return average time per kernel launch
    return ms / static_cast<float>(timed_iters);
}

// ----------------------------------------------------------------------------
// Host main
// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Problem size and work per thread
    const int N      = (argc > 1 ? std::atoi(argv[1]) : (1 << 20)); // default: 1M elements
    const int ITERS  = (argc > 2 ? std::atoi(argv[2]) : 256);       // default: 256 iterations
    const int BLOCK  = 256;
    const int GRID   = (N + BLOCK - 1) / BLOCK;
    const int WARMUP = 5;
    const int RUNS   = 20;

    std::cout << "N = " << N << ", ITERS = " << ITERS
              << ", BLOCK = " << BLOCK << ", GRID = " << GRID << "\n";

    // Host buffers
    std::vector<double> h_a(N), h_b(N), h_out_scalar(N), h_out_tensor(N);

    // Initialize inputs with some values
    for (int i = 0; i < N; ++i) {
        h_a[i] = 0.5 + 0.000001 * i;
        h_b[i] = 1.0 + 0.000002 * i;
    }

    // Device buffers
    double *d_a = nullptr, *d_b = nullptr, *d_out_scalar = nullptr, *d_out_tensor = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a,           N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b,           N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out_scalar,  N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out_tensor,  N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------------
    // Time scalar kernel
    // ------------------------------------------------------------------------
    float t_scalar_ms = run_and_time_kernel(
        dim3(GRID), dim3(BLOCK),
        WARMUP, RUNS,
        scalar_kernel,
        d_a, d_b, d_out_scalar, N, ITERS
    );

    // ------------------------------------------------------------------------
    // Time tensor kernel
    // ------------------------------------------------------------------------
    float t_tensor_ms = run_and_time_kernel(
        dim3(GRID), dim3(BLOCK),
        WARMUP, RUNS,
        tensor_kernel,
        d_a, d_b, d_out_tensor, N, ITERS
    );

    // Copy results back to host to check numerical differences
    CUDA_CHECK(cudaMemcpy(h_out_scalar.data(), d_out_scalar, N * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_tensor.data(), d_out_tensor, N * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // Validate maximum absolute difference
    double max_abs_diff = 0.0;
    for (int i = 0; i < N; ++i) {
        double diff = std::fabs(h_out_scalar[i] - h_out_tensor[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }

    std::cout << "Scalar kernel   avg time: " << t_scalar_ms << " ms\n";
    std::cout << "Tensor<1> kernel avg time: " << t_tensor_ms << " ms\n";
    std::cout << "Ratio (tensor / scalar): " << (t_tensor_ms / t_scalar_ms) << "x\n";
    std::cout << "Max |scalar - tensor|: " << max_abs_diff << "\n";

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out_scalar));
    CUDA_CHECK(cudaFree(d_out_tensor));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
