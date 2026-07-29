#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>

// =========================
// CUDA CODE
// =========================

static void checkCuda(cudaError_t err, const char *label)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
        std::exit(1);
    }
}

static void checkCublas(cublasStatus_t status, const char *label)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "cuBLAS error at %s: %d\n", label, static_cast<int>(status));
        std::exit(1);
    }
}

__global__
void vectoradd_kernel(float *vec1, float *vec2, float *vecres, int size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size)
    {
        vecres[tid] = vec1[tid] + vec2[tid];
    }
}

// =========================
// VECTOR ADD EXAMPLE
// =========================
void run_vectoradd_example()
{
    int size = 1024;
    int blocksize = 256;
    int numBlocks = (size + blocksize - 1) / blocksize;

    std::vector<float> h_vec1(size);
    std::vector<float> h_vec2(size);
    std::vector<float> h_vecres(size);

    for (int i = 0; i < size; ++i)
    {
        h_vec1[i] = static_cast<float>(i);
        h_vec2[i] = static_cast<float>(i + 1);
    }

    float *d_vec1 = nullptr;
    float *d_vec2 = nullptr;
    float *d_vecres = nullptr;

    checkCuda(cudaMalloc(&d_vec1, size * sizeof(float)), "cudaMalloc vec1");
    checkCuda(cudaMalloc(&d_vec2, size * sizeof(float)), "cudaMalloc vec2");
    checkCuda(cudaMalloc(&d_vecres, size * sizeof(float)), "cudaMalloc vecres");

    checkCuda(cudaMemcpy(d_vec1, h_vec1.data(), size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy vec1");
    checkCuda(cudaMemcpy(d_vec2, h_vec2.data(), size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy vec2");

    vectoradd_kernel<<<numBlocks, blocksize>>>(d_vec1, d_vec2, d_vecres, size);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "kernel sync");

    checkCuda(cudaMemcpy(h_vecres.data(), d_vecres, size * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy vecres");

    std::cout << "Sample vector add results:\n";
    for (int i = 0; i < 10; ++i)
    {
        std::cout << h_vec1[i] << " + " << h_vec2[i] << " = " << h_vecres[i] << "\n";
    }

    checkCuda(cudaFree(d_vec1), "cudaFree vec1");
    checkCuda(cudaFree(d_vec2), "cudaFree vec2");
    checkCuda(cudaFree(d_vecres), "cudaFree vecres");
}

// =========================
// BENCHMARK / cuBLAS COMPARISON
// =========================
void benchmark_vectoradd_vs_cublas(int size, int warmupIterations, int iterations)
{
    const int blockSize = 256;
    const int numBlocks = (size + blockSize - 1) / blockSize;

    std::vector<float> h_a(size);
    std::vector<float> h_b(size);
    std::vector<float> h_result(size);

    for (int i = 0; i < size; ++i)
    {
        h_a[i] = static_cast<float>(i % 17 + 1);
        h_b[i] = static_cast<float>((i * 3) % 23 + 1);
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_result = nullptr;

    checkCuda(cudaMalloc(&d_a, size * sizeof(float)), "cudaMalloc benchmark a");
    checkCuda(cudaMalloc(&d_b, size * sizeof(float)), "cudaMalloc benchmark b");
    checkCuda(cudaMalloc(&d_result, size * sizeof(float)), "cudaMalloc benchmark result");

    checkCuda(cudaMemcpy(d_a, h_a.data(), size * sizeof(float), cudaMemcpyHostToDevice), "copy a to device");
    checkCuda(cudaMemcpy(d_b, h_b.data(), size * sizeof(float), cudaMemcpyHostToDevice), "copy b to device");
    checkCuda(cudaMemcpy(d_result, h_b.data(), size * sizeof(float), cudaMemcpyHostToDevice), "copy initial result to device");

    std::cout << "Warming up GPU...\n";
    for (int i = 0; i < warmupIterations; ++i)
    {
        vectoradd_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, size);
        checkCuda(cudaGetLastError(), "warmup kernel launch");
        checkCuda(cudaDeviceSynchronize(), "warmup kernel sync");

        checkCuda(cudaMemcpy(d_result, h_b.data(), size * sizeof(float), cudaMemcpyHostToDevice), "warmup reset result");
    }

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");

    std::cout << "Benchmarking custom CUDA kernel...\n";
    checkCuda(cudaEventRecord(start), "record start event");
    for (int i = 0; i < iterations; ++i)
    {
        vectoradd_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, size);
        checkCuda(cudaGetLastError(), "kernel launch during benchmark");
        checkCuda(cudaMemcpy(d_result, h_b.data(), size * sizeof(float), cudaMemcpyHostToDevice), "reset result between iterations");
    }
    checkCuda(cudaEventRecord(stop), "record stop event");
    checkCuda(cudaEventSynchronize(stop), "sync stop event");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
    std::cout << "Custom kernel average time: " << (ms / iterations) << " ms per launch\n";

    cublasHandle_t handle = nullptr;
    checkCublas(cublasCreate(&handle), "cublasCreate");
    const float alpha = 1.0f;

    std::cout << "Benchmarking cuBLAS axpy...\n";
    checkCuda(cudaMemcpy(d_result, h_b.data(), size * sizeof(float), cudaMemcpyHostToDevice), "reset result before cublas");
    checkCuda(cudaEventRecord(start), "record cublas start");
    for (int i = 0; i < iterations; ++i)
    {
        checkCublas(cublasSaxpy(handle, size, &alpha, d_a, 1, d_result, 1), "cublasSaxpy");
        checkCuda(cudaDeviceSynchronize(), "cublas sync");
        checkCuda(cudaMemcpy(d_result, h_b.data(), size * sizeof(float), cudaMemcpyHostToDevice), "reset cublas result");
    }
    checkCuda(cudaEventRecord(stop), "record cublas stop");
    checkCuda(cudaEventSynchronize(stop), "sync cublas stop");

    checkCuda(cudaEventElapsedTime(&ms, start, stop), "cublas elapsed time");
    std::cout << "cuBLAS average time: " << (ms / iterations) << " ms per call\n";

    checkCuda(cudaMemcpy(h_result.data(), d_result, size * sizeof(float), cudaMemcpyDeviceToHost), "copy final result");
    std::cout << "Verification sample: " << h_result[0] << ", " << h_result[1] << ", " << h_result[2] << "\n";

    checkCuda(cudaEventDestroy(start), "destroy start event");
    checkCuda(cudaEventDestroy(stop), "destroy stop event");
    checkCublas(cublasDestroy(handle), "cublasDestroy");
    checkCuda(cudaFree(d_a), "cudaFree benchmark a");
    checkCuda(cudaFree(d_b), "cudaFree benchmark b");
    checkCuda(cudaFree(d_result), "cudaFree benchmark result");
}

int main()
{
    run_vectoradd_example();
    std::cout << "\nComparing custom kernel against cuBLAS...\n";
    benchmark_vectoradd_vs_cublas(1 << 20, 3, 1000);
    return 0;
}