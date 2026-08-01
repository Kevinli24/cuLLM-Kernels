#ifndef __UTIL__
#define __UTIL__

#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <memory>

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t error = (call);                               \
    if (error != cudaSuccess) {                               \
        std::cerr << "CUDA error at " << __FILE__ << ":"      \
                  << __LINE__ << ": "                         \
                  << cudaGetErrorString(error) << '\n';       \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)
                                              \
    
auto cudaMallocFunc = [](std::size_t size) -> void* {
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
};

struct cudaDeleteHelper
{
    int device_id_;

    cudaDeleteHelper() : device_id_(0) {}
    cudaDeleteHelper(int id) : device_id_(id) {}

    void operator()(void *ptr) const
    {
        if (ptr)
        {
            std::cerr << "Free GPU memory\n";
            CUDA_CHECK(cudaSetDevice(device_id_));
            CUDA_CHECK(cudaFree(ptr));
        }
    }

    ~cudaDeleteHelper() = default;
};

template <typename T>
using cudaUniquePtr = std::unique_ptr<T, cudaDeleteHelper>;

#endif