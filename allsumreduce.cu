#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include "util.h"

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

// Interleaved Addressing
/* 
__global__
void allsumreduce(float *dnums, float *dres, int size, int stride)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size/stride)
    {
        tid *= stride;
        dres[tid] = dnums[tid] + dnums[tid+(stride/2)];
    }
    
}


__host__
int main()
{
    int size = 1024;
    int blocksize = 256;
    int numblocks = (size + blocksize - 1) / blocksize;
    int stride = 2;

    float *hnums = new float[size];
    float *hres = new float[size];

    for (int i = 0; i < size; ++i)
    {
        hnums[i] = (float)i;
    }

    float *dnums, *dres;

    cudaMalloc(&dnums, size * sizeof(float));
    cudaMalloc(&dres, size * sizeof(float));
    cudaMemcpy(dnums, hnums, size * sizeof(float), cudaMemcpyHostToDevice);

    float *input = dnums;
    float *output = dres;

    while (stride <= size)
    {
        allsumreduce<<<numblocks, blocksize>>>(input, output, size, stride);
        float *temp = input;
        input = output;
        output = temp;
        stride *= 2;
    }
    cudaMemcpy(hres, input, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("result = %f\n", hres[0]); // expected answer: 523,776

    cudaFree(dnums);
    cudaFree(dres);

    delete[] hnums;
    delete[] hres;

    return 0;
}
*/

/*
// Sequential Addressing
__global__

void seqaddress(float *vector, int size, int halve)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size/halve)
    {
        vector[tid] = vector[tid] + vector[tid + size/halve];
    }
}


__host__
int main()
{
    int size = 1024;
    int blocksize = 256;
    int numblocks = (size + blocksize - 1) / blocksize;

    float *hnums = new float[size];
    float *hres = new float[size];

    for (int i = 0; i < size; ++i)
    {
        hnums[i] = (float)i;
    }

    float *dnums;

    cudaMalloc(&dnums, size * sizeof(float));
    cudaMemcpy(dnums, hnums, size * sizeof(float), cudaMemcpyHostToDevice);

    int halve = 2;

    while (halve <= size)
    {
        seqaddress<<<numblocks, blocksize>>>(dnums, size, halve);
        halve *= 2;
    }

    cudaMemcpy(hres, dnums, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("result = %f\n", hres[0]); // expected answer: 523,776

    cudaFree(dnums);

    delete[] hnums;
    delete[] hres;

    return 0;

}
*/

/*
// Sequential Addressing with Shared Memory
__global__
void seqaddsmem(float *input, float *output)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();


    for (unsigned int j = blockDim.x / 2; j > 0; j >>= 1)
    {
        if (tid < j)
        {
            sdata[tid] = sdata[tid] + sdata[tid + j];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }

}

__host__
int main()
{
    int size = 1024;
    int blocksize = 256;
    int numblocks = (size + blocksize - 1) / blocksize;
    int shmem = blocksize * sizeof(float);

    float *hin = new float[size];
    float *hout = new float[size];

    for (int i = 0; i < size; ++i)
    {
        hin[i] = (float)i;
    }

    float *din, *dout, *dout2;

    cudaMalloc(&din, size * sizeof(float));
    cudaMemcpy(din, hin, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&dout, size * sizeof(float));
    cudaMemset(dout, 0, size * sizeof(float));
    cudaMalloc(&dout2, size * sizeof(float));

    seqaddsmem<<<numblocks, blocksize, shmem>>>(din, dout);
    seqaddsmem<<<1, blocksize, shmem>>>(dout, dout2);
    cudaMemcpy(hout, dout2, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("result = %f\n", hout[0]); // expected answer: 523,776

    cudaFree(din);
    cudaFree(dout);
    cudaFree(dout2);

    delete[] hin;
    delete[] hout;

    return 0;

}
*/



/*
// First Add during Load
__global__
void firstadd(float *input, float *output)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = input[i] + input[i + blockDim.x];
    __syncthreads();

    for (unsigned int j = blockDim.x / 2; j > 0; j >>= 1)
    {
        if (tid < j)
        {
            sdata[tid] += sdata[tid + j];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }

}


__host__
int main()
{
    int size = 1024;
    int blocksize = 256;
    int numblocks = (size + (2*blocksize) - 1) / (2*blocksize);
    int shmem = blocksize * sizeof(float);

    float *hin = new float[size];
    float *hout = new float[size];

    for (int i = 0; i < size; ++i)
    {
        hin[i] = (float)i;
    }

    float *din, *dout, *dout2;

    cudaMalloc(&din, size * sizeof(float));
    cudaMemcpy(din, hin, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&dout, size * sizeof(float));
    cudaMemset(dout, 0, size * sizeof(float));
    cudaMalloc(&dout2, size * sizeof(float));

    firstadd<<<numblocks, blocksize, shmem>>>(din, dout);
    firstadd<<<1, blocksize, shmem>>>(dout, dout2);
    cudaMemcpy(hout, dout2, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("result = %f\n", hout[0]); // expected answer: 523,776

    cudaFree(din);
    cudaFree(dout);
    cudaFree(dout2);

    delete[] hin;
    delete[] hout;

    return 0;

}
*/


// Unroll last warp (32 threads)
/*
__device__ 
void lastwarpreduce(float *sdata, unsigned int tid)
{
    sdata[tid] = sdata[tid] + sdata[tid + 32];
    __syncwarp();
    sdata[tid] = sdata[tid] + sdata[tid + 16];
    __syncwarp();
    sdata[tid] = sdata[tid] + sdata[tid + 8];
    __syncwarp();
    sdata[tid] = sdata[tid] + sdata[tid + 4];
    __syncwarp();
    sdata[tid] = sdata[tid] + sdata[tid + 2];
    __syncwarp();
    if (tid == 0)
    {
        sdata[tid] = sdata[tid] + sdata[tid + 1];
    }
}


__global__
void firstadd2(float *input, float *output)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = input[i] + input[i + blockDim.x];
    __syncthreads();

    for (unsigned int j = blockDim.x / 2; j > 32; j >>= 1)
    {
        if (tid < j)
        {
            sdata[tid] += sdata[tid + j];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        lastwarpreduce(sdata, tid);
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }

}


__host__
int main()
{
    int size = 1024;
    int blocksize = 256;
    int numblocks = (size + (2*blocksize) - 1) / (2*blocksize);
    int shmem = blocksize * sizeof(float);

    float *hin = new float[size];
    float *hout = new float[size];

    for (int i = 0; i < size; ++i)
    {
        hin[i] = (float)i;
    }

    float *din, *dout, *dout2;

    cudaMalloc(&din, size * sizeof(float));
    cudaMemcpy(din, hin, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&dout, size * sizeof(float));
    cudaMemset(dout, 0, size * sizeof(float));
    cudaMalloc(&dout2, size * sizeof(float));

    firstadd2<<<numblocks, blocksize, shmem>>>(din, dout);
    firstadd2<<<1, blocksize, shmem>>>(dout, dout2);
    cudaMemcpy(hout, dout2, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("result = %f\n", hout[0]); // expected answer: 523,776

    cudaFree(din);
    cudaFree(dout);
    cudaFree(dout2);

    delete[] hin;
    delete[] hout;

    return 0;

} 
*/

/*
// Completely Unroll
template <unsigned int blocksize>
__device__ 
void lastwarpreduce2(float *sdata, unsigned int tid)
{
    if (blocksize >= 64)
    {
        if (tid < 32)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 32];
            __syncwarp();
        }
    }
    if (blocksize >= 32)
    {
        if (tid < 16)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 16];
            __syncwarp();
        }
    }
    if (blocksize >= 16)
    {
        if (tid < 8)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 8];
            __syncwarp();
        }
    }
    if (blocksize >= 8)
    {
        if (tid < 4)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 4];
            __syncwarp();
        }
    }
    if (blocksize >= 4)
    {
        if (tid < 2)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 2];
            __syncwarp();
        }
    }
    if (blocksize >= 2)
    {
        if (tid < 1)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 1];
        }
    }
}

template <unsigned int blocksize>
__global__
void firstadd3(float *input, float *output)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = input[i] + input[i + blockDim.x];
    __syncthreads();

    if (blocksize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 256];
            __syncthreads();
        }
    }
    if (blocksize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 128];
            __syncthreads();
        }
    }
    if (blocksize >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 64];
            __syncthreads();
        }
    }
    if (tid < 32)
    {
        lastwarpreduce2<blocksize>(sdata, tid);
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }

}


__host__
int main()
{
    int size = 1024;
    int blocksize = 256;
    int numblocks = (size + (2*blocksize) - 1) / (2*blocksize);

    float *hin = new float[size];
    float *hout = new float[size];

    for (int i = 0; i < size; ++i)
    {
        hin[i] = (float)i;
    }

    float *din, *dout, *dout2;

    cudaMalloc(&din, size * sizeof(float));
    cudaMemcpy(din, hin, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&dout, size * sizeof(float));
    cudaMemset(dout, 0, size * sizeof(float));
    cudaMalloc(&dout2, size * sizeof(float));

    switch (blocksize)
    {
        case 512:
            firstadd3<512><<<numblocks, 512, 512 * sizeof(float)>>>(din, dout);
            firstadd3<512><<<numblocks, 512, 512 * sizeof(float)>>>(dout, dout2);
            break;
        case 256:
            firstadd3<256><<<numblocks, 256, 256 * sizeof(float)>>>(din, dout);
            firstadd3<256><<<numblocks, 256, 256 * sizeof(float)>>>(dout, dout2);
            break;
        case 128:
            firstadd3<128><<<numblocks, 128, 128 * sizeof(float)>>>(din, dout);
            firstadd3<128><<<numblocks, 128, 128 * sizeof(float)>>>(dout, dout2);
            break;
    }

    cudaMemcpy(hout, dout2, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("result = %f\n", hout[0]); // expected answer: 523,776

    cudaFree(din);
    cudaFree(dout);
    cudaFree(dout2);

    delete[] hin;
    delete[] hout;

    return 0;

} */

// Multiple Adds / Threads

template <unsigned int blocksize>
__device__ 
void lastwarpreduce3(float *sdata, unsigned int tid)
{
    if (blocksize >= 64)
    {
        if (tid < 32)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 32];
        }
        __syncwarp();
    }
    if (blocksize >= 32)
    {
        if (tid < 16)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 16];
        }
        __syncwarp();
    }
    if (blocksize >= 16)
    {
        if (tid < 8)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 8];
        }
        __syncwarp();
    }
    if (blocksize >= 8)
    {
        if (tid < 4)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 4];
        }
        __syncwarp();
    }
    if (blocksize >= 4)
    {
        if (tid < 2)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 2];
        }
        __syncwarp();
    }
    if (blocksize >= 2)
    {
        if (tid < 1)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 1];
        }
    }
}

template <unsigned int blocksize>
__global__
void firstadd4(float *input, float *output, unsigned int size)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int gridsize = blockDim.x * 2 * gridDim.x;
    sdata[tid] = 0;

    while (i < size)
    {
        sdata[tid] += input[i] + input[i + blocksize];
        i += gridsize;
    }
    __syncthreads();

    if (blocksize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 256];
            __syncthreads();
        }
    }
    if (blocksize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 128];
            __syncthreads();
        }
    }
    if (blocksize >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 64];
            __syncthreads();
        }
    }
    if (tid < 32)
    {
        lastwarpreduce3<blocksize>(sdata, tid);
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }

}


__host__
int main1()
{
    int size = 1 << 24;
    int blocksize = 1024;
    int numblocks = (size + (2*blocksize) - 1) / (2*blocksize);

    float *hin = new float[size];
    float *hout = new float[size];

    for (int i = 0; i < size; ++i)
    {
        hin[i] = (float)i;
    }

    float *din, *dout, *dout2;

    //cudaUniquePtr<float> din_buf, dout_buf, dout2;
    //din_buf = cudaUniquePtr<float>((float*)cudaMallocFunc(size), cudaDeleteHelper(0));

    cudaMalloc(&din, size * sizeof(float));
    cudaMemcpy(din, hin, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&dout, size * sizeof(float));
    cudaMemset(dout, 0, size * sizeof(float));
    cudaMalloc(&dout2, size * sizeof(float));
    
    // Start benchmark code
    int iters = 1000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup
    firstadd4<256><<<numblocks, 256, 256 * sizeof(float)>>>(din, dout, size);

    //firstadd4<256><<<numblocks, 256, 256*sizeof(float)>>>(din_buf.get())

    firstadd4<256><<<1, 256, 256 * sizeof(float)>>>(dout, dout2, numblocks);
    cudaDeviceSynchronize();

    // timed loop
    cudaEventRecord(start);
    for (int k = 0; k < iters; ++k)
    {
        firstadd4<256><<<numblocks, 256, 256 * sizeof(float)>>>(din, dout, size);
        firstadd4<256><<<1, 256, 256 * sizeof(float)>>>(dout, dout2, numblocks);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float totalMs = 0.0f;
    cudaEventElapsedTime(&totalMs, start, stop);
    float ms = totalMs / iters;               // average per two-pass reduction
    // End benchmark code

    switch (blocksize)
    {
        case 512:
            firstadd4<512><<<numblocks, 512, 512 * sizeof(float)>>>(din, dout, size);
            firstadd4<512><<<numblocks, 512, 512 * sizeof(float)>>>(dout, dout2, size);
            break;
        case 256:
            firstadd4<256><<<numblocks, 256, 256 * sizeof(float)>>>(din, dout, size);
            firstadd4<256><<<numblocks, 256, 256 * sizeof(float)>>>(dout, dout2, size);
            break;
        case 128:
            firstadd4<128><<<numblocks, 128, 128 * sizeof(float)>>>(din, dout, size);
            firstadd4<128><<<numblocks, 128, 128 * sizeof(float)>>>(dout, dout2, size);
            break;
    }

    cudaMemcpy(hout, dout2, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("result = %f\n", hout[0]); // expected answer: 523,776

    // Start benchmark code
    double seconds = ms / 1000.0;
    double bytes   = (double)size * sizeof(float);      // input read once
    double gbps    = bytes / seconds / 1e9;

    // fill these from your other kernel runs (same harness) to get real speedups
    double prevTime     = ms;   // <-- PREVIOUS kernel's ms
    double baselineTime = ms;   // <-- REDUCE-0's ms

    double stepSpeedup = prevTime / ms;
    double cumSpeedup  = baselineTime / ms;

    printf("\n");
    printf("Time (ms)   Bandwidth (GB/s)   Step SpeedUp   Cumulative SpeedUp\n");
    printf("%-11.5f %-18.2f %-14.2f %-.2f\n", ms, gbps, stepSpeedup, cumSpeedup);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // End benchmark code

    cudaFree(din);
    cudaFree(dout);
    cudaFree(dout2);

    delete[] hin;
    delete[] hout;

    return 0;

} 

// New Implementation - with __shfl_down_sync

__device__ 
__forceinline__ float warpsumreduce(float val)
{
    for (unsigned int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <unsigned int blocksize>
__global__
void firstadd5(float *input, float *output, unsigned int size)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int gridsize = blockDim.x * 2 * gridDim.x;
    sdata[tid] = 0;

    while (i < size)
    {
        sdata[tid] += input[i] + input[i + blocksize];
        i += gridsize;
    }
    __syncthreads();

    if (blocksize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 256];
            __syncthreads();
        }
    }
    if (blocksize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 128];
            __syncthreads();
        }
    }
    if (blocksize >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 64];
            __syncthreads();
        }
    }
    if (tid < 32)
    {
        float v = sdata[tid] + sdata[tid + 32];
        v = warpsumreduce(v);
        if (tid == 0)
        {
        output[blockIdx.x] = v;
        }
    }

}


__host__
int main2()
{
    int size = 1 << 24;
    int blocksize = 256;
    int numblocks = (size + (2*blocksize) - 1) / (2*blocksize);

    float *hin = new float[size];
    float *hout = new float[size];

    for (int i = 0; i < size; ++i)
    {
        hin[i] = (float)i;
    }

    float *din, *dout, *dout2;

    cudaMalloc(&din, size * sizeof(float));
    cudaMemcpy(din, hin, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&dout, size * sizeof(float));
    cudaMemset(dout, 0, size * sizeof(float));
    cudaMalloc(&dout2, size * sizeof(float));

    // Benchmark code
    int iters = 1000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    firstadd5<256><<<numblocks, 256, 256 * sizeof(float)>>>(din, dout, size);
    firstadd5<256><<<1, 256, 256 * sizeof(float)>>>(dout, dout2, numblocks);
    cudaDeviceSynchronize();

    // Timed loop
    cudaEventRecord(start);
    for (int k = 0; k < iters; ++k)
    {
        firstadd5<256><<<numblocks, 256, 256 * sizeof(float)>>>(din, dout, size);
        firstadd5<256><<<1, 256, 256 * sizeof(float)>>>(dout, dout2, numblocks);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float totalMs = 0.0f;
    cudaEventElapsedTime(&totalMs, start, stop);
    float ms = totalMs / iters;               // average per two-pass reduction
    // End benchmark bode

    switch (blocksize)
    {
        case 512:
            firstadd5<512><<<numblocks, 512, 512 * sizeof(float)>>>(din, dout, size);
            firstadd5<512><<<numblocks, 512, 512 * sizeof(float)>>>(dout, dout2, size);
            break;
        case 256:
            firstadd5<256><<<numblocks, 256, 256 * sizeof(float)>>>(din, dout, size);
            firstadd5<256><<<numblocks, 256, 256 * sizeof(float)>>>(dout, dout2, size);
            break;
        case 128:
            firstadd5<128><<<numblocks, 128, 128 * sizeof(float)>>>(din, dout, size);
            firstadd5<128><<<numblocks, 128, 128 * sizeof(float)>>>(dout, dout2, size);
            break;
    }

    cudaMemcpy(hout, dout2, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("result = %f\n", hout[0]); // expected answer: 523,776

    // Start benchmark code: Compute and print metrics
    double seconds = ms / 1000.0;
    double bytes   = (double)size * sizeof(float);      // input elements read once
    double gbps    = bytes / seconds / 1e9;

    // --- for step / cumulative speedup, paste in your earlier kernels' times ---
    // --- (measure each version the same way, then fill these two numbers) ------
    double prevTime     = ms;   // <-- replace with the PREVIOUS kernel's ms
    double baselineTime = ms;   // <-- replace with REDUCE-0's ms

    double stepSpeedup = prevTime / ms;
    double cumSpeedup  = baselineTime / ms;

    printf("\n");
    printf("Time (ms)   Bandwidth (GB/s)   Step SpeedUp   Cumulative SpeedUp\n");
    printf("%-11.5f %-18.2f %-14.2f %-.2f\n", ms, gbps, stepSpeedup, cumSpeedup);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // End benchmark code

    cudaFree(din);
    cudaFree(dout);
    cudaFree(dout2);

    delete[] hin;
    delete[] hout;

    return 0;
}

#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>

#include <cmath>
#include <cstdlib>
#include <iostream>

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

int main3()
{
    constexpr int size = 1 << 24;
    constexpr int warmupIters = 1;
    constexpr int benchmarkIters = 1000;

    float* hin = new float[size];

    for (int i = 0; i < size; ++i) {
        hin[i] = static_cast<float>(i);
    }

    float* din = nullptr;
    float* dresult = nullptr;

    CUDA_CHECK(cudaMalloc(&din, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dresult, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        din,
        hin,
        size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // Determine CUB's temporary-storage requirement.
    void* dtemp = nullptr;
    size_t tempBytes = 0;

    CUDA_CHECK(cub::DeviceReduce::Sum(
        dtemp,
        tempBytes,
        din,
        dresult,
        size
    ));

    CUDA_CHECK(cudaMalloc(&dtemp, tempBytes));

    // Warmup
    for (int i = 0; i < warmupIters; ++i) {
        CUDA_CHECK(cub::DeviceReduce::Sum(
            dtemp,
            tempBytes,
            din,
            dresult,
            size
        ));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < benchmarkIters; ++i) {
        CUDA_CHECK(cub::DeviceReduce::Sum(
            dtemp,
            tempBytes,
            din,
            dresult,
            size
        ));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));

    const double averageMs =
        static_cast<double>(totalMs) / benchmarkIters;

    float gpuResult = 0.0f;

    CUDA_CHECK(cudaMemcpy(
        &gpuResult,
        dresult,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // Sum of 0 + 1 + ... + (size - 1)
    const double expected =
        static_cast<double>(size) *
        static_cast<double>(size - 1) / 2.0;

    const double seconds = averageMs / 1000.0;

    // Match your convention: count only the input read.
    const double bytes =
        static_cast<double>(size) * sizeof(float);

    const double bandwidthGBs =
        bytes / seconds / 1.0e9;

    const double elementsPerSecond =
        static_cast<double>(size) / seconds;

    const double relativeError =
        std::abs(static_cast<double>(gpuResult) - expected) /
        expected;

    std::cout << "CUB DeviceReduce::Sum\n";
    std::cout << "Result:               " << gpuResult << '\n';
    std::cout << "Expected:             " << expected << '\n';
    std::cout << "Relative error:       " << relativeError << '\n';
    std::cout << "Temporary storage:    " << tempBytes << " bytes\n";
    std::cout << "Average time:         " << averageMs << " ms\n";
    std::cout << "Effective bandwidth:  " << bandwidthGBs << " GB/s\n";
    std::cout << "Throughput:           "
              << elementsPerSecond / 1.0e9
              << " billion elements/s\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dtemp));
    CUDA_CHECK(cudaFree(dresult));
    CUDA_CHECK(cudaFree(din));

    delete[] hin;

    return 0;
}

int main()
{
    //main1();
    std::cout << "\nREDUCE 6\n";
    main1();
    std::cout << "\nCUSTOM REDUCTION\n";
    main2();
    std::cout << "\nCUB REDUCTION\n";
    main3();
    return 0;
}