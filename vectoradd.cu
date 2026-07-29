#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <iostream>

__global__
void vectoradd_kernel(float *vec1, float *vec2, float *vecres, int size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size)
    {
        vecres[tid] = vec1[tid] + vec2[tid];
    }
}

__host__
int main()
{
    int size = 1024;
    int blocksize = 256;
    int numBlocks = (size + blocksize - 1) / blocksize; // ceiling func

    float *h_vec1 = new float[size];
    float *h_vec2 = new float[size];
    float *h_vecres = new float[size];

    for (int i = 0; i < size; ++i)
    {
        h_vec1[i] = (float)i;
        h_vec2[i] = (float)(i + 1);
    }

    float *d_vec1, *d_vec2, *d_vecres;

    cudaMalloc(&d_vec1, size * sizeof(float));
    cudaMalloc(&d_vec2, size * sizeof(float));
    cudaMalloc(&d_vecres, size * sizeof(float));

    cudaMemcpy(d_vec1, h_vec1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, h_vec2, size * sizeof(float), cudaMemcpyHostToDevice);

    vectoradd_kernel<<<numBlocks, blocksize>>>(d_vec1, d_vec2, d_vecres, size);
    cudaMemcpy(h_vecres, d_vecres, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i)
    {
        printf("%f + %f = %f\n", h_vec1[i], h_vec2[i], h_vecres[i]);
    }

    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_vecres);

    delete[] h_vec1;
    delete[] h_vec2;
    delete[] h_vecres;

    return 0;
}