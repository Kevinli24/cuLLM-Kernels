#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <cmath>

__global__
void sumouter(float *dnums, float *dres, int size, int stride)
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
        sumouter<<<numblocks, blocksize>>>(input, output, size, stride);
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