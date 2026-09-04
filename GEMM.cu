#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>


// Initial/Naive Implementation of GEMM
/*
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ 
void sgemm_naive(int M, int N, int K, float alpha, 
const float *A, const float *B, float beta, float *C)
{
    // compute position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // 'if' condition is needed for when M or N aren't multiples of 32
    if (x < M && y < N)
    {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i)
        {
            tmp += A[x * K + i] * B[i * N + y];
            // Each thread computes dot product: iterate over A by column and iterate over B
            // by row
        }
        // Assign each temp result to an entry in C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

int main()
{
    int M = 1024, N = 1024, K = 1024;
    float alpha = 1.0f, beta = 0.0f;

    size_t szA = M * K * sizeof(float);
    size_t szB = K * N * sizeof(float);
    size_t szC = M * N * sizeof(float);

    // host memory
    float *hA = (float*)malloc(szA), *hB = (float*)malloc(szB), *hC = (float*)malloc(szC);
    for (int i = 0; i < M * K; ++i) hA[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) hB[i] = 2.0f;
    for (int i = 0; i < M * N; ++i) hC[i] = 0.0f;

    // device memory
    float *dA, *dB, *dC;
    cudaMalloc(&dA, szA); cudaMalloc(&dB, szB); cudaMalloc(&dC, szC);
    cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, szC, cudaMemcpyHostToDevice);

    // put as many blocks into grid needed to span matrix C (output matrix)
    dim3 gridspace(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);

    // each block responsible for calculating a 32x32 chunk of C
    // and each thread independently computes one entry of C
    dim3 blockspace(32, 32, 1);
    sgemm_naive<<<gridspace, blockspace>>>(M, N, K, alpha, dA, dB, beta, dC);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, szC, cudaMemcpyDeviceToHost);
    printf("C[0] = %f (expect %f)\n", hC[0], 2.0f * K);  

    cudaFree(dA); 
    cudaFree(dB); 
    cudaFree(dC);
    free(hA); 
    free(hB); 
    free(hC);

    return 0;
}
*/
// Using Global Memory Coalescing
/*
#define BLOCKSIZE 32
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))


__global__ 
void sgemm_naive(int M, int N, int K, float alpha, 
const float *A, const float *B, float beta, float *C)
{
    // compute position in C that this thread is responsible for
    const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    // 'if' condition is needed for when M or N aren't multiples of 32
    if (x < M && y < N)
    {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i)
        {
            tmp += A[x * K + i] * B[i * N + y];
            // Each thread computes dot product: iterate over A by column and iterate over B
            // by row
        }
        // Assign each temp result to an entry in C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

int main()
{
    int M = 1024, N = 1024, K = 1024;
    float alpha = 1.0f, beta = 0.0f;

    size_t szA = M * K * sizeof(float);
    size_t szB = K * N * sizeof(float);
    size_t szC = M * N * sizeof(float);

    // host memory
    float *hA = (float*)malloc(szA), *hB = (float*)malloc(szB), *hC = (float*)malloc(szC);
    for (int i = 0; i < M * K; ++i) hA[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) hB[i] = 2.0f;
    for (int i = 0; i < M * N; ++i) hC[i] = 0.0f;

    // device memory
    float *dA, *dB, *dC;
    cudaMalloc(&dA, szA); cudaMalloc(&dB, szB); cudaMalloc(&dC, szC);
    cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, szC, cudaMemcpyHostToDevice);

    // put as many blocks into grid needed to span matrix C (output matrix)
    dim3 gridspace(CEIL_DIV(M, 32), CEIL_DIV(N, 32));

    // make blockspace 1-D, but still same size
    dim3 blockspace(32 * 32);
    sgemm_naive<<<gridspace, blockspace>>>(M, N, K, alpha, dA, dB, beta, dC);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, szC, cudaMemcpyDeviceToHost);
    printf("C[0] = %f (expect %f)\n", hC[0], 2.0f * K);  

    cudaFree(dA); 
    cudaFree(dB); 
    cudaFree(dC);
    free(hA); 
    free(hB); 
    free(hC);

    return 0;
}
*/
// Using Shared Memory
/*
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define BLOCKSIZE 32

__global__ void sgemm_shared(int M, int N, int K, float alpha,
                             const float *A, const float *B, float beta, float *C)
{
    // block coordinates: which BLOCKSIZE x BLOCKSIZE tile of C this block owns
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;

    // thread coordinates within the tile (1D block -> 2D, same trick as kernel 2)
    const int threadRow = threadIdx.x / BLOCKSIZE;
    const int threadCol = threadIdx.x % BLOCKSIZE;

    // shared memory tiles for A and B
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // advance pointers to start positions
    A += cRow * BLOCKSIZE * K;                    // row = cRow, col = 0
    B += cCol * BLOCKSIZE;                        // row = 0, col = cCol
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row = cRow, col = cCol

    float tmp = 0.0f;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        __syncthreads();

        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();
    }
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

int main()
{
    int M = 1024, N = 1024, K = 1024;
    float alpha = 1.0f, beta = 0.0f;

    size_t szA = M * K * sizeof(float);
    size_t szB = K * N * sizeof(float);
    size_t szC = M * N * sizeof(float);

    float *hA = (float*)malloc(szA), *hB = (float*)malloc(szB), *hC = (float*)malloc(szC);
    for (int i = 0; i < M * K; ++i) hA[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) hB[i] = 2.0f;
    for (int i = 0; i < M * N; ++i) hC[i] = 0.0f;

    float *dA, *dB, *dC;
    cudaMalloc(&dA, szA); cudaMalloc(&dB, szB); cudaMalloc(&dC, szC);
    cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, szC, cudaMemcpyHostToDevice);

    dim3 grid(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
    dim3 block(BLOCKSIZE * BLOCKSIZE);   // 1D, 1024 threads
    sgemm_shared<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, szC, cudaMemcpyDeviceToHost);
    printf("C[0] = %f (expect %f)\n", hC[0], 2.0f * K);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
*/

// 1D Blocktiling Kernel
// allocate thread-local cache for results in registerfile
float threadResults[TM] = {0.0};

// outer loop over block tiles
for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
  // populate the SMEM caches (same as before)
  As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
  Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
  __syncthreads();

  // advance blocktile for outer loop
  A += BK;
  B += BK * N;

  // calculate per-thread results
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // we make the dotproduct loop the outside loop, which facilitates
    // reuse of the Bs entry, which we can cache in a tmp var.
    float Btmp = Bs[dotIdx * BN + threadCol];
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
      threadResults[resIdx] +=
          As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
    }
  }
  __syncthreads();
}