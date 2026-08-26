# CUDA Parallel Reduction

CUDA implementations of a floating-point sum reduction, developed progressively from simple global-memory approaches to an optimized shared-memory and warp-shuffle implementation.

## What is included

The source explores these reduction strategies:

1. **Interleaved addressing** — repeatedly combines elements separated by an increasing stride.
2. **Sequential addressing** — reduces the active range by halves in place.
3. **Shared-memory reduction** — loads each block into shared memory and reduces within the block.
4. **First add during load** — each thread loads and adds two input values, reducing global-memory traffic and the number of active threads.
5. **Warp-unrolled reduction** — manually unrolls the final warp-level steps.
6. **Completely unrolled reduction** — uses compile-time block-size specialization.
7. **Grid-stride reduction** — allows each block to process multiple input tiles.
8. **Warp-shuffle reduction** — uses `__shfl_down_sync` for the final warp reduction, avoiding shared-memory traffic in that stage.

The active implementation uses a two-pass reduction: the first kernel produces one partial sum per block, and a second launch reduces those partial sums to a single value.

The program also benchmarks the custom implementation against `cub::DeviceReduce::Sum`, using the same `2^24`-element input and 1,000 timed iterations.

## Algorithm

For an input array `x`:

```text
input array
    ↓  firstadd5: each block computes a partial sum
partial sums
    ↓  firstadd5: one block reduces the partial sums
final sum
```

Each thread accumulates two values per iteration, and the grid-stride loop allows the same launch configuration to cover arrays larger than the initial grid. Within a block, values are reduced in shared memory. The last 32 values are combined with warp shuffle instructions.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit and `nvcc`
- A CUDA-capable C++ compiler
- CUB, which is included with modern CUDA Toolkit installations
- `util.h` if it is included by the source file

The warp-shuffle implementation requires hardware supporting CUDA warp shuffle intrinsics.

## Building

The source contains a standard `main()` that runs the earlier custom reduction, the warp-shuffle custom reduction, and the CUB baseline.

Compile it with:

```bash
nvcc -O3 -arch=sm_XX reduction.cu -o reduction
```

Replace `sm_XX` with the compute capability of the target GPU, for example `sm_86` or `sm_89`. Replace `reduction.cu` with the actual source filename.

Run it with:

```bash
./reduction
```

## Benchmarking

The custom benchmark uses CUDA events and reports:

- Average runtime per two-pass reduction
- Approximate input bandwidth in GB/s
- Step speedup
- Cumulative speedup

The CUB benchmark additionally reports:

- Result and expected result
- Relative error
- Temporary-storage usage
- Average runtime
- Effective bandwidth
- Throughput in billions of elements per second

The custom benchmark's `prevTime` and `baselineTime` variables are placeholders, so its step and cumulative speedup columns are not meaningful until those values are replaced.

The direct custom-versus-CUB comparison is still available through their reported average runtimes and effective bandwidths. Both use the same input size and iteration count, although the code should use identical warmup counts and error-checking procedures for the fairest comparison.

The default benchmark configuration is:

```text
Input size: 2^24 floats
Block size: 256 threads
Timed iterations: 1000
```

## Correctness

The sample initializes the input as:

```cpp
x[i] = i;
```

For an input size of 1024, the expected sum is:

```text
0 + 1 + ... + 1023 = 523,776
```

For the `2^24`-element benchmark, the expected result is calculated as:

```cpp
size * (size - 1) / 2
```

Because floating-point addition is not associative, parallel reduction can produce small numerical differences from the exact result. The CUB benchmark reports relative error to quantify this difference.

## CUB Baseline

The program uses `cub::DeviceReduce::Sum` as an optimized library baseline.

Before benchmarking, it queries CUB for the required temporary-storage size:

```cpp
void* dtemp = nullptr;
size_t tempBytes = 0;

cub::DeviceReduce::Sum(
    dtemp,
    tempBytes,
    din,
    dresult,
    size
);
```

It then allocates the requested storage, performs a warmup, and measures 1,000 reductions using CUDA events.

The benchmark reports:

```text
CUB DeviceReduce::Sum
Result
Expected result
Relative error
Temporary storage
Average time
Effective bandwidth
Throughput
```

## Current limitations

- Several kernels assume the input size is compatible with the launch configuration; production code should add bounds checks for every load.
- CUDA API calls and kernel launches should be checked consistently with `cudaGetLastError()` and `cudaDeviceSynchronize()` during debugging.
- Both benchmarks report effective bandwidth based on input bytes only and do not include all intermediate-output or temporary-storage traffic.
- The custom reduction uses two kernel launches, while CUB may use a different internal strategy.
- The custom and CUB benchmark harnesses should use identical warmup counts and synchronization procedures before drawing strong performance conclusions.
- The custom benchmark's step and cumulative speedup values remain placeholders.
- Results depend heavily on GPU architecture, clock state, compiler flags, and CUDA version.

## Possible next steps

- Add command-line options for input size, block size, and reduction implementation.
- Move each reduction version into its own source file.
- Add automated correctness tests for non-power-of-two input sizes.
- Add bounds checks for all global-memory reads.
- Apply the `CUDA_CHECK` macro consistently throughout the custom implementations.
- Check kernel launches with `cudaGetLastError()`.
- Normalize the custom and CUB benchmark harnesses.
- Profile with Nsight Compute.
- Measure memory throughput, occupancy, branch efficiency, and instruction mix.
- Document the custom-versus-CUB results for a specific GPU.
