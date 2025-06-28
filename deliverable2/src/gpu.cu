#include <iostream>
#include "include/gpu.h"
#include "include/tester.h"
#include "include/type_alias.h"
#include "include/utils.h"

#define MAX_THREAD_PER_BLOCK 1024u
#define MAX_BLOCK 256u

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define READ_FOR_THREAD 2

std::tuple<u32, u32, u32> parameters_for_baseline(const GpuCoo<u32, float>& matrix)
{
    const u32 n_threads = std::min(MAX_THREAD_PER_BLOCK, matrix.NON_ZERO);
    const u32 n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads);
    return {n_blocks, n_threads, 0};
}

std::tuple<u32, u32, u32> parameters_for_basic(const GpuCoo<u32, float>& matrix)
{
    const u32 n_threads = std::min(MAX_THREAD_PER_BLOCK, matrix.NON_ZERO);
    const u32 n_blocks = std::min(MAX_BLOCK, CEIL_DIV(matrix.NON_ZERO, MAX_THREAD_PER_BLOCK));
    return {n_blocks, n_threads, 0};
}

std::tuple<u32, u32, u32> parameters_for_prefix_sum(const GpuCoo<u32, float>& matrix)
{
    const u32 n_threads = std::min(MAX_THREAD_PER_BLOCK, CEIL_DIV(matrix.NON_ZERO, READ_FOR_THREAD));
    const u32 n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads * READ_FOR_THREAD);
    const u32 shm = n_threads * READ_FOR_THREAD * sizeof(float);
    std::cout << "Normal " << n_threads << " " << n_blocks << " " << shm << std::endl;
    return {n_blocks, n_threads, shm};
}

std::tuple<u32, u32, u32> parameters_for_prefix_sum_bkp(const GpuCoo<u32, float>& matrix)
{
    const u32 n_threads = std::min(MAX_THREAD_PER_BLOCK, matrix.NON_ZERO);
    const u32 n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads);
    const u32 shm = n_threads * sizeof(float);
    std::cout << "Backup " << n_threads << " " << n_blocks << " " << shm << std::endl;
    return {n_blocks, n_threads, shm};
}


// ASSUME the result vector is zeroed before calling this function
__global__ void kernel_baseline(const u32* x, const u32* y, const float* val, const float* vec, float* res, const u32 NON_ZERO)
{
    const u32 el = blockIdx.x * blockDim.x + threadIdx.x;
    if (el < NON_ZERO)
        atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
}

// ASSUME the result vector is zeroed before calling this function
__global__ void kernel_full_strided(const u32* x, const u32* y, const float* val, const float* vec, float* res,
                                    const u32 NON_ZERO)
{
    const auto per_thread = CEIL_DIV(NON_ZERO, gridDim.x * blockDim.x);
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;

    const u32 start = tid * per_thread;
    const u32 end = MIN(start + per_thread, NON_ZERO);

    for (u32 el = start; el < end; el++)
        atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
}

// ASSUME the result vector is zeroed before calling this function
__global__ void kernel_full_jump(const u32* x, const u32* y, const float* val, const float* vec, float* res, const u32 NON_ZERO)
{
    const u32 n_threads = gridDim.x * blockDim.x;
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (u32 el = tid; el < NON_ZERO; el += n_threads)
        atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
}

// ASSUME the result vector is zeroed before calling this function
__global__ void kernel_warp_jump(const u32* x, const u32* y, const float* val, const float* vec, float* res, const u32 NON_ZERO)
{
    const u32 per_thread = CEIL_DIV(NON_ZERO, gridDim.x * blockDim.x);
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;

    const u32 wrap_id = tid / warpSize;
    const u32 start = wrap_id * (per_thread * warpSize) + (tid - wrap_id * warpSize);
    const u32 incr = warpSize;

    for (u32 i = 0; i < per_thread; i++)
    {
        if (const u32 el = start + i * incr; el < NON_ZERO)
            atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
    }
}

// ASSUME the result vector is zeroed before calling this function
__global__ void kernel_block_jump(const u32* x, const u32* y, const float* val, const float* vec, float* res, const u32 NON_ZERO)
{
    const u32 cell_per_block = CEIL_DIV(NON_ZERO, gridDim.x);
    const u32 start = blockIdx.x * cell_per_block + threadIdx.x;
    const u32 end = MIN(NON_ZERO, (blockIdx.x + 1) * cell_per_block);

    for (u32 el = start; el < end; el += blockDim.x)
        atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
}

// ASSUME the result vector is zeroed before calling this function
__global__ void kernel_block_jump_unsafe(const u32* x, const u32* y, const float* val, const float* vec, float* res,
                                         const u32 NON_ZERO)
{
    const u32 cell_per_block = CEIL_DIV(NON_ZERO, gridDim.x);
    const u32 start = blockIdx.x * cell_per_block + threadIdx.x;
    const u32 end = MIN(NON_ZERO, (blockIdx.x + 1) * cell_per_block);

    for (u32 el = start; el < end; el += blockDim.x)
        res[y[el]] += val[el] * vec[x[el]];
}


// ASSUME the result vector is zeroed before calling this function
// Compute multiplication
// Compute prefix sum (only fist step)
// Then using edges atomically push to global memory
__global__ void kernel_prefix_sum(const u32* x, const u32* y, const float* val, const float* vec, float* res, const u32 NON_ZERO)
{
    extern __shared__ float prefix[]; // NOLINT(*-redundant-declaration)
    const u32 read_per_block = blockDim.x * READ_FOR_THREAD;
    const u32 base_block = blockIdx.x * read_per_block;

    // Multiplication
    for (int i = 0; i < READ_FOR_THREAD; i++)
    {
        const u32 idx = threadIdx.x * READ_FOR_THREAD + i;
        prefix[idx] = base_block + idx < NON_ZERO ? val[base_block + idx] * vec[x[base_block + idx]] : 0;
    }
    __syncthreads();

    // Partial prefix sum
    for (u32 s = 1; s <= read_per_block / 2; s <<= 1)
    {
        for (int i = 0; i < READ_FOR_THREAD; i++)
        {
            const u32 idx = threadIdx.x * READ_FOR_THREAD + i;
            if (idx + s < read_per_block)
                prefix[idx + s] += prefix[idx];
        }

        __syncthreads();
    }

    // Edge detection and memory write
    for (int i = 0; i < READ_FOR_THREAD; i++)
    {
        const u32 idx = threadIdx.x * READ_FOR_THREAD + i;
        if (base_block + idx + 1 > NON_ZERO)
        {
        }
        else if (idx + 1 == blockDim.x || base_block + idx + 1 == NON_ZERO)
        {
            atomicAdd(&res[y[base_block + idx]], prefix[idx]);
        }
        else if (y[base_block + idx] < y[base_block + idx + 1])
        {
            atomicAdd(&res[y[base_block + idx]], prefix[idx]);
            atomicAdd(&res[y[base_block + idx + 1]], -prefix[idx]);
        }
    }
}


// ASSUME the result vector is zeroed before calling this function
// Compute multiplication
// Compute prefix sum (only fist step)
// Then using edges atomically push to global memory
__global__ void kernel_prefix_sum_bkp(const u32* x, const u32* y, const float* val, const float* vec, float* res,
                                      const u32 NON_ZERO)
{
    extern __shared__ float prefix[]; // NOLINT(*-redundant-declaration)
    const u32 base_block = blockIdx.x * blockDim.x;
    const u32 tid = base_block + threadIdx.x;

    // Multiplication
    prefix[threadIdx.x] = tid < NON_ZERO ? val[tid] * vec[x[tid]] : 0;
    __syncthreads();

    // Partial prefix sum
    for (u32 s = 1; s <= blockDim.x / 2; s <<= 1)
    {
        if (threadIdx.x + s < blockDim.x)
            prefix[threadIdx.x + s] += prefix[threadIdx.x];
        __syncthreads();
    }

    // Edge detection and memory write
    if (tid + 1 > NON_ZERO)
    {
    }
    else if (threadIdx.x + 1 == blockDim.x || tid + 1 == NON_ZERO)
    {
        atomicAdd(&res[y[tid]], prefix[threadIdx.x]);
    }
    else if (y[tid] < y[tid + 1])
    {
        atomicAdd(&res[y[tid]], prefix[threadIdx.x]);
        atomicAdd(&res[y[tid + 1]], -prefix[threadIdx.x]);
    }
}
