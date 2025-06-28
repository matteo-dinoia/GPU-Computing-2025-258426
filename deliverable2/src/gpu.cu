#include <cstdint>
#include <iostream>
#include "include/tester.h"
#include "include/utils.h"
#include "include/type_alias.h"

#define MAX_THREAD_PER_BLOCK 1024u
#define MAX_BLOCK 256u

#define MIN(a, b) ((a) < (b) ? (a) : (b))


std::pair<u32, u32> parameters_for_baseline(const GpuCoo<u32, float>& matrix)
{
    const u32 n_threads = std::min<u32>(MAX_THREAD_PER_BLOCK, matrix.NON_ZERO);
    const u32 n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads);
    return std::make_pair(n_blocks, n_threads);
}

std::pair<u32, u32> parameters_for_basic(const GpuCoo<u32, float>& matrix)
{
    const u32 n_threads = std::min<u32>(MAX_THREAD_PER_BLOCK, matrix.NON_ZERO);
    const u32 n_blocks = std::min(MAX_BLOCK, CEIL_DIV(matrix.NON_ZERO, MAX_THREAD_PER_BLOCK));
    return std::make_pair(n_blocks, n_threads);
}


// ASSUME the result vector is zeroed before calling this function
__global__ void kernel_baseline(const u32* x, const u32* y, const float* val, const float* vec, float* res,
                                const u32 NON_ZERO)
{
    const u32 el = blockIdx.x * blockDim.x + threadIdx.x;
    if (el < NON_ZERO)
        atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
}

// ASSUME the result vector is zeroed before calling this function
__global__ void kernel_full_strided(const u32* x, const u32* y, const float* val, const float* vec, float* res,
                                    const u32 NON_ZERO)
{
    const u32 n_threads = gridDim.x * blockDim.x;
    const auto per_thread = CEIL_DIV(NON_ZERO, n_threads);
    const u32 start_i = blockIdx.x * blockDim.x + threadIdx.x;

    const u32 start = start_i * per_thread;
    const u32 end = MIN(start + per_thread, NON_ZERO);

    for (u32 el = start; el < end; el++)
        atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
}

// ASSUME the result vector is zeroed before calling this function
__global__ void kernel_full_jump(const u32* x, const u32* y, const float* val, const float* vec, float* res,
                                 const u32 NON_ZERO)
{
    const u32 n_threads = gridDim.x * blockDim.x;
    const auto per_thread = CEIL_DIV(NON_ZERO, n_threads);
    const u32 start_i = blockIdx.x * blockDim.x + threadIdx.x;

    const u32 start = start_i;
    const u32 end = MIN(start_i + per_thread * n_threads, NON_ZERO);

    for (u32 el = start; el < end; el += n_threads)
        atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
}

// // ASSUME the result vector is zeroed before calling this function
// __global__ void kernel_warp_jump(const u32* x, const u32* y, const float* val, const float* vec, float* res,
//                                  const u32 NON_ZERO)
// {
//     const u32 per_thread = CEIL_DIV(NON_ZERO, gridDim.x * blockDim.x);
//     const u32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//     const u32 warp_id = thread_id * warpSize;
//
//     const u32 t = warp_id / warpSize;
//     const u32 start = (t * per_thread) + (thread_id - t);
//     const u32 end = MIN(NON_ZERO, (t + warpSize) * per_thread);
//
//     for (u32 el = start; el < end; el += warpSize)
//         atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
// }

// ASSUME the result vector is zeroed before calling this function
__global__ void kernel_warp_jump(const u32* x, const u32* y, const float* val, const float* vec, float* res,
                                 const u32 NON_ZERO)
{
    const u32 n_threads = gridDim.x * blockDim.x;
    const u32 per_thread = CEIL_DIV(NON_ZERO, n_threads);
    const u32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    const u32 wrap_id = thread_id / warpSize;
    const u32 start = wrap_id * (per_thread * warpSize) + (thread_id - wrap_id * warpSize);
    const u32 incr = warpSize;

    for (u32 i = 0; i < per_thread; i++)
    {
        const u32 el = start + i * incr;
        if (el < NON_ZERO)
            return;
        atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
    }
}

// ASSUME the result vector is zeroed before calling this function
__global__ void kernel_warp_jump_bkp(const u32* x, const u32* y, const float* val, const float* vec, float* res,
                                     const u32 NON_ZERO)
{
    const u32 n_threads = gridDim.x * blockDim.x;
    const u32 per_thread = CEIL_DIV(NON_ZERO, n_threads);
    const u32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    const u32 wrap_id = thread_id / warpSize;
    const u32 start = wrap_id * (per_thread * warpSize) + (thread_id - wrap_id * warpSize);
    const u32 incr = warpSize;

    for (u32 i = 0; i < per_thread; i++)
    {
        if (const u32 el = start + i * incr; el < NON_ZERO)
            atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
    }
}

// ASSUME the result vector is zeroed before calling this function
__global__ void kernel_block_jump(const u32* x, const u32* y, const float* val, const float* vec, float* res,
                                  const u32 NON_ZERO)
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
__global__ void kernel_block_jump_bkp(const u32* x, const u32* y, const float* val, const float* vec, float* res,
                                      const u32 NON_ZERO)
{
    const u32 n_threads = gridDim.x * blockDim.x;
    const auto per_thread = CEIL_DIV(NON_ZERO, n_threads);

    const u32 start = blockIdx.x * (per_thread * blockDim.x) + threadIdx.x;
    const u32 incr = blockDim.x;

    for (u32 i = 0; i < per_thread; i++)
    {
        if (const u32 el = start + i * incr; el < NON_ZERO)
            atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
    }
}
