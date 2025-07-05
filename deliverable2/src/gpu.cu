#include <iostream>
#include "include/gpu.h"
#include "include/type_alias.h"
#include "include/utils.h"


// ASSUME the result vector is zeroed before calling this function
__global__ void kernel_baseline(const u32* x, const u32* y, const MV* val, const MV* vec, MV* res, const u32 NON_ZERO)
{
    const u32 start = blockIdx.x * blockDim.x;
    const u32 el = start + threadIdx.x;
    if (el < NON_ZERO)
        atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
}

__global__ void kernel_block_jump_shared(const u32* x, const u32* y, const MV* val, const MV* vec, MV* res,
                                         const u32 NON_ZERO)
{
    // ReSharper disable once CppTooWideScope
    extern __shared__ MV prefix[]; // NOLINT(*-redundant-declaration)
    const u32 start = blockIdx.x * blockDim.x;
    const u32 el = start + threadIdx.x;
    if (el < NON_ZERO)
    {
        prefix[2 * threadIdx.x] = val[el];
        prefix[2 * threadIdx.x + 1] = vec[x[el]];

        atomicAdd(&res[y[el]], prefix[2 * threadIdx.x] * prefix[2 * threadIdx.x + 1]);
    }
}

// ASSUME the result vector is zeroed before calling this function
__global__ void kernel_full_strided(const u32* x, const u32* y, const MV* val, const MV* vec, MV* res,
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
__global__ void kernel_full_jump(const u32* x, const u32* y, const MV* val, const MV* vec, MV* res, const u32 NON_ZERO)
{
    const u32 n_threads = gridDim.x * blockDim.x;
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (u32 el = tid; el < NON_ZERO; el += n_threads)
        atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
}

// ASSUME the result vector is zeroed before calling this function
__global__ void kernel_warp_jump(const u32* x, const u32* y, const MV* val, const MV* vec, MV* res, const u32 NON_ZERO)
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
__global__ void kernel_block_jump(const u32* x, const u32* y, const MV* val, const MV* vec, MV* res, const u32 NON_ZERO)
{
    const u32 cell_per_block = CEIL_DIV(NON_ZERO, gridDim.x * blockDim.x) * blockDim.x;
    const u32 start = blockIdx.x * cell_per_block + threadIdx.x;
    const u32 end = MIN(NON_ZERO, (blockIdx.x + 1) * cell_per_block);

    for (u32 el = start; el < end; el += blockDim.x)
        atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
}


// ASSUME the result vector is zeroed before calling this function
__global__ void kernel_block_jump_unsafe(const u32* x, const u32* y, const MV* val, const MV* vec, MV* res,
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
__global__ void kernel_prefix_sum_multiple_read(const u32* x, const u32* y, const MV* val, const MV* vec, MV* res,
                                                const u32 NON_ZERO)
{
    extern __shared__ MV prefix[]; // NOLINT(*-redundant-declaration)
    const u32 read_per_block = blockDim.x * READ_FOR_THREAD;
    const u32 base_block = blockIdx.x * read_per_block;

    // Multiplication
    for (int i = 0; i < READ_FOR_THREAD; i++)
    {
        const u32 idx = threadIdx.x + i * blockDim.x;
        prefix[idx] = base_block + idx < NON_ZERO ? val[base_block + idx] * vec[x[base_block + idx]] : 0;
        printf("MUL [%d] %d %f ([%d] %d %d %f -> %f)\n", threadIdx.x, idx, prefix[idx], base_block + idx,
               x[base_block + idx], y[base_block + idx], val[base_block + idx], vec[x[base_block + idx]]);
    }
    __syncthreads();
    if (threadIdx.x == 0)
        printf("\n");

    // Partial prefix sum
    for (u32 s = 1; s <= read_per_block / 2; s <<= 1)
    {
        for (int i = READ_FOR_THREAD - 1; i >= 0; i--)
        {
            const u32 idx = threadIdx.x + i * blockDim.x;
            if (idx + s < read_per_block)
                prefix[idx + s] += prefix[idx];
        }

        for (int i = 0; i < READ_FOR_THREAD; i++)
        {
            const u32 idx = threadIdx.x + i * blockDim.x;
            printf("SUM [%d] %d %f ([%d] %d %d %f -> %f)\n", threadIdx.x, idx, prefix[idx], base_block + idx,
                   x[base_block + idx], y[base_block + idx], val[base_block + idx], vec[x[base_block + idx]]);
        }

        if (threadIdx.x == 0)
            printf("\n");
        __syncthreads();
    }

    // Edge detection and memory write
    for (int i = 0; i < READ_FOR_THREAD; i++)
    {
        const u32 idx = threadIdx.x + i * blockDim.x;
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
__global__ void kernel_prefix_sum_32_max(const u32* x, const u32* y, const MV* val, const MV* vec, MV* res,
                                         const u32 NON_ZERO)
{
    extern __shared__ MV prefix[]; // NOLINT(*-redundant-declaration)
    const u32 base_block = blockIdx.x * blockDim.x;
    const u32 tid = base_block + threadIdx.x;

    // Multiplication
    prefix[threadIdx.x] = tid < NON_ZERO ? val[tid] * vec[x[tid]] : 0;
    __syncthreads();

    // printf("MUL %d %f (%d %d %f -> %f)\n", threadIdx.x, prefix[threadIdx.x], x[tid], y[tid], val[tid], vec[x[tid]]);
    // if (threadIdx.x == 0)
    //     printf("\n");

    // Partial prefix sum
    for (u32 s = 1; s <= blockDim.x / 2; s <<= 1)
    {
        if (threadIdx.x + s < blockDim.x)
            prefix[threadIdx.x + s] += prefix[threadIdx.x];

        __syncthreads();
    }

    // printf("SUM %d %f (%d %d %f)\n", threadIdx.x, prefix[threadIdx.x], x[tid], y[tid], val[tid]);
    // if (threadIdx.x == 0)
    //     printf("\n");

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


// ASSUME the result vector is zeroed before calling this function
// Compute multiplication
// Compute prefix sum (only fist step)
// Then using edges atomically push to global memory
__global__ void kernel_prefix_sum_warp(const u32* x, const u32* y, const MV* val, const MV* vec, MV* res,
                                       const u32 NON_ZERO)
{
    const u32 base_block = blockIdx.x * blockDim.x;
    const u32 tid = base_block + threadIdx.x;
    const u32 tid_warp = threadIdx.x & (warpSize - 1);

    // Multiplication
    MV prefix_i = tid < NON_ZERO ? val[tid] * vec[x[tid]] : 0;

    // printf("MUL %d %f (%d %d %f -> %f)\n", threadIdx.x, prefix_i, x[tid], y[tid], val[tid], vec[x[tid]]);
    // if (threadIdx.x == 0)
    //     printf("\n");

    // Partial prefix sum
    for (u32 s = 1; s <= warpSize / 2; s <<= 1)
    {
        const MV to_add = __shfl_up_sync(0xffffffff, prefix_i, s);
        if (tid_warp >= s)
            prefix_i += to_add;
        // printf("SUM %d %f\n", threadIdx.x, prefix_i);
        // if (threadIdx.x == 0)
        //     printf("\n");
    }


    // Edge detection and memory write
    if (tid + 1 > NON_ZERO)
    {
    }
    else if (tid_warp + 1 == warpSize || tid + 1 == NON_ZERO)
    {
        atomicAdd(&res[y[tid]], prefix_i);
    }
    else if (y[tid] < y[tid + 1])
    {
        atomicAdd(&res[y[tid]], prefix_i);
        atomicAdd(&res[y[tid + 1]], -prefix_i);
    }
}


// ASSUME the result vector is zeroed before calling this function
// Compute multiplication
// Compute prefix sum (only fist step)
// Then using edges atomically push to global memory
__global__ void kernel_prefix_sum_warp_merged(const u32* x, const u32* y, const MV* val, const MV* vec, MV* res,
                                              const u32 NON_ZERO)
{
    extern __shared__ MV first_elements[]; // NOLINT(*-redundant-declaration)
    const u32 base_block = blockIdx.x * blockDim.x;
    const u32 tid = base_block + threadIdx.x;
    const u32 tid_warp = threadIdx.x & (warpSize - 1);
    const u32 id_warp = threadIdx.x >> 5;
    // HARDCODED WARP SIZE FOR PERFORMANCE

    // Multiplication
    MV prefix_i = tid < NON_ZERO ? val[tid] * vec[x[tid]] : 0;
    MV prefix_base = 0;

    // printf("MUL %d %f (%d %d %f -> %f)\n", threadIdx.x, prefix_i, x[tid], y[tid], val[tid], vec[x[tid]]);
    // if (threadIdx.x == 0)
    //     printf("\n");

    // Partial prefix sum
    for (u32 s = 1; s <= warpSize >> 1; s <<= 1)
    {
        const MV to_add = __shfl_up_sync(0xffffffff, prefix_i, s);
        if (tid_warp >= s)
            prefix_i += to_add;
    }

    // Put sum in first element of each warp
    if (tid_warp == 0)
    {
        first_elements[id_warp] = prefix_i;
        // printf("FIRSTS %d %f\n", id_warp, prefix_i);
        // if (id_warp == 0)
        //     printf("\n");
    }

    __syncthreads();
    if (threadIdx.x < 32)
    {
        float first_el_of_warp = first_elements[tid_warp];
        for (u32 s = 1; s <= warpSize >> 1; s <<= 1)
        {
            const float to_add = __shfl_up_sync(0xffffffff, first_el_of_warp, s);
            if (tid_warp >= s)
                first_el_of_warp += to_add;
        }
        first_elements[tid_warp] = first_el_of_warp;
    }
    __syncthreads();
    if (tid_warp == 0)
    {
        prefix_base = id_warp > 1 ? first_elements[id_warp - 1] : 0;
        // printf("FIRSTS_SUM %d %f\n", id_warp, prefix_i);
        // if (id_warp == 0)
        //     printf("\n");
    }

    // Partial prefix sum
    for (u32 s = 1; s <= warpSize / 2; s <<= 1)
    {
        const float to_add = __shfl_up_sync(0xffffffff, prefix_base, s);
        if (tid_warp >= s)
            prefix_base += to_add;
    }
    prefix_i += prefix_base;

    // printf("SUM %d %f\n", threadIdx.x, prefix_i);
    // if (threadIdx.x == 0)
    //     printf("\n");


    // Edge detection and memory write
    if (tid + 1 > NON_ZERO)
    {
    }
    else if (threadIdx.x + 1 == blockDim.x || tid + 1 == NON_ZERO)
    {
        atomicAdd(&res[y[tid]], prefix_i);
    }
    else if (y[tid] < y[tid + 1])
    {
        atomicAdd(&res[y[tid]], prefix_i);
        atomicAdd(&res[y[tid + 1]], -prefix_i);
    }
}


// ASSUME the result vector is zeroed before calling this function
// Compute multiplication
// Compute prefix sum (only fist step)
// Then using edges atomically push to global memory
__global__ void kernel_prefix_sum_warp_with_block_jump(const u32* x, const u32* y, const MV* val, const MV* vec,
                                                       MV* res, const u32 NON_ZERO)
{
    const u32 cell_per_block = CEIL_DIV(NON_ZERO, gridDim.x * blockDim.x) * blockDim.x;
    const u32 start = blockIdx.x * cell_per_block + threadIdx.x;
    const u32 end = MIN(NON_ZERO, (blockIdx.x + 1) * cell_per_block);

    for (u32 tid = start; tid < end; tid += blockDim.x)
    {
        const u32 tid_warp = threadIdx.x & (warpSize - 1);

        // Multiplication
        MV prefix_i = tid < NON_ZERO ? val[tid] * vec[x[tid]] : 0;

        // printf("MUL %d %f (%d %d %f -> %f)\n", threadIdx.x, prefix_i, x[tid], y[tid], val[tid], vec[x[tid]]);
        // if (threadIdx.x == 0)
        //     printf("\n");

        // Partial prefix sum
        for (u32 s = 1; s <= warpSize / 2; s <<= 1)
        {
            const float to_add = __shfl_up_sync(0xffffffff, prefix_i, s);
            if (tid_warp >= s)
                prefix_i += to_add;
            // printf("SUM %d %f\n", threadIdx.x, prefix_i);
            // if (threadIdx.x == 0)
            //     printf("\n");
        }


        // Edge detection and memory write
        if (tid + 1 > NON_ZERO)
        {
        }
        else if (tid_warp + 1 == warpSize || tid + 1 == NON_ZERO)
        {
            atomicAdd(&res[y[tid]], prefix_i);
        }
        else if (y[tid] < y[tid + 1])
        {
            atomicAdd(&res[y[tid]], prefix_i);
            atomicAdd(&res[y[tid + 1]], -prefix_i);
        }
    }
}

// ASSUME the result vector is zeroed before calling this function
// Compute multiplication
// Compute prefix sum (only fist step)
// Then using edges atomically push to global memory
__global__ void kernel_prefix_sum_unlimited(const u32* x, const u32* y, const MV* val, const MV* vec, MV* res,
                                            const u32 NON_ZERO)
{
    extern __shared__ MV prefix[]; // NOLINT(*-redundant-declaration)
    const u32 base_block = blockIdx.x * blockDim.x;
    const u32 tid = base_block + threadIdx.x;
    u32 pout = 0, pin = 1;

    // Multiplication
    prefix[pout * blockDim.x + threadIdx.x] = tid < NON_ZERO ? val[tid] * vec[x[tid]] : 0;
    __syncthreads();

    // printf("MUL %d %f (%d %d %f -> %f)\n", threadIdx.x, prefix[pout * blockDim.x + threadIdx.x], x[tid], y[tid],
    //        val[tid], vec[x[tid]]);
    // if (threadIdx.x == 0)
    //     printf("\n");

    // Partial prefix sum
    for (u32 s = 1; s <= blockDim.x / 2; s <<= 1)
    {
        // swap double buffer indices
        pout = 1 - pout;
        pin = 1 - pout;

        if (threadIdx.x >= s)
            prefix[pout * blockDim.x + threadIdx.x] =
                prefix[pin * blockDim.x + threadIdx.x] + prefix[pin * blockDim.x + threadIdx.x - s];
        else
            prefix[pout * blockDim.x + threadIdx.x] = prefix[pin * blockDim.x + threadIdx.x];
        __syncthreads();

        // printf("SUM %d %f old %f (%d %d %f)\n", threadIdx.x, prefix[pout * blockDim.x + threadIdx.x],
        //        prefix[pin * blockDim.x + threadIdx.x], x[tid], y[tid], val[tid]);
        // if (threadIdx.x == 0)
        //     printf("\n");
    }


    // Edge detection and memory write
    if (tid + 1 > NON_ZERO)
    {
    }
    else if (threadIdx.x + 1 == blockDim.x || tid + 1 == NON_ZERO)
    {
        atomicAdd(&res[y[tid]], prefix[pout * blockDim.x + threadIdx.x]);
    }
    else if (y[tid] < y[tid + 1])
    {
        atomicAdd(&res[y[tid]], prefix[pout * blockDim.x + threadIdx.x]);
        atomicAdd(&res[y[tid + 1]], -prefix[pout * blockDim.x + threadIdx.x]);
    }
}

__global__ void kernel_prefix_sum_max_32_work_efficient(const u32* x, const u32* y, const MV* val, const MV* vec,
                                                        MV* res, const u32 NON_ZERO)
{
    extern __shared__ MV prefix[]; // NOLINT(*-redundant-declaration)
    const u32 shared_size = blockDim.x * 2;
    const u32 base_block = blockIdx.x * shared_size;
    const u32 thid = threadIdx.x;

    // Multiplication
    const u32 in1 = 2 * thid;
    const u32 in2 = 2 * thid + 1;
    prefix[in1] = base_block + in1 + 1 < NON_ZERO ? val[base_block + in1 + 1] * vec[x[base_block + in1 + 1]] : 0;
    prefix[in2] = base_block + in2 + 1 < NON_ZERO && threadIdx.x != blockDim.x - 1
        ? val[base_block + in2 + 1] * vec[x[base_block + in2 + 1]]
        : 0;
    __syncthreads();
    // printf("MUL %d %f (upper is %d %d %f -> %f)\nMUL %d %f (upper is %d %d %f -> %f)\n", base_block + in1,
    // prefix[in1],
    //        x[base_block + in1], y[base_block + in1], val[base_block + in1], vec[x[base_block + in1]], base_block +
    //        in2, prefix[in2], x[base_block + in2], y[base_block + in2], val[base_block + in2], vec[x[base_block +
    //        in2]]);
    // if (threadIdx.x == 0)
    //     printf("\n");

    // Partial prefix sum
    // build sum in place up the tree
    u32 offset = 1;
    for (u32 d = shared_size >> 1; d > 0; d >>= 1)
    {
        if (thid < d)
        {
            const u32 ai = offset * (2 * thid + 1) - 1;
            const u32 bi = offset * (2 * thid + 2) - 1;
            prefix[bi] += prefix[ai];
        }
        offset <<= 1;
        __syncthreads();
    }
    // printf("P1 %d %f\nP1 %d %f\n", base_block + in1, prefix[in1], base_block + in2, prefix[in2]);
    // if (threadIdx.x == 0)
    //     printf("\n");

    // clear the last element
    if (thid == 0)
        prefix[shared_size - 1] = val[base_block] * vec[x[base_block]];
    __syncthreads();

    // printf("P1.5 %d %f\nP1.5 %d %f\n", base_block + in1, prefix[in1], base_block + in2, prefix[in2]);
    // if (threadIdx.x == 0)
    //     printf("\n");

    // traverse down tree & build scan
    for (u32 d = 1; d < shared_size; d <<= 1)
    {
        offset >>= 1;
        if (thid < d)
        {
            const u32 ai = offset * (2 * thid + 1) - 1;
            const u32 bi = offset * (2 * thid + 2) - 1;

            const MV t = prefix[ai];
            prefix[ai] = prefix[bi];
            prefix[bi] += t;
        }
        __syncthreads();
    }

    // printf("P2 %d %f\nP2 %d %f\n", base_block + in1, prefix[in1], base_block + in2, prefix[in2]);
    // if (threadIdx.x == 0)
    //     printf("\n");

    // Edge detection and memory write
    for (u32 el = 2 * thid; el < 2 * (thid + 1); el++)
    {
        const u32 tid = base_block + el;
        if (tid + 1 > NON_ZERO)
        {
        }
        else if (el + 1 == shared_size || tid + 1 == NON_ZERO)
        {
            atomicAdd(&res[y[tid]], prefix[el]);
        }
        else if (y[tid] < y[tid + 1])
        {
            atomicAdd(&res[y[tid]], prefix[el]);
            atomicAdd(&res[y[tid + 1]], -prefix[el]);
        }
    }
}


__global__ void kernel_prefix_sum_we_32_conflict_free(const u32* x, const u32* y, const MV* val, const MV* vec, MV* res,
                                                      const u32 NON_ZERO)
{
    extern __shared__ MV prefix[]; // NOLINT(*-redundant-declaration)
    const u32 shared_size = blockDim.x * 2;
    const u32 base_block = blockIdx.x * shared_size;
    const u32 thid = threadIdx.x;

    // Multiplication
    u32 in1 = 2 * thid;
    in1 += CONFLICT_FREE_OFFSET(in1);
    u32 in2 = 2 * thid + 1;
    in2 += CONFLICT_FREE_OFFSET(in2);
    prefix[in1] = base_block + in1 + 1 < NON_ZERO ? val[base_block + in1 + 1] * vec[x[base_block + in1 + 1]] : 0;
    prefix[in2] = base_block + in2 + 1 < NON_ZERO && threadIdx.x != blockDim.x - 1
        ? val[base_block + in2 + 1] * vec[x[base_block + in2 + 1]]
        : 0;
    __syncthreads();
    // printf("MUL %d %f (upper is %d %d %f -> %f)\nMUL %d %f (upper is %d %d %f -> %f)\n", base_block + in1,
    // prefix[in1],
    //        x[base_block + in1], y[base_block + in1], val[base_block + in1], vec[x[base_block + in1]], base_block +
    //        in2, prefix[in2], x[base_block + in2], y[base_block + in2], val[base_block + in2], vec[x[base_block +
    //        in2]]);
    // if (threadIdx.x == 0)
    //     printf("\n");

    // Partial prefix sum
    // build sum in place up the tree
    u32 offset = 1;
    for (u32 d = shared_size >> 1; d > 0; d >>= 1)
    {
        if (thid < d)
        {
            u32 ai = offset * (2 * thid + 1) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            u32 bi = offset * (2 * thid + 2) - 1;
            bi += CONFLICT_FREE_OFFSET(bi);

            prefix[bi] += prefix[ai];
        }
        offset <<= 1;
        __syncthreads();
    }
    // printf("P1 %d %f\nP1 %d %f\n", base_block + in1, prefix[in1], base_block + in2, prefix[in2]);
    // if (threadIdx.x == 0)
    //     printf("\n");

    // clear the last element
    if (thid == 0)
        prefix[shared_size - 1] = val[base_block] * vec[x[base_block]];
    __syncthreads();

    // printf("P1.5 %d %f\nP1.5 %d %f\n", base_block + in1, prefix[in1], base_block + in2, prefix[in2]);
    // if (threadIdx.x == 0)
    //     printf("\n");

    // traverse down tree & build scan
    for (u32 d = 1; d < shared_size; d <<= 1)
    {
        offset >>= 1;
        if (thid < d)
        {
            u32 ai = offset * (2 * thid + 1) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            u32 bi = offset * (2 * thid + 2) - 1;
            bi += CONFLICT_FREE_OFFSET(bi);

            const MV t = prefix[ai];
            prefix[ai] = prefix[bi];
            prefix[bi] += t;
        }
        __syncthreads();
    }

    // printf("P2 %d %f\nP2 %d %f\n", base_block + in1, prefix[in1], base_block + in2, prefix[in2]);
    // if (threadIdx.x == 0)
    //     printf("\n");

    // Edge detection and memory write
    for (u32 i = 2 * thid; i < 2 * (thid + 1); i++)
    {
        const u32 el = i + CONFLICT_FREE_OFFSET(i);
        const u32 tid = base_block + el;
        if (tid + 1 > NON_ZERO)
        {
        }
        else if (el + 1 == shared_size || tid + 1 == NON_ZERO)
        {
            atomicAdd(&res[y[tid]], prefix[el]);
        }
        else if (y[tid] < y[tid + 1])
        {
            atomicAdd(&res[y[tid]], prefix[el]);
            atomicAdd(&res[y[tid + 1]], -prefix[el]);
        }
    }
}


__global__ void kernel_prefix_sum_we_unlimited_conflict_free(const u32* x, const u32* y, const MV* val, const MV* vec,
                                                             MV* res, const u32 NON_ZERO)
{
    extern __shared__ MV prefix[]; // NOLINT(*-redundant-declaration)
    const u32 shared_size = blockDim.x * 2;
    const u32 base_block = blockIdx.x * shared_size;
    const u32 thid = threadIdx.x;
    u32 pout = 0, pin = 1;

    // Multiplication
    u32 in1 = 2 * thid;
    in1 += CONFLICT_FREE_OFFSET(in1);
    u32 in2 = 2 * thid + 1;
    in2 += CONFLICT_FREE_OFFSET(in2);
    prefix[pout * shared_size + in1] =
        base_block + in1 + 1 < NON_ZERO ? val[base_block + in1 + 1] * vec[x[base_block + in1 + 1]] : 0;
    prefix[pout * shared_size + in2] = base_block + in2 + 1 < NON_ZERO && threadIdx.x != blockDim.x - 1
        ? val[base_block + in2 + 1] * vec[x[base_block + in2 + 1]]
        : 0;
    __syncthreads();
    // printf("MUL %d %f (upper is %d %d %f -> %f)\nMUL %d %f (upper is %d %d %f -> %f)\n", base_block + in1,
    // prefix[in1],
    //        x[base_block + in1], y[base_block + in1], val[base_block + in1], vec[x[base_block + in1]], base_block +
    //        in2, prefix[in2], x[base_block + in2], y[base_block + in2], val[base_block + in2], vec[x[base_block +
    //        in2]]);
    // if (threadIdx.x == 0)
    //     printf("\n");

    // Partial prefix sum
    // build sum in place up the tree
    u32 offset = 1;
    for (u32 d = shared_size >> 1; d > 0; d >>= 1)
    {
        pout = 1 - pout;
        pin = 1 - pin;
        prefix[pout * shared_size + in1] = prefix[pin * shared_size + in1];
        prefix[pout * shared_size + in2] = prefix[pin * shared_size + in2];
        __syncthreads();

        if (thid < d)
        {
            u32 ai = offset * (2 * thid + 1) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            u32 bi = offset * (2 * thid + 2) - 1;
            bi += CONFLICT_FREE_OFFSET(bi);

            prefix[pout * shared_size + bi] = prefix[pin * shared_size + bi] + prefix[pin * shared_size + ai];
        }
        offset <<= 1;
        __syncthreads();
    }
    // printf("P1 %d %f\nP1 %d %f\n", base_block + in1, prefix[in1], base_block + in2, prefix[in2]);
    // if (threadIdx.x == 0)
    //     printf("\n");

    // clear the last element
    if (thid == 0)
        prefix[pout * shared_size + shared_size - 1] = val[base_block] * vec[x[base_block]];
    __syncthreads();

    // printf("P1.5 %d %f\nP1.5 %d %f\n", base_block + in1, prefix[in1], base_block + in2, prefix[in2]);
    // if (threadIdx.x == 0)
    //     printf("\n");

    // traverse down tree & build scan
    for (u32 d = 1; d < shared_size; d <<= 1)
    {
        pout = 1 - pout;
        pin = 1 - pin;
        prefix[pout * shared_size + in1] = prefix[pin * shared_size + in1];
        prefix[pout * shared_size + in2] = prefix[pin * shared_size + in2];
        __syncthreads();

        offset >>= 1;
        if (thid < d)
        {
            u32 ai = offset * (2 * thid + 1) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            u32 bi = offset * (2 * thid + 2) - 1;
            bi += CONFLICT_FREE_OFFSET(bi);

            const MV t = prefix[pin * shared_size + ai];
            prefix[pout * shared_size + ai] = prefix[pin * shared_size + bi];
            prefix[pout * shared_size + bi] = prefix[pin * shared_size + bi] + t;
        }
        __syncthreads();
    }

    // printf("P2 %d %f\nP2 %d %f\n", base_block + in1, prefix[in1], base_block + in2, prefix[in2]);
    // if (threadIdx.x == 0)
    //     printf("\n");

    // Edge detection and memory write
    for (u32 i = 2 * thid; i < 2 * (thid + 1); i++)
    {
        const u32 el = i + CONFLICT_FREE_OFFSET(i);
        const u32 tid = base_block + el;
        if (tid + 1 > NON_ZERO)
        {
        }
        else if (el + 1 == shared_size || tid + 1 == NON_ZERO)
        {
            atomicAdd(&res[y[tid]], prefix[pout * shared_size + el]);
        }
        else if (y[tid] < y[tid + 1])
        {
            atomicAdd(&res[y[tid]], prefix[pout * shared_size + el]);
            atomicAdd(&res[y[tid + 1]], -prefix[pout * shared_size + el]);
        }
    }
}
