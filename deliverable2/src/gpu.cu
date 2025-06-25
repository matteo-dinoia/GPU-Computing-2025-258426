#define MIN(a, b) ((a) <= (b) ? (a) : (b))
#include <cstdint>
#include "include/utils.h"


// Kernel function to add the elements of two arrays
// ASSUME it is zeroed the res vector
__global__ void spmv_baseline(const uint32_t* x, const uint32_t* y, const float* val, const float* vec, float* res,
                              const uint32_t NON_ZERO)
{
    if (const uint32_t el = blockIdx.x * blockDim.x + threadIdx.x; el < NON_ZERO)
    {
        atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
    }
}

// Kernel function to add the elements of two arrays
// ASSUME it is zeroed the res vector
__global__ void spmv_full_strided(const uint32_t* x, const uint32_t* y, const float* val, const float* vec, float* res,
                                  const uint32_t NON_ZERO)
{
    const uint32_t n_threads = gridDim.x * blockDim.x;
    const auto per_thread = CEIL_DIV(NON_ZERO, n_threads);
    const uint32_t start_i = blockIdx.x * blockDim.x + threadIdx.x;

    const uint32_t start = start_i * per_thread;
    const uint32_t end = MIN(start + per_thread, NON_ZERO);
    constexpr uint32_t incr = 1;

    for (uint32_t el = start; el < end; el += incr)
    {
        atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
    }
}

// Kernel function to add the elements of two arrays
// ASSUME it is zeroed the res vector
__global__ void spmv_full_jump(const uint32_t* x, const uint32_t* y, const float* val, const float* vec, float* res,
                               const uint32_t NON_ZERO)
{
    const uint32_t n_threads = gridDim.x * blockDim.x;
    const auto per_thread = CEIL_DIV(NON_ZERO, n_threads);
    const uint32_t start_i = blockIdx.x * blockDim.x + threadIdx.x;

    const uint32_t start = start_i;
    const uint32_t end = MIN(start_i + per_thread * n_threads, NON_ZERO);
    const uint32_t incr = n_threads;

    for (uint32_t el = start; el < end; el += incr)
    {
        atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
    }
}

// Kernel function to add the elements of two arrays
// ASSUME it is zeroed the res vector
__global__ void spmv_warp_jump(const uint32_t* x, const uint32_t* y, const float* val, const float* vec, float* res,
                               const uint32_t NON_ZERO)
{
    const uint32_t n_threads = gridDim.x * blockDim.x;
    const uint32_t per_thread = CEIL_DIV(NON_ZERO, n_threads);
    const uint32_t start_i = blockIdx.x * blockDim.x + threadIdx.x;

    const uint32_t wrap_id = start_i / warpSize;
    const uint32_t start = wrap_id * (per_thread * warpSize) + (start_i - wrap_id * warpSize);
    const uint32_t incr = warpSize;

    for (uint32_t i = 0; i < per_thread; i++)
    {
        if (const uint32_t el = start + i * incr; el < NON_ZERO)
        {
            atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
        }
    }
}

// Kernel function to add the elements of two arrays
// ASSUME it is zeroed the res vector
__global__ void spmv_block_jump(const uint32_t* x, const uint32_t* y, const float* val, const float* vec, float* res,
                                const uint32_t NON_ZERO)
{
    const uint32_t n_threads = gridDim.x * blockDim.x;
    const auto per_thread = CEIL_DIV(NON_ZERO, n_threads);

    const uint32_t start = blockIdx.x * (per_thread * blockDim.x) + threadIdx.x;
    const uint32_t incr = blockDim.x;

    for (uint32_t i = 0; i < per_thread; i++)
    {
        if (const uint32_t el = start + i * incr; el < NON_ZERO)
        {
            atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
        }
    }
}
