#include "include/gpu.h"
#include "include/type_alias.h"
#include "include/utils.h"

std::tuple<u32, u32, u32> parameters_for_baseline(const GpuCoo<u32, MV>& matrix)
{
    const u32 n_threads = MIN(MAX_THREAD_PER_BLOCK, matrix.NON_ZERO);
    const u32 n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads);
    return {n_blocks, n_threads, 0};
}

std::tuple<u32, u32, u32> parameters_for_basic(const GpuCoo<u32, MV>& matrix)
{
    const u32 n_threads = MIN(MAX_THREAD_PER_BLOCK, matrix.NON_ZERO);
    const u32 n_blocks = MIN(MAX_BLOCK, CEIL_DIV(matrix.NON_ZERO, n_threads));
    return {n_blocks, n_threads, 0};
}

std::tuple<u32, u32, u32> parameters_for_basic_with_2_shm(const GpuCoo<u32, MV>& matrix)
{
    const u32 n_threads = MIN(1024, matrix.NON_ZERO);
    const u32 n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads);
    return {n_blocks, n_threads, 2 * n_threads * sizeof(MV)};
}

std::tuple<u32, u32, u32> parameters_for_prefix_sum_multiple_read(const GpuCoo<u32, MV>& matrix)
{
    const u32 max_thread = lowest_greater_2_power(CEIL_DIV(matrix.NON_ZERO, READ_FOR_THREAD));
    const u32 n_threads = MIN(MAX_THREAD_PER_BLOCK, max_thread);
    const u32 n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads * READ_FOR_THREAD);
    const u32 shm = n_threads * READ_FOR_THREAD * sizeof(MV);
    return {n_blocks, n_threads, shm};
}

std::tuple<u32, u32, u32> parameters_for_prefix_sum(const GpuCoo<u32, MV>& matrix)
{
    const u32 max_thread = lowest_greater_2_power(matrix.NON_ZERO);
    const u32 n_threads = MIN(THREAD_PER_BLOCK_WE, max_thread);
    const u32 n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads);
    const u32 shm = n_threads * sizeof(MV) * 2;
    return {n_blocks, n_threads, shm};
}

std::tuple<u32, u32, u32> parameters_for_prefix_sum_max_32(const GpuCoo<u32, MV>& matrix)
{
    const u32 max_thread = lowest_greater_2_power(matrix.NON_ZERO);
    const u32 n_threads = MIN(32, max_thread);
    const u32 n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads);
    const u32 shm = n_threads * sizeof(MV);
    return {n_blocks, n_threads, shm};
}

std::tuple<u32, u32, u32> parameters_for_prefix_sum_max_32_efficient(const GpuCoo<u32, MV>& matrix)
{
    const u32 max_thread = lowest_greater_2_power(matrix.NON_ZERO) / 2;
    const u32 n_threads = MIN(32, max_thread);
    const u32 n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads * 2);
    const u32 shm = n_threads * 2 * sizeof(MV);
    return {n_blocks, n_threads, shm};
}

std::tuple<u32, u32, u32> parameters_for_prefix_sum_we_unlimited(const GpuCoo<u32, MV>& matrix)
{
    const u32 max_thread = lowest_greater_2_power(matrix.NON_ZERO) / 2;
    const u32 n_threads = MIN(THREAD_PER_BLOCK_WE, max_thread);
    const u32 n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads * 2);
    const u32 shm = n_threads * 4 * sizeof(MV);
    return {n_blocks, n_threads, shm};
}
std::tuple<u32, u32, u32> parameters_prefix_sum_warp(const GpuCoo<u32, MV>& matrix)
{
    const u32 n_threads = MIN(CEIL_DIV(matrix.NON_ZERO, 32) * 32, MAX_THREAD_PER_BLOCK);
    const u32 n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads);
    return {n_blocks, n_threads, 0};
}

std::tuple<u32, u32, u32> parameters_prefix_sum_warp_with_block_jump(const GpuCoo<u32, MV>& matrix)
{
    const u32 n_threads = MIN(CEIL_DIV(matrix.NON_ZERO, 32) * 32, MAX_THREAD_PER_BLOCK);
    const u32 n_blocks = MIN(MAX_BLOCK, CEIL_DIV(matrix.NON_ZERO, n_threads));
    return {n_blocks, n_threads, 0};
}

std::tuple<u32, u32, u32> parameters_prefix_sum_warp_merged(const GpuCoo<u32, MV>& matrix)
{
    // n_thread is 32 * n
    const u32 n_threads = MIN(CEIL_DIV(matrix.NON_ZERO, 32) * 32, MAX_THREAD_PER_BLOCK);
    const u32 n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads);
    const u32 shm = n_threads / 32 * sizeof(MV);
    return {n_blocks, n_threads, shm};
}
