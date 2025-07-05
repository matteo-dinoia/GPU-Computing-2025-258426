#include "include/gpu.h"
#include "include/type_alias.h"
#include "include/utils.h"

std::tuple<MI, MI, MI> parameters_for_baseline(const GpuCoo<MI, MV>& matrix)
{
    const MI n_threads = MIN(MAX_THREAD_PER_BLOCK, matrix.NON_ZERO);
    const MI n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads);
    return {n_blocks, n_threads, 0};
}

std::tuple<MI, MI, MI> parameters_for_basic(const GpuCoo<MI, MV>& matrix)
{
    const MI n_threads = MIN(MAX_THREAD_PER_BLOCK, matrix.NON_ZERO);
    const MI n_blocks = MIN(MAX_BLOCK, CEIL_DIV(matrix.NON_ZERO, n_threads));
    return {n_blocks, n_threads, 0};
}

std::tuple<MI, MI, MI> parameters_for_basic_with_2_shm(const GpuCoo<MI, MV>& matrix)
{
    const MI n_threads = MIN(1024, matrix.NON_ZERO);
    const MI n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads);
    return {n_blocks, n_threads, 2 * n_threads * sizeof(MV)};
}

std::tuple<MI, MI, MI> parameters_for_prefix_sum_multiple_read(const GpuCoo<MI, MV>& matrix)
{
    const MI max_thread = lowest_greater_2_power(CEIL_DIV(matrix.NON_ZERO, READ_FOR_THREAD));
    const MI n_threads = MIN(MAX_THREAD_PER_BLOCK, max_thread);
    const MI n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads * READ_FOR_THREAD);
    const MI shm = n_threads * READ_FOR_THREAD * sizeof(MV);
    return {n_blocks, n_threads, shm};
}

std::tuple<MI, MI, MI> parameters_for_prefix_sum(const GpuCoo<MI, MV>& matrix)
{
    const MI max_thread = lowest_greater_2_power(matrix.NON_ZERO);
    const MI n_threads = MIN(THREAD_PER_BLOCK_WE, max_thread);
    const MI n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads);
    const MI shm = n_threads * sizeof(MV) * 2;
    return {n_blocks, n_threads, shm};
}

std::tuple<MI, MI, MI> parameters_for_prefix_sum_max_32(const GpuCoo<MI, MV>& matrix)
{
    const MI max_thread = lowest_greater_2_power(matrix.NON_ZERO);
    const MI n_threads = MIN(32, max_thread);
    const MI n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads);
    const MI shm = n_threads * sizeof(MV);
    return {n_blocks, n_threads, shm};
}

std::tuple<MI, MI, MI> parameters_for_prefix_sum_max_32_efficient(const GpuCoo<MI, MV>& matrix)
{
    const MI max_thread = lowest_greater_2_power(matrix.NON_ZERO) / 2;
    const MI n_threads = MIN(32, max_thread);
    const MI n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads * 2);
    const MI shm = n_threads * 2 * sizeof(MV);
    return {n_blocks, n_threads, shm};
}

std::tuple<MI, MI, MI> parameters_for_prefix_sum_we_unlimited(const GpuCoo<MI, MV>& matrix)
{
    const MI max_thread = lowest_greater_2_power(matrix.NON_ZERO) / 2;
    const MI n_threads = MIN(THREAD_PER_BLOCK_WE, max_thread);
    const MI n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads * 2);
    const MI shm = n_threads * 4 * sizeof(MV);
    return {n_blocks, n_threads, shm};
}
std::tuple<MI, MI, MI> parameters_prefix_sum_warp(const GpuCoo<MI, MV>& matrix)
{
    const MI n_threads = MIN(CEIL_DIV(matrix.NON_ZERO, 32) * 32, MAX_THREAD_PER_BLOCK);
    const MI n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads);
    return {n_blocks, n_threads, 0};
}

std::tuple<MI, MI, MI> parameters_prefix_sum_warp_with_block_jump(const GpuCoo<MI, MV>& matrix)
{
    const MI n_threads = MIN(CEIL_DIV(matrix.NON_ZERO, 32) * 32, MAX_THREAD_PER_BLOCK);
    const MI n_blocks = MIN(MAX_BLOCK, CEIL_DIV(matrix.NON_ZERO, n_threads));
    return {n_blocks, n_threads, 0};
}

std::tuple<MI, MI, MI> parameters_prefix_sum_warp_merged(const GpuCoo<MI, MV>& matrix)
{
    // n_thread is 32 * n
    const MI n_threads = MIN(CEIL_DIV(matrix.NON_ZERO, 32) * 32, MAX_THREAD_PER_BLOCK);
    const MI n_blocks = CEIL_DIV(matrix.NON_ZERO, n_threads);
    const MI shm = n_threads / 32 * sizeof(MV);
    return {n_blocks, n_threads, shm};
}
