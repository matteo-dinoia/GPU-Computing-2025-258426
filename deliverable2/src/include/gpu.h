#pragma once
#include <string_view>
#include <tuple>
#include "tester.h"
#include "type_alias.h"
// Is h file of both gpu.cu and parameters.cu

// General parameters
#define MAX_THREAD_PER_BLOCK 1024
#define MAX_BLOCK 256
// Specific parameters
#define THREAD_PER_BLOCK_WE 128
#define READ_FOR_THREAD 2
// Parameter for conflict free
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

// Types definitions
typedef void (*KernelFunc)(const MI*, const MI*, const MV*, const MV*, MV*, MI);
typedef std::tuple<MI, MI, MI> (*KernelParameterGetter)(const GpuCoo<MI, MV>&);
struct SmpvKernel
{
    std::string_view name;
    KernelFunc execute;
    KernelParameterGetter parameter_getter;
};
#define KERNEL_DEF(name) __global__ void name(const MI*, const MI*, const MV*, const MV*, MV*, MI)
#define PAR_GETTER_DEF(name) std::tuple<MI, MI, MI> name(const GpuCoo<MI, MV>&)
#define WRAPPER_DEF(name, par_getter) constexpr SmpvKernel name = {#name, kernel_##name, (par_getter)}


// Define kernel function (with helper macro)
KERNEL_DEF(kernel_baseline);
KERNEL_DEF(kernel_full_strided);
KERNEL_DEF(kernel_full_jump);
KERNEL_DEF(kernel_warp_jump);
KERNEL_DEF(kernel_block_jump);
KERNEL_DEF(kernel_block_jump_unsafe);
KERNEL_DEF(kernel_prefix_sum_32_max);
KERNEL_DEF(kernel_prefix_sum_unlimited);
KERNEL_DEF(kernel_prefix_sum_max_32_work_efficient);
KERNEL_DEF(kernel_prefix_sum_we_32_conflict_free);
KERNEL_DEF(kernel_prefix_sum_we_unlimited_conflict_free);
KERNEL_DEF(kernel_block_jump_shared);
KERNEL_DEF(kernel_prefix_sum_warp);
KERNEL_DEF(kernel_prefix_sum_warp_2x);
KERNEL_DEF(kernel_prefix_sum_warp_with_block_jump);
KERNEL_DEF(kernel_prefix_sum_warp_merged);

// Define parameter getter function (with helper macro)
PAR_GETTER_DEF(parameters_for_baseline);
PAR_GETTER_DEF(parameters_for_basic);
PAR_GETTER_DEF(parameters_for_prefix_sum);
PAR_GETTER_DEF(parameters_for_prefix_sum_max_32);
PAR_GETTER_DEF(parameters_for_prefix_sum_max_32_efficient);
PAR_GETTER_DEF(parameters_for_prefix_sum_we_unlimited);
PAR_GETTER_DEF(parameters_for_basic_with_2_shm);
PAR_GETTER_DEF(parameters_prefix_sum_warp);
PAR_GETTER_DEF(parameters_prefix_sum_warp_with_block_jump);
PAR_GETTER_DEF(parameters_prefix_sum_warp_merged);

// Define wrappers (with helper macro)
WRAPPER_DEF(baseline, parameters_for_baseline);
WRAPPER_DEF(full_strided, parameters_for_basic);
WRAPPER_DEF(full_jump, parameters_for_basic);
WRAPPER_DEF(warp_jump, parameters_for_basic);
WRAPPER_DEF(block_jump, parameters_for_basic);
WRAPPER_DEF(block_jump_unsafe, parameters_for_basic);
WRAPPER_DEF(prefix_sum_32_max, parameters_for_prefix_sum_max_32);
WRAPPER_DEF(prefix_sum_unlimited, parameters_for_prefix_sum);
WRAPPER_DEF(prefix_sum_max_32_work_efficient, parameters_for_prefix_sum_max_32_efficient);
WRAPPER_DEF(prefix_sum_we_32_conflict_free, parameters_for_prefix_sum_max_32_efficient);
WRAPPER_DEF(prefix_sum_we_unlimited_conflict_free, parameters_for_prefix_sum_we_unlimited);
WRAPPER_DEF(block_jump_shared, parameters_for_basic_with_2_shm);
WRAPPER_DEF(prefix_sum_warp, parameters_prefix_sum_warp);
WRAPPER_DEF(prefix_sum_warp_2x, parameters_prefix_sum_warp);
WRAPPER_DEF(prefix_sum_warp_with_block_jump, parameters_prefix_sum_warp_with_block_jump);
WRAPPER_DEF(prefix_sum_warp_merged, parameters_prefix_sum_warp_merged);
