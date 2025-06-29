#pragma once
#include <tuple>
#include "tester.h"
#include "type_alias.h"

typedef void (*KernelFunc)(const u32*, const u32*, const MV*, const MV*, MV*, u32);

__global__ void kernel_baseline(const u32*, const u32*, const MV*, const MV*, MV*, u32);
__global__ void kernel_full_strided(const u32*, const u32*, const MV*, const MV*, MV*, u32);
__global__ void kernel_full_jump(const u32*, const u32*, const MV*, const MV*, MV*, u32);
__global__ void kernel_warp_jump(const u32*, const u32*, const MV*, const MV*, MV*, u32);
__global__ void kernel_block_jump(const u32*, const u32*, const MV*, const MV*, MV*, u32);
__global__ void kernel_block_jump_unsafe(const u32*, const u32*, const MV*, const MV*, MV*, u32);
__global__ void kernel_prefix_sum_32_max(const u32*, const u32*, const MV*, const MV*, MV*, u32);
__global__ void kernel_prefix_sum_unlimited(const u32*, const u32*, const MV*, const MV*, MV*, u32);

typedef std::tuple<u32, u32, u32> (*KernelParameterGetter)(const GpuCoo<u32, MV>&);
std::tuple<u32, u32, u32> parameters_for_baseline(const GpuCoo<u32, MV>&);
std::tuple<u32, u32, u32> parameters_for_basic(const GpuCoo<u32, MV>&);
std::tuple<u32, u32, u32> parameters_for_prefix_sum(const GpuCoo<u32, MV>&);
std::tuple<u32, u32, u32> parameters_for_prefix_sum_max_32(const GpuCoo<u32, MV>& matrix);

struct SmpvKernel
{
    std::string_view name;
    KernelFunc execute;
    KernelParameterGetter parameter_getter;
};

constexpr SmpvKernel baseline = {"baseline", kernel_baseline, parameters_for_baseline};
constexpr SmpvKernel full_strided = {"full_strided", kernel_full_strided, parameters_for_basic};
constexpr SmpvKernel full_jump = {"full_jump", kernel_full_jump, parameters_for_basic};
constexpr SmpvKernel warp_jump = {"warp_jump", kernel_warp_jump, parameters_for_basic};
constexpr SmpvKernel block_jump = {"block_jump", kernel_block_jump, parameters_for_basic};
constexpr SmpvKernel block_jump_unsafe = {"block_jump_unsafe", kernel_block_jump_unsafe, parameters_for_basic};
constexpr SmpvKernel prefix_sum_32_max = {"prefix_sum_32_max", kernel_prefix_sum_32_max,
                                          parameters_for_prefix_sum_max_32};
constexpr SmpvKernel prefix_sum_unlimited = {"prefix_sum_bkp_unlimited", kernel_prefix_sum_unlimited,
                                             parameters_for_prefix_sum};
