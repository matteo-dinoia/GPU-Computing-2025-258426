#pragma once
#include "type_alias.h"
#include "tester.h"

typedef void (*KernelFunc)(const u32*, const u32*, const float*, const float*, float*, u32);

void kernel_baseline(const u32*, const u32*, const float*, const float*, float*, u32);
void kernel_full_strided(const u32*, const u32*, const float*, const float*, float*, u32);
void kernel_full_jump(const u32*, const u32*, const float*, const float*, float*, u32);
void kernel_warp_jump(const u32*, const u32*, const float*, const float*, float*, u32);
void kernel_warp_jump_bkp(const u32*, const u32*, const float*, const float*, float*, u32);
void kernel_block_jump(const u32*, const u32*, const float*, const float*, float*, u32);
void kernel_block_jump_unsafe(const u32*, const u32*, const float*, const float*, float*, u32);
void kernel_block_jump_bkp(const u32*, const u32*, const float*, const float*, float*, u32);

typedef std::pair<u32, u32> (*KernelParameterGetter)(const GpuCoo<u32, float>&);
std::pair<u32, u32> parameters_for_baseline(const GpuCoo<u32, float>&);
std::pair<u32, u32> parameters_for_basic(const GpuCoo<u32, float>&);

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
constexpr SmpvKernel warp_jump_bkp = {"warp_jump_bkp", kernel_warp_jump_bkp, parameters_for_basic};
constexpr SmpvKernel block_jump = {"block_jump", kernel_block_jump, parameters_for_basic};
constexpr SmpvKernel block_jump_unsafe = {"block_jump_unsafe", kernel_block_jump_unsafe, parameters_for_basic};
constexpr SmpvKernel block_jump_bkp = {" block_jump_bkp", kernel_block_jump_bkp, parameters_for_basic};

