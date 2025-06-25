#pragma once
#include <cstdint>

typedef void (*kernel_func)(const uint32_t*, const uint32_t*, const float*, const float*, float*, uint32_t);

void spmv_baseline(const uint32_t*, const uint32_t*, const float*, const float*, float*, uint32_t);
void spmv_full_strided(const uint32_t*, const uint32_t*, const float*, const float*, float*, uint32_t);
void spmv_full_jump(const uint32_t*, const uint32_t*, const float*, const float*, float*, uint32_t);
void spmv_warp_jump(const uint32_t*, const uint32_t*, const float*, const float*, float*, uint32_t);
void spmv_block_jump(const uint32_t*, const uint32_t*, const float*, const float*, float*, uint32_t);
