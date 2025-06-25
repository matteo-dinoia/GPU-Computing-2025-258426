#pragma once
#include <cstdint>

template <typename IT, typename VT>
struct GpuCoo
{
    IT NON_ZERO;
    IT ROWS;
    IT COLS;
    IT* xs;
    IT* ys;
    VT* vals;
};


void execution(const GpuCoo<uint32_t, float>&, float*, float*, float*);
