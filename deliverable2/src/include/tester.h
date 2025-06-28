#pragma once
#include <cstdint>
#include "type_alias.h"

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


void execution(const GpuCoo<uint32_t, float>&, const float*, float*, float*);
