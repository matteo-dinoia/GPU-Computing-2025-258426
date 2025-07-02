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


void execution(const GpuCoo<u32, MV>&, const MV*, MV*, MV*);
