#pragma once
#include <memory>

#include "type_alias.h"

struct CudaDeleter
{
    void operator()(void* x) const { free(x); }
};

template <typename IT, typename VT>
struct GpuCoo
{
    IT NON_ZERO;
    IT ROWS;
    IT COLS;
    std::unique_ptr<IT, CudaDeleter> xs;
    std::unique_ptr<IT, CudaDeleter> ys;
    std::unique_ptr<VT, CudaDeleter> vals;
};


void execution(const GpuCoo<MI, MV>&, const MV*, MV*, MV*, const char*);
