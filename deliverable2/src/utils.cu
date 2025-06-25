#include <cmath>
#include <iostream>
#include "include/utils.h"

using std::cout, std::endl;

// TODO fix negative number causing problems
#define DOUBLE_RAND_MIN 0
#define DOUBLE_RAND_MAX 1000

void randomize_dense_vec(float* vec, const uint32_t N)
{
    for (uint32_t i = 0; i < N; i++)
    {
        const double value = static_cast<double>(rand()) / RAND_MAX;
        vec[i] = static_cast<float>(value * (DOUBLE_RAND_MAX - DOUBLE_RAND_MIN) + DOUBLE_RAND_MIN);
    }
}

uint32_t diff_size(const float* v, const float* control, const uint32_t LEN)
{
    uint32_t n_error = 0;

    for (uint32_t i = 0; i < LEN; i++)
        if (std::fabs(v[i] - control[i]) > control[i] * 0.01)
            n_error++;

    return n_error;
}
