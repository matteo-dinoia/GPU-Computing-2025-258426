#include <cmath>
#include <iostream>
#include "include/utils.h"

using std::cout, std::endl;

#define DOUBLE_RAND_MIN (-1000)
#define DOUBLE_RAND_MAX 1000

void randomize_dense_vec(float* vec, const uint32_t N)
{
    for (int i = 0; i < N; i++)
    {
        const double value = static_cast<double>(rand()) / RAND_MAX;
        vec[i] = static_cast<float>(value * (DOUBLE_RAND_MAX - DOUBLE_RAND_MIN) + DOUBLE_RAND_MIN);
    }
}

int diff_size(const float* v, const float* control, const uint32_t LEN)
{
    int n_error = 0;

    for (int i = 0; i < LEN; i++)
    {
        if (std::fabs(v[i] - control[i]) > control[i] * 0.01)
        {
            n_error++;
        }
    }

    return n_error;
}
