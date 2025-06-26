#include <cmath>
#include <iostream>
#include "include/utils.h"
#include "include/type_alias.h"
#include <random>

using std::cout, std::endl;

void randomize_dense_vec(float* vec, const u32 N)
{
    std::uniform_real_distribution<float> unif(-1000, 1000);
    std::default_random_engine re; // NOLINT(*-msc51-cpp)

    for (u32 i = 0; i < N; i++)
        vec[i] = unif(re);
}

void print_diff_info(const float* v, const float* control, const u32 LEN, const std::string_view name)
{
    u32 n_error = 0;
    u32 i_first_err = 0;

    for (u32 i = 0; i < LEN; i++)
    {
        if (std::fabs(v[i] - control[i]) > std::fabs(control[i] * 0.1))
        {
            if (n_error == 0)
                i_first_err = i;
            n_error++;
        }
    }

    if (n_error > 0)
    {
        cout << "ERROR/s in " << name << " there are " << n_error << " over " << LEN
            << " [ first at index " << i_first_err << " " << v[i_first_err]
            << " insted of " << control[i_first_err] << "]" << endl;
    }
}

void print_min_max(const float* v, const u32 len)
{
    float max = -100;
    float min = 100;

    for (u32 i = 0; i < len; i++)
    {
        max = std::max(v[i], max);
        min = std::min(v[i], min);
    }

    cout << "Vector is in range [" << min << ", " << max << "]" << endl;
}
