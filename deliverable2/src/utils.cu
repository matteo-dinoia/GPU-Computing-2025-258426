#include <iostream>
#include <random>
#include "include/type_alias.h"
#include "include/utils.h"

using std::cout, std::endl;

#define RAND_BOUND 10
#define RAND_PREC 1000

void randomize_dense_vec(MV* vec, const u32 N)
{
    for (u32 i = 0; i < N; i++)
    {
        const auto tmp = static_cast<MV>(rand() % (RAND_BOUND * RAND_PREC * 2)); // NOLINT(*-msc50-cpp)
        vec[i] = tmp / RAND_PREC - RAND_BOUND;
    }
}

void print_diff_info(const MV* v, const MV* control, const u32 LEN, const std::string_view name)
{
    u32 n_error = 0;
    u32 i_first_err = 0;

    for (u32 i = 0; i < LEN; i++)
    {
        if (std::fabs(v[i] - control[i]) > std::max(std::fabs(control[i] * 0.1), 0.01))
        {
            if (n_error == 0)
                i_first_err = i;
            n_error++;
        }
    }

    if (n_error > 0)
    {
        cout << "ERROR/s in " << name << " there are " << n_error << " over " << LEN << " [ first at index "
             << i_first_err << " where found " << v[i_first_err] << " insted of expected " << control[i_first_err]
             << "]" << endl;

        if (LEN < 20)
        {
            PRINT_VEC(v, LEN);
            PRINT_VEC(control, LEN);
        }
    }
}

void print_min_max(const MV* v, const u32 len)
{
    MV max = -100;
    MV min = 100;

    for (u32 i = 0; i < len; i++)
    {
        max = std::max(v[i], max);
        min = std::min(v[i], min);
    }

    cout << "Vector is in range [" << min << ", " << max << "]" << endl;
}

bool is_sorted_indexes(const u32* v, const u32 len)
{
    for (u32 i = 1; i < len; i++)
    {
        if (v[i - 1] > v[i])
            return false;
    }
    return true;
}

u32 lowest_greater_2_power(const u32 n)
{
    u32 res = 1;
    while (res < n)
        res <<= 1;
    return res;
}
