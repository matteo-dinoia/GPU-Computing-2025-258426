#pragma once
#include "type_alias.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define UNUSED(x) (void)(x)

#define eprintf(...) fprintf(stderr, __VA_ARGS__)

#define PRINT_VEC(vec, len)                                                                                            \
    for (auto i = (len); i > 0; i--)                                                                                   \
    {                                                                                                                  \
        cout << (vec)[(len) - i] << "\t";                                                                              \
    }                                                                                                                  \
    cout << endl;

#define CEIL_DIV(a, b) (static_cast<MI>(std::ceil(static_cast<MV>((a)) / static_cast<MV>((b)))))

void randomize_dense_vec(MV*, MI);

void print_min_max(const MV*, MI);

void print_diff_info(const MV*, const MV*, MI, std::string_view);

bool is_sorted_indexes(const MI* v, MI len);

MI lowest_greater_2_power(MI n);
