#pragma once
#include "type_alias.h"


#define UNUSED(x) (void)(x)

#define eprintf(...) fprintf (stderr, __VA_ARGS__)

#define PRINT_VEC(format_str, vec, N)   \
    for (int i = 0; i < (N); i++) {     \
        printf((format_str), (vec)[i]); \
    }                                   \
    printf("\n");

#define CEIL_DIV(a, b) (static_cast<u32>(std::ceil(static_cast<float>((a)) / static_cast<float>((b)))))

void randomize_dense_vec(float*, u32);

uint32_t diff_size(const float*, const float*, uint32_t);
void print_min_max(const float*, u32);

