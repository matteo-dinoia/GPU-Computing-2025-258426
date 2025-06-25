#pragma once
#include <cstdint>

#define UNUSED(x) (void)(x)

#define eprintf(...) fprintf (stderr, __VA_ARGS__)

#define PRINT_VEC(format_str, vec, N)   \
    for (int i = 0; i < (N); i++) {     \
        printf((format_str), (vec)[i]); \
    }                                   \
    printf("\n");

#define CEIL_DIV(a, b) (static_cast<uint32_t>(std::ceil(static_cast<float>((a)) / static_cast<float>((b)))))

void randomize_dense_vec(float*, uint32_t);

uint32_t diff_size(const float*, const float*, uint32_t);
