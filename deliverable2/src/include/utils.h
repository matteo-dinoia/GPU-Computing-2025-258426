#pragma once
#include "type_alias.h"


#define UNUSED(x) (void)(x)

#define eprintf(...) fprintf(stderr, __VA_ARGS__)

#define PRINT_VEC(format_str, vec, N)                                                                                  \
    for (int i = 0; i < (N); i++)                                                                                      \
    {                                                                                                                  \
        printf((format_str), (vec)[i]);                                                                                \
    }                                                                                                                  \
    printf("\n");

#define CEIL_DIV(a, b) (static_cast<u32>(std::ceil(static_cast<MV>((a)) / static_cast<MV>((b)))))

void randomize_dense_vec(MV*, u32);

void print_min_max(const MV*, u32);

void print_diff_info(const MV*, const MV*, u32, std::string_view);

bool is_sorted_indexes(const u32* v, u32 len);

template <typename TypeName>
void print_arr(TypeName* arr, const u32 len);