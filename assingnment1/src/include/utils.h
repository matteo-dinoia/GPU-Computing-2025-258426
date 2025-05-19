#pragma once
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#define PRINT_VEC(format_str, vec, N)   \
    for (int i = 0; i < (N); i++) {     \
        printf((format_str), (vec)[i]); \
    }                                   \
    printf("\n");


void gen_random_vec_float(float *, const int);

void print_matrix_float(const float *, const int, const int);

int diff_size(float *, float *, int);