#include <iostream>

// ASSUME it is zeroed the res vector
void gemm_sparse_cpu(const int *cx, const int *cy, const float *vals, const float *vec, float *res, const int NON_ZERO) {
    if (cx == NULL || cy == NULL || vals == NULL || vec == NULL || res == NULL) {
        printf("NULL pointeri in GEMM sparse\n");
        return;
    }

    for (int i = 0; i < NON_ZERO; i++) {
        const int row = cy[i];
        const int col = cx[i];

        res[row] += vec[col] * vals[i];
    }
}