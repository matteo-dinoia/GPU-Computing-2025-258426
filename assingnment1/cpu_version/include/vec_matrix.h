#ifndef VEC_UTILS_H
#define VEC_UTILS_H

#define PRINT_VEC(format_str, vec, N)   \
    for (int i = 0; i < (N); i++) {     \
        printf((format_str), (vec)[i]); \
    }                                   \
    printf("\n");


void gen_random_vec_double(float *vec, const int N) {
    for (int i = 0; i < N; i++) {
        vec[i] = rand() % 10; /* TODO FIX */
    }
}

void print_matrix_double(const float *res, const int M, const int N) {
    if (res == NULL) {
        printf("NULL pointer in print\n");
        return;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.4f\t", res[i * N + j]);
        }
        printf("\n");
    }
}

bool compare_double(const double *a, const double *b, const int N) {
    if (a == NULL || b == NULL) {
        return false;
    }

    for (int i = 0; i < N; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

#endif
