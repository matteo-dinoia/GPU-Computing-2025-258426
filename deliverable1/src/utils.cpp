#include <iostream>
#include "include/utils.h"

using std::cout, std::endl;

void gen_random_vec_float(float *vec, const int N) {
    for (int i = 0; i < N; i++) {
        vec[i] = rand() % 1000;
    }
}

void print_matrix_float(const float *res, const int M, const int N) {
    if (res == NULL) {
        cout << "NULL pointer in print\n"
             << endl;
        return;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            cout << res[i * N + j] << "\t" << endl;
        }
        printf("\n");
    }
}

int diff_size(float *v, float *control, int LEN) {
    int n_error = 0;

    for (int i = 0; i < LEN; i++) {
        if (fabs(v[i] - control[i]) > control[i] * 0.01) {
            n_error++;
        }
    }

    return n_error;
}