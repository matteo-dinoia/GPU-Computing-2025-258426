#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <strings.h>

#include "include/time_utils.h"
#include "include/vec_matrix.h"
#include "include/mtx.h"

#define WARM_CYCLES 100
#define TEST_CYCLES 400
#define INPUT_FILENAME "../datasets/mycielskian3.mtx"

#define OK true
#define ERR false

// ASSUME it is zeroed the res vector
void gemm_sparse_coo(const int *cx, const int *cy, const M_TYPE *vals, const M_TYPE *vec, M_TYPE *res, const int NON_ZERO) {
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

// ASSUME it is zeroed the res vector
void gemm_sparse_csr(const int *cx, const int *csr_y, const M_TYPE *vals, const M_TYPE *vec, M_TYPE *res, const int ROWS) {
    if (cx == NULL || csr_y == NULL || vals == NULL || vec == NULL || res == NULL) {
        printf("NULL pointer in GEMM sparse\n");
        return;
    }

    for (int row = 0; row < ROWS; row++) {
        const int start = csr_y[row];
        const int end = csr_y[row + 1];

        for (int i = start; i < end; i++) {
            const int col = cx[i];
            res[row] += vec[col] * vals[i];
        }
    }
}

// Assume csr_y being of length correct ROWS
// Assume it is sorted by (0,0), (0,1), ..., (1,0),...
void convert_to_csr(const int *cy, int *csr_y, const int NON_ZERO, const int ROWS) {
    if (cy == NULL || csr_y == NULL) {
        printf("NULL pointer in convert_to_csr\n");
        return;
    }

    bzero(csr_y, sizeof(int) * (ROWS + 1));

    for (int i = 0; i < NON_ZERO; i++) {
        csr_y[cy[i] + 1]++;
    }

    int sum = 0;
    for (int i = 0; i < ROWS + 1; i++) {
        sum = csr_y[i] = csr_y[i] + sum;
    }
}

void print_sparse(const int *cx, const int *cy, const M_TYPE *vals, const int NON_ZERO) {
    for (int i = 0; i < NON_ZERO; i++) {
        printf("%.2e in (col %d, row %d)\n", vals[i], cx[i], cy[i]);
    }
    printf("\n");
}

int main() {
    int ROWS, COLS, NON_ZERO;
    int ret = 0;
    FILE *file;
    TIMER_DEF(0);
    TIMER_DEF(1);
    srand(time(0) * 5);

    // HEADER READING --------------------------------------------------------------
    TIMER_TIME(0, {
        file = fopen(INPUT_FILENAME, "r");
        const bool status = read_mtx_header(file, &ROWS, &COLS, &NON_ZERO);
        if (status == ERR) {
            fclose(file);
            return -1;
        }

    });
    printf("READ HEADER: %fms\n", TIMER_ELAPSED(0) / 1.e3);

    // VARIABLE CREATION ----------------------------------------------------------
    int *x = calloc(NON_ZERO, sizeof(int));
    int *y = calloc(NON_ZERO, sizeof(int));
    int *csr_y = calloc((NON_ZERO + 1), sizeof(int));
    M_TYPE *vals = calloc(NON_ZERO, sizeof(M_TYPE));
    M_TYPE *vec = calloc(COLS, sizeof(M_TYPE));
    M_TYPE *res1 = calloc(ROWS, sizeof(M_TYPE));
    M_TYPE *res2 = calloc(ROWS, sizeof(M_TYPE));
    double *times_coo = calloc(TEST_CYCLES, sizeof(double));
    double *times_csr = calloc(TEST_CYCLES, sizeof(double));

    // DATA CREATION ---------------------------------------------------------------
    TIMER_TIME(0, {
        const bool status = read_mtx_data(file, x, y, vals, NON_ZERO);
        if (status == ERR) {
            goto free;
        }

    });

    gen_random_vec_double(vec, COLS);

    // CSR CONVERTION --------------------------------------------------------------
    TIMER_TIME(0, {
        convert_to_csr(y, csr_y, NON_ZERO, ROWS);
    });
    printf("CONVERT TO CSR: %fms\n", TIMER_ELAPSED(0) / 1.e3);

    // TIMING OF GEMM_SPARSE --------------------------------------------------------
    int wrong = 0;
    for (int cycle = -WARM_CYCLES; cycle < TEST_CYCLES; cycle++) {
        // Reset
        bzero(res1, ROWS * sizeof(double));
        bzero(res2, ROWS * sizeof(double));

        // TEST
        TIMER_TIME(0, gemm_sparse_coo(x, y, vals, vec, res1, NON_ZERO));
        if (cycle >= 0)
            times_coo[cycle] = TIMER_ELAPSED(0) / 1.e6;

        TIMER_TIME(1, gemm_sparse_csr(x, csr_y, vals, vec, res2, ROWS));
        if (cycle >= 0)
            times_csr[cycle] = TIMER_ELAPSED(1) / 1.e6;

        for (int i = 0; i < ROWS; i++) {
            if ((res1[i] - res2[i]) / res1[i] < 0.01)
                wrong++;
        }
    }

    // TIMING ------------------------------------------------------------------
    print_time_data("COO", times_coo, TEST_CYCLES, 2 * NON_ZERO);
    print_time_data("CSR", times_csr, TEST_CYCLES, 2 * NON_ZERO);
    printf("WRONG %d", wrong);

// FREE  -------------------------------------------------------------------
free:
    free(x);
    free(y);
    free(csr_y);
    free(vals);
    free(vec);
    free(res1);
    free(res2);
    free(times_coo);
    free(times_csr);
    fclose(file);
    return ret;
}
