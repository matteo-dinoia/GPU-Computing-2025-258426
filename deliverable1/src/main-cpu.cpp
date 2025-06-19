#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <strings.h>
#include <fstream>

#include "include/time_utils.h"
#include "include/utils.h"
#include "include/mtx.h"

#define WARM_CYCLES 1
#define TEST_CYCLES 20

#define OK true
#define ERR false

using std::cout, std::endl;

// ASSUME it is zeroed the res vector
void spmv_coo(const int *cx, const int *cy, const float *vals, const float *vec, float *res, const int NON_ZERO) {
    if (cx == NULL || cy == NULL || vals == NULL || vec == NULL || res == NULL) {
        printf("NULL pointeri in spmv sparse\n");
        return;
    }

    int i;
    for (i = 0; i < NON_ZERO; i++) {
        res[cy[i]] += vec[cx[i]] * vals[i];
    }
}

// ASSUME it is zeroed the res vector
void spmv_csr(const int *cx, const int *csr_y, const float *vals, const float *vec, float *res, const int ROWS) {
    if (cx == NULL || csr_y == NULL || vals == NULL || vec == NULL || res == NULL) {
        printf("NULL pointer in spmv sparse\n");
        return;
    }

    int start, end, i;
    for (int row = 0; row < ROWS; row++) {
        start = csr_y[row];
        end = csr_y[row + 1];

        for (i = start; i < end; i++) {
            res[row] += vec[cx[i]] * vals[i];
        }
    }
}

// ASSUME it is zeroed the res vector
void optimized_spmv_csr(const int *cx, const int *csr_y, const float *vals, const float *vec, float *res, const int ROWS, const int NON_ZERO) {
    if (cx == NULL || csr_y == NULL || vals == NULL || vec == NULL || res == NULL) {
        printf("NULL pointer in spmv sparse\n");
        return;
    }

    int i = 0;
    int end = 0;
    for (int row = 0; row < ROWS; row++) {
        end = csr_y[row + 1];

        for (; i < end; i++) {
            res[row] += vec[cx[i]] * vals[i];
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

    int old = 0;
    for (int i = 0; i < ROWS + 1; i++) {
        csr_y[i] += old;
        old = csr_y[i];
    }
}

void print_sparse(const int *cx, const int *cy, const float *vals, const int NON_ZERO) {
    for (int i = 0; i < NON_ZERO; i++) {
        printf("%.2e in (col %d, row %d)\n", vals[i], cx[i], cy[i]);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    srand(time(NULL));
    TIMER_DEF(0);
    TIMER_DEF(1);
    TIMER_DEF(2);
    int wrong;
    bool status = OK;

    int *csr_y = NULL;
    float *vec = NULL, *res1 = NULL, *res2 = NULL;
    double *times_coo = NULL, *times_csr = NULL, *times_opt_csr = NULL;
    struct Coo matrix = {0, 0, 0, NULL, NULL, NULL};

    if (argc < 2) {
        cout << "FATAL: require filename argument" << endl;
        return 1;
    }

    // File opener
    std::ifstream file(argv[1]);
    std::ios_base::sync_with_stdio(false);
    file.tie(NULL);
    if (!file.is_open()) {
        cout << "FATAL: couldn't open the file" << endl;
        status = ERR;
        goto free;
    }

    // Reading header
    TIMER_TIME(2, { status = read_mtx_header(file, &matrix); });
    if (status == ERR) {
        cout << "FATAL: fail to read header" << endl;
        goto free;
    }
    cout << "# READ HEADER: " << TIMER_ELAPSED(2) / 1.e3 << "ms" << endl;
    cout << "# Header: " << matrix.ROWS << " " << matrix.COLS << " " << matrix.NON_ZERO << endl;

    // VARIABLE CREATION ----------------------------------------------------------
    matrix.xs = (int *)calloc(matrix.NON_ZERO, sizeof(int));
    matrix.ys = (int *)calloc(matrix.NON_ZERO, sizeof(int));
    matrix.vals = (float *)calloc(matrix.NON_ZERO, sizeof(float));
    csr_y = (int *)calloc(matrix.ROWS + 1, sizeof(int));
    vec = (float *)calloc(matrix.COLS, sizeof(float));
    res1 = (float *)calloc(matrix.ROWS, sizeof(float));
    res2 = (float *)calloc(matrix.ROWS, sizeof(float));
    times_coo = (double *)calloc(TEST_CYCLES, sizeof(double));
    times_csr = (double *)calloc(TEST_CYCLES, sizeof(double));
    times_opt_csr = (double *)calloc(TEST_CYCLES, sizeof(double));

    // DATA CREATION ---------------------------------------------------------------
    // Reading data
    TIMER_TIME(2, { status = read_mtx_data(file, &matrix); });
    if (status == ERR) {
        cout << "FATAL: fail to read data" << endl;
        goto free;
    }
    cout << "# READ DATA: " << TIMER_ELAPSED(2) / 1.e3 << "ms" << endl;

    gen_random_vec_float(vec, matrix.COLS);

    // CSR CONVERTION --------------------------------------------------------------
    TIMER_TIME(0, {
        convert_to_csr(matrix.ys, csr_y, matrix.NON_ZERO, matrix.ROWS);
    });
    printf("CONVERT TO CSR: %fms\n", TIMER_ELAPSED(0) / 1.e3);

    // TIMING OF spmv_SPARSE --------------------------------------------------------
    wrong = 0;
    for (int cycle = -WARM_CYCLES; cycle < TEST_CYCLES; cycle++) {
        // Reset
        bzero(res1, matrix.ROWS * sizeof(float));
        bzero(res2, matrix.ROWS * sizeof(float));

        // TEST
        TIMER_TIME(0, spmv_coo(matrix.xs, matrix.ys, matrix.vals, vec, res1, matrix.NON_ZERO));
        if (cycle >= 0)
            times_coo[cycle] = TIMER_ELAPSED(0) / 1.e6;

        TIMER_TIME(1, spmv_csr(matrix.xs, csr_y, matrix.vals, vec, res2, matrix.ROWS));
        if (cycle >= 0)
            times_csr[cycle] = TIMER_ELAPSED(1) / 1.e6;

        for (int i = 0; i < matrix.ROWS; i++) {
            if ((res1[i] - res2[i]) / res1[i] > 0.01) {
                wrong++;
            }
        }

        TIMER_TIME(1, optimized_spmv_csr(matrix.xs, csr_y, matrix.vals, vec, res2, matrix.ROWS, matrix.NON_ZERO));
        if (cycle >= 0)
            times_opt_csr[cycle] = TIMER_ELAPSED(1) / 1.e6;

        for (int i = 0; i < matrix.ROWS; i++) {
            if ((res1[i] - res2[i]) / res1[i] > 0.01) {
                wrong++;
            }
        }
    }

    // TIMING ------------------------------------------------------------------
    print_time_data("COO", times_coo, TEST_CYCLES, 2 * matrix.NON_ZERO);
    print_time_data("CSR", times_csr, TEST_CYCLES, 2 * matrix.NON_ZERO);
    print_time_data("OPT CSR", times_opt_csr, TEST_CYCLES, 2 * matrix.NON_ZERO);
    printf("WRONG %d\n", wrong);
// FREE  -------------------------------------------------------------------
free:
    free(matrix.xs);
    free(matrix.ys);
    free(matrix.vals);
    free(csr_y);
    free(vec);
    free(res1);
    free(res2);
    free(times_coo);
    free(times_csr);
    free(times_opt_csr);
    file.close();
    return status == OK ? 0 : 1;
}
