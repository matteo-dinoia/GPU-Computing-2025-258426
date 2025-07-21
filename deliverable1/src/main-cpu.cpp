#include <cstring>
#include <iostream>
#include <strings.h>
#include <fstream>

#include "include/time_utils.h"
#include "include/utils.h"
#include "../distributed_mmio/include/mmio.h"

#define WARM_CYCLES 10
#define TEST_CYCLES 20

#define OK true
#define ERR false

using std::cout, std::endl;

typedef uint32_t u32;

// ASSUME it is zeroed the res vector
void spmv_coo(const u32 *cx, const u32 *cy, const float *vals, const float *vec, float *res, const int NON_ZERO) {
    if (cx == nullptr || cy == nullptr || vals == nullptr || vec == nullptr || res == nullptr) {
        printf("nullptr pointer in sparse matrix or vector or res\n");
        return;
    }

    for (u32 i = 0; i < NON_ZERO; i++) {
        res[cy[i]] += vec[cx[i]] * vals[i];
    }
}

// ASSUME it is zeroed the res vector
void spmv_csr(const u32 *cx, const u32 *csr_y, const float *vals, const float *vec, float *res, const int ROWS) {
    if (cx == nullptr || csr_y == nullptr || vals == nullptr || vec == nullptr || res == nullptr) {
        printf("nullptr pointer in sparse matrix or vector or res\n");
        return;
    }

    for (u32 row = 0; row < ROWS; row++) {
        const u32 start = csr_y[row];
        const u32 end = csr_y[row + 1];

        for (u32 i = start; i < end; i++)
            res[row] += vec[cx[i]] * vals[i];
    }
}

// ASSUME it is zeroed the res vector
void optimized_spmv_csr(const u32 *cx, const u32 *csr_y, const float *vals, const float *vec, float *res, const int ROWS, const int NON_ZERO) {
    if (cx == nullptr || csr_y == nullptr || vals == nullptr || vec == nullptr || res == nullptr) {
        printf("nullptr pointer in spmv sparse\n");
        return;
    }

    u32 i = 0;
    for (u32 row = 0; row < ROWS; row++) {
        const u32 end = csr_y[row + 1];

        for (; i < end; i++)
            res[row] += vec[cx[i]] * vals[i];
    }
}

// Assume csr_y being of length correct ROWS
// Assume it is sorted by (0,0), (0,1), ..., (1,0),...
void convert_to_csr(const u32 *cy, u32 *csr_y, const int NON_ZERO, const int ROWS) {
    if (cy == nullptr || csr_y == nullptr) {
        printf("nullptr pointer in convert_to_csr\n");
        return;
    }

    bzero(csr_y, sizeof(int) * (ROWS + 1));

    for (int i = 0; i < NON_ZERO; i++) {
        csr_y[cy[i] + 1]++;
    }

    u32 old = 0;
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
    srand(time(nullptr));
    TIMER_DEF(0);
    TIMER_DEF(1);
    TIMER_DEF(2);

    u32 *csr_row = nullptr;
    float *vec = nullptr, *res1 = nullptr, *res2 = nullptr;
    double *times_coo = nullptr, *times_csr = nullptr, *times_opt_csr = nullptr;


    if (argc < 2) {
        cout << "FATAL: require filename argument" << endl;
        return 1;
    }

    // File opener
    TIMER_START(2);
    COO_local<u32, float>* coo = Distr_MMIO_sorted_COO_local_read<u32, float>(argv[1], false);
    if (coo == nullptr)
        return -1;

    if (coo->val == nullptr)
    {
        coo->val = (float *) calloc(coo->nnz, sizeof(float));
        memset(coo->val, 1, coo->nnz);
    }
    TIMER_STOP(2);
    cout << "# Header: " << coo->nrows << " " << coo->ncols << " " << coo->nnz << endl;
    cout << "# READ DATA: " << TIMER_ELAPSED(2) / 1.e3 << "ms" << endl;


    // VARIABLE CREATION ----------------------------------------------------------
    csr_row = (u32 *)calloc(coo->nnz, sizeof(int));
    vec = (float *)calloc(coo->ncols, sizeof(float));
    res1 = (float *)calloc(coo->nrows, sizeof(float));
    res2 = (float *)calloc(coo->nrows, sizeof(float));
    times_coo = (double *)calloc(TEST_CYCLES, sizeof(double));
    times_csr = (double *)calloc(TEST_CYCLES, sizeof(double));
    times_opt_csr = (double *)calloc(TEST_CYCLES, sizeof(double));

    // RANDOMIZATION ----------------------------------------------------------------
    TIMER_START(2);
    gen_random_vec_float(vec, coo->ncols);
    TIMER_STOP(2);
    cout << "# RANDOMIZE VECTOR: " << TIMER_ELAPSED(2) / 1.e3 << "ms" << endl;

    // CSR CONVERTION --------------------------------------------------------------
    TIMER_TIME(0, {
        convert_to_csr(coo->row, csr_row, coo->nnz, coo->nrows);
    });
    printf("CONVERT TO COO: %fms\n", TIMER_ELAPSED(0) / 1.e3);

    // TIMING OF spmv_SPARSE --------------------------------------------------------
    int wrong = 0;
    for (int cycle = -WARM_CYCLES; cycle < TEST_CYCLES; cycle++) {
        // Reset
        bzero(res1, coo->nrows * sizeof(float));
        bzero(res2, coo->nrows * sizeof(float));

        // TEST

        TIMER_TIME(0, spmv_coo(coo->col, coo->row, coo->val, vec, res1, coo->nnz));
        if (cycle >= 0)
            times_coo[cycle] = TIMER_ELAPSED(0) / 1.e6;

        TIMER_TIME(1, spmv_csr(coo->col, csr_row, coo->val, vec, res2, coo->nrows));
        if (cycle >= 0)
            times_csr[cycle] = TIMER_ELAPSED(1) / 1.e6;

        for (int i = 0; i < coo->nrows; i++) {
            if ((res1[i] - res2[i]) / res1[i] > 0.01) {
                wrong++;
            }
        }


        TIMER_TIME(1, optimized_spmv_csr(coo->col, coo->row, coo->val, vec, res2, coo->nrows, coo->nnz));
        if (cycle >= 0)
            times_opt_csr[cycle] = TIMER_ELAPSED(1) / 1.e6;


        for (int i = 0; i < coo->nrows; i++) {
            if ((res1[i] - res2[i]) / res1[i] > 0.01) {
                wrong++;
            }
        }

    }

    // TIMING ------------------------------------------------------------------
    print_time_data("COO", times_coo, TEST_CYCLES, coo->nnz);
    print_time_data("CSR", times_csr, TEST_CYCLES, coo->nnz);
    print_time_data("OPT CSR", times_opt_csr, TEST_CYCLES, coo->nnz);
    printf("WRONG %d\n", wrong);
// FREE  -------------------------------------------------------------------
free:
    Distr_MMIO_COO_local_destroy(&coo);
    free(csr_row);
    free(vec);
    free(res1);
    free(res2);
    free(times_coo);
    free(times_csr);
    free(times_opt_csr);
    return 0;
}
