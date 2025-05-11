#include <iostream>

#include "include/mtx.h"
#include "include/time_utils.h"

using std::cout;
using std::endl;

#define INPUT_FILENAME "datasets/mawi_201512020330.mtx"

bool read_data(struct Coo *);
void execution(const struct Coo, float *, float *, float *);
int timed_main();


int main() {
    int ret;
    TIMER_DEF(1);

    TIMER_TIME(1, {
        ret = timed_main();
    });

    cout << "TOTAL PROGRAM TIME: " << TIMER_ELAPSED(1) / 1.e3 << "ms" << endl;
    return ret;
}

int timed_main() {
    // Data allocation
    struct Coo matrix = {0, 0, 0, NULL, NULL};
    float *vec = NULL, *res = NULL, *res_control = NULL;

    // Data read
    bool ret = read_data(&matrix);

    // Execution
    if (ret == OK) {
        cudaMallocManaged(&vec, matrix.COLS * sizeof(float));
        cudaMallocManaged(&res, matrix.ROWS * sizeof(float));
        res_control = (float *)malloc(matrix.ROWS * sizeof(float));

        execution(matrix, vec, res, res_control);
    }


    // Free memory
    cudaFree(matrix.xs);
    cudaFree(matrix.ys);
    cudaFree(matrix.vals);
    cudaFree(res);
    free(res_control);
    return 0;
}


bool read_data(struct Coo *matrix) {
    FILE *file = fopen(INPUT_FILENAME, "r");
    TIMER_DEF(2);

    // Reading header
    TIMER_TIME(2, {
        const bool status = read_mtx_header(file, matrix);
        if (status == ERR) {
            cout << "FATAL: fail to read header" << endl;
            return ERR;
        }
    });
    cout << "READ HEADER: " << TIMER_ELAPSED(2) / 1.e3 << "ms" << endl;
    cout << "Header: " << matrix->ROWS << " " << matrix->COLS << " " << matrix->NON_ZERO << endl;

    // Alloc memory
    cudaMallocManaged(&matrix->xs, matrix->NON_ZERO * sizeof(int));
    cudaMallocManaged(&matrix->ys, matrix->NON_ZERO * sizeof(int));
    cudaMallocManaged(&matrix->vals, matrix->NON_ZERO * sizeof(float));

    // Reading data
    TIMER_TIME(2, {
        const bool status = read_mtx_data(file, matrix);
        if (status == ERR) {
            cout << "FATAL: fail to read data" << endl;
            return ERR;
        }
    });
    cout << "READ DATA: " << TIMER_ELAPSED(2) / 1.e3 << "ms" << endl;

    return OK;
}