#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include "include/mtx.h"
#include "include/time_utils.h"
#include "include/tester.h"

using std::cout, std::endl;

int timed_main(const char *);

int main(int argc, char **argv) {
    int ret = 1;
    TIMER_DEF(1);

    TIMER_TIME(1, {
        if (argc >= 2)
            ret = timed_main(argv[1]);
        else
            cout << "FATAL: require filename argument" << endl;
    });

    cout << "TOTAL PROGRAM TIME: " << TIMER_ELAPSED(1) / 1.e3 << "ms" << endl;
    return ret;
}

int timed_main(const char *input_file) {
    // Data allocation
    struct Coo matrix = {0, 0, 0, NULL, NULL, NULL};
    float *vec = NULL;
    float *res = NULL;
    float *res_control = NULL;
    bool status = OK;
    TIMER_DEF(2);

    // File opener
    std::ifstream file(input_file);
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

    // Alloc memory
    cudaMallocManaged(&matrix.xs, matrix.NON_ZERO * sizeof(int));
    cudaMallocManaged(&matrix.ys, matrix.NON_ZERO * sizeof(int));
    cudaMallocManaged(&matrix.vals, matrix.NON_ZERO * sizeof(float));

    // Reading data
    TIMER_TIME(2, { status = read_mtx_data(file, &matrix); });
    if (status == ERR) {
        cout << "FATAL: fail to read data" << endl;
        goto free;
    }
    cout << "# READ DATA: " << TIMER_ELAPSED(2) / 1.e3 << "ms" << endl;

    // Execution
    cudaMallocManaged(&vec, matrix.COLS * sizeof(float));
    cudaMallocManaged(&res, matrix.ROWS * sizeof(float));
    res_control = (float *)malloc(matrix.ROWS * sizeof(float));

    execution(matrix, vec, res, res_control);

free:
    // Free memory and close resources
    file.close();
    free(res_control);
    cudaFree(matrix.xs);
    cudaFree(matrix.ys);
    cudaFree(matrix.vals);
    cudaFree(vec);
    cudaFree(res);
    return status == OK ? 0 : 1;
}