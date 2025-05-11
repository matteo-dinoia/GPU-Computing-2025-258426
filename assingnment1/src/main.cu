#include <iostream>
#include <math.h>
#include <strings.h>

#include "include/cpu.h"
#include "include/gpu.h"
#include "include/mtx.h"
#include "include/time_utils.h"

using std::cout;
using std::endl;
using std::min;
using std::max;

#define MAX_THREAD_PER_BLOCK 1024
#define MAX_WARP 32
#define MAX_BLOCK 256
#define INPUT_FILENAME "datasets/mawi_201512020330.mtx"

#define CYCLES 10
#define WARMUP_CYCLES 0

#define OK true
#define ERR false

bool read_data(struct Coo *);
void execution(const struct Coo, float *, float *, float *);
int timed_main();
int diff_size(float *, float *, int);

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

void execution(const struct Coo matrix, float *vec, float *res, float *res_control) {
    int n_blocks = min(MAX_BLOCK, (int)ceil(matrix.NON_ZERO / (float)MAX_THREAD_PER_BLOCK));
    int n_thread_per_block = min(MAX_THREAD_PER_BLOCK, matrix.NON_ZERO);
    cout << "Starting with <<<" << n_blocks << ", " << n_thread_per_block << ">>>" << endl;

    srand(time(0));
    TIMER_DEF(0);
    GPU_TIMER_DEF();
    int n_error = 0;

    // Time definition
    int N_GPU_KERNEL = 2;
    float gpu_times[N_GPU_KERNEL] = {0};
    double sum_times[N_GPU_KERNEL] = {0};
    double cpu_time = 0;
    double sum_cpu_times = 0;

    // Execute multiple time
    int cycle;
    for (int cycle = -WARMUP_CYCLES; cycle < CYCLES; cycle++) {
        // initialize vec arrays with random values
        for (int i = 0; i < matrix.COLS; i++) {
            vec[i] = rand() % 50;
        }

        // Run cpu version
        bzero(res_control, matrix.ROWS * sizeof(float));
        TIMER_TIME(0, gemm_sparse_cpu(matrix.xs, matrix.ys, matrix.vals, vec, res_control, matrix.NON_ZERO));
        cpu_time = TIMER_ELAPSED(0) / 1.e3;

        // KERNEL 1
        bzero(res, matrix.ROWS * sizeof(float));
        GPU_TIMER_START();
        SpMV_A<<<n_blocks, n_thread_per_block>>>(matrix.xs, matrix.ys, matrix.vals, vec, res, matrix.NON_ZERO);
        GPU_TIMER_STOP(&gpu_times[0]);
        // Check for errors
        n_error += diff_size(res, res_control, matrix.ROWS);

        // KERNEL 2
        bzero(res, matrix.ROWS * sizeof(float));
        GPU_TIMER_START();
        SpMV_B<<<n_blocks, n_thread_per_block>>>(matrix.xs, matrix.ys, matrix.vals, vec, res, matrix.NON_ZERO);
        GPU_TIMER_STOP(&gpu_times[1]);
        // Check for errors
        n_error += diff_size(res, res_control, matrix.ROWS);

        if (cycle >= 0) {
            cout << "|--> Kernel Time (id " << cycle << "): ";
            for (int i = 0; i < N_GPU_KERNEL; i++) {
                sum_times[i] += gpu_times[i];
                cout << "[id=" << i << "] => " << gpu_times[i] << "ms ";
            }
            cout << "[cpu " << cpu_time << " ms]" << endl;

            sum_cpu_times += cpu_time;
        }
    }

    cout << "|-----> Kernel Time (average): ";
    for (int i = 0; i < N_GPU_KERNEL; i++) {
        cout << "[id=" << i << "] => " << sum_times[i] / cycle << "ms ";
    }
    cout << "[cpu " << sum_cpu_times / cycle << " ms]\n\n"
         << endl;

    if (n_error > 0) {
        cout << "There were " << n_error << " errors in the array (cycle " << cycle - 1 << ")" << endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int diff_size(float *v, float *control, int LEN) {
    int n_error = 0;

    for (int i = 0; i < LEN; i++) {
        if (fabs(v[i] - control[i]) > control[i] * 0.01) {
            //cout << "INFO: data error: " << a[i] << " vs expected " << control[i] << endl;
            n_error++;
        }
    }

    return n_error;
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