#include <iostream>
#include <math.h>
#include <strings.h>

#include "include/cpu.h"
#include "include/gpu.h"
#include "include/mtx.h"
#include "include/time_utils.h"
#include "include/tester.h"

#define MAX_THREAD_PER_BLOCK 1024
#define MAX_WARP 32
#define MAX_BLOCK 256

#define CYCLES 10
#define WARMUP_CYCLES 0

using std::cout;
using std::endl;

int diff_size(float *, float *, int);

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
        TIMER_TIME(0, spmv_cpu(matrix.xs, matrix.ys, matrix.vals, vec, res_control, matrix.NON_ZERO));
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