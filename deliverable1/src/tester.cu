#include <iostream>
#include <math.h>
#include <strings.h>

#include "include/cpu.h"
#include "include/gpu.h"
#include "include/mtx.h"
#include "include/time_utils.h"
#include "include/tester.h"
#include "include/utils.h"

#define MAX_THREAD_PER_BLOCK 1024
#define MAX_BLOCK 256

#define CYCLES 10
#define WARMUP_CYCLES 1

using std::cout, std::endl;

int diff_size(float *, float *, int);


void execution(const struct Coo matrix, float *vec, float *res, float *res_control) {
    srand(time(0));
    TIMER_DEF(0);
    GPU_TIMER_DEF();

    // Time definition
    const int N_GPU_KERNEL = 5;
    kernel_func kernels[N_GPU_KERNEL] = {spmv_a, spmv_b, spmv_c, spmv_d, spmv_e};
    float gpu_times[N_GPU_KERNEL] = {0};
    double sum_times[N_GPU_KERNEL] = {0};
    //int gpu_errors[N_GPU_KERNEL] = {0};

    // initialize vec arrays with random values
    for (int i = 0; i < matrix.COLS; i++) {
        vec[i] = rand() % 50;
    }

    int *xy;
    cudaMallocManaged(&xy, matrix.NON_ZERO * 2 * sizeof(int));
    for (int i = 0; i < matrix.NON_ZERO; i++) {
        xy[2 * i] = matrix.xs[i];
        xy[2 * i + 1] = matrix.ys[i];
    }

    // Run cpu version
    bzero(res_control, matrix.ROWS * sizeof(float));
    TIMER_TIME(0, spmv_cpu(matrix.xs, matrix.ys, matrix.vals, vec, res_control, matrix.NON_ZERO));
    const double cpu_time = TIMER_ELAPSED(0) / 1.e3;

    // Execute multiple time
    int n_blocks = std::min(MAX_BLOCK, (int)ceil(matrix.NON_ZERO / (float)MAX_THREAD_PER_BLOCK));
    int n_thread_per_block = std::min(MAX_THREAD_PER_BLOCK, matrix.NON_ZERO);
    cout << "# Starting with <<<" << n_blocks << ", " << n_thread_per_block << ">>>\n"
         << endl;

    bzero(sum_times, N_GPU_KERNEL * sizeof(*sum_times));

    int cycle;
    for (cycle = -WARMUP_CYCLES; cycle < CYCLES; cycle++) {
        // KERNELS
        for (int i = 0; i < N_GPU_KERNEL; i++) {
            bzero(res, matrix.ROWS * sizeof(float));
            GPU_TIMER_START();
            kernels[i]<<<n_blocks, n_thread_per_block>>>(matrix.xs, matrix.ys, matrix.vals, vec, res, matrix.NON_ZERO);
            GPU_TIMER_STOP(&gpu_times[i]);

            //gpu_errors[i] += diff_size(res, res_control, matrix.ROWS);
            //cout << "ERROR of gpu " << i << " are " << gpu_errors[i] << endl;
        }
        bzero(res, matrix.ROWS * sizeof(float));

        // Save times
        if (cycle >= 0) {
            cout << "|--> Kernel Time (id " << cycle << "): ";
            for (int i = 0; i < N_GPU_KERNEL; i++) {
                sum_times[i] += gpu_times[i];
                cout << "[id=" << i << "]=> " << gpu_times[i] << "ms ";
            }
            cout << endl;
        }
    }

    cout << "|" << endl;
    cout << "|-----> Kernel Time (average): (kernel_id, block_size, avg_time, gflops)" << endl;
    for (int i = 0; i < N_GPU_KERNEL; i++) {
        double avg = sum_times[i] / cycle;
        double gflops = 2 * (matrix.NON_ZERO / avg / 1e6);
        double gbs = 6 * 4 * (matrix.NON_ZERO / avg / 1e6);
        cout << "|---------->[ id =" << i << " ]=> " << avg << "ms (" << gflops << " Gflops " << gbs << " Gbs)" << endl; //with # error " << gpu_errors[i] << endl;
    }
    cout << endl;

    double avg = cpu_time; // / cycle;
    double gflops = 2 * (matrix.NON_ZERO / avg / 1e6);
    double gbs = 6 * 4 * (matrix.NON_ZERO / avg / 1e6);
    cout << "\n\n|---------->[  cpu  ]=> " << avg << " ms (" << gflops << " Gflops " << gbs << " Gbs) \n\n"
         << endl;

    cudaFree(xy);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}