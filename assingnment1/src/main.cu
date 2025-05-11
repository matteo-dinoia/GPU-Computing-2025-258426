// ES 6.1
// Use "module load GCC/12.3.0" to enable highlighting
#include <iostream>
#include <math.h>
#include <strings.h>
#include "include/mtx.h"
#include "include/time_utils.h"
#include <cuda_runtime.h>

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

// Kernel function to add the elements of two arrays
// ASSUME it is zeroed the res vector
__global__ void SpMV_A(const int *x, const int *y, const float *val, const float *vec, float *res, int NON_ZERO) {
    int n_threads = gridDim.x * blockDim.x;
    int per_thread = (int)ceil(NON_ZERO / (float)n_threads);
    int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    int start = start_i * per_thread;

    for (int i = 0; i < per_thread; i++) {
        const int el = start + i;
        if (el < NON_ZERO) {
            atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
            //printf("%d %d %f %f %f\n", y[el], x[el], val[el], res[y[el]], vec[x[el]]);
        }
    }
}

// Kernel function to add the elements of two arrays
// ASSUME it is zeroed the res vector
__global__ void SpMV_B(const int *x, const int *y, const float *val, const float *vec, float *res, int NON_ZERO) {
    int n_threads = gridDim.x * blockDim.x;
    int per_thread = (int)ceil(NON_ZERO / (float)n_threads);
    int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < per_thread; i++) {
        const int el = start_i + i * n_threads;
        if (el < NON_ZERO) {
            atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
            //printf("%d %d %f %f %f\n", y[el], x[el], val[el], res[y[el]], vec[x[el]]);
        }
    }
}

// ASSUME it is zeroed the res vector
void gemm_sparse_cpu(const int *cx, const int *cy, const float *vals, const float *vec, float *res, const int NON_ZERO) {
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

int run() {
    TIMER_DEF(1);
    TIMER_START(1);
    cudaEvent_t start, stop;
    TIMER_DEF(0);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    struct Coo matrix;
    FILE *file;
    // HEADER READING --------------------------------------------------------------
    TIMER_TIME(0, {
        file = fopen(INPUT_FILENAME, "r");
        const bool status = read_mtx_header(file, &matrix);
        if (status == ERR) {
            //TODO DON'T LEAK
            cout << "FATAL: fail to read header" << endl;
            return -1;
        }
    });
    cout << "READ HEADER: " << TIMER_ELAPSED(0) / 1.e3 << "ms" << endl;
    cout << "Header: " << matrix.ROWS << " " << matrix.COLS << " " << matrix.NON_ZERO << endl;

    float *vec, *res, *res_control;
    cudaMallocManaged(&matrix.xs, matrix.NON_ZERO * sizeof(int));
    cudaMallocManaged(&matrix.ys, matrix.NON_ZERO * sizeof(int));
    cudaMallocManaged(&matrix.vals, matrix.NON_ZERO * sizeof(float));
    cudaMallocManaged(&vec, matrix.COLS * sizeof(float));
    cudaMallocManaged(&res, matrix.ROWS * sizeof(float));
    res_control = (float *)malloc(matrix.ROWS * sizeof(float));

    // CREAZIONE DATA --------------------------------------------------------------
    TIMER_TIME(0, {
        const bool status = read_mtx_data(file, &matrix);
        if (status == ERR) {
            //TODO DON'T LEAK
            cout << "FATAL: fail to read data" << endl;
            return -1;
        }
    });
    cout << "READ DATA: " << TIMER_ELAPSED(0) / 1.e3 << "ms" << endl;

    int N_GPU_KERNEL = 2;
    float gpu_times[N_GPU_KERNEL] = {0};
    double cpu_time = 0;
    double sum_times[N_GPU_KERNEL] = {0};
    double sum_cpu_times = 0;
    srand(time(0));
    int n_error = 0;
    int cycle;

    int n_blocks = min(MAX_BLOCK, (int)ceil(matrix.NON_ZERO / (float)MAX_THREAD_PER_BLOCK));
    int n_thread_per_block = min(MAX_THREAD_PER_BLOCK, matrix.NON_ZERO);
    cout << "Starting with <<<" << n_blocks << ", " << n_thread_per_block << ">>>" << endl;

    // Allocate Unified Memory accessible from CPU or GPU
    for (cycle = -WARMUP_CYCLES; cycle < CYCLES /*&& n_error == 0*/; cycle++) {
        n_error = 0; //TODO REmove
        // initialize vec arrays with random values
        for (int i = 0; i < matrix.COLS; i++) {
            vec[i] = rand() % 50; //TODO Use float value
        }

        // Run cpu version
        bzero(res_control, matrix.ROWS * sizeof(float));
        TIMER_TIME(0, gemm_sparse_cpu(matrix.xs, matrix.ys, matrix.vals, vec, res_control, matrix.NON_ZERO));
        cpu_time = TIMER_ELAPSED(0) / 1.e3;

        // KERNEL 1
        cudaEventRecord(start);
        bzero(res, matrix.ROWS * sizeof(float));
        SpMV_A<<<n_blocks, n_thread_per_block>>>(matrix.xs, matrix.ys, matrix.vals, vec, res, matrix.NON_ZERO);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_times[0], start, stop);
        // Check for errors
        for (int i = 0; i < matrix.ROWS; i++) {
            if (fabs(res_control[i] - res[i]) > res_control[i] * 0.01) {
                cout << "INFO: data error: " << res[i] << " vs " << res_control[i] << endl;
                n_error++;
            }
        }
        cout << "After kernel 1 errors are: " << n_error << endl;

        // KERNEL 2
        cudaEventRecord(start);
        bzero(res, matrix.ROWS * sizeof(float));
        SpMV_B<<<n_blocks, n_thread_per_block>>>(matrix.xs, matrix.ys, matrix.vals, vec, res, matrix.NON_ZERO);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_times[1], start, stop);
        // Check for errors
        for (int i = 0; i < matrix.ROWS; i++) {
            if (fabs(res_control[i] - res[i]) > res_control[i] * 0.01) {
                cout << "INFO: data error: " << res[i] << " vs " << res_control[i] << endl;
                n_error++;
            }
        }
        cout << "After kernel 2 errors are:  " << n_error << endl;

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
        cout << "[id=" << i << "] => " << sum_times[i] / CYCLES << "ms ";
    }
    cout << "[cpu " << sum_cpu_times / CYCLES << " ms]\n\n"
         << endl;

    if (n_error > 0) {
        cout << "There were " << n_error << " errors in the array (cycle " << cycle - 1 << ")" << endl;
    }

    // Free memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(matrix.xs);
    cudaFree(matrix.ys);
    cudaFree(matrix.vals);
    cudaFree(res);
    cudaFree(res_control);

    // full time
    TIMER_STOP(1);
    cout << "TOTAL PROGRAM TIME: " << TIMER_ELAPSED(1) / 1.e3 << "ms" << endl;

    return 0;
}
