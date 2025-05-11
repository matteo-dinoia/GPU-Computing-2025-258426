// ES 6.1
#include <iostream>
#include <math.h>
#include <strings.h>
#include "include/mtx.h"
#include "include/time_utils.h"
#include <cuda_runtime.h>

#define MAX_THREAD_PER_BLOCK 1024
#define MAX_WARP 32
#define MAX_BLOCK 256
#define INPUT_FILENAME "../mawi_201512020330.mtx"

#define CYCLES 1
#define WARMUP_CYCLES 0

#define OK true
#define ERR false

#define eprintf(...) fprintf(stderr, __VA_ARGS__)
using namespace std;

// Kernel function to add the elements of two arrays
// ASSUME it is zeroed the res vector
__global__ void SpMV_A(const int *x, const int *y, const M_TYPE *val, const M_TYPE *vec, M_TYPE *res, int NON_ZERO) {
    int n_threads = gridDim.x * blockDim.x;
    int per_thread = (int)ceil(NON_ZERO / (float)n_threads);
    int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    int start = start_i * per_thread;

    for (int i = start; i < start + per_thread; i++) {
        if (i < NON_ZERO) {
            atomicAdd(&res[y[i]], val[i] * vec[x[i]]);
            //printf("%d %d %f %f %f\n", y[i], x[i], val[i], res[y[i]], vec[x[i]]);
        }
    }
}

// Kernel function to add the elements of two arrays
// ASSUME it is zeroed the res vector
__global__ void SpMV_B(const int *x, const int *y, const M_TYPE *val, const M_TYPE *vec, M_TYPE *res, int NON_ZERO) {
    int n_threads = gridDim.x * blockDim.x;
    int per_thread = (int)ceil(NON_ZERO / (float)n_threads);
    int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    int start = start_i * per_thread;

    for (int i = start; i < n_threads * per_thread; i += n_threads) {
        if (i < NON_ZERO) {
            atomicAdd(&res[y[i]], val[i] * vec[x[i]]);
            //printf("%d %d %f %f %f\n", y[i], x[i], val[i], res[y[i]], vec[x[i]]);
        }
    }
}

// ASSUME it is zeroed the res vector
void gemm_sparse_cpu(const int *cx, const int *cy, const M_TYPE *vals, const M_TYPE *vec, M_TYPE *res, const int NON_ZERO) {
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

int main(void) {
    cudaEvent_t start, stop;
    TIMER_DEF(0);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int NON_ZERO, ROWS, COLS;
    FILE *file;
    // HEADER READING --------------------------------------------------------------
    TIMER_TIME(0, {
        file = fopen(INPUT_FILENAME, "r");
        const bool status = read_mtx_header(file, &ROWS, &COLS, &NON_ZERO);
        if (status == ERR) {
            //TODO DON'T LEAK
            cout << "FATAL: fail to read header" << endl;
            return -1;
        }
    });
    cout << "READ HEADER: " << TIMER_ELAPSED(0) / 1.e3 << "ms" << endl;

    int *x, *y;
    M_TYPE *vals, *vec, *res, *res_control;

    cudaMallocManaged(&x, NON_ZERO * sizeof(int));
    cudaMallocManaged(&y, NON_ZERO * sizeof(int));
    cudaMallocManaged(&vals, NON_ZERO * sizeof(M_TYPE));
    cudaMallocManaged(&vec, COLS * sizeof(M_TYPE));
    cudaMallocManaged(&res, ROWS * sizeof(M_TYPE));
    res_control = (float *)malloc(ROWS * sizeof(float));

    // CREAZIONE DATA --------------------------------------------------------------
    TIMER_TIME(0, {
        const bool status = read_mtx_data(file, x, y, vals, NON_ZERO);
        if (status == ERR) {
            //TODO DON'T LEAK
            cout << "FATAL: fail to read data" << endl;
            return -1;
        }
    });
    cout << "READ DATA: " << TIMER_ELAPSED(0) / 1.e3 << "ms" << endl;

    double sum_times = 0;
    double sum_cpu_times = 0;
    srand(time(0));
    int n_block = min(MAX_BLOCK, (int)ceil(NON_ZERO / (float)MAX_THREAD_PER_BLOCK));
    int n_error = 0;
    int cycle;

    // Allocate Unified Memory accessible from CPU or GPU
    for (cycle = -WARMUP_CYCLES; cycle < CYCLES && n_error == 0; cycle++) {
        // initialize vec arrays with random values
        for (int i = 0; i < COLS; i++) {
            vec[i] = rand() % 50; //TODO Use float value
        }

        // Run cpu version
        bzero(res_control, ROWS * sizeof(float));
        TIMER_TIME(0, gemm_sparse_cpu(x, y, vals, vec, res_control, NON_ZERO));
        float cpu_time = TIMER_ELAPSED(0) / 1.e3;

        // Run kernel on 1M elements on the GPU
        bzero(res, ROWS * sizeof(float));
        cudaEventRecord(start);
        SpMV_A<<<n_block, MAX_THREAD_PER_BLOCK>>>(x, y, vals, vec, res, NON_ZERO);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        // Check for errors (all values should be 3.0f)
        for (int i = 0; i < ROWS; i++) {
            if (fabs(res_control[i] - res[i]) > res_control[i] * 0.01) {
                //cout << "INFO: data error: " << res[i] << " vs " << res_control[i] << endl;
                n_error++;
            }
        }

        // Get times
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if (cycle >= 0) {
            cout << "|--> Kernel Time (id " << cycle << "): " << milliseconds << "ms [cpu took " << cpu_time << " ms]"
                 << endl;
            sum_times += milliseconds;
            sum_cpu_times += cpu_time;
        }
    }

    cout << "|-----> Kernel Time (average): " << sum_times / CYCLES << " ms [cpu took " << sum_cpu_times / CYCLES << " ms]\n\n"
         << endl;

    if (n_error > 0) {
        cout << "There were " << n_error << " errors in the array (cycle " << cycle - 1 << ")" << endl;
    }

    // Free memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(x);
    cudaFree(y);
    cudaFree(res);
}
