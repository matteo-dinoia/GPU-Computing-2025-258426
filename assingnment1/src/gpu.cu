#include <iostream>

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
