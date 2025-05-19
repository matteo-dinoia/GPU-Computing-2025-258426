#include <iostream>

// Kernel function to add the elements of two arrays
// ASSUME it is zeroed the res vector
__global__ void spmv_a(const int *x, const int *y, const float *val, const float *vec, float *res, int NON_ZERO) {
    const int n_threads = gridDim.x * blockDim.x;
    const int per_thread = (int)ceil(NON_ZERO / (float)n_threads);
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    const int start = start_i * per_thread;
    const int end = start + per_thread;
    const int incr = 1;

    for (int el = start; el < end; el += incr) {
        if (el < NON_ZERO) {
            atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
            //printf("%d %d %f %f %f\n", y[el], x[el], val[el], res[y[el]], vec[x[el]]);
        }
    }
}

// Kernel function to add the elements of two arrays
// ASSUME it is zeroed the res vector
__global__ void spmv_b(const int *x, const int *y, const float *val, const float *vec, float *res, int NON_ZERO) {
    const int n_threads = gridDim.x * blockDim.x;
    const int per_thread = (int)ceil(NON_ZERO / (float)n_threads);
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    const int start = start_i;
    const int end = start_i + per_thread * n_threads;
    const int incr = n_threads;

    for (int el = start; el < end; el += incr) {
        if (el < NON_ZERO) {
            atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
            //printf("%d %d %f %f %f\n", y[el], x[el], val[el], res[y[el]], vec[x[el]]);
        }
    }
}

// Kernel function to add the elements of two arrays
// ASSUME it is zeroed the res vector
__global__ void spmv_c(const int *x, const int *y, const float *val, const float *vec, float *res, int NON_ZERO) {
    const int n_threads = gridDim.x * blockDim.x;
    const int per_thread = (int)ceil(NON_ZERO / (float)n_threads);
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    const int wrap_id = start_i / warpSize;
    const int start = wrap_id * (per_thread * warpSize) + (start_i - wrap_id * warpSize);
    const int incr = warpSize;

    for (int i = 0; i < per_thread; i++) {
        const int el = start + i * incr;
        if (el < NON_ZERO) {
            atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
        }
    }
}

// Kernel function to add the elements of two arrays
// ASSUME it is zeroed the res vector
__global__ void spmv_ci(const int *x, const int *y, const float *val, const float *vec, float *res, int NON_ZERO) {
    const int n_threads = gridDim.x * blockDim.x;
    const int per_thread = (int)ceil(NON_ZERO / (float)n_threads);
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    const int wrap_id = start_i / warpSize;
    const int start = wrap_id * (per_thread * warpSize) + (start_i - wrap_id * warpSize);
    const int incr = warpSize;

    for (int i = 0; i < per_thread; i++) {
        const int el = start + i * incr;
        if (el < NON_ZERO) {
            atomicAdd(&res[x[el]], val[el] * vec[y[el]]);
        }
    }
}

// Kernel function to add the elements of two arrays
// ASSUME it is zeroed the res vector
__global__ void spmv_cc(const int *xy, const int *null, const float *val, const float *vec, float *res, int NON_ZERO) {
    const int n_threads = gridDim.x * blockDim.x;
    const int per_thread = (int)ceil(NON_ZERO / (float)n_threads);
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    const int wrap_id = start_i / warpSize;
    const int start = wrap_id * (per_thread * warpSize) + (start_i - wrap_id * warpSize);
    const int incr = warpSize;

    for (int i = 0; i < per_thread; i++) {
        const int el = start + i * incr;
        if (el < NON_ZERO) {
            atomicAdd(&res[xy[el * 2 + 1]], val[el] * vec[xy[el * 2]]);
        }
    }
}


// Kernel function to add the elements of two arrays
// ASSUME it is zeroed the res vector
__global__ void spmv_d(const int *x, const int *y, const float *val, const float *vec, float *res, int NON_ZERO) {
    const int n_threads = gridDim.x * blockDim.x;
    const int per_thread = (int)ceil(NON_ZERO / (float)n_threads);

    const int start = blockIdx.x * (per_thread * blockDim.x) + threadIdx.x;
    const int incr = blockDim.x;

    for (int i = 0; i < per_thread; i++) {
        const int el = start + i * incr;
        if (el < NON_ZERO) {
            atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
        }
    }
}

// Kernel function to add the elements of two arrays
// ASSUME it is zeroed the res vector
__global__ void spmv_e(const int *x, const int *y, const float *val, const float *vec, float *res, int NON_ZERO) {
    const int n_threads = gridDim.x * blockDim.x;
    const int per_thread = (int)ceil(NON_ZERO / (float)n_threads);
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    const int wrap_id = start_i / warpSize;
    const int start = per_thread * (warpSize * wrap_id) + threadIdx.x;
    const int incr = warpSize;

    for (int i = 0; i < per_thread; i++) {
        const int el = start + i * incr;
        if (el < NON_ZERO) {
            atomicAdd(&res[y[el]], val[el] * vec[x[el]]);
        }
    }
}

// Function pointer type for a kernel