#ifndef GPU_H
#define GPU_H


__global__ void SpMV_A(const int *, const int *, const float *, const float *, float *, int);
__global__ void SpMV_B(const int *, const int *, const float *, const float *, float *, int);


#endif