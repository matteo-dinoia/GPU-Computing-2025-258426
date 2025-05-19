#pragma once

typedef void (*kernel_func)(const int *, const int *, const float *, const float *, float *, int);

void spmv_a(const int *, const int *, const float *, const float *, float *, int);
void spmv_b(const int *, const int *, const float *, const float *, float *, int);
void spmv_c(const int *, const int *, const float *, const float *, float *, int);
void spmv_d(const int *, const int *, const float *, const float *, float *, int);
void spmv_e(const int *, const int *, const float *, const float *, float *, int);
void spmv_ci(const int *, const int *, const float *, const float *, float *, int);
void spmv_cc(const int *, const int *, const float *, const float *, float *, int);