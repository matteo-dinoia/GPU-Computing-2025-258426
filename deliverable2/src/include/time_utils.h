#pragma once

#define TIMER_DEF(n) struct timeval temp_1_##n = {0, 0}, temp_2_##n = {0, 0}
#define TIMER_START(n) gettimeofday(&temp_1_##n, (struct timezone*)0)
#define TIMER_STOP(n) gettimeofday(&temp_2_##n, (struct timezone*)0)
#define TIMER_TIME(n, op)                                                                                              \
    TIMER_START(n);                                                                                                    \
    op;                                                                                                                \
    TIMER_STOP(n)
#define TIMER_ELAPSED_MS(n)                                                                                            \
    ((temp_2_##n.tv_sec - temp_1_##n.tv_sec) * 1.e3 + (temp_2_##n.tv_usec - temp_1_##n.tv_usec) / 1.e3)

#define GPU_TIMER_DEF()                                                                                                \
    cudaEvent_t gpu_start, gpu_stop;                                                                                   \
    cudaEventCreate(&gpu_start);                                                                                       \
    cudaEventCreate(&gpu_stop)
#define GPU_TIMER_START() cudaEventRecord(gpu_start)
#define GPU_TIMER_STOP(pointer_pos)                                                                                    \
    cudaEventRecord(gpu_stop);                                                                                         \
    cudaEventSynchronize(gpu_stop);                                                                                    \
    cudaEventElapsedTime(pointer_pos, gpu_start, gpu_stop)
#define GPU_TIMER_DESTROY()                                                                                            \
    cudaEventDestroy(gpu_start);                                                                                       \
    cudaEventDestroy(gpu_stop);


double average(const double*, int);
double variance(const double*, double, int);
void print_time_data(const char*, const double*, int, int);
