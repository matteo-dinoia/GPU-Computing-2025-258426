#ifndef LAB1_EX2_LIB
#define LAB1_EX2_LIB

#include <sys/time.h>


#define TIMER_DEF(n) struct timeval temp_1_##n = {0, 0}, temp_2_##n = {0, 0}
#define TIMER_START(n) gettimeofday(&temp_1_##n, (struct timezone *)0)
#define TIMER_STOP(n) gettimeofday(&temp_2_##n, (struct timezone *)0)
#define TIMER_TIME(n, op) \
    TIMER_START(n);       \
    op;                   \
    TIMER_STOP(n)
#define TIMER_ELAPSED(n) ((temp_2_##n.tv_sec - temp_1_##n.tv_sec) * 1.e6 + (temp_2_##n.tv_usec - temp_1_##n.tv_usec))

#define GPU_TIMER_DEF()      \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop)
#define GPU_TIMER_START() \
    cudaEventRecord(start)
#define GPU_TIMER_STOP(pointer_pos) \
    cudaEventRecord(stop);          \
    cudaEventSynchronize(stop);     \
    cudaEventElapsedTime(pointer_pos, start, stop)


double average(const double *, const int);
double variance(const double *, const double, const int);
void print_time_data(const char *, const double *, const int, const int);

#endif
