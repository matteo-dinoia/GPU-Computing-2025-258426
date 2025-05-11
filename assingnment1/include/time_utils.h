#ifndef LAB1_EX2_LIB
#define LAB1_EX2_LIB

#include <sys/time.h>

#define TIMER_DEF(n)	 struct timeval temp_1_##n={0,0}, temp_2_##n={0,0}
#define TIMER_START(n)	 gettimeofday(&temp_1_##n, (struct timezone*)0)
#define TIMER_STOP(n)	 gettimeofday(&temp_2_##n, (struct timezone*)0)
#define TIMER_TIME(n, op)   TIMER_START(n); (op); TIMER_STOP(n);
#define TIMER_ELAPSED(n) ((temp_2_##n.tv_sec-temp_1_##n.tv_sec)*1.e6+(temp_2_##n.tv_usec-temp_1_##n.tv_usec))

double average(const double *times, const int N)
{
	double average = 0;
    for (int i = 0; i < N; i++) {
        average += times[i] / N;
    }
    return average;
}

double variance(const double *times, const double average, const int N)
{
    double variance = 0;
    for (int i = 0; i < N; i++) {
        const double diff = average - times[i];
        variance += diff * diff / N;
    }
    return variance;
}

void print_time_data(const char *name, const double *times,  const int N, const int OP) {
    double mean = average(times, N);
    double var = variance(times, mean, N);

    const double flops1 = OP / mean;
    printf("MY IMP TIME OF '%s' avarage %fms (%f MFLOP/S) over %d size and %d tests [with var = %e]\n",
            name, mean * 1e3, flops1 / 1e6, OP, N, var);
}

#endif

