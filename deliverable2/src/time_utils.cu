#include <iostream>
#include "include/time_utils.h"

using std::cout;
using std::endl;


double average(const double *times, const int N) {
    double average = 0;
    for (int i = 0; i < N; i++) {
        average += times[i] / N;
    }
    return average;
}

double variance(const double *times, const double average, const int N) {
    double variance = 0;
    for (int i = 0; i < N; i++) {
        const double diff = average - times[i];
        variance += diff * diff / N;
    }
    return variance;
}

void print_time_data(const char *name, const double *times, const int N, const int OP) {
    const double mean = average(times, N);
    const double var = variance(times, mean, N);

    const double flops1 = OP / mean;
    printf("MY IMP TIME OF '%s' avarage %fms (%f MFLOP/S) over %d size and %d tests [with var = %e]\n",
           name, mean * 1e3, flops1 / 1e6, OP, N, var);
}
