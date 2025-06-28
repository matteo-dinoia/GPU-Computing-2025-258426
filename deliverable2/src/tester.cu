#include <iostream>
#include <vector>
#include "include/gpu.h"
#include "include/tester.h"
#include "include/time_utils.h"
#include "include/utils.h"

using std::cout, std::endl;

#define CYCLES 3
#define WARMUP_CYCLES 1

inline float test_kernel(const SmpvKernel* kernel, const GpuCoo<u32, float>& matrix, const float* vec, float* res)
{
    float time;
    GPU_TIMER_DEF();

    bzero(res, matrix.ROWS * sizeof(float));
    const auto [blocks, threads, shm] = kernel->parameter_getter(matrix);

    GPU_TIMER_START();
    if (shm == 0)
        kernel->execute<<<blocks, threads>>>(matrix.xs, matrix.ys, matrix.vals, vec, res, matrix.NON_ZERO);
    else
        kernel->execute<<<blocks, threads, shm>>>(matrix.xs, matrix.ys, matrix.vals, vec, res, matrix.NON_ZERO);
    GPU_TIMER_STOP(&time);

    GPU_TIMER_DESTROY();
    return time;
}

void execution(const GpuCoo<uint32_t, float>& matrix, const float* vec, float* res, float* res_control)
{
    int cycle;

    // Kernel used in the testing
    const std::vector kernels = {
        baseline, full_strided, full_jump, warp_jump, warp_jump_bkp, block_jump, block_jump_bkp
    };

    // Time definition
    float gpu_times[kernels.size()];
    double sum_times[kernels.size()] = {};

    // Execute multiple time
    for (cycle = -WARMUP_CYCLES; cycle < CYCLES; cycle++)
    {
        // Kernels
        gpu_times[0] = test_kernel(&kernels[0], matrix, vec, res_control);
        // print_min_max(res_control, matrix.ROWS);
        for (u32 i = 1; i < kernels.size(); i++)
        {
            gpu_times[i] = test_kernel(&kernels[i], matrix, vec, res);
            if (cycle >= 0)
                print_diff_info(res, res_control, matrix.ROWS, kernels[i].name);
        }

        // Save times
        if (cycle >= 0)
        {
            cout << "|--> Kernel Time (id " << cycle << "): ";
            for (u32 i = 0; i < kernels.size(); i++)
            {
                sum_times[i] += gpu_times[i];
                cout << "[" << kernels[i].name << "]=> " << gpu_times[i] << "ms ";
            }
            cout << endl;
        }
    }

    cout << "|" << endl;
    cout << "|-----> Kernel Time (average): (kernel_id, block_size, avg_time, gflops)" << endl;
    for (u32 i = 0; i < kernels.size(); i++)
    {
        const double avg = sum_times[i] / cycle;
        const double gflops = 2 * (matrix.NON_ZERO / avg / 1e6);
        const double gbs = 6 * 4 * (matrix.NON_ZERO / avg / 1e6);
        cout << "|---------->[" << kernels[i].name << "]=> " << avg << "ms (" << gflops << " Gflops " << gbs << " Gbs)"
             << endl;
    }
    cout << endl;
}
