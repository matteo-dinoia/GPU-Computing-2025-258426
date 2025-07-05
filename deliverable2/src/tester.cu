#include <iostream>
#include <vector>
#include "include/gpu.h"
#include "include/tester.h"
#include "include/time_utils.h"
#include "include/utils.h"

using std::cout, std::endl;

#define CYCLES 1
#define WARMUP_CYCLES 0
#define PRINT_INTERMEDIATE false
#define CHECK_CORRECT false

inline float test_kernel(const SmpvKernel* kernel, const GpuCoo<u32, MV>& matrix, const MV* vec, MV* res)
{
    float time;
    GPU_TIMER_DEF();

    bzero(res, matrix.ROWS * sizeof(MV));
    const auto [blocks, threads, shm] = kernel->parameter_getter(matrix);
    // cout << "Running " << kernel->name << " with " << blocks << " " << threads << " " << shm << endl;

    GPU_TIMER_START();
    if (shm == 0)
        kernel->execute<<<blocks, threads>>>(matrix.xs, matrix.ys, matrix.vals, vec, res, matrix.NON_ZERO);
    else
        kernel->execute<<<blocks, threads, shm>>>(matrix.xs, matrix.ys, matrix.vals, vec, res, matrix.NON_ZERO);
    GPU_TIMER_STOP(&time);
    if (cudaGetLastError() != cudaSuccess)
    {
        cout << "Kernel " << kernel->name << " failed" << endl;
        cout << "Runned " << kernel->name << " with " << blocks << " " << threads << " " << shm << endl;
        return -1;
    }

    GPU_TIMER_DESTROY();
    return time;
}

void execution(const GpuCoo<u32, MV>& matrix, const MV* vec, MV* res, MV* res_control)
{
    int cycle;


    // Kernel used in the testing
    const std::vector kernels = {baseline,
                                 block_jump,
                                 warp_jump,
                                 prefix_sum_unlimited,
                                 prefix_sum_32_max,
                                 prefix_sum_max_32_work_efficient,
                                 prefix_sum_we_32_conflict_free,
                                 prefix_sum_we_unlimited_conflict_free,
                                 prefix_sum_warp,
                                 prefix_sum_warp_2x,
                                 prefix_sum_warp_with_block_jump,
                                 prefix_sum_warp_merged};

    // Time definition
    float gpu_times[kernels.size()];
    double sum_times[kernels.size()] = {};

    // Execute multiple time
    bool failed = false;
    u32 failed_idx = 0;
    for (cycle = -WARMUP_CYCLES; cycle < CYCLES && !failed; cycle++)
    {
        // Kernels
        gpu_times[0] = test_kernel(&kernels[0], matrix, vec, res_control);
        // print_min_max(res_control, matrix.ROWS);

        for (u32 i = 1; i < kernels.size() && !failed; i++)
        {
            gpu_times[i] = test_kernel(&kernels[i], matrix, vec, res);
            if (gpu_times[i] < 0)
            {
                failed = true;
                failed_idx = i;
            }
            else if (cycle >= 0 && CHECK_CORRECT)
                print_diff_info(res, res_control, matrix.ROWS, kernels[i].name);
        }

        // Save times
        if (cycle >= 0 && !failed)
        {
            cout << "|--> Kernel Time (id " << cycle << "): ";

            for (u32 i = 0; i < kernels.size(); i++)
            {
                if (gpu_times[i] < 0.0 || sum_times[i] < 0.0)
                    sum_times[i] = -1;
                else
                    sum_times[i] += gpu_times[i];
#if PRINT_INTERMEDIATE == true
                cout << "[" << kernels[i].name << "]=> " << gpu_times[i] << "ms ";
#endif
            }

            cout << endl;
        }
    }


    if (failed)
    {
        cout << "\nFAILED when running kernel " << kernels[failed_idx].name << "\n" << endl;
    }
    else
    {
        cout << "|" << endl;
        cout << "|-----> Kernel Time (average): (kernel_id, block_size, avg_time, gflops)" << endl;
        for (u32 i = 0; i < kernels.size(); i++)
        {
            if (sum_times[i] > 0.0)
            {
                const double avg = sum_times[i] / cycle;
                const double gflops = 2 * (matrix.NON_ZERO / avg / 1e6);
                const double gbs = 6 * 4 * (matrix.NON_ZERO / avg / 1e6);
                cout << "|---------->[" << kernels[i].name << "]=> " << avg << "ms (" << gflops << " Gflops " << gbs
                     << " Gbs)" << endl;
            }
            else if (sum_times[i] == 0.0)
            {
                cout << "|---------->[" << kernels[i].name << "]=> not runned" << endl;
            }
            else
            {
                cout << "|---------->[" << kernels[i].name << "]=> failed" << endl;
            }
        }
        cout << endl;
    }
}
