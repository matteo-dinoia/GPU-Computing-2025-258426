#include <iostream>
#include <vector>
#include "include/gpu.h"
#include "include/tester.h"
#include <fstream>
#include <sys/time.h>
#include "include/time_utils.h"
#include "include/utils.h"

using std::cout, std::endl, std::cerr;


// Quick way to run only once
#if false
#define CYCLES 1
#define WARMUP_CYCLES 0
#else
#define CYCLES 5
#define WARMUP_CYCLES 1
#endif

// WARNING: enabling this option will produce false positive errors
// as there is no way to account for float imprecision over line dense or big
// matrix.
#define CHECK_CORRECT false


// Return a time of kernel execution and the total time with
// also the time to reset the result vector.
inline std::pair<float, float> test_kernel(const SmpvKernel* kernel, const GpuCoo<MI, MV>& matrix, const MV* vec,
                                           MV* res)
{
    float time;
    GPU_TIMER_DEF();
    TIMER_DEF(0);

    const auto [blocks, threads, shm] = kernel->parameter_getter(matrix);

    // Reset result vector to 0
    TIMER_START(0);
    bzero(res, matrix.ROWS * sizeof(MV));
    // Execute the kernel
    GPU_TIMER_START();
    if (shm == 0)
        kernel->execute<<<blocks, threads>>>(matrix.xs, matrix.ys, matrix.vals, vec, res, matrix.NON_ZERO);
    else
        kernel->execute<<<blocks, threads, shm>>>(matrix.xs, matrix.ys, matrix.vals, vec, res, matrix.NON_ZERO);
    GPU_TIMER_STOP(&time);
    TIMER_STOP(0);
    float total_time = TIMER_ELAPSED_MS(0);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cout << "Kernel " << kernel->name << " failed with " << err << endl;
        cout << "Runned " << kernel->name << " with " << blocks << " " << threads << " " << shm << endl;
        return std::make_pair(-1, total_time);
    }

    GPU_TIMER_DESTROY();
    return std::make_pair(time, total_time);
}

void execution(const GpuCoo<MI, MV>& matrix, const MV* vec, MV* res, MV* res_control, const char* output_csv_filename)
{
    std::ofstream csv;
    csv.open(output_csv_filename);

    int cycle;

    // Kernel used in the testing
    const std::vector kernels = {baseline, full_strided, full_jump, block_jump, warp_jump,
                                 prefix_sum_unlimited, prefix_sum_32_max,
                                 prefix_sum_max_32_work_efficient,
                                 prefix_sum_we_32_conflict_free,
                                 prefix_sum_we_unlimited_conflict_free,
                                 prefix_sum_warp, prefix_sum_s_warp, prefix_sum_s_warp_jump_block,
                                 prefix_sum_s_warp_jump_block_unroll,
                                 //prefix_sum_warp_2x,
                                 prefix_sum_warp_with_block_jump, prefix_sum_warp_merged};

    // Time definition
    float total_time[kernels.size()];
    double sum_total_time[kernels.size()];
    float gpu_times[kernels.size()];
    double sum_gpu_times[kernels.size()] = {};

    // Init CSV file
    for (u32 i = 0; i < kernels.size(); i++)
        csv << kernels[i].name << (i + 1 != kernels.size() ? ", " : "");
    csv << endl;

    // Execute multiple time
    bool failed = false;
    u32 failed_idx = 0;
    for (cycle = -WARMUP_CYCLES; cycle < CYCLES && !failed; cycle++)
    {
        // Kernels
        for (u32 i = 0; i < kernels.size() && !failed; i++)
        {
            auto t = test_kernel(&kernels[i], matrix, vec, i == 0 ? res_control : res);
            gpu_times[i] = t.first;
            total_time[i] = t.second;

            // print_min_max(res_control, matrix.ROWS);
            if (gpu_times[i] < 0)
            {
                failed = true;
                failed_idx = i;
            }
            else if (i != 0 && cycle >= 0 && CHECK_CORRECT)
                print_diff_info(res, res_control, matrix.ROWS, kernels[i].name);
        }

        // Save times and print csv / partial results
        if (cycle >= 0 && !failed)
        {
            for (u32 i = 0; i < kernels.size(); i++)
            {
                csv << gpu_times[i] << (i + 1 != kernels.size() ? ", " : "");
                if (gpu_times[i] < 0.0 || sum_gpu_times[i] < 0.0)
                {
                    sum_gpu_times[i] = -1;
                    sum_total_time[i] = -1;
                }
                else
                {
                    sum_gpu_times[i] += gpu_times[i];
                    sum_total_time[i] += total_time[i];
                }
            }
            csv << endl;
            cout << "|--> Kernel Time (cycle " << cycle << ") " << endl;
        }
    }


    if (failed)
    {
        cerr << "\nFAILED when running kernel " << kernels[failed_idx].name << "\n" << endl;
        return;
    }

    // Print compact results
    const double avg0 = sum_gpu_times[0] / cycle;
    cout << "|" << endl;
    cout << "|-----> Kernel Time (average): (kernel_id, block_size, avg_time, gflops)" << endl;
    for (u32 i = 0; i < kernels.size(); i++)
    {
        if (sum_gpu_times[i] > 0.0)
        {
            const double avg = sum_gpu_times[i] / cycle;
            const double gflops = 2 * (matrix.NON_ZERO / avg / 1e6);
            const double gbs = 6 * 4 * (matrix.NON_ZERO / avg / 1e6);
            cout << "|---------->[" << kernels[i].name << "]=> ";
            for (u32 j = 0; j < 40 - kernels[i].name.size(); j++)
                cout << " ";
            cout << "[" << avg0 / avg << "x] " << avg << "ms (" << gflops << " Gflops " << gbs << " Gbs)" << endl;
        }
        else if (sum_gpu_times[i] == 0.0)
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
