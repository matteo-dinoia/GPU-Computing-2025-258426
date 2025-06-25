#include <iostream>
#include <cmath>
#include <strings.h>

#include "include/gpu.h"
#include "include/time_utils.h"
#include "include/tester.h"
#include "include/utils.h"

using std::cout, std::endl;

#define MAX_THREAD_PER_BLOCK 1024u
#define MAX_BLOCK 256u

#define CYCLES 1
#define WARMUP_CYCLES 0

void execution(const GpuCoo<uint32_t, float>& matrix, const float* vec, float* res, float* res_control)
{
    GPU_TIMER_DEF();

    // Time definition
    constexpr int N_GPU_KERNEL = 5;
    constexpr kernel_func kernels[N_GPU_KERNEL] = {
        spmv_baseline, spmv_full_strided, spmv_full_jump, spmv_block_jump, spmv_warp_jump
    };
    float gpu_times[N_GPU_KERNEL] = {};
    double sum_times[N_GPU_KERNEL] = {};

    // Execute multiple time
    uint32_t n_blocks = std::min(MAX_BLOCK, CEIL_DIV(matrix.NON_ZERO, MAX_THREAD_PER_BLOCK));
    uint32_t n_thread_per_block = std::min<uint32_t>(MAX_THREAD_PER_BLOCK, matrix.NON_ZERO);
    cout << "# Starting with <<<" << n_blocks << ", " << n_thread_per_block << ">>>\n"
        << endl;

    bzero(sum_times, N_GPU_KERNEL * sizeof(*sum_times));

    int cycle;
    for (cycle = -WARMUP_CYCLES; cycle < CYCLES; cycle++)
    {
        // BASELINE KERNEL
        bzero(res_control, matrix.ROWS * sizeof(float));
        int tmp_bl = CEIL_DIV(matrix.NON_ZERO, n_thread_per_block);
        GPU_TIMER_START();
        kernels[0]<<<tmp_bl, n_thread_per_block>>
            >(matrix.xs, matrix.ys, matrix.vals, vec, res_control, matrix.NON_ZERO);
        GPU_TIMER_STOP(&gpu_times[0]);

        // OTHER KERNELS
        for (int i = 1; i < N_GPU_KERNEL; i++)
        {
            bzero(res, matrix.ROWS * sizeof(float));
            GPU_TIMER_START();
            kernels[i]<<<n_blocks, n_thread_per_block>>>(matrix.xs, matrix.ys, matrix.vals, vec, res, matrix.NON_ZERO);
            GPU_TIMER_STOP(&gpu_times[i]);

            cout << "ERROR of gpu " << i << " are " << diff_size(res, res_control, matrix.ROWS) << endl;
        }

        // Save times
        if (cycle >= 0)
        {
            cout << "|--> Kernel Time (id " << cycle << "): ";
            for (int i = 0; i < N_GPU_KERNEL; i++)
            {
                sum_times[i] += gpu_times[i];
                cout << "[id=" << i << "]=> " << gpu_times[i] << "ms ";
            }
            cout << endl;
        }
    }

    cout << "|" << endl;
    cout << "|-----> Kernel Time (average): (kernel_id, block_size, avg_time, gflops)" << endl;
    for (int i = 0; i < N_GPU_KERNEL; i++)
    {
        const double avg = sum_times[i] / cycle;
        const double gflops = 2 * (matrix.NON_ZERO / avg / 1e6);
        const double gbs = 6 * 4 * (matrix.NON_ZERO / avg / 1e6);
        cout << "|---------->[ id =" << i << " ]=> " << avg << "ms (" << gflops << " Gflops " << gbs << " Gbs)" << endl;
        //with # error " << gpu_errors[i] << endl;
    }
    cout << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
