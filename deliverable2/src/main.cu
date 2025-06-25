#include <iostream>
#include <fstream>
#include <unistd.h>

#include <sys/time.h>
#include "tester.h"
#include "include/time_utils.h"
#include "include/utils.h"
#include "../distributed_mmio/include/mmio.h"

#define eprintf(...) fprintf (stderr, __VA_ARGS__)

using std::cout, std::endl;

int timed_main(const char*);


int main(const int argc, char** argv)
{
    int ret = 1;
    TIMER_DEF(1);
    srand(time(nullptr));

    TIMER_START(1);
    if (argc >= 2)
        ret = timed_main(argv[1]);
    else
        cout << "FATAL: require filename argument" << endl;
    TIMER_STOP(1);

    cout << "TOTAL PROGRAM TIME: " << TIMER_ELAPSED_MS(1) << "ms" << endl;
    return ret;
}


int timed_main(const char* input_file)
{
    TIMER_DEF(0);
    TIMER_DEF(1);
    TIMER_DEF(2);
    TIMER_DEF(3);
    TIMER_DEF(4);

    // Data allocation
    GpuCoo<uint32_t, float> matrix = {0, 0, 0, nullptr, nullptr, nullptr};
    float* vec = nullptr;
    float* res = nullptr;
    float* res_control = nullptr;

    // Reading matrix data
    TIMER_START(0);
    const COO_local<uint32_t, float>* coo = Distr_MMIO_COO_local_read<uint32_t, float>(input_file);
    if (coo == nullptr)
    {
        printf("Failed to import graph from file [%s]\n", input_file);
        return -1;
    }
    TIMER_STOP(0);

    // Alloc memory
    TIMER_START(1);
    matrix.NON_ZERO = coo->nnz;
    matrix.COLS = coo->ncols;
    matrix.ROWS = coo->nrows;
    cudaMallocManaged(&matrix.xs, matrix.NON_ZERO * sizeof(int));
    cudaMallocManaged(&matrix.ys, matrix.NON_ZERO * sizeof(int));
    cudaMallocManaged(&matrix.vals, matrix.NON_ZERO * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(matrix.xs, coo->col, matrix.NON_ZERO * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.ys, coo->row, matrix.NON_ZERO * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.vals, coo->val, matrix.NON_ZERO * sizeof(float), cudaMemcpyHostToDevice);

    // Alloc memory for other part
    cudaMallocManaged(&vec, matrix.COLS * sizeof(float));
    cudaMallocManaged(&res, matrix.ROWS * sizeof(float));
    cudaMallocManaged(&res_control, matrix.ROWS * sizeof(float));
    TIMER_STOP(1);

    // Generation of random vector
    TIMER_START(2);
    randomize_dense_vec(vec, matrix.ROWS);
    TIMER_STOP(2);

    // Execution
    TIMER_START(3);
    execution(matrix, vec, res, res_control);
    TIMER_STOP(3);

    // Free memory and close resources
    TIMER_START(4);
    cudaFree(matrix.xs);
    cudaFree(matrix.ys);
    cudaFree(matrix.vals);
    cudaFree(vec);
    cudaFree(res_control);
    cudaFree(res);
    TIMER_STOP(4);

    // Print time
    cout << "Time elapsed for reading: " << TIMER_ELAPSED_MS(0) << " ms" << endl;
    cout << "Time elapsed for allocation and copy: " << TIMER_ELAPSED_MS(1) << " ms" << endl;
    cout << "Time elapsed for vector generation: " << TIMER_ELAPSED_MS(2) << " ms" << endl;
    cout << "Time elapsed for full tester: " << TIMER_ELAPSED_MS(3) << " ms" << endl;
    cout << "Time elapsed for free : " << TIMER_ELAPSED_MS(4) << " ms" << endl;
    return 0;
}
