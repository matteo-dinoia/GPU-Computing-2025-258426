#include <chrono>
#include <iostream>
#include <sys/time.h>
#include "../distributed_mmio/include/mmio.h"
#include "include/tester.h"
#include "include/time_utils.h"
#include "include/type_alias.h"
#include "include/utils.h"

using std::cout, std::endl;

int timed_main(const char*);


int main(const int argc, char** argv)
{
    int ret = 1;
    srand(std::chrono::system_clock::now().time_since_epoch().count());
    TIMER_DEF(1);

    TIMER_START(1);
    if (argc >= 2)
        ret = timed_main(argv[1]);
    else
        cout << "FATAL: require filename argument" << endl;
    TIMER_STOP(1);

    cout << "TOTAL PROGRAM TIME: " << TIMER_ELAPSED_MS(1) << "ms" << endl;
    return ret;
}

// Matrix is inverted so that is ordered by row
int timed_main(const char* input_file)
{
    cout << "\n* Started" << endl;
    TIMER_DEF(0);
    TIMER_DEF(1);
    TIMER_DEF(2);
    TIMER_DEF(3);
    TIMER_DEF(4);
    TIMER_DEF(5);

    // Data allocation
    GpuCoo<MI, MV> matrix = {0, 0, 0, nullptr, nullptr, nullptr};
    MV* vec = nullptr;
    MV* res = nullptr;
    MV* res_control = nullptr;

    // Reading matrix data
    TIMER_START(0);
    const COO_local<MI, MV>* coo = Distr_MMIO_COO_local_read<MI, MV>(input_file);
    if (coo == nullptr)
    {
        printf("Failed to import graph from file [%s]\n", input_file);
        return -1;
    }
    TIMER_STOP(0);
    cout << "* Read data (" << coo->nrows << " " << coo->ncols << " " << coo->nnz << ")" << endl;

    // Alloc memory
    TIMER_START(1);
    matrix.NON_ZERO = coo->nnz;
    matrix.COLS = coo->nrows;
    matrix.ROWS = coo->ncols;
    cudaMallocManaged(&matrix.xs, matrix.NON_ZERO * sizeof(MI));
    cudaMallocManaged(&matrix.ys, matrix.NON_ZERO * sizeof(MI));
    cudaMallocManaged(&matrix.vals, matrix.NON_ZERO * sizeof(MV));

    // Alloc memory for other part
    cudaMallocManaged(&vec, matrix.COLS * sizeof(MV));
    cudaMallocManaged(&res, matrix.ROWS * sizeof(MV));
    cudaMallocManaged(&res_control, matrix.ROWS * sizeof(MV));
    TIMER_STOP(1);
    cout << "* Allocated  memory" << endl;

    // Copy data to GPU
    TIMER_START(2);
    cudaMemcpy(matrix.xs, coo->row, matrix.NON_ZERO * sizeof(MI), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.ys, coo->col, matrix.NON_ZERO * sizeof(MI), cudaMemcpyHostToDevice);
    if (coo->val != nullptr)
        cudaMemcpy(matrix.vals, coo->val, matrix.NON_ZERO * sizeof(MV), cudaMemcpyHostToDevice);
    else
        cudaMemset(matrix.vals, 1, matrix.NON_ZERO * sizeof(MV));
    TIMER_STOP(2);
    cout << "* Copied COO to GPU memory" << endl;

    // Generation of random vector
    TIMER_START(3);
    randomize_dense_vec(vec, matrix.COLS);
    TIMER_STOP(3);
    print_min_max(vec, matrix.COLS);
    cout << "* Randomized Vector" << endl;

    // Execution
    cout << "* Starting with nz=" << matrix.NON_ZERO << " cols=" << matrix.COLS << " rows=" << matrix.ROWS << "\n"
        << endl;
    TIMER_START(4);
    execution(matrix, vec, res, res_control);
    TIMER_STOP(4);
    cout << "* Terminated execution" << endl;

    // Free memory and close resources
    TIMER_START(5);
    cudaFree(matrix.xs);
    cudaFree(matrix.ys);
    cudaFree(matrix.vals);
    cudaFree(vec);
    cudaFree(res_control);
    cudaFree(res);
    TIMER_STOP(5);
    cout << "* Finished Deallocating\n" << endl;

    // Print time
    cout << "Time elapsed for reading: " << TIMER_ELAPSED_MS(0) << " ms" << endl;
    cout << "Time elapsed for allocation: " << TIMER_ELAPSED_MS(1) << " ms" << endl;
    cout << "Time elapsed for allocation: " << TIMER_ELAPSED_MS(1) << " ms" << endl;
    cout << "Time elapsed for copy: " << TIMER_ELAPSED_MS(2) << " ms" << endl;
    cout << "Time elapsed for vector generation: " << TIMER_ELAPSED_MS(3) << " ms" << endl;
    cout << "Time elapsed for full tester: " << TIMER_ELAPSED_MS(4) << " ms" << endl;
    cout << "Time elapsed for free : " << TIMER_ELAPSED_MS(5) << " ms" << endl;
    return 0;
}
