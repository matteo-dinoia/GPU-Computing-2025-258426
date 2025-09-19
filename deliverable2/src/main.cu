#include <chrono>
#include <iostream>
#include <memory>
#include <sys/time.h>
#include "../distributed_mmio/include/mmio.h"
#include "include/tester.h"
#include "include/time_utils.h"
#include "include/type_alias.h"
#include "include/utils.h"

using std::cout, std::endl;

struct CooDeleter
{
    void operator()(COO_local<MI, MV>* x) const { Distr_MMIO_COO_local_destroy(&x); }
};



template <typename T>
std::unique_ptr<T, CudaDeleter> newCudaMemory(const size_t lenght){
    auto  myCudaMalloc = [](const size_t sizeBytes) { void* ptr; cudaMallocManaged(&ptr, sizeBytes); return ptr; };
    return std::unique_ptr<T, CudaDeleter>(static_cast<T*>(myCudaMalloc(lenght * sizeof(T))));
}

int timed_main(const char*, const char*);


int main(const int argc, char** argv)
{
    int ret = 1;
    srand(std::chrono::system_clock::now().time_since_epoch().count());
    TIMER_DEF(1);

    TIMER_START(1);
    if (argc >= 3)
        ret = timed_main(argv[1], argv[2]);
    else if (argc >= 2)
        ret = timed_main(argv[1], "/dev/null");
    else
        cout << "FATAL: require at least datasets filename argument" << endl;
    TIMER_STOP(1);

    cout << "TOTAL PROGRAM TIME: " << TIMER_ELAPSED_MS(1) << "ms" << endl;
    return ret;
}

// Matrix is inverted so that is ordered by row
int timed_main(const char* input_filename, const char* output_csv_filename)
{
    cout << "\n* Started" << endl;
    TIMER_DEF(0);
    TIMER_DEF(1);
    TIMER_DEF(2);
    TIMER_DEF(3);
    TIMER_DEF(4);
    TIMER_DEF(5);

    // Data allocation


    // Reading matrix data
    TIMER_START(0);
    const auto coo = std::unique_ptr<COO_local<MI, MV>, CooDeleter>(Distr_MMIO_sorted_COO_local_read<MI, MV>(input_filename, false));
    if (coo == nullptr)
    {
        printf("Failed to import graph from file [%s]\n", input_filename);
        return -1;
    }
    TIMER_STOP(0);
    cout << "* Read data (" << coo->nrows << " " << coo->ncols << " " << coo->nnz << ")" << endl;

    // Alloc memory
    TIMER_START(1);
    const GpuCoo<MI, MV> matrix = {coo->nnz, coo->ncols, coo->nrows,
        newCudaMemory<MI>(coo->nnz), newCudaMemory<MI>(coo->nnz), newCudaMemory<MV>(coo->nnz)};


    // Alloc memory for other part
    const auto vec = newCudaMemory<MV>(matrix.COLS);
    const auto res = newCudaMemory<MV>(matrix.ROWS);
    const auto res_control = newCudaMemory<MV>(matrix.ROWS);
    TIMER_STOP(1);
    cout << "* Allocated  memory" << endl;

    // Copy data to GPU
    TIMER_START(2);
    cudaMemcpy(matrix.xs.get(), coo->col, matrix.NON_ZERO * sizeof(MI), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.ys.get(), coo->row, matrix.NON_ZERO * sizeof(MI), cudaMemcpyHostToDevice);
    if (coo->val != nullptr)
        cudaMemcpy(matrix.vals.get(), coo->val, matrix.NON_ZERO * sizeof(MV), cudaMemcpyHostToDevice);
    else
        cudaMemset(matrix.vals.get(), 1, matrix.NON_ZERO * sizeof(MV));
    TIMER_STOP(2);
    cout << "* Copied COO to GPU memory" << endl;

    // Check errors
    if (cudaGetLastError() != cudaSuccess)
    {
        cout << "FATAL: could not initialize memory (possibly because no GPU)" << endl;
        return -1;
    }

    // Generation of random vector
    TIMER_START(3);
    randomize_dense_vec(vec.get(), matrix.COLS);
    TIMER_STOP(3);
    print_min_max(vec.get(), matrix.COLS);
    cout << "* Randomized Vector" << endl;

    // Execution
    cout << "* Starting with nz=" << matrix.NON_ZERO << " cols=" << matrix.COLS << " rows=" << matrix.ROWS << "\n"
        << endl;
    TIMER_START(4);
    execution(matrix, vec.get(), res.get(), res_control.get(), output_csv_filename);
    TIMER_STOP(4);
    cout << "* Terminated execution" << endl;

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
