#include <iostream>
#include <limits>
#include <chrono>
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"
#include <iostream>

using namespace std::chrono;
using namespace std;

auto start = high_resolution_clock::now();

int main() {

    auto transA = oneapi::mkl::transpose::nontrans;
    auto transB = oneapi::mkl::transpose::nontrans;

    int m = 100;  // A = m*k
    int k = 100;  // B = k*n
    int n = 100;  // C = m*n

    int lda = k;
    int ldb = n;
    int ldc = n;

    double alpha = 1.0;
    double beta = 0.0;

    sycl::queue device_queue{sycl::cpu_selector{}};

    auto A = sycl::malloc_shared<double>(m * k, device_queue);
    auto B = sycl::malloc_shared<double>(k * n, device_queue);
    auto C = sycl::malloc_shared<double>(m * n, device_queue);

    if (!A || !B || !C) {
        std::cerr << "Could not allocate memory for matrices." << std::endl;
        exit(1);
    }

        // Initialize matrix data.
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            A[i * lda + j] = 1;        // tu wartość 1

    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++)
            B[i * ldb + j] = 1;        // tu wartość 2

    std::cerr << "Launching oneMKL GEMM calculation..." << std::endl;
    oneapi::mkl::blas::row_major::gemm(device_queue, transA, transB, m, n, k,
                                           alpha, A, lda, B, ldb, beta, C, ldc);

        // While calculation occurs, compute reference result to check accuracy.
    //    std::cerr << "Performing reference calculation..." << std::endl;
  //  for (int i = 0; i < m; i++)
      //  for (int h = 0; h < k; h++)
          //  for (int j = 0; j < n; j++)
             //   C_reference[i * ldc + j] += A[i * lda + h] * B[h * ldb + j];
        
    device_queue.wait_and_throw();

    free(A, device_queue);
    free(B, device_queue);
    free(C, device_queue);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout<<"TIME: "<<duration.count();

    return 0;
}