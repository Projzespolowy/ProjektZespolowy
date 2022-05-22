#include <iostream>
#include <limits>
#include <chrono>
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"
#include <iostream>

using namespace std::chrono;
using namespace std;

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

    sycl::queue cpu_device_queue{sycl::cpu_selector{}};
    sycl::queue gpu_device_queue{sycl::gpu_selector{}};

    auto A_CPU = sycl::malloc_shared<double>(m * k, cpu_device_queue);
    auto B_CPU = sycl::malloc_shared<double>(k * n, cpu_device_queue);
    auto C_CPU = sycl::malloc_shared<double>(m * n, cpu_device_queue);

    auto A_GPU = sycl::malloc_shared<double>(m * k, gpu_device_queue);
    auto B_GPU = sycl::malloc_shared<double>(k * n, gpu_device_queue);
    auto C_GPU = sycl::malloc_shared<double>(m * n, gpu_device_queue);

    if (!A_CPU || !B_CPU || !C_CPU || !A_GPU || !B_GPU || !C_GPU) {
        std::cerr << "Could not allocate memory for matrices." << std::endl;
        exit(1);
    }

        // Initialize matrix data.
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++) {
            A_CPU[i * lda + j] = 1;
            A_GPU[i * lda + j] = 1;        // tu wartość 1
        }

    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++) {
            B_CPU[i * ldb + j] = 1;
            B_GPU[i * ldb + j] = 1;        // tu wartość 2
        }


    std::cerr << "Launching oneMKL CPU GEMM calculations..." << std::endl;
    auto cpu_start = high_resolution_clock::now();
    oneapi::mkl::blas::row_major::gemm(cpu_device_queue, transA, transB, m, n, k,
                                           alpha, A_CPU, lda, B_CPU, ldb, beta, C_CPU, ldc);

        // While calculation occurs, compute reference result to check accuracy.
    //    std::cerr << "Performing reference calculation..." << std::endl;
  //  for (int i = 0; i < m; i++)
      //  for (int h = 0; h < k; h++)
          //  for (int j = 0; j < n; j++)
             //   C_reference[i * ldc + j] += A[i * lda + h] * B[h * ldb + j];
        
    cpu_device_queue.wait_and_throw();
    auto cpu_end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<microseconds>(cpu_end - cpu_start);
    cout <<"CPU TIME: "<<cpu_duration.count() << std::endl;
    free(C_CPU, cpu_device_queue);
    free(A_CPU, cpu_device_queue);
    free(B_CPU, cpu_device_queue);

    std::cerr << "Launching oneMKL GPU GEMM calculations..." << std::endl;
    auto gpu_start = high_resolution_clock::now();
    oneapi::mkl::blas::row_major::gemm(gpu_device_queue, transA, transB, m, n, k,
                                           alpha, A_GPU, lda, B_GPU, ldb, beta, C_GPU, ldc);

        // While calculation occurs, compute reference result to check accuracy.
    //    std::cerr << "Performing reference calculation..." << std::endl;
  //  for (int i = 0; i < m; i++)
      //  for (int h = 0; h < k; h++)
          //  for (int j = 0; j < n; j++)
             //   C_reference[i * ldc + j] += A[i * lda + h] * B[h * ldb + j];
        
    gpu_device_queue.wait_and_throw();
    auto gpu_end = high_resolution_clock::now();
    auto gpu_duration = duration_cast<microseconds>(gpu_end - gpu_start);
    cout <<"GPU TIME: "<<gpu_duration.count() << std::endl;
    free(C_GPU, gpu_device_queue);
    free(A_GPU, gpu_device_queue);
    free(B_GPU, gpu_device_queue);

    return 0;
}