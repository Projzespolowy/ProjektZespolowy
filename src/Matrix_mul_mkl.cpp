#include <iostream>
#include <limits>
#include <chrono>

#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"
#include <iostream>

using namespace std::chrono;
using namespace std;

int main() {
    int a;

    cout << "Podaj rozmiar: "; 
    cin >> a;

    auto transA = oneapi::mkl::transpose::nontrans;
    auto transB = oneapi::mkl::transpose::nontrans;

    int m = a;  // A = m*k
    int k = a;  // B = k*n
    int n = a;  // C = m*n

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
        cout << "Could not allocate memory for matrices." << endl;
        exit(1);
    }

        // Initialize matrix data.
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            A[i * lda + j] = rand() %100;       
 

    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++)
            B[i * ldb + j] = rand() %100;       


    auto start = high_resolution_clock::now();
    cout << "Launching oneMKL GEMM calculation..." << endl;
    oneapi::mkl::blas::row_major::gemm(device_queue, transA, transB, m, n, k,
                                           alpha, A, lda, B, ldb, beta, C, ldc);
    device_queue.wait_and_throw();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "TIME: " << duration.count() << "ms" << endl;
    free(A, device_queue);
    free(B, device_queue);
    free(C, device_queue);

    return 0;
}