#include <cublas_v2.h>
#include <stdio.h>

#include "utils.h"

void cublas_sgemm(cublasHandle_t handle,
                  float* a,
                  int lda,
                  float* b,
                  int ldb,
                  float* c,
                  int ldc,
                  int m,
                  int n,
                  int k,
                  U_shape_t us,
                  V_shape_t vs,
                  tiling_info_t ti) {
  const float alpha = 1.f;
  const float beta = 0.f;
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;

  auto err = cublasSgemm(handle, transa, transb, n, m, k, &alpha, b, ldb, a, lda, &beta, c, ldc);
  cudaDeviceSynchronize();
}