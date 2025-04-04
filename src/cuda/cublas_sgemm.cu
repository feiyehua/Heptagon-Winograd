#include <cublas_v2.h>
#include <stdio.h>

#include "utils.h"

void cublas_sgemm(cublasHandle_t handle,
                  float* a,
                  int lda,
                  long long int strideA,
                  float* b,
                  int ldb,
                  long long int strideB,
                  float* c,
                  int ldc,
                  long long int strideC,
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

  cublasSgemmStridedBatched(handle,
                            transa,
                            transb,
                            n,
                            m,
                            k,
                            &alpha,
                            b,
                            ldb,
                            strideB,
                            a,
                            lda,
                            strideA,
                            &beta,
                            c,
                            ldc,
                            strideC,
                            ti.tile_in_h * ti.tile_in_w);
}