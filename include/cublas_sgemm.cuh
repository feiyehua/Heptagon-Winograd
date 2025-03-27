#ifndef __CUBLAS_SGEMM_CUH__
#define __CUBLAS_SGEMM_CUH__
#include <cublas_v2.h>

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
                  tiling_info_t ti);
#endif