#include <cublas_v2.h>
#include <stdio.h>
#include "utils.h"

void cublas_sgemm(float* a,
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
  // cudaPitchedPtr device_M_tensor;
  // cudaExtent device_M_tensor_extent = make_cudaExtent(
  //     vs.num_tiles * sizeof(float) * us.oc, ti.tile_in_w, ti.tile_in_h);
  // cudaMalloc3D(&device_M_tensor, device_M_tensor_extent);
  const float alpha = 1.f;
  const float beta = 0.f;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;

  auto err=cublasSgemm(handle,
              transa,
              transb,
              n,
              m,
              k,
              &alpha,
              b,
              ldb,
              a,
              lda,
              &beta,
              c,
              ldc);
  cudaDeviceSynchronize();
  cublasDestroy(handle);
  cudaFree(a);
  cudaFree(b);
  // printf("%d", err);
}