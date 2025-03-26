#include <cuda_runtime.h>
#include <stdio.h>

#include "utils.h"
// Call:
// sgemm<flat,32><<<(DIV_UP(m, 32),DIV_UP(n,32)), (32, 32)>>>(A, B, C, m, n, k);
// Assuming tensor B is transposed
template <typename T, int STRIDE>
__global__ void sgemm(T* A, T* B, T* C, size_t m, size_t n, size_t k) {
  // const int STRIDE = 32;
  // In every block we have 32*32=1024 threads
  // They share the same block of memory
  __shared__ T shared_A[STRIDE][STRIDE];
  __shared__ T shared_B[STRIDE][STRIDE];
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t tx = threadIdx.x;
  int64_t ty = threadIdx.y;
  int64_t A_y = k;
  int64_t B_y = k;
  int64_t C_y = n;
  T sum = 0;
  for (int j = 0; j < k / STRIDE; j++) {
    shared_A[tx][ty] = x<m?A[x * A_y + j * STRIDE + ty]:0;
    shared_B[ty][tx] = y<n?B[y * B_y + j * STRIDE + tx]:0;
    __syncthreads();
    for (int i = 0; i < STRIDE; i++) {
      sum += shared_A[tx][i] * shared_B[ty][i];
    }
    __syncthreads();
  }
  if (x<m&&ROUND(k, STRIDE) + ty < k) shared_A[tx][ty] = A[x * A_y + ROUND(k, STRIDE) + ty];
  if (y<n&&ROUND(k, STRIDE) + tx < k) shared_B[ty][tx] = B[y * B_y + ROUND(k, STRIDE) + tx];
  __syncthreads();
  for (int i = 0; i < STRIDE && ROUND(k, STRIDE) + i < k; i++) {
    sum += shared_A[tx][i] * shared_B[ty][i];
  }
  if((x<m&&y<n)) C[x * C_y + y]=sum;
}