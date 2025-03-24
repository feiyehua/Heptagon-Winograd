#include<cuda_runtime.h>
#include<stdio.h>

// Assuming tensor B is transposed
template<typename T>
__global__ void sgemm(T* A,T* B,T* C,size_t m,size_t n,size_t k)
{
  const int STRIDE = 32;
  // In every block we have 32*32=1024 threads
  // They share the same block of memory
  __shared__ T shared_A[STRIDE][STRIDE];
  __shared__ T shared_B[STRIDE][STRIDE];

}