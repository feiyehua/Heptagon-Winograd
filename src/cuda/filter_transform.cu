#include "filter_transform.cuh"

__host__ void device_filter_transform(
    // collapsed_dim_size = us.oc * us.ic = fs.oc * fs.ic
    float *__restrict__ packed_filter,  //  packed_filter collapsed_dim_size*fs.h*fs.w
                                        // collapsed_dim_size*us.h*us.w
    const filter_shape_t fs,
    const U_shape_t us,
    const int64_t collapsed_dim_size,
    float **device_U_tensor,
    int *ldu,
    Device_Memory_Pool &device_Memory_Pool) {
  float *__restrict__ device_filter;
  float *__restrict__ device_packed_U;
  device_Memory_Pool.poolMalloc((void **)&device_filter, sizeof(float) * collapsed_dim_size * fs.h * fs.w);
  device_Memory_Pool.poolMalloc((void **)&device_packed_U, sizeof(float) * collapsed_dim_size * us.h * us.w);
  cudaMemcpy(
      device_filter, packed_filter, sizeof(float) * collapsed_dim_size * fs.h * fs.w, cudaMemcpyHostToDevice);
  thread_filter_transform<<<us.oc, us.ic>>>(
      device_filter, device_packed_U, fs, us, collapsed_dim_size);

  *device_U_tensor = device_packed_U;
  *ldu = us.ic;
}


__global__ void thread_filter_transform(float *__restrict__ packed_filter,
                                        // float *__restrict__ U,
                                        float *__restrict__ packed_U,
                                        const filter_shape_t fs,
                                        const U_shape_t us,
                                        const int64_t collapsed_dim_size) {
  //   typedef float(*packed_filter_tensor_t)[fs.h][fs.w];
  //   typedef float(*U_tensor_t)[us.h][us.w];
  //   packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;
  //   U_tensor_t U_tensor = (U_tensor_t)U;

  float z0, z1, z2, z3, z4, z5, z6;

  /*
G =
⎡1/4     0     0  ⎤
⎢                 ⎥
⎢-1/6  -1/6   -1/6⎥
⎢                 ⎥
⎢-1/6   1/6   -1/6⎥
⎢                 ⎥
⎢1/24  1/12   1/6 ⎥
⎢                 ⎥
⎢1/24  -1/12  1/6 ⎥
⎢                 ⎥
⎣ 0      0     1  ⎦
*/
  // for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
// parallel computation for each id
  float U[TILE_IN_H][TILE_IN_W];
#pragma unroll
  // fs.w=3
  for (int64_t w = 0; w < fs.w; ++w) {
    // non-sequential memory access
    // rewrite for better memory access performance
    z6 = packed_filter[idx * fs.h * fs.w + 0 * fs.w + w];

    z0 = (1.0f / 4.0f) * z6;
    z1 = (-1.0f / 6.0f) * z6;
    z2 = (-1.0f / 6.0f) * z6;
    z3 = (1.0f / 24.0f) * z6;
    z4 = (1.0f / 24.0f) * z6;

    z6 = packed_filter[idx * fs.h * fs.w + 1 * fs.w + w];

    z1 += (-1.0f / 6.0f) * z6;
    z2 += (1.0f / 6.0f) * z6;
    z3 += (1.0f / 12.0f) * z6;
    z4 += (-1.0f / 12.0f) * z6;

    z6 = packed_filter[idx * fs.h * fs.w + 2 * fs.w + w];

    z1 += (-1.0f / 6.0f) * z6;
    z2 += (-1.0f / 6.0f) * z6;
    z3 += (1.0f / 6.0f) * z6;
    z4 += (1.0f / 6.0f) * z6;
    z5 = z6;

    U[0][w] = z0;
    U[1][w] = z1;
    U[2][w] = z2;
    U[3][w] = z3;
    U[4][w] = z4;
    U[5][w] = z5;
  }
#pragma unroll
  // us.h=6
  for (int64_t h = 0; h < us.h; ++h) {
    z6 = U[h][0];

    z0 = (1.0f / 4.0f) * z6;
    z1 = (-1.0f / 6.0f) * z6;
    z2 = (-1.0f / 6.0f) * z6;
    z3 = (1.0f / 24.0f) * z6;
    z4 = (1.0f / 24.0f) * z6;

    z6 = U[h][1];

    z1 += (-1.0f / 6.0f) * z6;
    z2 += (1.0f / 6.0f) * z6;
    z3 += (1.0f / 12.0f) * z6;
    z4 += (-1.0f / 12.0f) * z6;

    z6 = U[h][2];

    z1 += (-1.0f / 6.0f) * z6;
    z2 += (-1.0f / 6.0f) * z6;
    z3 += (1.0f / 6.0f) * z6;
    z4 += (1.0f / 6.0f) * z6;
    z5 = z6;

    packed_U[h * us.w * collapsed_dim_size + 0 * collapsed_dim_size + blockIdx.x * us.ic + threadIdx.x] = z0;
    packed_U[h * us.w * collapsed_dim_size + 1 * collapsed_dim_size + blockIdx.x * us.ic + threadIdx.x] = z1;
    packed_U[h * us.w * collapsed_dim_size + 2 * collapsed_dim_size + blockIdx.x * us.ic + threadIdx.x] = z2;
    packed_U[h * us.w * collapsed_dim_size + 3 * collapsed_dim_size + blockIdx.x * us.ic + threadIdx.x] = z3;
    packed_U[h * us.w * collapsed_dim_size + 4 * collapsed_dim_size + blockIdx.x * us.ic + threadIdx.x] = z4;
    packed_U[h * us.w * collapsed_dim_size + 5 * collapsed_dim_size + blockIdx.x * us.ic + threadIdx.x] = z5;
  }
  // }
}