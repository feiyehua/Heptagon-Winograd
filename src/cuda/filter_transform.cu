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
  float *__restrict__ device_U;
  float *__restrict__ device_packed_U;
  device_Memory_Pool.poolMalloc((void **)&device_filter, sizeof(float) * collapsed_dim_size * fs.h * fs.w);
  device_Memory_Pool.poolMalloc((void **)&device_U, sizeof(float) * collapsed_dim_size * us.h * us.w);
  device_Memory_Pool.poolMalloc((void **)&device_packed_U, sizeof(float) * collapsed_dim_size * us.h * us.w);
  cudaMemcpy(
      device_filter, packed_filter, sizeof(float) * collapsed_dim_size * fs.h * fs.w, cudaMemcpyHostToDevice);
  thread_filter_transform<<<us.oc, us.ic>>>(
      device_filter, device_U, device_packed_U, fs, us, collapsed_dim_size);
  // cudaDeviceSynchronize();
  *device_U_tensor = device_packed_U;
  *ldu = us.ic;
}

// __device__ void device_filter_packing(float *__restrict__ filter_tensor,
//                                       float *__restrict__ packed_filter_tensor,
//                                       const U_shape_t fs) {
//   // typedef float(*filter_tensor_t)[fs.ic][fs.h][fs.w];
//   // typedef float(*packed_filter_tensor_t)[fs.w][fs.oc][fs.ic];
//   // filter_tensor_t filter_tensor = (filter_tensor_t)filter;
//   // packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;

//   // get packed filter frome filter tensor
//   for (int64_t h = 0; h < fs.h; ++h)
//     for (int64_t w = 0; w < fs.w; ++w)
// #pragma unroll
//       for (int64_t oc = 0; oc < fs.oc; oc++)
// #pragma unroll
//         for (int64_t ic = 0; ic < fs.ic; ic++)
//           packed_filter_tensor[h * fs.w * fs.oc * fs.ic + w * fs.oc * fs.ic + oc * fs.ic +
//                                ic] = filter_tensor[oc * fs.ic * fs.h * fs.w + ic * fs.h * fs.w + h * fs.w +
//                                                    w];
// }

__global__ void thread_filter_transform(float *__restrict__ packed_filter,
                                        float *__restrict__ U,
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

    U[idx * us.h * us.w + 0 * us.w + w] = z0;
    U[idx * us.h * us.w + 1 * us.w + w] = z1;
    U[idx * us.h * us.w + 2 * us.w + w] = z2;
    U[idx * us.h * us.w + 3 * us.w + w] = z3;
    U[idx * us.h * us.w + 4 * us.w + w] = z4;
    U[idx * us.h * us.w + 5 * us.w + w] = z5;
  }
#pragma unroll
  // us.h=6
  for (int64_t h = 0; h < us.h; ++h) {
    z6 = U[idx * us.h * us.w + h * us.w + 0];

    z0 = (1.0f / 4.0f) * z6;
    z1 = (-1.0f / 6.0f) * z6;
    z2 = (-1.0f / 6.0f) * z6;
    z3 = (1.0f / 24.0f) * z6;
    z4 = (1.0f / 24.0f) * z6;

    z6 = U[idx * us.h * us.w + h * us.w + 1];

    z1 += (-1.0f / 6.0f) * z6;
    z2 += (1.0f / 6.0f) * z6;
    z3 += (1.0f / 12.0f) * z6;
    z4 += (-1.0f / 12.0f) * z6;

    z6 = U[idx * us.h * us.w + h * us.w + 2];

    z1 += (-1.0f / 6.0f) * z6;
    z2 += (-1.0f / 6.0f) * z6;
    z3 += (1.0f / 6.0f) * z6;
    z4 += (1.0f / 6.0f) * z6;
    z5 = z6;

    U[idx * us.h * us.w + h * us.w + 0] = z0;
    packed_U[h * us.w * collapsed_dim_size + 0 * collapsed_dim_size + blockIdx.x * us.ic + threadIdx.x] = z0;
    U[idx * us.h * us.w + h * us.w + 1] = z1;
    packed_U[h * us.w * collapsed_dim_size + 1 * collapsed_dim_size + blockIdx.x * us.ic + threadIdx.x] = z1;
    U[idx * us.h * us.w + h * us.w + 2] = z2;
    packed_U[h * us.w * collapsed_dim_size + 2 * collapsed_dim_size + blockIdx.x * us.ic + threadIdx.x] = z2;
    U[idx * us.h * us.w + h * us.w + 3] = z3;
    packed_U[h * us.w * collapsed_dim_size + 3 * collapsed_dim_size + blockIdx.x * us.ic + threadIdx.x] = z3;
    U[idx * us.h * us.w + h * us.w + 4] = z4;
    packed_U[h * us.w * collapsed_dim_size + 4 * collapsed_dim_size + blockIdx.x * us.ic + threadIdx.x] = z4;
    U[idx * us.h * us.w + h * us.w + 5] = z5;
    packed_U[h * us.w * collapsed_dim_size + 5 * collapsed_dim_size + blockIdx.x * us.ic + threadIdx.x] = z5;
  }
  // }
}