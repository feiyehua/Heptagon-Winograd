#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

#include "cublas_sgemm.cuh"
#include "device_memory_pool.h"
#include "filter_transform.cuh"
#include "image_transform.cuh"
#include "output_transform.cuh"
#include "utils.h"

void cudaHostMalloc(float **ptr, size_t size, unsigned int flags) { cudaHostAlloc(ptr, size, flags); }

// get V tensor = BT*d*B
void image_transform(float *__restrict__ packed_image,
                     float *__restrict__ V,
                     const V_shape_t vs,
                     const tiling_info_t ti,
                     const int64_t collapsed_dim_size) {
  // collapsed_dim_size = vs.ic * vs.num_tiles
  // collapsed the tensor for better performance?
  // should transform first, then pack
  typedef float(*packed_image_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*V_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  packed_image_tensor_t packed_image_tensor = (packed_image_tensor_t)packed_image;
  V_tensor_t V_tensor = (V_tensor_t)V;

  float z0, z1, z2, z3, z4, z5, z6;

  /*
  BT =
⎡4  0   -5  0   1  0⎤
⎢                   ⎥
⎢0  -4  -4  1   1  0⎥
⎢                   ⎥
⎢0  4   -4  -1  1  0⎥
⎢                   ⎥
⎢0  -2  -1  2   1  0⎥
⎢                   ⎥
⎢0  2   -1  -2  1  0⎥
⎢                   ⎥
⎣0  4   0   -5  0  1⎦
  */
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    // ti.tile_in_w = 6
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      z6 = packed_image_tensor[0][w][idx];

      z0 = 4.0f * z6;

      z6 = packed_image_tensor[1][w][idx];

      z1 = -4.0f * z6;
      z2 = 4.0f * z6;
      z3 = -2.0f * z6;
      z4 = 2.0f * z6;
      z5 = 4.0f * z6;

      z6 = packed_image_tensor[2][w][idx];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = packed_image_tensor[3][w][idx];

      z1 += z6;
      z2 += -z6;
      z3 += 2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = packed_image_tensor[4][w][idx];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = packed_image_tensor[5][w][idx];

      z5 += z6;

      V_tensor[0][w][idx] = z0;
      V_tensor[1][w][idx] = z1;
      V_tensor[2][w][idx] = z2;
      V_tensor[3][w][idx] = z3;
      V_tensor[4][w][idx] = z4;
      V_tensor[5][w][idx] = z5;
    }
    // ti.tile_in_h = 6
    for (int64_t h = 0; h < ti.tile_in_h; ++h) {
      z6 = V_tensor[h][0][idx];

      z0 = 4.0f * z6;

      z6 = V_tensor[h][1][idx];

      z1 = -4.0f * z6;
      z2 = 4.0f * z6;
      z3 = -2.0f * z6;
      z4 = 2.0f * z6;
      z5 = 4.0f * z6;

      z6 = V_tensor[h][2][idx];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = V_tensor[h][3][idx];

      z1 += z6;
      z2 += -z6;
      z3 += 2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = V_tensor[h][4][idx];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = V_tensor[h][5][idx];

      z5 += z6;

      V_tensor[h][0][idx] = z0;
      V_tensor[h][1][idx] = z1;
      V_tensor[h][2][idx] = z2;
      V_tensor[h][3][idx] = z3;
      V_tensor[h][4][idx] = z4;
      V_tensor[h][5][idx] = z5;
    }
  }
}

// get U tensor = G*g*GT
void filter_transform(float *__restrict__ packed_filter,
                      float *__restrict__ U,
                      const filter_shape_t fs,
                      const U_shape_t us,
                      const int64_t collapsed_dim_size) {
  typedef float(*packed_filter_tensor_t)[fs.h][fs.w];
  typedef float(*U_tensor_t)[us.h][us.w];
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;
  U_tensor_t U_tensor = (U_tensor_t)U;

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
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    // parallel computation for each id
    for (int64_t w = 0; w < fs.w; ++w) {
      // non-sequential memory access
      // rewrite for better memory access performance
      z6 = packed_filter_tensor[idx][0][w];

      z0 = (1.0f / 4.0f) * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = packed_filter_tensor[idx][1][w];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (1.0f / 6.0f) * z6;
      z3 += (1.0f / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = packed_filter_tensor[idx][2][w];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += (1.0f / 6.0f) * z6;
      z4 += (1.0f / 6.0f) * z6;
      z5 = z6;

      U_tensor[idx][0][w] = z0;
      U_tensor[idx][1][w] = z1;
      U_tensor[idx][2][w] = z2;
      U_tensor[idx][3][w] = z3;
      U_tensor[idx][4][w] = z4;
      U_tensor[idx][5][w] = z5;
    }

    for (int64_t h = 0; h < us.h; ++h) {
      z6 = U_tensor[idx][h][0];

      z0 = (1.0f / 4.0f) * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = U_tensor[idx][h][1];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (1.0f / 6.0f) * z6;
      z3 += (1.0f / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = U_tensor[idx][h][2];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += (1.0f / 6.0f) * z6;
      z4 += (1.0f / 6.0f) * z6;
      z5 = z6;

      U_tensor[idx][h][0] = z0;
      U_tensor[idx][h][1] = z1;
      U_tensor[idx][h][2] = z2;
      U_tensor[idx][h][3] = z3;
      U_tensor[idx][h][4] = z4;
      U_tensor[idx][h][5] = z5;
    }
  }
}
// Calculate AT*...*A
void output_transform(float *__restrict__ M,  // input tensor
                      float *__restrict__ Y,  // output tensor
                      const tiling_info_t ti,
                      const int64_t collapsed_dim_size) {
  typedef float(*M_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*Y_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  M_tensor_t M_tensor = (M_tensor_t)M;
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;
  float z0, z1, z2, z3, z4;
  /*
  AT =
  ⎡1  1  1   1  1   0⎤
  ⎢                  ⎥
  ⎢0  1  -1  2  -2  0⎥
  ⎢                  ⎥
  ⎢0  1  1   4  4   0⎥
  ⎢                  ⎥
  ⎣0  1  -1  8  -8  1⎦
  */
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {  // processing tiles
#pragma omp parallel for

    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      z4 = M_tensor[0][w][idx];
      z0 = z4;

      z4 = M_tensor[1][w][idx];
      z0 = z0 + z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = M_tensor[2][w][idx];
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = M_tensor[3][w][idx];
      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = M_tensor[4][w][idx];
      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = M_tensor[5][w][idx];
      z3 += z4;

      Y_tensor[0][w][idx] = z0;
      Y_tensor[1][w][idx] = z1;
      Y_tensor[2][w][idx] = z2;
      Y_tensor[3][w][idx] = z3;
    }
#pragma omp parallel for

    for (int64_t h = 0; h < ti.tile_out_h; ++h) {
      z4 = Y_tensor[h][0][idx];

      z0 = z4;

      z4 = Y_tensor[h][1][idx];
      z0 += z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = Y_tensor[h][2][idx];
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = Y_tensor[h][3][idx];
      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = Y_tensor[h][4][idx];
      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = Y_tensor[h][5][idx];

      z3 += z4;

      Y_tensor[h][0][idx] = z0;
      Y_tensor[h][1][idx] = z1;
      Y_tensor[h][2][idx] = z2;
      Y_tensor[h][3][idx] = z3;
    }
  }
}

void filter_packing(float *__restrict__ filter, float *__restrict__ packed_filter, const U_shape_t fs) {
  typedef float(*filter_tensor_t)[fs.ic][fs.h][fs.w];
  typedef float(*packed_filter_tensor_t)[fs.w][fs.oc][fs.ic];
  filter_tensor_t filter_tensor = (filter_tensor_t)filter;
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;

  // get packed filter frome filter tensor
  for (int64_t h = 0; h < fs.h; ++h)
    for (int64_t w = 0; w < fs.w; ++w)
      for (int64_t oc = 0; oc < fs.oc; oc++)
        for (int64_t ic = 0; ic < fs.ic; ic++)
          packed_filter_tensor[h][w][oc][ic] = filter_tensor[oc][ic][h][w];
}

void image_packing(float *__restrict__ image,
                   float *__restrict__ packed_image,
                   const image_shape_t is,
                   const tiling_info_t ti) {
  typedef float(*packedImage_tensor_t)[is.ic][ti.tile_in_h][ti.tile_in_w];
  typedef float(*image_tensor_t)[is.ic][is.h][is.w];
  packedImage_tensor_t packed_image_tensor = (packedImage_tensor_t)packed_image;
  image_tensor_t image_tensor = (image_tensor_t)image;

// batch个image，每个image有ts.num_tile_per_image个tiles，对每个tiles求卷积
#pragma omp parallel for collapse(2)
  for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
    for (int64_t ic = 0; ic < is.ic; ic++) {
      for (int64_t h = 0; h < ti.tile_in_h; ++h) {
        for (int64_t w = 0; w < ti.tile_in_w; ++w) {
          tile_index_t tidx = get_tile_index(tile, ti);
          int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
          // Something to be done here
          // 即：tiling size 为4*4，防止数组越界；超出范围的用0填充
          // image数组已经给出来了，似乎是无法通过一些小trick去掉分支？
          if (hh * 4 + h < is.h && ww * 4 + w < is.w)
            packed_image_tensor[tile][ic][h][w] = image_tensor[batch][ic][(hh * 4 + h)][(ww * 4 + w)];
          else
            packed_image_tensor[tile][ic][h][w] = 0;
        }
      }
    }
  }
}

void output_unpacking_store(float *__restrict__ Y,
                            float *__restrict__ out,
                            const out_shape_t os,
                            const tiling_info_t ti) {
  typedef float(*Y_tensor_t)[ti.num_tiles][ti.tile_in_h][ti.tile_out_w];
  typedef float(*out_tensor_t)[os.oc][os.h][os.w];
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;
  out_tensor_t out_tensor = (out_tensor_t)out;
#pragma omp parallel for collapse(2)
  for (int64_t h = 0; h < ti.tile_out_h; ++h) {
    for (int64_t w = 0; w < ti.tile_out_w; ++w) {
      for (int64_t oc = 0; oc < os.oc; oc++) {
        for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
          tile_index_t tidx = get_tile_index(tile, ti);
          int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
          if (hh * 4 + h < os.h && ww * 4 + w < os.w)
            out_tensor[batch][oc][(hh * 4 + h)][(ww * 4 + w)] = Y_tensor[oc][tile][h][w];
        }
      }
    }
  }
}
/*
AT*((G*g)(BT*d)) =
⎡d[0]⋅g[0] + d[1]⋅g[1] + d[2]⋅g[2]⎤
⎢                                 ⎥
⎢d[1]⋅g[0] + d[2]⋅g[1] + d[3]⋅g[2]⎥
⎢                                 ⎥
⎢d[2]⋅g[0] + d[3]⋅g[1] + d[4]⋅g[2]⎥
⎢                                 ⎥
⎣d[3]⋅g[0] + d[4]⋅g[1] + d[5]⋅g[2]⎦
*/
void sgemm(const int64_t M, const int64_t N, const int64_t K, float *A, float *B, float *C) {
  typedef float(*A_tensor_t)[K];
  typedef float(*B_tensor_t)[K];
  typedef float(*C_tensor_t)[M];
  A_tensor_t A_tensor = (A_tensor_t)A;
  B_tensor_t B_tensor = (B_tensor_t)B;
  C_tensor_t C_tensor = (C_tensor_t)C;

  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      // 矩阵元素置零
      C_tensor[n][m] = 0;
      // 计算对应元素乘积
      for (int64_t k = 0; k < K; ++k) {
        // packing improves performance here
        C_tensor[n][m] += A_tensor[m][k] * B_tensor[n][k];
      }
    }
  }
}

void winograd_convolution(
    float *__restrict__ image, /**< float [batch_num][input_channel_num][image_height][image_width] */
    const int image_height,
    const int image_width,
    const int input_channel_num,
    float *__restrict__ filter, /**< float [output_channel_num][input_channel_num][FLT_H][FLT_W] */
    const int output_channel_num,
    const int batch_num,
    float *__restrict__ out) {
  std::chrono::system_clock::time_point start;
  std::chrono::system_clock::time_point end;
  std::chrono::milliseconds duration;
  static Device_Memory_Pool device_Memory_Pool;
  static bool initialized = 0;
  static cublasHandle_t handle;
  static std::thread cublasHandleCreate;
  static std::thread cudaHostMallocThread;
  static float *packed_image;
  // cublasCreate(&handle);
  if (!initialized) {
    std::thread tmp1(cublasCreate, &handle);
    std::swap(tmp1, cublasHandleCreate);
    std::thread tmp2(cudaHostMalloc, &packed_image, (size_t)1<<(size_t)31, cudaHostAllocMapped);  // 20000000000
    // alloc enough memory to store packed_image and out tensor
    std::swap(tmp2, cudaHostMallocThread);
    // cudaHostAlloc(
    //   &packed_image, sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic,
    //   cudaHostAllocMapped);
    device_Memory_Pool.init();
  } else {
    device_Memory_Pool.poolFree();
  }
  initialized = 1;
  /* new vars of shape */
  // image shape
  const image_shape_t is = {.bs = batch_num, .ic = input_channel_num, .h = image_height, .w = image_width};
  // filter shape
  const filter_shape_t fs = {.oc = output_channel_num, .ic = input_channel_num, .h = FLT_H, .w = FLT_W};
  // output shape
  const out_shape_t os = get_output_shape(is, fs);
  // tiling info
  const tiling_info_t ti = get_tiling_info(is, os);
  // U shape
  const U_shape_t us = get_U_shape(fs, ti);
  // V shape
  // vs.ic=is.ic=input_channel_num
  // vs.num_tiles=ts.num_tiles =  DIV_UP(os.h, 4) * DIV_UP(os.w, 4) * batch_num;
  const V_shape_t vs = get_V_shape(is, ti);

  cudaPitchedPtr device_M_tensor;
  cudaExtent device_M_tensor_extent = make_cudaExtent(
      vs.num_tiles * sizeof(float) * us.oc, ti.tile_in_w, ti.tile_in_h);
  device_Memory_Pool.poolMalloc3D(&device_M_tensor, device_M_tensor_extent);
  // std::thread device_M_tensor_alloc_thread(alloc_M_Tensor_Memory, std::ref(device_M_tensor), vs, us, ti);
  // allocate memory
  //  float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);

  //  = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);

  //进行两次变换
  float *device_U_tensor = NULL;
  int ldu = 0;
  device_filter_transform(filter, fs, us, us.oc * us.ic, &device_U_tensor, &ldu, device_Memory_Pool);

  // filter_transform(filter, transformed_filter, fs, us, us.oc * us.ic);
  // filter_packing(transformed_filter, U, us);

  // parallel accelerate!
  // 150ms
  float *device_V_tensor;
  int ldv = 0;
  if (cudaHostMallocThread.joinable()) {
    cudaHostMallocThread.join();
  }
  float *host_out_tensor = packed_image + ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic;
  float *device_out_tensor;
  cudaHostGetDevicePointer(&device_out_tensor, packed_image, 0);
  device_out_tensor += ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic;
  image_packing(image, packed_image, is, ti);
  device_image_transform(packed_image, is, ti, vs, &device_V_tensor, &ldv, device_Memory_Pool);

  // cudaFreeHost(packed_image);
  // 425ms
  // image_transform(packed_image, V, vs, ti, vs.ic * vs.num_tiles);
  // ti.tile_in_h = ti.tile_in_w = 6
  // #pragma omp parallel for collapse(2)
  // alloc_M_Tensor_Memory(device_M_tensor, vs, us, ti);

  if (cublasHandleCreate.joinable()) {
    cublasHandleCreate.join();
  }
  // device_M_tensor_alloc_thread.join();

  for (int64_t h = 0; h < ti.tile_in_h; ++h) {
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      // 定义出U V M Tensor指针
      typedef float(*U_tensor_t)[ti.tile_in_w][us.oc][ldu];
      typedef float(*V_tensor_t)[ti.tile_in_w][ldv];
      typedef float(*M_tensor_t)[ti.tile_in_w][device_M_tensor.pitch / sizeof(float)];
      // 每次循环的时候都会定义一遍？不过估计会被编译器优化掉
      U_tensor_t U_tensor = (U_tensor_t)device_U_tensor;
      V_tensor_t V_tensor = (V_tensor_t)device_V_tensor;
      M_tensor_t M_tensor = (M_tensor_t)device_M_tensor.ptr;
      // 90ms
      cublas_sgemm(handle,
                   (float *)U_tensor[h][w],
                   us.ic,
                   (float *)V_tensor[h][w],
                   vs.ic,
                   (float *)M_tensor[h][w],
                   vs.num_tiles,
                   us.oc,
                   vs.num_tiles,
                   us.ic,
                   us,
                   vs,
                   ti);
      // sgemm(vs.num_tiles,
      //       us.oc,
      //       us.ic,
      //       (float *)(V_tensor[h][w]),
      //       (float *)(U_tensor[h][w]),
      //       (float *)(M_tensor[h][w]));
    }
  }
  // cublasDestroy(handle);
  // cudaFree(device_U_tensor);
  // cudaFree(device_V_tensor);
  // cudaFree(device_U_tensor);
  // cudaFree(device_V_tensor);
  // 6000ms
  device_output_transform(
      device_M_tensor, device_out_tensor, out, ti, us.oc * vs.num_tiles, us, vs, os, device_Memory_Pool);
  // output_transform(M, Y, ti, us.oc * vs.num_tiles);
  // 5000ms
  // output_unpacking_store(Y, out, os, ti);

  // memcpy(out, host_out_tensor, sizeof(float) * os.bs * os.oc * os.h * os.w);
  // free(packed_filter);
  // cudaFreeHost(packed_image);
}