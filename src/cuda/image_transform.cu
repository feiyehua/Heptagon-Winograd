#include "image_transform.cuh"

void allocate_packed_image_memory(void **ptr, size_t size, unsigned int flags) {
  cudaHostAlloc(ptr, size, flags);
}
// 计算某个tile对应的横纵坐标
__device__ inline tile_index_t device_get_tile_index(int64_t tile, tiling_info_t ts) {
  tile_index_t ti;
  ti.b = tile / ts.num_tile_per_image;
  tile = tile % ts.num_tile_per_image;
  ti.th = tile / ts.tiles_on_w;
  ti.tw = tile % ts.tiles_on_w;
  return ti;
}

__global__ void image_packing(const cudaPitchedPtr device_image,
                              const cudaPitchedPtr device_packed_image,
                              const image_shape_t is,
                              const tiling_info_t ti) {
  // typedef float(*packedImage_tensor_t)[ti.tile_in_w][ti.num_tiles][is.ic];
  // typedef float(*image_tensor_t)[is.ic][is.h][is.w];
  // packedImage_tensor_t packed_image_tensor = (packedImage_tensor_t)packed_image;
  // image_tensor_t image_tensor = (image_tensor_t)image;

  // batch个image，每个image有ts.num_tile_per_image个tiles，对每个tiles求卷积
  float *device_image_tensor = (float *)device_image.ptr;
  float *device_packed_image_tensor = (float *)device_packed_image.ptr;

  int64_t x_index = (blockIdx.x * blockDim.x + threadIdx.x);
  int64_t tile = x_index / is.ic;
  int64_t ic = x_index % is.ic;

  int64_t device_packed_image_tensor_z = device_packed_image.pitch / sizeof(float);
  int64_t device_packed_image_tensor_yz = device_packed_image.ysize * device_packed_image_tensor_z;

  int64_t device_image_tensor_z = device_image.pitch / sizeof(float);
  int64_t device_image_tensor_yz = device_image.ysize * device_image_tensor_z;

  if (tile >= ti.num_tiles) {
    return;
  }
#pragma unroll
  for (int64_t h = 0; h < ti.tile_in_h; ++h) {
#pragma unroll
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      tile_index_t tidx = device_get_tile_index(tile, ti);
      int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
      // Something to be done here
      // 即：tiling size 为4*4，防止数组越界；超出范围的用0填充
      // image数组已经给出来了，似乎是无法通过一些小trick去掉分支？
      if (hh * 4 + h < is.h && ww * 4 + w < is.w)
        device_packed_image_tensor[x_index * device_packed_image.ysize * device_packed_image.pitch /
                                       sizeof(float) +
                                   h * device_packed_image.pitch / sizeof(float) + w]

            = device_image_tensor[(batch * is.ic + ic) * device_image.ysize * device_image.pitch /
                                      sizeof(float) +
                                  (hh * 4 + h) * device_image.pitch / sizeof(float) + (ww * 4 + w)];
      else {
        device_packed_image_tensor[x_index * device_packed_image.ysize * device_packed_image.pitch /
                                       sizeof(float) +
                                   h * device_packed_image.pitch / sizeof(float) + w]

            = 0;
      }
    }
  }
}
// get V tensor = BT*d*B
__global__ void image_transform(const cudaPitchedPtr device_image,
                                const cudaPitchedPtr V,
                                const V_shape_t vs,
                                const tiling_info_t ti,
                                const image_shape_t is,
                                const int64_t collapsed_dim_size) {

  float *device_image_tensor = (float *)device_image.ptr;
  float packed_image_tensor[TILE_IN_H][TILE_IN_W];
  float *V_tensor = (float *)V.ptr;

  int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
  int64_t x_index = (blockIdx.x * blockDim.x + threadIdx.x);

  int64_t tile = x_index / is.ic;
  int64_t ic = x_index % is.ic;

  int64_t device_image_tensor_z = device_image.pitch / sizeof(float);
  int64_t device_image_tensor_yz = device_image.ysize * device_image_tensor_z;

  if (tile >= ti.num_tiles) {
    return;
  }
#pragma unroll
  for (int64_t h = 0; h < ti.tile_in_h; ++h) {
#pragma unroll
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      tile_index_t tidx = device_get_tile_index(tile, ti);
      int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
      // Something to be done here
      // 即：tiling size 为4*4，防止数组越界；超出范围的用0填充
      // image数组已经给出来了，似乎是无法通过一些小trick去掉分支？
      if (hh * 4 + h < is.h && ww * 4 + w < is.w)
        packed_image_tensor[h][w]

            = device_image_tensor[(batch * is.ic + ic) * device_image.ysize * device_image.pitch /
                                      sizeof(float) +
                                  (hh * 4 + h) * device_image.pitch / sizeof(float) + (ww * 4 + w)];
      else {
        packed_image_tensor[h][w] = 0;
      }
    }
  }

  if (idx >= collapsed_dim_size) return;

  int64_t V_tensor_yz = V.ysize * V.pitch / sizeof(float);
  int64_t V_tensor_z = V.pitch / sizeof(float);

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
// ti.tile_in_w = 6
#pragma unroll
  for (int64_t w = 0; w < ti.tile_in_w; ++w) {
    z6 = packed_image_tensor[0][w];

    z0 = 4.0f * z6;

    z6 = packed_image_tensor[1][w];

    z1 = -4.0f * z6;
    z2 = 4.0f * z6;
    z3 = -2.0f * z6;
    z4 = 2.0f * z6;
    z5 = 4.0f * z6;

    z6 = packed_image_tensor[2][w];

    z0 += -5.0f * z6;
    z1 += -4.0f * z6;
    z2 += -4.0f * z6;
    z3 += -z6;
    z4 += -z6;

    z6 = packed_image_tensor[3][w];

    z1 += z6;
    z2 += -z6;
    z3 += 2.0f * z6;
    z4 += -2.0f * z6;
    z5 += -5.0f * z6;

    z6 = packed_image_tensor[4][w];

    z0 += z6;
    z1 += z6;
    z2 += z6;
    z3 += z6;
    z4 += z6;

    z6 = packed_image_tensor[5][w];

    z5 += z6;

    V_tensor[0 * V_tensor_yz + w * V_tensor_z + idx] = z0;
    V_tensor[1 * V_tensor_yz + w * V_tensor_z + idx] = z1;
    V_tensor[2 * V_tensor_yz + w * V_tensor_z + idx] = z2;
    V_tensor[3 * V_tensor_yz + w * V_tensor_z + idx] = z3;
    V_tensor[4 * V_tensor_yz + w * V_tensor_z + idx] = z4;
    V_tensor[5 * V_tensor_yz + w * V_tensor_z + idx] = z5;
  }
  // ti.tile_in_h = 6

#pragma unroll
  for (int64_t h = 0; h < ti.tile_in_h; ++h) {
    z6 = V_tensor[h * V_tensor_yz + 0 * V_tensor_z + idx];

    z0 = 4.0f * z6;

    z6 = V_tensor[h * V_tensor_yz + 1 * V_tensor_z + idx];

    z1 = -4.0f * z6;
    z2 = 4.0f * z6;
    z3 = -2.0f * z6;
    z4 = 2.0f * z6;
    z5 = 4.0f * z6;

    z6 = V_tensor[h * V_tensor_yz + 2 * V_tensor_z + idx];

    z0 += -5.0f * z6;
    z1 += -4.0f * z6;
    z2 += -4.0f * z6;
    z3 += -z6;
    z4 += -z6;

    z6 = V_tensor[h * V_tensor_yz + 3 * V_tensor_z + idx];

    z1 += z6;
    z2 += -z6;
    z3 += 2.0f * z6;
    z4 += -2.0f * z6;
    z5 += -5.0f * z6;

    z6 = V_tensor[h * V_tensor_yz + 4 * V_tensor_z + idx];

    z0 += z6;
    z1 += z6;
    z2 += z6;
    z3 += z6;
    z4 += z6;

    z6 = V_tensor[h * V_tensor_yz + 5 * V_tensor_z + idx];

    z5 += z6;

    V_tensor[h * V_tensor_yz + 0 * V_tensor_z + idx] = z0;
    V_tensor[h * V_tensor_yz + 1 * V_tensor_z + idx] = z1;
    V_tensor[h * V_tensor_yz + 2 * V_tensor_z + idx] = z2;
    V_tensor[h * V_tensor_yz + 3 * V_tensor_z + idx] = z3;
    V_tensor[h * V_tensor_yz + 4 * V_tensor_z + idx] = z4;
    V_tensor[h * V_tensor_yz + 5 * V_tensor_z + idx] = z5;
  }
}

void device_image_transform(float *__restrict__ image,
                            const image_shape_t is,
                            const tiling_info_t ti,
                            const V_shape_t vs,
                            float **V_tensor,
                            int *ldv,
                            Device_Memory_Pool &device_Memory_Pool) {
  cudaPitchedPtr device_image;

  device_Memory_Pool.poolMalloc(&device_image.ptr, sizeof(float) * is.w * is.h * is.bs * is.ic);
  device_image.pitch = sizeof(float) * is.w;
  device_image.ysize = is.h;
  device_image.xsize = sizeof(float) * is.w;
  cudaMemcpy(device_image.ptr, image, sizeof(float) * is.w * is.h * is.bs * is.ic, cudaMemcpyHostToDevice);
  // 既然在这里需要拷贝一次内存，那应该就可以填充一些多余的0，实现padding，去掉image_packing中的分支。(deprecated)

  //分配V_tensor内存
  cudaExtent V_tensor_extent = make_cudaExtent(
      sizeof(float) * vs.ic * vs.num_tiles, ti.tile_in_w, ti.tile_in_h);
  cudaPitchedPtr device_V_tensor;
  device_Memory_Pool.poolMalloc3D(&device_V_tensor, V_tensor_extent);

  image_transform<<<DIV_UP(vs.num_tiles * vs.ic, 128), 128>>>(
      device_image, device_V_tensor, vs, ti,is, vs.ic * vs.num_tiles);

  *V_tensor = (float *)device_V_tensor.ptr;
  *ldv = device_V_tensor.pitch / (sizeof(float));
}
