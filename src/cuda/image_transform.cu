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
  if (tile >= ti.num_tiles) {
    return;
  }
  for (int64_t h = 0; h < ti.tile_in_h; ++h) {
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      tile_index_t tidx = device_get_tile_index(tile, ti);
      int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
      // Something to be done here
      // 即：tiling size 为4*4，防止数组越界；超出范围的用0填充
      // image数组已经给出来了，似乎是无法通过一些小trick去掉分支？
      device_packed_image_tensor[x_index * device_packed_image.ysize * device_packed_image.pitch /
                                     sizeof(float) +
                                 h * device_packed_image.pitch / sizeof(float) + w]

          = device_image_tensor[(batch * is.ic + ic) * device_image.ysize * device_image.pitch /
                                    sizeof(float) +
                                (hh * 4 + h) * device_image.pitch / sizeof(float) + (ww * 4 + w)];
    }
  }
}
// get V tensor = BT*d*B
__global__ void image_transform(const cudaPitchedPtr device_packed_image,
                                const cudaPitchedPtr V,
                                const V_shape_t vs,
                                const tiling_info_t ti,
                                const int64_t collapsed_dim_size) {
  // collapsed_dim_size = vs.ic * vs.num_tiles
  // collapsed the tensor for better performance?
  // should transform first, then pack
  // typedef float(*packed_image_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  // typedef float(*V_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  // packed_image_tensor_t packed_image_tensor = (packed_image_tensor_t)packed_image;
  // V_tensor_t V_tensor = (V_tensor_t)V;

  float *packed_image_tensor = (float *)device_packed_image.ptr;
  float *V_tensor = (float *)V.ptr;

  int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
  int64_t x_index = (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= collapsed_dim_size) return;

  int64_t device_packed_image_z = device_packed_image.pitch / sizeof(float);
  int64_t device_packed_image_yz = device_packed_image.ysize * device_packed_image_z;

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
    z6 = packed_image_tensor[x_index * device_packed_image_yz + 0 * device_packed_image_z + w];

    z0 = 4.0f * z6;

    z6 = packed_image_tensor[x_index * device_packed_image_yz + 1 * device_packed_image_z + w];

    z1 = -4.0f * z6;
    z2 = 4.0f * z6;
    z3 = -2.0f * z6;
    z4 = 2.0f * z6;
    z5 = 4.0f * z6;

    z6 = packed_image_tensor[x_index * device_packed_image_yz + 2 * device_packed_image_z + w];

    z0 += -5.0f * z6;
    z1 += -4.0f * z6;
    z2 += -4.0f * z6;
    z3 += -z6;
    z4 += -z6;

    z6 = packed_image_tensor[x_index * device_packed_image_yz + 3 * device_packed_image_z + w];

    z1 += z6;
    z2 += -z6;
    z3 += 2.0f * z6;
    z4 += -2.0f * z6;
    z5 += -5.0f * z6;

    z6 = packed_image_tensor[x_index * device_packed_image_yz + 4 * device_packed_image_z + w];

    z0 += z6;
    z1 += z6;
    z2 += z6;
    z3 += z6;
    z4 += z6;

    z6 = packed_image_tensor[x_index * device_packed_image_yz + 5 * device_packed_image_z + w];

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
  // 直接使用内存映射访问packed_image
  // float *device_packed_image;
  // cudaHostGetDevicePointer(&device_packed_image, packed_image, 0);
  //分配device_packed_image内存
  // cudaPitchedPtr device_packed_image;
  // cudaExtent device_packed_image_extent = make_cudaExtent(
  //     sizeof(float) * ti.tile_in_w, ti.tile_in_h, ti.num_tiles * is.ic);
  // cudaError_t err = cudaMalloc3D(&device_packed_image,
  // device_packed_image_extent);

  // 将packed_image拷贝到GPU内存上
  // cudaMemcpy3DParms device_packed_image_copy_parms = {0};
  // device_packed_image_copy_parms.srcPtr.ptr = packed_image;
  // device_packed_image_copy_parms.srcPtr.xsize = ti.num_tiles * is.ic;
  // device_packed_image_copy_parms.srcPtr.ysize = ti.tile_in_h;
  // device_packed_image_copy_parms.srcPtr.pitch = ti.tile_in_w * sizeof(float);
  // device_packed_image_copy_parms.dstPtr = device_packed_image;
  // device_packed_image_copy_parms.extent = device_packed_image_extent;
  // device_packed_image_copy_parms.kind = cudaMemcpyHostToDevice;
  // cudaMemcpy3D(&device_packed_image_copy_parms);

  // float *__restrict__ device_image;
  cudaPitchedPtr device_image;
  cudaExtent extent = make_cudaExtent(sizeof(float) * ti.tiles_on_w * ti.tile_in_w,
                                      ti.tiles_on_h * ti.tile_in_h,
                                      is.bs * is.ic);  // 分配足够的内存，去掉image_packing中的分支
  device_Memory_Pool.poolMalloc3D(&device_image, extent);
  cudaMemset(device_image.ptr, 0, device_image.pitch * ti.tiles_on_h * ti.tile_in_h * is.bs * is.ic);
  cudaExtent image_extent = make_cudaExtent(sizeof(float) * is.w, is.h, is.bs * is.ic);
  cudaMemcpy3DParms device_image_copy_parms = {0};
  device_image_copy_parms.srcPtr.ptr = image;
  device_image_copy_parms.srcPtr.xsize = is.w * sizeof(float);
  device_image_copy_parms.srcPtr.ysize = is.h;
  device_image_copy_parms.srcPtr.pitch = is.w * sizeof(float);
  device_image_copy_parms.dstPtr = device_image;
  device_image_copy_parms.extent = image_extent;
  device_image_copy_parms.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&device_image_copy_parms);
  //既然在这里需要拷贝一次内存，那应该就可以填充一些多余的0，实现padding，去掉image_packing中的分支。

  // 分配packed_image内存
  cudaPitchedPtr device_packed_image;
  cudaExtent device_packed_image_extent = make_cudaExtent(
      sizeof(float) * ti.tile_in_w, ti.tile_in_h, ti.num_tiles * is.ic);
  device_Memory_Pool.poolMalloc3D(&device_packed_image, device_packed_image_extent);
  image_packing<<<DIV_UP(ti.num_tiles * is.ic, 1024), 1024>>>(device_image, device_packed_image, is, ti);
  cudaDeviceSynchronize();

  //分配V_tensor内存
  cudaExtent V_tensor_extent = make_cudaExtent(
      sizeof(float) * vs.ic * vs.num_tiles, ti.tile_in_w, ti.tile_in_h);
  cudaPitchedPtr device_V_tensor;
  device_Memory_Pool.poolMalloc3D(&device_V_tensor, V_tensor_extent);

  image_transform<<<DIV_UP(vs.num_tiles * vs.ic, 1024), 1024>>>(
      device_packed_image, device_V_tensor, vs, ti, vs.ic * vs.num_tiles);
  cudaDeviceSynchronize();

  *V_tensor = (float *)device_V_tensor.ptr;
  *ldv = device_V_tensor.pitch / (sizeof(float));
}
