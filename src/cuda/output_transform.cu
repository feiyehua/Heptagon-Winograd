#include "output_transform.cuh"
__global__ void output_transform(cudaPitchedPtr M,  // input tensor
                                 cudaPitchedPtr Y,  // output tensor
                                 const tiling_info_t ti,
                                 const int64_t collapsed_dim_size) {
  float* M_tensor = (float*)M.ptr;
  float* Y_tensor = (float*)Y.ptr;
  float z0, z1, z2, z3, z4;
  int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
  int64_t M_tensor_z = M.pitch / sizeof(float);
  int64_t M_tensor_yz = M_tensor_z * M.ysize;

  int64_t Y_tensor_z = Y.pitch / sizeof(float);
  int64_t Y_tensor_yz = Y.ysize * Y_tensor_z;

  if (idx >= collapsed_dim_size) {
    return;
  }
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
#pragma unroll
  for (int64_t w = 0; w < ti.tile_in_w; ++w) {
    z4 = M_tensor[0 * M_tensor_yz + w * M_tensor_z + idx];
    z0 = z4;

    z4 = M_tensor[1 * M_tensor_yz + w * M_tensor_z + idx];
    z0 = z0 + z4;
    z1 = z4;
    z2 = z4;
    z3 = z4;

    z4 = M_tensor[2 * M_tensor_yz + w * M_tensor_z + idx];
    z0 += z4;
    z1 += -z4;
    z2 += z4;
    z3 += -z4;

    z4 = M_tensor[3 * M_tensor_yz + w * M_tensor_z + idx];
    z0 += z4;
    z1 += 2.0f * z4;
    z2 += 4.0f * z4;
    z3 += 8.0f * z4;

    z4 = M_tensor[4 * M_tensor_yz + w * M_tensor_z + idx];
    z0 += z4;
    z1 += -2.0f * z4;
    z2 += 4.0f * z4;
    z3 += -8.0f * z4;

    z4 = M_tensor[5 * M_tensor_yz + w * M_tensor_z + idx];
    z3 += z4;

    Y_tensor[0 * Y_tensor_z + w + idx * Y_tensor_yz] = z0;
    Y_tensor[1 * Y_tensor_z + w + idx * Y_tensor_yz] = z1;
    Y_tensor[2 * Y_tensor_z + w + idx * Y_tensor_yz] = z2;
    Y_tensor[3 * Y_tensor_z + w + idx * Y_tensor_yz] = z3;
  }
#pragma unroll
  for (int64_t h = 0; h < ti.tile_out_h; ++h) {
    z4 = Y_tensor[h * Y_tensor_z + 0 + idx * Y_tensor_yz];

    z0 = z4;

    z4 = Y_tensor[h * Y_tensor_z + 1 + idx * Y_tensor_yz];
    z0 += z4;
    z1 = z4;
    z2 = z4;
    z3 = z4;

    z4 = Y_tensor[h * Y_tensor_z + 2 + idx * Y_tensor_yz];
    z0 += z4;
    z1 += -z4;
    z2 += z4;
    z3 += -z4;

    z4 = Y_tensor[h * Y_tensor_z + 3 + idx * Y_tensor_yz];
    z0 += z4;
    z1 += 2.0f * z4;
    z2 += 4.0f * z4;
    z3 += 8.0f * z4;

    z4 = Y_tensor[h * Y_tensor_z + 4 + idx * Y_tensor_yz];
    z0 += z4;
    z1 += -2.0f * z4;
    z2 += 4.0f * z4;
    z3 += -8.0f * z4;

    z4 = Y_tensor[h * Y_tensor_z + 5 + idx * Y_tensor_yz];

    z3 += z4;

    Y_tensor[h * Y_tensor_z + 0 + idx * Y_tensor_yz] = z0;
    Y_tensor[h * Y_tensor_z + 1 + idx * Y_tensor_yz] = z1;
    Y_tensor[h * Y_tensor_z + 2 + idx * Y_tensor_yz] = z2;
    Y_tensor[h * Y_tensor_z + 3 + idx * Y_tensor_yz] = z3;
  }
}

void alloc_M_Tensor_Memory(cudaPitchedPtr& M_tensor, V_shape_t vs, U_shape_t us, tiling_info_t ti) {
  cudaPitchedPtr device_M_tensor;
  cudaExtent device_M_tensor_extent = make_cudaExtent(
      vs.num_tiles * sizeof(float) * us.oc, ti.tile_in_w, ti.tile_in_h);
  cudaMalloc3D(&device_M_tensor, device_M_tensor_extent);
  M_tensor = device_M_tensor;
}

void device_output_transform(cudaPitchedPtr device_M_tensor,  // input tensor
                             float* __restrict__ Y,           // output tensor
                             const tiling_info_t ti,
                             const int64_t collapsed_dim_size,
                             const U_shape_t us,
                             const V_shape_t vs) {
  // 将M_tensor拷贝到GPU内存上
  // cudaMemcpy3DParms device_M_tensor_copy_parms = {0};
  // device_M_tensor_copy_parms.srcPtr.ptr = M;
  // device_M_tensor_copy_parms.srcPtr.xsize = ti.tile_in_h;
  // device_M_tensor_copy_parms.srcPtr.ysize = ti.tile_in_w;
  // device_M_tensor_copy_parms.srcPtr.pitch = vs.num_tiles * us.oc * sizeof(float);
  // device_M_tensor_copy_parms.dstPtr = device_M_tensor;
  // device_M_tensor_copy_parms.extent = device_M_tensor_extent;
  // device_M_tensor_copy_parms.kind = cudaMemcpyHostToDevice;
  // cudaMemcpy3D(&device_M_tensor_copy_parms);

  // 分配Y_tensor内存
  cudaPitchedPtr device_Y_tensor;
  cudaExtent device_Y_tensor_extent = make_cudaExtent(
      sizeof(float) * ti.tile_out_w, ti.tile_in_h, us.oc * vs.num_tiles);
  cudaMalloc3D(&device_Y_tensor, device_Y_tensor_extent);

  //计算Y_tensor
  output_transform<<<DIV_UP(us.oc * vs.num_tiles, 1024), 1024>>>(
      device_M_tensor, device_Y_tensor, ti, us.oc * vs.num_tiles);
  cudaDeviceSynchronize();

  // 将Y_tensor复制回host
  cudaMemcpy3DParms host_Y_tensor_copy_parms = {0};
  host_Y_tensor_copy_parms.srcPtr = device_Y_tensor;
  host_Y_tensor_copy_parms.dstPtr.ptr = Y;
  host_Y_tensor_copy_parms.dstPtr.pitch = sizeof(float) * ti.tile_out_w;
  host_Y_tensor_copy_parms.dstPtr.ysize = ti.tile_in_h;
  host_Y_tensor_copy_parms.dstPtr.xsize = us.oc * vs.num_tiles;
  host_Y_tensor_copy_parms.extent = device_Y_tensor_extent;
  host_Y_tensor_copy_parms.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&host_Y_tensor_copy_parms);

  //
  cudaFree(device_Y_tensor.ptr);
  cudaFree(device_M_tensor.ptr);
}