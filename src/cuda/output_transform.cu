#include "output_transform.cuh"

__device__ inline tile_index_t device_get_tile_index(int64_t tile, tiling_info_t ts) {
  tile_index_t ti;
  ti.b = tile / ts.num_tile_per_image;
  tile = tile % ts.num_tile_per_image;
  ti.th = tile / ts.tiles_on_w;
  ti.tw = tile % ts.tiles_on_w;
  return ti;
}

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


__global__ void device_output_unpacking_store(cudaPitchedPtr device_Y_tensor,
                                              float* __restrict__ device_out_tensor,
                                              const out_shape_t os,
                                              const tiling_info_t ti) {
  // typedef float(*Y_tensor_t)[ti.num_tiles][ti.tile_in_h][ti.tile_out_w];
  // typedef float(*out_tensor_t)[os.oc][os.h][os.w];
  float* Y_tensor = (float*)device_Y_tensor.ptr;
  float* out_tensor = (float*)device_out_tensor;
  int64_t Y_tensor_z = device_Y_tensor.pitch / sizeof(float);
  int64_t Y_tensor_yz = device_Y_tensor.ysize * Y_tensor_z;

  int64_t out_tensor_z = ROUND_UP(os.w , 4);  // device_out_tensor.pitch / sizeof(float);
  int64_t out_tensor_yz = ROUND_UP(os.h , 4) * out_tensor_z;

  int64_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  int64_t oc = tid % os.oc;
  int64_t tile = tid / os.oc;
  tile_index_t tidx = device_get_tile_index(tile, ti);
  int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
  if (batch >= os.bs) return;
#pragma unroll
  for (int64_t h = 0; h < ti.tile_out_h; ++h) {
    FETCH_FLOAT4(out_tensor[(batch * os.oc + oc) * out_tensor_yz + (hh * 4 + h) * out_tensor_z +
                            (ww * 4)]) = FETCH_FLOAT4(Y_tensor[(oc * ti.num_tiles + tile) * Y_tensor_yz +
                                                               h * Y_tensor_z]);
  }
}


void device_output_transform(cudaPitchedPtr device_M_tensor,         // input tensor
                             float* __restrict__ device_out1_tensor,  // output tensor
                             float* __restrict__ out,
                             const tiling_info_t ti,
                             const int64_t collapsed_dim_size,
                             const U_shape_t us,
                             const V_shape_t vs,
                             const out_shape_t os,
                             Device_Memory_Pool& device_Memory_Pool) {

  // 分配Y_tensor内存
  cudaPitchedPtr device_Y_tensor;
  cudaExtent device_Y_tensor_extent = make_cudaExtent(
      sizeof(float) * ti.tile_in_w, ti.tile_in_h, us.oc * vs.num_tiles);

  device_Memory_Pool.poolMalloc3D(&device_Y_tensor, device_Y_tensor_extent);

  //计算Y_tensor
  output_transform<<<DIV_UP(us.oc * vs.num_tiles, 1024), 1024>>>(
      device_M_tensor, device_Y_tensor, ti, us.oc * vs.num_tiles);

  // 分配out_tensor内存
  cudaPitchedPtr device_out_tensor;
  cudaExtent device_out_tensor_extent = make_cudaExtent(
      sizeof(float) * ROUND_UP(os.w ,4), ROUND_UP(os.h,4), os.oc * os.bs);
  //等待Y_tensor计算完成
  device_Memory_Pool.poolMalloc3D(&device_out_tensor, device_out_tensor_extent);
  cudaDeviceSynchronize();

  device_output_unpacking_store<<<DIV_UP(os.oc * ti.num_tiles, 1024), 1024>>>(
      device_Y_tensor, (float*)device_out_tensor.ptr, os, ti);
  cudaDeviceSynchronize();

  // 将Y_tensor复制回host
  cudaMemcpy3DParms host_out_tensor_copy_parms = {0};
  host_out_tensor_copy_parms.srcPtr = device_out_tensor;
  host_out_tensor_copy_parms.dstPtr.ptr = out;
  host_out_tensor_copy_parms.dstPtr.pitch = sizeof(float) * os.w;
  host_out_tensor_copy_parms.dstPtr.ysize = os.h;
  host_out_tensor_copy_parms.dstPtr.xsize = os.w;
  host_out_tensor_copy_parms.extent = make_cudaExtent(
      sizeof(float) * os.w, os.h, os.oc * os.bs);  // device_out_tensor_extent;  //
  host_out_tensor_copy_parms.kind = cudaMemcpyHostToHost;
  cudaMemcpy3D(&host_out_tensor_copy_parms);
}