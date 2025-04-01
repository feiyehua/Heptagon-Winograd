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

#define FETCH_FLOAT4(float_var) (reinterpret_cast<float4*>(&(float_var))[0])

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

  int64_t out_tensor_z = os.w;  // device_out_tensor.pitch / sizeof(float);
  int64_t out_tensor_yz = os.h * out_tensor_z;

  int64_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  int64_t oc = tid % os.oc;
  int64_t tile = tid / os.oc;
  tile_index_t tidx = device_get_tile_index(tile, ti);
  int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
  float tmp[TILE_OUT_H][TILE_OUT_W];
  if (batch >= os.bs) return;
#pragma unroll
  for (int64_t h = 0; h < min(ti.tile_out_h, os.h - 4 * hh); ++h) {
#pragma unroll
    for (int64_t w = 0; w < min(ti.tile_out_w, os.h - 4 * ww); ++w) {
      {
        if (hh * 4 + h < os.h && ww * 4 + w < os.w)
          out_tensor[(batch * os.oc + oc) * out_tensor_yz + (hh * 4 + h) * out_tensor_z +
                     (ww * 4 + w)] = Y_tensor[(oc * ti.num_tiles + tile) * Y_tensor_yz + h * Y_tensor_z + w];
        // tmp[h][w] = Y_tensor[(oc * ti.num_tiles + tile) * Y_tensor_yz + h * Y_tensor_z + w];
      }
    }
  }
  // assert(tmp[1][1] != 0);
  // cudaMemcpy2DAsync(out_tensor + (batch * os.oc + oc) * out_tensor_yz + (hh * 4) * out_tensor_z + ww * 4,
  //                   os.w * sizeof(float),
  //                   tmp,
  //                   TILE_OUT_W * sizeof(float),
  //                   min(TILE_OUT_W, os.w - ww * 4) * sizeof(float),
  //                   min(TILE_OUT_H, os.h - hh * 4),
  //                   cudaMemcpyDeviceToHost,
  //                   0);
}

void alloc_M_Tensor_Memory(cudaPitchedPtr& M_tensor, V_shape_t vs, U_shape_t us, tiling_info_t ti) {
  cudaPitchedPtr device_M_tensor;
  cudaExtent device_M_tensor_extent = make_cudaExtent(
      vs.num_tiles * sizeof(float) * us.oc, ti.tile_in_w, ti.tile_in_h);
  cudaMalloc3D(&device_M_tensor, device_M_tensor_extent);
  M_tensor = device_M_tensor;
}

void device_output_transform(cudaPitchedPtr device_M_tensor,  // input tensor
                             float* __restrict__ out,         // output tensor
                             const tiling_info_t ti,
                             const int64_t collapsed_dim_size,
                             const U_shape_t us,
                             const V_shape_t vs,
                             const out_shape_t os,
                             Device_Memory_Pool& device_Memory_Pool) {
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

  // 分配out_tensor内存
  // cudaPitchedPtr device_out_tensor;
  // cudaExtent device_out_tensor_extent = make_cudaExtent(
  //     sizeof(float) * (os.w + ti.tile_out_w), os.h + ti.tile_out_h, os.oc * os.bs);
  // //等待Y_tensor计算完成
  // cudaMalloc3D(&device_out_tensor, device_out_tensor_extent);
  float* device_out_tensor;
  device_out_tensor = out;
  // cudaHostGetDevicePointer(&device_out_tensor,out,0);
  cudaDeviceSynchronize();

  device_output_unpacking_store<<<DIV_UP(os.oc * ti.num_tiles, 1024), 1024>>>(
      device_Y_tensor, device_out_tensor, os, ti);
  device_Memory_Pool.free(device_M_tensor.ptr);
  cudaDeviceSynchronize();
  cudaStreamSynchronize(0);

  // 将Y_tensor复制回host
  // cudaMemcpy3DParms host_out_tensor_copy_parms = {0};
  // host_out_tensor_copy_parms.srcPtr = device_out_tensor;
  // host_out_tensor_copy_parms.dstPtr.ptr = out;
  // host_out_tensor_copy_parms.dstPtr.pitch = sizeof(float) * os.w;
  // host_out_tensor_copy_parms.dstPtr.ysize = os.h;
  // host_out_tensor_copy_parms.dstPtr.xsize = us.oc * os.bs;
  // host_out_tensor_copy_parms.extent = make_cudaExtent(
  //     sizeof(float) * os.w, os.h, os.oc * os.bs);  // device_out_tensor_extent;  //
  // host_out_tensor_copy_parms.kind = cudaMemcpyDeviceToHost;
  // cudaMemcpy3D(&host_out_tensor_copy_parms);

  //
  device_Memory_Pool.free(device_Y_tensor.ptr);
  // device_Memory_Pool.free(device_out_tensor.ptr);
  // cudaFree(device_Y_tensor.ptr);
  // cudaFree(device_M_tensor.ptr);
}