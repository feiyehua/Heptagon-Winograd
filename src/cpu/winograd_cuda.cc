#include "winograd_cuda.h"

void winograd_cuda(
    float *__restrict__ image, /**< float [batch_num][input_channel_num][image_height][image_width] */
    const int image_height,
    const int image_width,
    const int input_channel_num,
    float *__restrict__ filter, /**< float [output_channel_num][input_channel_num][FLT_H][FLT_W] */
    const int output_channel_num,
    const int batch_num,
    float *__restrict__ out,
    Device_Memory_Pool &device_Memory_Pool,
    cublasHandle_t &handle,
    int cuda_device_num) {
  cudaSetDevice(cuda_device_num);
  float *device_out_tensor;
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

  //进行两次变换
  float *device_U_tensor = NULL;
  int ldu = 0;
  device_filter_transform(filter, fs, us, us.oc * us.ic, &device_U_tensor, &ldu, device_Memory_Pool);

  // parallel accelerate!
  // 150ms
  float *device_V_tensor;
  int ldv = 0;

  device_image_transform(image, is, ti, vs, &device_V_tensor, &ldv, device_Memory_Pool);

  cublas_sgemm(handle,
               device_U_tensor,
               us.ic,
               us.oc * ldu,
               device_V_tensor,
               vs.ic,
               ldv,
               (float*)device_M_tensor.ptr,
               vs.num_tiles,
               device_M_tensor.pitch / sizeof(float),
               us.oc,
               vs.num_tiles,
               us.ic,
               us,
               vs,
               ti);
  // 6000ms
  device_output_transform(
      device_M_tensor, device_out_tensor, out, ti, us.oc * vs.num_tiles, us, vs, os, device_Memory_Pool);
}