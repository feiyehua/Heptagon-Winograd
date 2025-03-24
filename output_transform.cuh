#ifndef __OUTPUT_TRANSFORM_CUH__
#define __OUTPUT_TRANSFORM_CUH__

#include <cuda_runtime.h>

#include "utils.h"
__global__ void output_transform(cudaPitchedPtr M,  // input tensor
                                 cudaPitchedPtr Y,  // output tensor
                                 const tiling_info_t ti,
                                 const int64_t collapsed_dim_size);

void device_output_transform(float* __restrict__ M,  // input tensor
                             float* __restrict__ Y,  // output tensor
                             const tiling_info_t ti,
                             const int64_t collapsed_dim_size,
                             const U_shape_t us,
                             const V_shape_t vs);
#endif