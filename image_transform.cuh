#ifndef __IMAGE_TRANSFORM_CUH__
#define __IMAGE_TRANSFORM_CUH__
#include <cuda_runtime.h>
#include <stdio.h>

#include "utils.h"

__device__ inline tile_index_t device_get_tile_index(int64_t tile, tiling_info_t ts);

__global__ void image_packing(const cudaPitchedPtr *device_image,
                              const cudaPitchedPtr *device_packed_image,
                              const image_shape_t is,
                              const tiling_info_t ti);

void device_image_transform(float *__restrict__ image,
                            float *__restrict__ V,
                            const image_shape_t is,
                            const tiling_info_t ti,
                            const V_shape_t vs);
#endif