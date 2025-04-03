#ifndef __WINOGRAD_CUDA__
#define __WINOGRAD_CUDA__
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_sgemm.cuh"
#include "device_memory_pool.h"
#include "filter_transform.cuh"
#include "image_transform.cuh"
#include "output_transform.cuh"
#include "utils.h"

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
    int cuda_device_num);

#endif