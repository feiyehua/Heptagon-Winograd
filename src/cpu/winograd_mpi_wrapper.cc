#include <mpi.h>

#include "device_memory_pool.h"
#include "utils.h"
#include "winograd_cuda.h"

void device_initialize(cublasHandle_t *handle, Device_Memory_Pool &device_Memory_Pool, int num) {
  cublasCreate(handle);
  device_Memory_Pool.init(num);
}

void winograd_mpi_wrapper(
    float *__restrict__ image, /**< float [batch_num][input_channel_num][image_height][image_width] */
    const int image_height,
    const int image_width,
    const int input_channel_num,
    float *__restrict__ filter, /**< float [output_channel_num][input_channel_num][FLT_H][FLT_W] */
    const int output_channel_num,
    const int batch_num,
    float *__restrict__ out,
    Device_Memory_Pool *device_Memory_Pool,
    cublasHandle_t *handle,
    int cuda_device_num,
    bool initialized) {
  if (!initialized) {
#pragma omp parallel for collapse(1)
    for (int i = 0; i < GPU_NUM; i++) {
      device_initialize(&handle[i], device_Memory_Pool[i], i);
    }
  } else {
    for (int i = 0; i < GPU_NUM; i++) {
      device_Memory_Pool[i].poolFree();
    }
  }

  initialized = 1;
#pragma omp parallel for collapse(1)
  for (int i = 0; i < GPU_NUM; i++) {
    int bn;
    if (i != GPU_NUM - 1) {
      bn = batch_num / GPU_NUM;
    } else {
      bn = batch_num / GPU_NUM + batch_num % GPU_NUM;
    }
    const int output_height = image_height - 2;
    const int output_width = image_width - 2;
    winograd_cuda(image + input_channel_num * image_height * image_width * batch_num / GPU_NUM * i,
                  image_height,
                  image_width,
                  input_channel_num,
                  filter,
                  output_channel_num,
                  bn,
                  out + output_channel_num * output_height * output_width * batch_num / GPU_NUM * i,
                  device_Memory_Pool[i],
                  handle[i],
                  i);
  }
}