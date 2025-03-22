#ifndef __FILTER_TRANSFORM_HPP__
#define __FILTER_TRANSFORM_HPP__
#include <cuda_runtime.h>
#include<stdio.h>
#include "utils.h"
__host__ void device_filter_transform(float *__restrict__ packed_filter,
                                        float *__restrict__ U,
                                        const filter_shape_t fs,
                                        const U_shape_t us,
                                        const int64_t collapsed_dim_size);
__global__ void thread_filter_transform(float *__restrict__ packed_filter,
                                        float *__restrict__ U,
                                        float *__restrict__ packed_U,
                                        const filter_shape_t fs,
                                        const U_shape_t us,
                                        const int64_t collapsed_dim_size);
#endif