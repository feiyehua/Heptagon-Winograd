#ifndef DEVICE_MEMORY_POOL_H
#define DEVICE_MEMORY_POOL_H

#include <cuda_runtime.h>

#include <iostream>
#include <queue>
#include <thread>

#include "utils.h"

class Device_Memory_Pool {
 public:
  Device_Memory_Pool();
  ~Device_Memory_Pool();

  // void free(void* ptr);

  void init();

  void poolFree();
  void poolMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent);

 private:
  void* startPtr;
  void* nextFree;
  std::queue<std::thread> memory_free_queue;
};

#endif  // DEVICE_MEMORY_POOL_H
