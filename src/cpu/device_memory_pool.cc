#include "device_memory_pool.h"

#include <cuda_runtime.h>

using std::queue;
using std::thread;
Device_Memory_Pool::Device_Memory_Pool() {}

Device_Memory_Pool::~Device_Memory_Pool() {
  while (!memory_free_queue.empty()) {
    memory_free_queue.front().join();
    memory_free_queue.pop();
  }
}

// void Device_Memory_Pool::free(void* ptr) {
//   cudaFree(ptr);
//   // this->memory_free_queue.push(thread(cudaFree, ptr));
// }

// Let pre-allocate 17GB of VRAM and assume it is sufficient
void Device_Memory_Pool::init() {
  cudaError_t err = cudaMalloc(&startPtr, sizeof(char) * ((size_t)1 << (size_t)33));
  nextFree = startPtr;
}

void Device_Memory_Pool::poolMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) {
  pitchedDevPtr->ptr = nextFree;
  // implement 16 byte alignment
  pitchedDevPtr->pitch = ROUND_UP(extent.width, 16);
  pitchedDevPtr->ysize = extent.height;
  pitchedDevPtr->xsize = extent.width;
  nextFree = (char*)nextFree + pitchedDevPtr->pitch * pitchedDevPtr->ysize * extent.depth;
}

void Device_Memory_Pool::poolFree() { nextFree = startPtr; }