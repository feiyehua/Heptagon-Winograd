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
  // cudaMemset(startPtr, 0, sizeof(char) * ((size_t)1 << (size_t)33));
  printf("init loc %lx,size %lu\n", startPtr, sizeof(char) * ((size_t)1 << (size_t)33));
  size_t avail, total;
  cudaMemGetInfo(&avail, &total);
  printf("avail,%lu,total,%lu\n", avail, total);
  nextFree = startPtr;
  // if (err != cudaSuccess) {
  //   std::cout << cudaGetErrorString(err) << std::endl;
  //   exit(-1);
  // }
}

void Device_Memory_Pool::poolMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) {
  pitchedDevPtr->ptr = nextFree;
  // implement 16 byte alignment
  pitchedDevPtr->pitch = ROUND_UP(extent.width, 16);
  pitchedDevPtr->ysize = extent.height;
  pitchedDevPtr->xsize = extent.width;
  // cudaMemset(nextFree, 0, pitchedDevPtr->pitch * pitchedDevPtr->ysize * extent.depth);
  nextFree = (char*)nextFree + pitchedDevPtr->pitch * pitchedDevPtr->ysize * extent.depth;
  printf("next free loc %lx,%d,%s\n", nextFree,__LINE__,__func__);
}

void Device_Memory_Pool::poolFree() {
  // cudaMemset(startPtr, 0, sizeof(char) * ((size_t)1 << (size_t)33));
  nextFree = startPtr;
  printf("next free loc %lx,%d,%s\n", nextFree, __LINE__, __func__);
}