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

void Device_Memory_Pool::free(void* ptr) { this->memory_free_queue.push(thread(cudaFree, ptr)); }
