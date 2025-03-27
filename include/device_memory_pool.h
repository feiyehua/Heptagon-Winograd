#ifndef DEVICE_MEMORY_POOL_H
#define DEVICE_MEMORY_POOL_H

#include <queue>
#include <thread>


class Device_Memory_Pool {
 public:
  Device_Memory_Pool();
  ~Device_Memory_Pool();

  void free(void* ptr);

 private:
  std::queue<std::thread> memory_free_queue;
};

#endif  // DEVICE_MEMORY_POOL_H
