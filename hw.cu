#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __forceinline__ uint32_t lane() {
  uint32_t id;
  asm("mov.u32 %0, %laneid;" : "=r"(id));
  return id;
}

__device__ __forceinline__ uint32_t sm() {
  uint32_t id;
  asm("mov.u32 %0, %smid;" : "=r"(id));
  return id;
}

__device__ __forceinline__ uint64_t clock64x () {
  uint64_t id;
  asm("mov.u64 %0, %clock64;" : "=l"(id));
  return id;
}

__global__ void kernel() {
  printf("%5" PRIu32 " %3" PRIu32 " %5" PRIu32 " %8" PRIu64 "\n",
       threadIdx.x, lane(), sm(), clock64x());
}

int main(int argc, char **argv) {
  enum { gridSize = 1 };
  int blockSize = 256;
  kernel<<<gridSize, blockSize>>>();
  cudaDeviceSynchronize();
}
