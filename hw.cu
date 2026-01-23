#include <stdio.h>

__device__ __forceinline__ unsigned int lane_id() {
  unsigned int id;
  asm("mov.u32 %0, %laneid;" : "=r"(id));
  return id;
}

__global__ void kernel() {
  unsigned int lane = lane_id();
  printf("Hello from GPU: threadIdx.x=%d lane=%u\n",
         threadIdx.x, lane);
}

int main() {
  int gridSize = 1;
  int blockSize = 10;
  kernel<<<gridSize, blockSize>>>();
  cudaDeviceSynchronize();
  printf("Hello World from CPU!\n");
}
