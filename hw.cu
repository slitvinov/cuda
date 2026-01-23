#include <stdio.h>

__device__ __forceinline__ unsigned int lane_id() {
  unsigned int id;
  asm("mov.u32 %0, %laneid;" : "=r"(id));
  return id;
}

__device__ __forceinline__ unsigned int sm_id() {
  unsigned int id;
  asm("mov.u32 %0, %smid;" : "=r"(id));
  return id;
}

__global__ void kernel() {
  unsigned int lane = lane_id();
  unsigned int sm = sm_id();
  printf("threadIdx.x=%d lane=%u smid=%u\n", threadIdx.x, lane, sm);
}

int main() {
  int gridSize = 1;
  int blockSize = 256;
  kernel<<<gridSize, blockSize>>>();
  cudaDeviceSynchronize();
}
