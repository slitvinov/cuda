#include <stdio.h>

__global__ void kernel() {
  printf("Hello World from GPU!\n");
}

int main() {
  int gridSize = 1;
  int blockSize = 10;
  kernel<<<gridSize, blockSize>>>();
  cudaDeviceSynchronize();
  printf("Hello World from CPU!\n");
}
