#include <stdio.h>

__global__ void kernel() {
  printf("threadIdx.x=%d\n", threadIdx.x);
}

int main() {
  kernel<<<1, 8>>>();
  cudaDeviceSynchronize();
}
