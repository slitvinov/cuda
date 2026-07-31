#include <stdio.h>
static __global__ void kernel() {
  printf("threadIdx.x=%d\n", threadIdx.x);
}
int main() {
  cudaError_t err;
  kernel<<<1, 2>>>();
  if ((err = cudaGetLastError()) != cudaSuccess)
    goto fail;
  if ((err = cudaDeviceSynchronize()) != cudaSuccess)
    goto fail;  
  return 0;
 fail:
    fprintf(stderr, "smoke: error: %s\n", cudaGetErrorString(err));
    exit(2);
}
