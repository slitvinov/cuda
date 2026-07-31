#include <stdio.h>
#include <inttypes.h>
#include <cuda/math>
#include "reg.h"
enum { threads = 32, n = 40 };

static __global__ void kernel(float *a, float *b, float *c) {
  uint64_t tid = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n)
    c[tid] = a[tid] + b[tid];
}
int main() {
  cudaError_t err;
  printf("smid tid threadIdx.x blockIdx.x\n");
  kernel<<<2, 3>>>();
  if ((err = cudaGetLastError()) != cudaSuccess)
    goto fail;
  if ((err = cudaDeviceSynchronize()) != cudaSuccess)
    goto fail;  
  return 0;
 fail:
    fprintf(stderr, "smoke: error: %s\n", cudaGetErrorString(err));
    exit(2);
}
