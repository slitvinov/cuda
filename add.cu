#include <stdio.h>
#include <inttypes.h>
#include "reg.h"
enum { th = 32, n = 40 };
static __managed__ float a[n], b[n], c[n];
static __global__ void kernel() {
  uint64_t tid = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n)
    c[tid] = a[tid] + b[tid];
}
int main() {
  cudaError_t err;
  uint64_t i;
  for (i = 0; i < n; i++) {
    a[i] = i;
    b[i] = 2 * i;
  }
  kernel<<< (n + th - 1) / th, th>>>();
  if ((err = cudaGetLastError()) != cudaSuccess)
    goto fail;
  if ((err = cudaDeviceSynchronize()) != cudaSuccess)
    goto fail;
  for (i = 0; i < n; i++)
    printf("%.1f ", c[i]);
  return 0;
 fail:
    fprintf(stderr, "smoke: error: %s\n", cudaGetErrorString(err));
    exit(2);
}
