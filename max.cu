#include <stdio.h>
#include <cuda/cmath>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <cassert>
enum {n = 32, th = 32};
__global__ void f(int *a) {
  int tid, s;
  __shared__ int b[n], c[n];
  tid = blockIdx.x * blockDim.x + threadIdx.x;
  b[tid] = a[tid];
  c[tid] = tid;
  #pragma unroll
  for (s = 1; s < n; s <<= 1) {
    __syncwarp();
    if (tid + s < n) {
      if (b[tid] < b[tid + s]) {
        b[tid] = b[tid + s];
        c[tid] = c[tid + s];
      }
    }
  }
  if (tid == 0) {
    a[0] = b[0];
    a[1] = c[0];
  }
}
int main(int argc, char **argv) {
  int bl, i, *a;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  assert(prop.warpSize == th);
  assert(n == th);
  cudaMallocManaged(&a, n * sizeof *a);
  std::mt19937 rng(argv[1] ? atoi(argv[1]) : 0);
  std::iota(a, a + n, 0);
  std::shuffle(a, a + n, rng);
  for (i = 0; i < n; i++)
    printf("%02d ", a[i]);
  printf("\n");
  bl = (n + th - 1) / th;
  f<<<bl, th>>>(a);
  cudaDeviceSynchronize();
  assert(a[0] == n - 1);
  for (i = 0; i < n; i++)
    printf(i == a[1] ? "^^ " : "   ");
  printf("\n");  
  cudaFree(a);
  return 0;
}
