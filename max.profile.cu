#include <stdio.h>
#include <cuda/cmath>
#include <inttypes.h>
#include <reg.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <cassert>
enum {n = 32, th = 32, iters = 100};
__global__ void f(int *a, int64_t *t) {
  int tid, s;
  uint64_t start, end;
  __shared__ int b[n], c[n];
  tid = blockIdx.x * blockDim.x + threadIdx.x;
  reg_clock64(&start);
  b[tid] = a[tid];
  c[tid] = tid;
  #pragma unroll
  for (s = n / 2; s > 0; s >>= 1) {
    __syncwarp();
    if (tid < s) {
      if (b[tid] < b[tid + s]) {
        b[tid] = b[tid + s];
        c[tid] = c[tid + s];
      }
    }
  }
  __syncwarp();
  reg_clock64(&end);
  if (tid == 0) {
    a[0] = b[0];
    a[1] = c[0];
    t[0] = end - start;
  }
}
int main(int argc, char **argv) {
  int bl, j, *a, h[n];
  int64_t *t, ht[iters];
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  assert(prop.warpSize == th);
  assert(n == th);
  cudaMalloc(&a, n * sizeof *a);
  cudaMalloc(&t, iters * sizeof *t);
  std::mt19937 rng(argv[1] ? atoi(argv[1]) : 0);
  for (j = 0; j < iters; j++) {
    std::iota(h, h + n, 0);
    std::shuffle(h, h + n, rng);
    bl = (n + th - 1) / th;
    cudaMemcpy(a, h, n * sizeof *a, cudaMemcpyHostToDevice);
    f<<<bl, th>>>(a, t + j);
  }
  cudaMemcpy(ht, t, iters * sizeof *t, cudaMemcpyDeviceToHost);
  for (j = 0; j < iters; j++)
    printf("%8" PRIu64 "\n", ht[j]);
  cudaFree(t);
  cudaFree(a);
  return 0;
}
