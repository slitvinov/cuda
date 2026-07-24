#include <stdio.h>
#include <inttypes.h>
#include <reg.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <cassert>
enum {n = 32, th = 32, warm = 100, iters = 10000000, NT = 10};
__global__ void f(int *a, int64_t *t) {
  uint64_t ts[NT];
  uint32_t smid, warpid;
  int k = 0, tid, s;
  __shared__ int b[n], c[n];
#define TS() reg_clock64(&ts[k++])
  reg_smid(&smid);
  reg_warpid(&warpid);
  TS();
  tid = blockIdx.x * blockDim.x + threadIdx.x;
  TS();
  b[tid] = a[tid];
  TS();
  c[tid] = tid;
  TS();
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
  TS();
  if (tid == 0) {
    a[0] = b[0];
    a[1] = c[0];
  }
  t[tid * NT] = smid;
  t[tid * NT + 1] = warpid;
  for (int i = 0; i < NT - 2; i++)
    if (i < k)
      t[tid * NT + i + 2] = ts[i];
}
int main(int argc, char **argv) {
  int bl, *a, *a2, h[n];
  size_t i, j;
  int64_t *t, *tjunk;
  int64_t ht[th * NT];
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  assert(prop.warpSize == th);
  assert(n == th);
  cudaMalloc(&a, n * sizeof *a);
  cudaMalloc(&a2, n * sizeof *a2);
  cudaMalloc(&tjunk, th * NT * sizeof *tjunk);
  size_t flushbytes = 2 * (size_t)prop.l2CacheSize;
  char *flush;
  cudaMalloc(&flush, flushbytes);
  cudaMalloc(&t, th * NT * sizeof *t);
  std::mt19937 rng(argv[1] ? atoi(argv[1]) : 0);
  bl = (n + th - 1) / th;
  for (j = 0; j < warm + iters; j++) {
    std::iota(h, h + n, 0);
    std::shuffle(h, h + n, rng);
    cudaMemcpy(a, h, n * sizeof *a, cudaMemcpyHostToDevice);
    cudaMemset(flush, j, flushbytes);
    cudaMemset(t, 0xff, th * NT * sizeof *t);
    f<<<bl, th>>>(a2, tjunk);
    f<<<bl, th>>>(a, t);
    cudaMemcpy(ht, t, th * NT * sizeof *t, cudaMemcpyDeviceToHost);
    if (j < warm)
      continue;
    for (int lane = 1; lane < th; lane++)
      for (int slot = 0; slot < NT; slot++)
        assert(ht[lane * NT + slot] == ht[slot]);
    printf("%" PRId64 " %" PRId64 " ", ht[0], ht[1]);
    for (i = 3; i < NT; i++)
      if (ht[i] != -1)
        printf("%" PRId64 " ", ht[i] - ht[2]);
    printf("\n");
  }
  cudaFree(flush);
  cudaFree(tjunk);
  cudaFree(a2);
  cudaFree(t);
  cudaFree(a);
  return 0;
}
