#include <stdio.h>
#include <inttypes.h>
#include <reg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <cassert>
enum {n = 32, iters = 1000};
struct Rec {
  uint32_t smid, warpid;
  uint64_t gt, ts[n];
};
__global__ void f(int *a, Rec *g, uint64_t *gsum) {
  Rec l;
  uint64_t ts[n], sum = 0;
  int k = 0, tid, s;
  reg_smid(r);
  for (i = 0; i < NT; i++) {
    reg_clock64(&ts[i]);
    sum += a[i];
  }
  // copy to l to g with memcopy
  // copy sum to gsum
}
int main(int argc, char **argv) {
  int bl, *a, *a2, h[n];
  size_t i, j;
  Rec *r, *rjunk, hr[th];
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  assert(prop.warpSize == th);
  assert(n == th);
  cudaMalloc(&a, n * sizeof *a);
  cudaMalloc(&a2, n * sizeof *a2);
  cudaMalloc(&rjunk, th * sizeof *rjunk);
  size_t flushbytes = 2 * (size_t)prop.l2CacheSize;
  char *flush;
  cudaMalloc(&flush, flushbytes);
  cudaMalloc(&r, th * sizeof *r);
  std::mt19937 rng(argv[1] ? atoi(argv[1]) : 0);
  bl = (n + th - 1) / th;
  for (j = 0; j < iters; j++) {
    std::iota(h, h + n, 0);
    std::shuffle(h, h + n, rng);
    cudaMemcpy(a, h, n * sizeof *a, cudaMemcpyHostToDevice);
    cudaMemset(flush, j, flushbytes);
    f<<<1, 1>>>(a2, rjunk);
    f<<<1, 1>>>(a, r);
    cudaMemcpy(hr, r, th * sizeof *r, cudaMemcpyDeviceToHost);
    printf("%" PRIu32 " %" PRIu32 " %" PRIu64 " ", hr[0].smid, hr[0].warpid,
           hr[0].gt);
    for (i = 1; i < NT; i++)
        printf("%" PRIu64 " ", hr[0].ts[i] - hr[0].ts[0]);
    printf("\n");
  }
  cudaFree(flush);
  cudaFree(rjunk);
  cudaFree(a2);
  cudaFree(r);
  cudaFree(a);
  return 0;
}
