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
enum { n = 8192, iters = 10, STRIDE = 32 };
struct Rec {
  uint32_t smid, warpid;
  uint64_t gt[n], ts[n];
};
__global__ void f(uint32_t *a, Rec *g, uint64_t *gsum) {
  Rec l;
  uint64_t t0, t1, sum = 0;
  uint32_t x;
  int i;
  reg_smid(&l.smid);
  reg_warpid(&l.warpid);
#pragma unroll 1
  for (i = 0; i < n; i++) {
    reg_globaltimer(&l.gt[i]);
    reg_clock64(&t0);
    x = a[i * STRIDE];
    sum += x;
    reg_clock64(&t1);
    l.ts[i] = t1 - t0;
  }
  memcpy(g, &l, sizeof l);
  *gsum = sum;
}
int main(int argc, char **argv) {
  uint32_t *a, *a2;
  size_t i, j, flushbytes;
  Rec hr, *r, *rjunk;
  uint64_t *gsum, gt = 0;
  char *flush;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  flushbytes = 2 * (size_t)prop.l2CacheSize;
  cudaMalloc(&a, n * STRIDE * sizeof *a);
  cudaMalloc(&a2, n * STRIDE * sizeof *a2);
  cudaMalloc(&rjunk, sizeof *rjunk);
  cudaMalloc(&r, sizeof *r);
  cudaMalloc(&gsum, sizeof *gsum);
  cudaMalloc(&flush, flushbytes);
  std::mt19937 rng(argv[1] ? atoi(argv[1]) : 0);
  for (j = 0; j < iters; j++) {
    cudaMemset(flush, j, flushbytes);
    f<<<1, 1>>>(a2, rjunk, gsum);
    f<<<1, 1>>>(a, r, gsum);
    cudaMemcpy(&hr, r, sizeof hr, cudaMemcpyDeviceToHost);
    if (j == 0)
      gt = hr.gt[0];
    for (i = 0; i < n; i++)
      printf("%" PRIu64 " %" PRIu64 "\n", hr.gt[i] - gt, hr.ts[i]);
  }
  cudaFree(flush);
  cudaFree(rjunk);
  cudaFree(a2);
  cudaFree(r);
  cudaFree(a);
  cudaFree(gsum);
  return 0;
}
