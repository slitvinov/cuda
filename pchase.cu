#include <stdio.h>
#include <inttypes.h>
#include <reg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
const uint64_t n = 1u << 20;
enum { iters = 10, STRIDE = 32 };
struct Rec {
  uint32_t smid, warpid;
  uint64_t gt[n], ts[n];
};
__global__ void f(uint32_t *a, Rec *g, uint64_t *gsum) {
  volatile Rec l;
  uint32_t smid, warpid;
  uint64_t t0, t1, gt, i, sum = 0;
  uint32_t x;
  reg_smid(&smid);
  reg_warpid(&warpid);
  l.smid = smid;
  l.warpid = warpid;
#pragma unroll 1
  for (i = 0; i < n; i++) {
    reg_globaltimer(&gt);
    reg_clock64(&t0);
    x = a[i * STRIDE];
    sum += x;
    reg_clock64(&t1);
    l.gt[i] = gt;
    l.ts[i] = t1 - t0;
  }
  memcpy(g, (const void *)&l, sizeof l);
  *gsum = sum;
}
int main() {
  uint32_t *a, *a2;
  uint64_t i, j, flushbytes, *gsum, gt = 0;
  Rec *hr, *r;
  char *flush;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaGetDeviceProperties failed\n");
    exit(2);
  }
  flushbytes = 2 * (size_t)prop.l2CacheSize;
  if (cudaMalloc(&a, n * STRIDE * sizeof *a) != cudaSuccess || 
      cudaMalloc(&a2, n * STRIDE * sizeof *a2) != cudaSuccess ||
      cudaMalloc(&r, sizeof *r) != cudaSuccess ||
      cudaMalloc(&gsum, sizeof *gsum)  != cudaSuccess ||
      cudaMalloc(&flush, flushbytes)  != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaMalloc failed\n");
    exit(2);
  }
  hr = (Rec *)malloc(sizeof *hr);
  if (!hr) {
    fprintf(stderr, "pchase: error: malloc failed\n");
    exit(2);
  }
  for (j = 0; j < iters; j++) {
    cudaMemset(flush, j, flushbytes);
    f<<<1, 1>>>(a2, r, gsum);
    f<<<1, 1>>>(a, r, gsum);
    cudaMemcpy(hr, r, sizeof *hr, cudaMemcpyDeviceToHost);
    if (j == 0)
      gt = hr->gt[0];
    for (i = 0; i < n; i++)
      printf("%" PRIu64 " %" PRIu64 "\n", hr->gt[i] - gt, hr->ts[i]);
  }
  if (cudaFree(flush) != cudaSuccess ||
      cudaFree(a2) != cudaSuccess ||
      cudaFree(r) != cudaSuccess ||
      cudaFree(a) != cudaSuccess ||
      cudaFree(gsum) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaFree failed\n");
    exit(2);
  }
  free(hr);
}
