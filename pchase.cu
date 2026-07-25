#include <stdio.h>
#include <inttypes.h>
#include <reg.h>
#include <stdint.h>
#include <stdlib.h>
#include <algorithm>
#include <numeric>
#include <random>
enum { n = 1u << 10, iters = 100000, STRIDE = 32 };
struct Rec {
  uint32_t smid, warpid;
  uint64_t gt[n], ts[n];
};
__global__ void f(uint32_t *a, const uint32_t *idx, Rec *g, uint64_t *gsum) {
  __shared__ uint32_t sidx[n];
  Rec l;
  uint32_t smid, warpid, x, off;
  uint64_t t0, t1, gt, i, sum = 0;
  reg_smid(&smid);
  reg_warpid(&warpid);
  l.smid = smid;
  l.warpid = warpid;
  for (i = 0; i < n; i++)
    sidx[i] = idx[i];
#pragma unroll 1
  for (i = 0; i < n; i++) {
    off = sidx[i] * STRIDE;
    reg_globaltimer(&gt);
    reg_clock64(&t0);
    x = a[off];
    sum += x;
    reg_clock64(&t1);
    l.gt[i] = gt;
    l.ts[i] = t1 - t0;
  }
  *g = l;
  *gsum = sum;
}
int main() {
  uint32_t *a, *a2, *d_idx, h_idx[n];
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
      cudaMalloc(&d_idx, n * sizeof *d_idx) != cudaSuccess ||
      cudaMalloc(&r, sizeof *r) != cudaSuccess ||
      cudaMalloc(&gsum, sizeof *gsum) != cudaSuccess ||
      cudaMalloc(&flush, flushbytes) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaMalloc failed\n");
    exit(2);
  }
  if (cudaMallocHost((void **)&hr, sizeof *hr) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaMallocHost failed\n");
    exit(2);
  }
  std::iota(h_idx, h_idx + n, 0u);
  std::mt19937 rng(0);
  for (j = 0; j < iters; j++) {
    std::shuffle(h_idx, h_idx + n, rng);
    cudaMemcpy(d_idx, h_idx, n * sizeof *d_idx, cudaMemcpyHostToDevice);
    cudaMemset(flush, j, flushbytes);
    f<<<1, 1>>>(a2, d_idx, r, gsum);
    f<<<1, 1>>>(a, d_idx, r, gsum);
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
      fprintf(stderr, "pchase: error: kernel launch failed\n");
      exit(2);
    }
    cudaMemcpy(hr, r, sizeof *hr, cudaMemcpyDeviceToHost);
    if (j == 0)
      gt = hr->gt[0];
    for (i = 0; i < n; i++)
      printf("%" PRIu64 " %" PRIu32 " %" PRIu64 " %" PRIu64 "\n", i, h_idx[i],
             hr->gt[i] - gt, hr->ts[i]);
  }
  if (cudaFree(flush) != cudaSuccess ||
      cudaFree(a2) != cudaSuccess ||
      cudaFree(d_idx) != cudaSuccess ||
      cudaFree(r) != cudaSuccess ||
      cudaFree(a) != cudaSuccess ||
      cudaFree(gsum) != cudaSuccess ||
      cudaFreeHost(hr) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaFree failed\n");
    exit(2);
  }
}
