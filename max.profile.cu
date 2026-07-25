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
enum {n = 32, th = 32, warm = 100, iters = 1000000, NT = 10};
struct Rec {
  uint32_t smid, warpid;
  uint64_t gt, ts[NT];
};
__global__ void f(int *a, Rec *r) {
  uint64_t ts[NT], gt;
  uint32_t smid, warpid;
  int k = 0, tid, s;
  __shared__ int b[n], c[n];
#define TS() reg_clock64(&ts[k++])
  reg_smid(&smid);
  reg_warpid(&warpid);
  reg_globaltimer(&gt);
#pragma unroll
  for (int i = 0; i < NT; i++)
    ts[i] = ~0ULL;
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
  r[tid].smid = smid;
  r[tid].warpid = warpid;
  r[tid].gt = gt;
#pragma unroll
  for (int i = 0; i < NT; i++)
    r[tid].ts[i] = ts[i];
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
  for (j = 0; j < warm + iters; j++) {
    std::iota(h, h + n, 0);
    std::shuffle(h, h + n, rng);
    cudaMemcpy(a, h, n * sizeof *a, cudaMemcpyHostToDevice);
    cudaMemset(flush, j, flushbytes);
    f<<<bl, th>>>(a2, rjunk);
    f<<<bl, th>>>(a, r);
    cudaMemcpy(hr, r, th * sizeof *r, cudaMemcpyDeviceToHost);
    if (j < warm)
      continue;
    for (int lane = 1; lane < th; lane++)
      assert(memcmp(&hr[lane], &hr[0], sizeof(Rec)) == 0);
    printf("%" PRIu32 " %" PRIu32 " %" PRIu64 " ", hr[0].smid, hr[0].warpid,
           hr[0].gt);
    for (i = 1; i < NT; i++)
      if (hr[0].ts[i] != ~0ULL)
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
