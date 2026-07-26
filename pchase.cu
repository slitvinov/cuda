#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <thread>
#include <chrono>
#include <time.h>
#include <assert.h>

#include "reg.h"
#include "tick.h"
#include "arg.h"
enum { n = 1 << 10, iters = 10000, STRIDE = 16 };
struct Rec {
  uint32_t smid, warpid;
  uint64_t gt[n], ts[n];
};
static __device__ __forceinline__ uint64_t load_cv_u64(const uint64_t *a) {
  uint64_t v;
  asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(v) : "l"(a) : "memory");
  return v;
}
__global__ void f(uint64_t *a, const uint32_t *idx, Rec *g, uint32_t target,
                  int *claim) {
  __shared__ uint32_t sidx[n];
  Rec l;
  uint32_t smid, warpid, off;
  uint64_t t0, t1, gt, i, x;
  reg_smid(&smid);
  if (smid != target || atomicAdd(claim, 1) != 0)
    return;
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
    x = load_cv_u64(a + off);
    tick_clock64((uint32_t)x, &t1);
    l.gt[i] = gt;
    l.ts[i] = t1 - t0;
  }
  *g = l;
}
char *argv0;
static void usage(void) {
  fprintf(stderr, "usage: %s [-s smid]\n", argv0);
  exit(2);
}
int main(int argc, char **argv) {
  uint64_t *a;
  uint32_t *d_idx, h_idx[n];
  uint64_t i, j, gt = 0;
  uint32_t target = 0, warp0 = 0;
  int *claim, K;
  Rec *hr, *r;
  cudaDeviceProp prop;
  ARGBEGIN {
  case 's':
    target = (uint32_t)atoi(EARGF(usage()));
    break;
  default:
    usage();
  } ARGEND;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaGetDeviceProperties failed\n");
    exit(2);
  }
  K = 4 * prop.multiProcessorCount;
  if (target >= (uint32_t)prop.multiProcessorCount) {
    fprintf(stderr, "pchase: error: target SM %u >= %d\n", target,
            prop.multiProcessorCount);
    exit(2);
  }
  if (cudaMalloc(&a, n * STRIDE * sizeof *a) != cudaSuccess ||
      cudaMalloc(&d_idx, n * sizeof *d_idx) != cudaSuccess ||
      cudaMalloc(&r, sizeof *r) != cudaSuccess ||
      cudaMalloc(&claim, sizeof *claim) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaMalloc failed\n");
    exit(2);
  }
  if (cudaMallocHost((void **)&hr, sizeof *hr) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaMallocHost failed\n");
    exit(2);
  }
  std::iota(h_idx, h_idx + n, 0u);
  std::mt19937 rng(0);
  std::uniform_int_distribution<int> jitter(0, 16000);
  for (j = 0; j < iters; ) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(jitter(rng)));
    std::shuffle(h_idx, h_idx + n, rng);
    cudaMemcpy(d_idx, h_idx, n * sizeof *d_idx, cudaMemcpyHostToDevice);
    cudaMemset(claim, 0, sizeof *claim);
    cudaMemset(r, 0xff, sizeof r->smid);
    f<<<K, 1>>>(a, d_idx, r, target, claim);
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
      fprintf(stderr, "pchase: error: kernel launch failed\n");
      exit(2);
    }
    cudaMemcpy(hr, r, sizeof *hr, cudaMemcpyDeviceToHost);
    if (hr->smid != target)
      continue;
    if (j == 0) {
      gt = hr->gt[0];
      warp0 = hr->warpid;
    }
    if (hr->warpid == warp0) {
      for (i = 0; i < n; i++)
	printf("%" PRIu64 " %" PRIu32 " %" PRIu64 " %" PRIu64 "\n", i, h_idx[i],
	       hr->gt[i] - gt, hr->ts[i]);
      j++;
    }
  }
  if (cudaFree(d_idx) != cudaSuccess ||
      cudaFree(r) != cudaSuccess ||
      cudaFree(claim) != cudaSuccess ||
      cudaFree(a) != cudaSuccess ||
      cudaFreeHost(hr) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaFree failed\n");
    exit(2);
  }
}
