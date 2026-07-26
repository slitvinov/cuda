#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <thread>
#include <chrono>

#include "reg.h"
#include "tick.h"
#include "arg.h"
static __device__ __forceinline__ uint64_t load_cv_u64(const uint64_t *a) {
  uint64_t v;
  asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(v) : "l"(a) : "memory");
  return v;
}
__global__ void f(uint64_t *a, const uint32_t *idx, uint64_t *g_gt,
                  uint64_t *g_ts, uint32_t *g_smid, uint32_t *g_warpid, int n,
                  int stride, uint32_t target, int *claim) {
  extern __shared__ uint32_t sidx[];
  uint32_t smid, warpid, off;
  uint64_t t0, t1, gt, x;
  int i;
  reg_smid(&smid);
  if (smid != target || atomicAdd(claim, 1) != 0)
    return;
  reg_warpid(&warpid);
  for (i = 0; i < n; i++)
    sidx[i] = idx[i];
#pragma unroll 1
  for (i = 0; i < n; i++) {
    off = sidx[i] * (uint32_t)stride;
    reg_globaltimer(&gt);
    reg_clock64(&t0);
    x = load_cv_u64(a + off);
    tick_clock64((uint32_t)x, &t1);
    g_gt[i] = gt;
    g_ts[i] = t1 - t0;
  }
  *g_smid = smid;
  *g_warpid = warpid;
}
char *argv0;
static void usage(void) {
  fprintf(stderr, "usage: %s smid n iters stride\n", argv0);
  exit(2);
}
static int argint(const char *s) {
  char *e;
  long v = strtol(s, &e, 0);
  if (*s == '\0' || *e != '\0')
    usage();
  return (int)v;
}
int main(int argc, char **argv) {
  uint64_t *a, *g_gt, *g_ts, *h_gt, *h_ts, gt = 0;
  uint32_t *d_idx, *g_smid, *g_warpid, *h_idx, target, warp0 = 0, smid, warpid;
  int *claim, K, n, iters, stride, i, j;
  cudaDeviceProp prop;
  std::mt19937 rng(0);
  std::uniform_int_distribution<int> jitter(0, 16000);
  ARGBEGIN {
  default:
    usage();
  } ARGEND;
  if (argc != 4)
    usage();
  target = (uint32_t)argint(argv[0]);
  n = argint(argv[1]);
  iters = argint(argv[2]);
  stride = argint(argv[3]);
  if (n < 1 || iters < 1 || stride < 1)
    usage();
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
  if (cudaMalloc(&a, (size_t)n * stride * sizeof *a) != cudaSuccess ||
      cudaMalloc(&d_idx, n * sizeof *d_idx) != cudaSuccess ||
      cudaMalloc(&g_gt, n * sizeof *g_gt) != cudaSuccess ||
      cudaMalloc(&g_ts, n * sizeof *g_ts) != cudaSuccess ||
      cudaMalloc(&g_smid, sizeof *g_smid) != cudaSuccess ||
      cudaMalloc(&g_warpid, sizeof *g_warpid) != cudaSuccess ||
      cudaMalloc(&claim, sizeof *claim) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaMalloc failed\n");
    exit(2);
  }
  if (cudaMallocHost((void **)&h_idx, n * sizeof *h_idx) != cudaSuccess ||
      cudaMallocHost((void **)&h_gt, n * sizeof *h_gt) != cudaSuccess ||
      cudaMallocHost((void **)&h_ts, n * sizeof *h_ts) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaMallocHost failed\n");
    exit(2);
  }
  if (cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)(n * sizeof(uint32_t))) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaFuncSetAttribute failed\n");
    exit(2);
  }
  std::iota(h_idx, h_idx + n, 0u);
  for (j = 0; j < iters;) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(jitter(rng)));
    std::shuffle(h_idx, h_idx + n, rng);
    cudaMemcpy(d_idx, h_idx, n * sizeof *d_idx, cudaMemcpyHostToDevice);
    cudaMemset(claim, 0, sizeof *claim);
    cudaMemset(g_smid, 0xff, sizeof *g_smid);
    f<<<K, 1, n * sizeof(uint32_t)>>>(a, d_idx, g_gt, g_ts, g_smid, g_warpid, n,
                                      stride, target, claim);
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
      fprintf(stderr, "pchase: error: kernel launch failed\n");
      exit(2);
    }
    cudaMemcpy(&smid, g_smid, sizeof smid, cudaMemcpyDeviceToHost);
    if (smid != target)
      continue;
    cudaMemcpy(&warpid, g_warpid, sizeof warpid, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gt, g_gt, n * sizeof *h_gt, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ts, g_ts, n * sizeof *h_ts, cudaMemcpyDeviceToHost);
    if (j == 0) {
      gt = h_gt[0];
      warp0 = warpid;
    }
    if (warpid == warp0) {
      for (i = 0; i < n; i++)
        printf("%d %" PRIu32 " %" PRIu64 " %" PRIu64 "\n", i, h_idx[i],
               h_gt[i] - gt, h_ts[i]);
      j++;
    }
  }
  if (cudaFree(a) != cudaSuccess ||
      cudaFree(d_idx) != cudaSuccess ||
      cudaFree(g_gt) != cudaSuccess ||
      cudaFree(g_ts) != cudaSuccess ||
      cudaFree(g_smid) != cudaSuccess ||
      cudaFree(g_warpid) != cudaSuccess ||
      cudaFree(claim) != cudaSuccess ||
      cudaFreeHost(h_idx) != cudaSuccess ||
      cudaFreeHost(h_gt) != cudaSuccess ||
      cudaFreeHost(h_ts) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaFree failed\n");
    exit(2);
  }
}
