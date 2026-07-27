#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <thread>
#include <chrono>

#include "reg.h"
#include "tick.h"
#include "arg.h"
struct Args {
  int n, stride;
  uint32_t target;
};
struct Rec {
  uint64_t id, clock64, lat;
  uint32_t smid, warpid;
};
__constant__ Args C;
__global__ void f(uint64_t *a, const uint32_t *idx, Rec *g_rec,
                  uint32_t *g_smid, uint32_t *g_warpid, int *claim) {
  uint32_t smid, warpid, off, id;
  uint64_t t0, t1, x;
  Rec r;
  int i;
  reg_smid(&smid);
  if (smid != C.target || atomicAdd(claim, 1) != 0)
    return;
  reg_warpid(&warpid);
  r.smid = smid;
  r.warpid = warpid;
#pragma unroll 1
  for (i = 0; i < C.n; i++) {
    id = idx[i];
    off = id * (uint32_t)C.stride;
    reg_clock64(&t0);
    x = __ldcg(&a[off]);
    tick_clock64((uint32_t)x, &t1);
    r.line = id;
    r.clk = t0;
    r.lat = t1 - t0;
    g_rec[i] = r;
  }
  *g_smid = smid;
  *g_warpid = warpid;
}
char *argv0;
static void usage(void) {
  fprintf(stderr,
          "usage: %s [-s smid] [-n lines] [-i iters] [-w stride] [-b base]\n",
          argv0);
  exit(2);
}
static int argint(const char *s) {
  char *e;
  long v;
  errno = 0;
  v = strtol(s, &e, 0);
  if (*s == '\0' || *e != '\0' || errno != 0)
    usage();
  return (int)v;
}
int main(int argc, char **argv) {
  uint64_t *a, clk0 = 0;
  Rec *g_rec, *h_rec;
  uint32_t *d_idx, *g_smid, *g_warpid, *h_idx, warp0 = 0, smid, warpid;
  int *claim, K, iters = -1, sm = -1, i, j;
  size_t flushbytes;
  char *flush, *base = 0, path[4096];
  FILE *raw = 0, *m;
  uint16_t bo = 1;
  cudaDeviceProp prop;
  Args ha = {-1, -1, 0};
  std::mt19937 rng(0);
  std::uniform_int_distribution<int> jitter(0, 16000);
  ARGBEGIN {
  case 's':
    sm = argint(EARGF(usage()));
    break;
  case 'n':
    ha.n = argint(EARGF(usage()));
    break;
  case 'i':
    iters = argint(EARGF(usage()));
    break;
  case 'w':
    ha.stride = argint(EARGF(usage()));
    break;
  case 'b':
    base = EARGF(usage());
    break;
  default:
    usage();
  } ARGEND;
  if (sm < 0 || ha.n < 1 || iters < 1 || ha.stride < 1 || !base)
    usage();
  ha.target = (uint32_t)sm;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaGetDeviceProperties failed\n");
    exit(2);
  }
  K = 4 * prop.multiProcessorCount;
  flushbytes = 2 * (size_t)prop.l2CacheSize;
  if (ha.target >= (uint32_t)prop.multiProcessorCount) {
    fprintf(stderr, "pchase: error: target SM %u >= %d\n", ha.target,
            prop.multiProcessorCount);
    exit(2);
  }
  if (cudaMalloc(&a, (size_t)ha.n * ha.stride * sizeof *a) != cudaSuccess ||
      cudaMalloc(&d_idx, ha.n * sizeof *d_idx) != cudaSuccess ||
      cudaMalloc(&g_rec, ha.n * sizeof *g_rec) != cudaSuccess ||
      cudaMalloc(&g_smid, sizeof *g_smid) != cudaSuccess ||
      cudaMalloc(&g_warpid, sizeof *g_warpid) != cudaSuccess ||
      cudaMalloc(&claim, sizeof *claim) != cudaSuccess ||
      cudaMalloc(&flush, flushbytes) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaMalloc failed\n");
    exit(2);
  }
  if (cudaMallocHost((void **)&h_idx, ha.n * sizeof *h_idx) != cudaSuccess ||
      cudaMallocHost((void **)&h_rec, ha.n * sizeof *h_rec) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaMallocHost failed\n");
    exit(2);
  }
  if (cudaMemcpyToSymbol(C, &ha, sizeof ha) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaMemcpyToSymbol failed\n");
    exit(2);
  }
  snprintf(path, sizeof path, "%s.raw", base);
  if (!(raw = fopen(path, "wb"))) {
    fprintf(stderr, "pchase: error: cannot open %s\n", path);
    exit(2);
  }
  std::iota(h_idx, h_idx + ha.n, 0u);
  for (j = 0; j < iters;) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(jitter(rng)));
    std::shuffle(h_idx, h_idx + ha.n, rng);
    if (cudaMemcpy(d_idx, h_idx, ha.n * sizeof *d_idx,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(flush, j, flushbytes) != cudaSuccess ||
        cudaMemset(claim, 0, sizeof *claim) != cudaSuccess ||
        cudaMemset(g_smid, 0xff, sizeof *g_smid) != cudaSuccess) {
      fprintf(stderr, "pchase: error: setup failed\n");
      exit(2);
    }
    f<<<K, 1>>>(a, d_idx, g_rec, g_smid, g_warpid, claim);
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
      fprintf(stderr, "pchase: error: kernel launch failed\n");
      exit(2);
    }
    if (cudaMemcpy(&smid, g_smid, sizeof smid,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      fprintf(stderr, "pchase: error: cudaMemcpy failed\n");
      exit(2);
    }
    if (smid == ha.target) {
      if (cudaMemcpy(&warpid, g_warpid, sizeof warpid,
                     cudaMemcpyDeviceToHost) != cudaSuccess ||
          cudaMemcpy(h_rec, g_rec, ha.n * sizeof *h_rec,
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "pchase: error: cudaMemcpy failed\n");
        exit(2);
      }
      if (j == 0) {
        clk0 = h_rec[0].clk;
        warp0 = warpid;
      }
      if (warpid == warp0) {
        for (i = 0; i < ha.n; i++) {
          uint64_t clk = h_rec[i].clk - clk0;
          if (h_rec[i].smid != ha.target || h_rec[i].warpid != warp0) {
            fprintf(stderr, "pchase: error: record smid %u warpid %u != %u %u\n",
                    h_rec[i].smid, h_rec[i].warpid, ha.target, warp0);
            exit(2);
          }
          if (fwrite(&h_rec[i].line, sizeof h_rec[i].line, 1, raw) != 1 ||
              fwrite(&clk, sizeof clk, 1, raw) != 1 ||
              fwrite(&h_rec[i].lat, sizeof h_rec[i].lat, 1, raw) != 1 ||
              fwrite(&h_rec[i].smid, sizeof h_rec[i].smid, 1, raw) != 1 ||
              fwrite(&h_rec[i].warpid, sizeof h_rec[i].warpid, 1, raw) != 1) {
            fprintf(stderr, "pchase: error: fwrite failed\n");
            exit(2);
          }
        }
        j++;
      }
    }
  }
  fclose(raw);
  snprintf(path, sizeof path, "%s.meta", base);
  if (!(m = fopen(path, "w"))) {
    fprintf(stderr, "pchase: error: cannot open %s\n", path);
    exit(2);
  }
  fprintf(m, "rows %ld\nendian %s\nline u64\nclk u64\nlat u64\nsmid u32\nwarpid u32\n",
          (long)iters * ha.n, *(char *)&bo ? "little" : "big");
  fclose(m);
  if (cudaFree(a) != cudaSuccess ||
      cudaFree(flush) != cudaSuccess ||
      cudaFree(d_idx) != cudaSuccess ||
      cudaFree(g_rec) != cudaSuccess ||
      cudaFree(g_smid) != cudaSuccess ||
      cudaFree(g_warpid) != cudaSuccess ||
      cudaFree(claim) != cudaSuccess ||
      cudaFreeHost(h_idx) != cudaSuccess ||
      cudaFreeHost(h_rec) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaFree failed\n");
    exit(2);
  }
}
