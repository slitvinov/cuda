#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>
#include <stddef.h>
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
#define FIELDS                                                                 \
  X(uint64_t, id, 1)                                                           \
  X(uint64_t, clock64, 1)                                                      \
  X(uint64_t, globaltimer, 1)                                                  \
  X(uint64_t, lat, 1)                                                          \
  X(uint32_t, warpid, 0)                                                       \
  X(uint32_t, iter, 0)
#define PTR1(t) t *
#define PTR0(t) t
struct Rec {
#define X(t, nm, arr) PTR##arr(t) nm;
  FIELDS
#undef X
  uint32_t smid;
};
static const struct Col {
  const char *name;
  size_t size;
  int arr;
} cols[] = {
#define X(t, nm, arr) {#nm, sizeof(t), arr},
    FIELDS
#undef X
};
enum { NCOL = sizeof cols / sizeof *cols };
__constant__ Args C;
__global__ void kernel(uint64_t *a, const uint32_t *idx, Rec *g, int *claim) {
  uint32_t smid, warpid, off, id;
  uint64_t t0, t1, x, gt;
  int i;
  reg_smid(&smid);
  if (smid != C.target || atomicAdd(claim, 1) != 0)
    return;
  reg_warpid(&warpid);
  g->smid = smid;
  g->warpid = warpid;
#pragma unroll 1
  for (i = 0; i < C.n; i++) {
    id = idx[i];
    off = id * (uint32_t)C.stride;
    reg_globaltimer(&gt);
    reg_clock64(&t0);
    x = __ldcg(&a[off]);
    tick_clock64((uint32_t)x, &t1);
    g->id[i] = id;
    g->clock64[i] = t0;
    g->globaltimer[i] = gt;
    g->lat[i] = t1 - t0;
  }
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
  uint64_t *a, clock0 = 0, *buf;
  Rec *g_rec;
  const void *src[NCOL];
  uint32_t *d_idx, *h_idx, warp0 = 0;
  long iters = -1;
  int *claim, K, sm = -1, i, j;
  size_t flushbytes, c, off, bufsz;
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
  bufsz = 0;
#define X(t, nm, arr) if (arr) bufsz += (size_t)ha.n * sizeof(t);
  FIELDS
#undef X
  if (cudaMalloc(&a, (size_t)ha.n * ha.stride * sizeof *a) != cudaSuccess ||
      cudaMalloc(&d_idx, ha.n * sizeof *d_idx) != cudaSuccess ||
      cudaMalloc(&claim, sizeof *claim) != cudaSuccess ||
      cudaMalloc(&flush, flushbytes) != cudaSuccess ||
      cudaMallocManaged(&g_rec, sizeof *g_rec) != cudaSuccess ||
      cudaMallocManaged(&buf, bufsz) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaMalloc failed\n");
    exit(2);
  }
  if (cudaMallocHost((void **)&h_idx, ha.n * sizeof *h_idx) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaMallocHost failed\n");
    exit(2);
  }
  off = 0;
#define SLICE1(t, nm)                                                          \
  g_rec->nm = (t *)((char *)buf + off);                                        \
  off += (size_t)ha.n * sizeof(t);
#define SLICE0(t, nm)
#define X(t, nm, arr) SLICE##arr(t, nm)
  FIELDS
#undef X
#undef SLICE0
#undef SLICE1
  c = 0;
#define SRC1(nm) g_rec->nm
#define SRC0(nm) &g_rec->nm
#define X(t, nm, arr) src[c++] = SRC##arr(nm);
  FIELDS
#undef X
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
    g_rec->smid = 0xffffffffu;
    if (cudaMemcpy(d_idx, h_idx, ha.n * sizeof *d_idx,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(flush, j, flushbytes) != cudaSuccess ||
        cudaMemset(claim, 0, sizeof *claim) != cudaSuccess) {
      fprintf(stderr, "pchase: error: setup failed\n");
      exit(2);
    }
    kernel<<<K, 1>>>(a, d_idx, g_rec, claim);
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
      fprintf(stderr, "pchase: error: kernel launch failed\n");
      exit(2);
    }
    if (g_rec->smid == ha.target) {
      if (j == 0) {
        clock0 = g_rec->clock64[0];
        warp0 = g_rec->warpid;
      }
      if (g_rec->warpid == warp0) {
        g_rec->iter = (uint32_t)j;
        for (i = 0; i < ha.n; i++)
          g_rec->clock64[i] -= clock0;
        for (i = 0; i < ha.n; i++)
          for (c = 0; c < NCOL; c++)
            if (fwrite((const char *)src[c] +
                           (cols[c].arr ? (size_t)i * cols[c].size : 0),
                       cols[c].size, 1, raw) != 1) {
              fprintf(stderr, "pchase: error: fwrite failed\n");
              exit(2);
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
  fprintf(m, "rows %ld\nendian %s\n", iters * ha.n,
          *(char *)&bo ? "little" : "big");
  for (c = 0; c < NCOL; c++)
    fprintf(m, "%s u%zu\n", cols[c].name, cols[c].size);
  fclose(m);
  if (cudaFree(a) != cudaSuccess ||
      cudaFree(flush) != cudaSuccess ||
      cudaFree(d_idx) != cudaSuccess ||
      cudaFree(buf) != cudaSuccess ||
      cudaFree(g_rec) != cudaSuccess ||
      cudaFree(claim) != cudaSuccess ||
      cudaFreeHost(h_idx) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaFree failed\n");
    exit(2);
  }
}
