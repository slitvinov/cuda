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
struct Rec {
  uint64_t *id, *clock64, *lat;
  uint32_t smid, warpid;
};
#define ROWFIELDS                                                              \
  X(uint64_t, id)                                                              \
  X(uint64_t, clock64)                                                         \
  X(uint64_t, lat)                                                             \
  X(uint32_t, smid)                                                            \
  X(uint32_t, warpid)                                                          \
  X(uint32_t, iter)
struct Row {
#define X(t, nm) t nm;
  ROWFIELDS
#undef X
};
static const struct {
  const char *name;
  size_t off, size;
} cols[] = {
#define X(t, nm) {#nm, offsetof(Row, nm), sizeof(t)},
    ROWFIELDS
#undef X
};
__constant__ Args C;
__global__ void f(uint64_t *a, const uint32_t *idx, Rec *g, int *claim) {
  uint32_t smid, warpid, off, id;
  uint64_t t0, t1, x;
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
    reg_clock64(&t0);
    x = __ldcg(&a[off]);
    tick_clock64((uint32_t)x, &t1);
    g->id[i] = id;
    g->clock64[i] = t0;
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
  uint64_t *a, clock0 = 0;
  uint64_t *d_id, *d_clock64, *d_lat, *h_id, *h_clock64, *h_lat;
  Rec rec, *g_rec;
  Row row;
  uint32_t *d_idx, *h_idx, warp0 = 0, smid, warpid;
  int *claim, K, iters = -1, sm = -1, i, j;
  size_t flushbytes, c;
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
      cudaMalloc(&d_id, ha.n * sizeof *d_id) != cudaSuccess ||
      cudaMalloc(&d_clock64, ha.n * sizeof *d_clock64) != cudaSuccess ||
      cudaMalloc(&d_lat, ha.n * sizeof *d_lat) != cudaSuccess ||
      cudaMalloc(&g_rec, sizeof *g_rec) != cudaSuccess ||
      cudaMalloc(&claim, sizeof *claim) != cudaSuccess ||
      cudaMalloc(&flush, flushbytes) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaMalloc failed\n");
    exit(2);
  }
  if (cudaMallocHost((void **)&h_idx, ha.n * sizeof *h_idx) != cudaSuccess ||
      cudaMallocHost((void **)&h_id, ha.n * sizeof *h_id) != cudaSuccess ||
      cudaMallocHost((void **)&h_clock64, ha.n * sizeof *h_clock64) != cudaSuccess ||
      cudaMallocHost((void **)&h_lat, ha.n * sizeof *h_lat) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaMallocHost failed\n");
    exit(2);
  }
  rec.id = d_id;
  rec.clock64 = d_clock64;
  rec.lat = d_lat;
  if (cudaMemcpyToSymbol(C, &ha, sizeof ha) != cudaSuccess ||
      cudaMemcpy(g_rec, &rec, sizeof rec, cudaMemcpyHostToDevice) != cudaSuccess) {
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
        cudaMemset(&g_rec->smid, 0xff, sizeof rec.smid) != cudaSuccess) {
      fprintf(stderr, "pchase: error: setup failed\n");
      exit(2);
    }
    f<<<K, 1>>>(a, d_idx, g_rec, claim);
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
      fprintf(stderr, "pchase: error: kernel launch failed\n");
      exit(2);
    }
    if (cudaMemcpy(&smid, &g_rec->smid, sizeof smid,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      fprintf(stderr, "pchase: error: cudaMemcpy failed\n");
      exit(2);
    }
    if (smid == ha.target) {
      if (cudaMemcpy(&warpid, &g_rec->warpid, sizeof warpid,
                     cudaMemcpyDeviceToHost) != cudaSuccess ||
          cudaMemcpy(h_id, d_id, ha.n * sizeof *h_id,
                     cudaMemcpyDeviceToHost) != cudaSuccess ||
          cudaMemcpy(h_clock64, d_clock64, ha.n * sizeof *h_clock64,
                     cudaMemcpyDeviceToHost) != cudaSuccess ||
          cudaMemcpy(h_lat, d_lat, ha.n * sizeof *h_lat,
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "pchase: error: cudaMemcpy failed\n");
        exit(2);
      }
      if (j == 0) {
        clock0 = h_clock64[0];
        warp0 = warpid;
      }
      if (warpid == warp0) {
        row.smid = smid;
        row.warpid = warpid;
        row.iter = (uint32_t)j;
        for (i = 0; i < ha.n; i++) {
          row.id = h_id[i];
          row.clock64 = h_clock64[i] - clock0;
          row.lat = h_lat[i];
          for (c = 0; c < sizeof cols / sizeof *cols; c++)
            if (fwrite((char *)&row + cols[c].off, cols[c].size, 1, raw) != 1) {
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
  fprintf(m, "rows %ld\nendian %s\n", (long)iters * ha.n,
          *(char *)&bo ? "little" : "big");
  for (c = 0; c < sizeof cols / sizeof *cols; c++)
    fprintf(m, "%s u%zu\n", cols[c].name, cols[c].size);
  fclose(m);
  if (cudaFree(a) != cudaSuccess ||
      cudaFree(flush) != cudaSuccess ||
      cudaFree(d_idx) != cudaSuccess ||
      cudaFree(d_id) != cudaSuccess ||
      cudaFree(d_clock64) != cudaSuccess ||
      cudaFree(d_lat) != cudaSuccess ||
      cudaFree(g_rec) != cudaSuccess ||
      cudaFree(claim) != cudaSuccess ||
      cudaFreeHost(h_idx) != cudaSuccess ||
      cudaFreeHost(h_id) != cudaSuccess ||
      cudaFreeHost(h_clock64) != cudaSuccess ||
      cudaFreeHost(h_lat) != cudaSuccess) {
    fprintf(stderr, "pchase: error: cudaFree failed\n");
    exit(2);
  }
}
