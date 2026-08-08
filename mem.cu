#include <cuda_runtime.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "arg.h"

static uint64_t ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000000000LL + t.tv_nsec;
}

char *argv0;
static void usage(void) {
  fprintf(stderr, "usage: %s -n samples -s maxbytes -o base\n", argv0);
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
  int n = -1, smax = -1, i, nsz = 0;
  uint64_t t0, t1, lat, k = 0;
  size_t sizes[64], sz, buf, off, span;
  const char *base = 0;
  char path[4096];
  FILE *f, *m;
  uint16_t bo = 1;
  char *h;
  char *d;
  cudaError_t err;

  ARGBEGIN {
  case 'n':
    n = argint(EARGF(usage()));
    break;
  case 's':
    smax = argint(EARGF(usage()));
    break;
  case 'o':
    base = EARGF(usage());
    break;
  default:
    usage();
  }
  ARGEND;
  if (n < 1 || smax < 1 || !base)
    usage();

  for (sz = 1; sz <= (size_t)smax; sz <<= 1)
    sizes[nsz++] = sz;
  buf = (size_t)smax * 2;
  span = buf - (size_t)smax; /* offsets in [0, span) keep off+size <= buf */

  snprintf(path, sizeof path, "%s.raw", base);
  if (!(f = fopen(path, "wb"))) {
    fprintf(stderr, "mem: error: cannot open %s\n", path);
    exit(2);
  }
  if ((err = cudaSetDeviceFlags(cudaDeviceScheduleSpin)) != cudaSuccess ||
      (err = cudaMallocHost((void **)&h, buf)) != cudaSuccess ||
      (err = cudaMalloc((void **)&d, buf)) != cudaSuccess) {
    fprintf(stderr, "mem: error: %s\n", cudaGetErrorString(err));
    exit(2);
  }
  memset(h, 0, buf);
  /* warmup: touch the whole span once */
  if ((err = cudaMemcpy(d, h, (size_t)smax, cudaMemcpyHostToDevice)) !=
          cudaSuccess ||
      (err = cudaDeviceSynchronize()) != cudaSuccess) {
    fprintf(stderr, "mem: error: %s\n", cudaGetErrorString(err));
    exit(2);
  }
  for (i = 0; i < n; i++) {
    int j;
    for (j = 0; j < nsz; j++) {
      sz = sizes[j];
      off = (k * 128) % span; /* cold, 128B-strided line each copy */
      t0 = ns();
      err = cudaMemcpy(d + off, h + off, sz, cudaMemcpyHostToDevice);
      t1 = ns();
      lat = t1 - t0;
      if (err != cudaSuccess) {
        fprintf(stderr, "mem: error: %s\n", cudaGetErrorString(err));
        exit(2);
      }
      if (fwrite(&t0, sizeof t0, 1, f) != 1 ||
          fwrite(&sz, sizeof sz, 1, f) != 1 ||
          fwrite(&lat, sizeof lat, 1, f) != 1) {
        fprintf(stderr, "mem: error: cannot write %s\n", path);
        exit(2);
      }
      k++;
    }
  }
  if (fclose(f)) {
    fprintf(stderr, "mem: error: cannot write %s\n", path);
    exit(2);
  }
  snprintf(path, sizeof path, "%s.meta", base);
  if (!(m = fopen(path, "w"))) {
    fprintf(stderr, "mem: error: cannot open %s\n", path);
    exit(2);
  }
  if (fprintf(m, "rows %ld\nendian %s\nt0 u8\nsize u8\nlat u8\n",
              (long)n * nsz, *(char *)&bo ? "little" : "big") < 0 ||
      fclose(m)) {
    fprintf(stderr, "mem: error: cannot write %s\n", path);
    exit(2);
  }
  return 0;
}
