#include <cuda_runtime.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "arg.h"

__global__ static void nop(void) {}

static uint64_t ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000000000LL + t.tv_nsec;
}

static void jitter(void) {
  struct timespec t;
  t.tv_sec = 0;
  t.tv_nsec = rand() % 16001;
  nanosleep(&t, NULL);
}

char *argv0;
static void usage(void) {
  fprintf(stderr, "usage: %s [-n measurements] [-o base]\n", argv0);
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
  int n = -1, i;
  uint64_t t0, t1, lat;
  const char *base = 0;
  char path[4096];
  FILE *f, *m;
  uint16_t bo = 1;
  cudaError_t err;

  ARGBEGIN {
  case 'n':
    n = argint(EARGF(usage()));
    break;
  case 'o':
    base = EARGF(usage());
    break;
  default:
    usage();
  }
  ARGEND;
  if (n < 1 || !base)
    usage();

  snprintf(path, sizeof path, "%s.raw", base);
  if (!(f = fopen(path, "wb"))) {
    fprintf(stderr, "launch: error: cannot open %s\n", path);
    exit(2);
  }
  srand(0);
  if ((err = cudaSetDeviceFlags(cudaDeviceScheduleSpin)) != cudaSuccess) {
    fprintf(stderr, "launch: error: %s\n", cudaGetErrorString(err));
    exit(2);
  }
  nop<<<1, 1>>>();
  if ((err = cudaGetLastError()) != cudaSuccess ||
      (err = cudaDeviceSynchronize()) != cudaSuccess) {
    fprintf(stderr, "launch: error: %s\n", cudaGetErrorString(err));
    exit(2);
  }
  for (i = 0; i < n; i++) {
    jitter();
    t0 = ns();
    nop<<<1, 1>>>();
    err = cudaDeviceSynchronize();
    t1 = ns();
    lat = t1 - t0;
    if (err != cudaSuccess || (err = cudaGetLastError()) != cudaSuccess) {
      fprintf(stderr, "launch: error: %s\n", cudaGetErrorString(err));
      exit(2);
    }
    if (fwrite(&t0, sizeof t0, 1, f) != 1 ||
        fwrite(&lat, sizeof lat, 1, f) != 1) {
      fprintf(stderr, "launch: error: cannot write %s\n", path);
      exit(2);
    }
  }
  if (fclose(f)) {
    fprintf(stderr, "launch: error: cannot write %s\n", path);
    exit(2);
  }
  snprintf(path, sizeof path, "%s.meta", base);
  if (!(m = fopen(path, "w"))) {
    fprintf(stderr, "launch: error: cannot open %s\n", path);
    exit(2);
  }
  if (fprintf(m, "rows %d\nendian %s\nt0 u8\nlat u8\n", n,
              *(char *)&bo ? "little" : "big") < 0 ||
      fclose(m)) {
    fprintf(stderr, "launch: error: cannot write %s\n", path);
    exit(2);
  }
}
