#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

struct Rec {
  uint64_t t0;
  uint64_t lat;
};

int main(int argc, char **argv) {
  int n = 1000000, i;
  uint64_t t0, t1;
  struct Rec *rec;
  const char *base = argc > 1 ? argv[1] : "launch";
  char path[4096];
  FILE *f;
  uint16_t bo = 1;
  cudaError_t err;

  if (!(rec = (struct Rec *)malloc((size_t)n * sizeof *rec))) {
    fprintf(stderr, "launch: error: malloc failed\n");
    exit(2);
  }
  srand(0);
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
    rec[i].t0 = t0;
    rec[i].lat = t1 - t0;
    if (err != cudaSuccess || (err = cudaGetLastError()) != cudaSuccess) {
      fprintf(stderr, "launch: error: %s\n", cudaGetErrorString(err));
      exit(2);
    }
  }
  snprintf(path, sizeof path, "%s.raw", base);
  if (!(f = fopen(path, "wb")) ||
      fwrite(rec, sizeof *rec, (size_t)n, f) != (size_t)n || fclose(f)) {
    fprintf(stderr, "launch: error: cannot write %s\n", path);
    exit(2);
  }
  snprintf(path, sizeof path, "%s.meta", base);
  if (!(f = fopen(path, "w"))) {
    fprintf(stderr, "launch: error: cannot write %s\n", path);
    exit(2);
  }
  fprintf(f, "rows %d\nendian %s\nt0 u8\nlat u8\n", n,
          *(char *)&bo ? "little" : "big");
  fclose(f);
  free(rec);
}
