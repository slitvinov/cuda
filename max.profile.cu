#include <stdio.h>
#include <inttypes.h>
#include <reg.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <cassert>
enum {n = 32, th = 32, warm = 100, iters = 1000, NT = 10};
__global__ void f(int *a, int64_t *t) {
  uint64_t ts[NT];
  uint32_t smid;
  int k = 0, tid, s;
  __shared__ int b[n], c[n];
#define TS() reg_clock64(&ts[k++])
  reg_smid(&smid);
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
    t[0] = smid;
    for (int i = 0; i < NT - 1; i++)
      if (i < k)
	t[i + 1] = ts[i];
  }
}
int main(int argc, char **argv) {
  int bl, *a, h[n];
  size_t i, j;
  int64_t *t, *ht;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  assert(prop.warpSize == th);
  assert(n == th);
  cudaMalloc(&a, n * sizeof *a);
  cudaMalloc(&t, (warm + iters) * NT * sizeof *t);
  cudaMemset(t, 0xff, (warm + iters) * NT * sizeof *t);
  std::mt19937 rng(argv[1] ? atoi(argv[1]) : 0);
  std::iota(h, h + n, 0);
  std::shuffle(h, h + n, rng);
  bl = (n + th - 1) / th;
  for (j = 0; j < warm + iters; j++) {
    cudaMemcpy(a, h, n * sizeof *a, cudaMemcpyHostToDevice);
    f<<<bl, th>>>(a, t + j * NT);
  }
  ht = (int64_t *)malloc((warm + iters) * NT * sizeof *ht);
  cudaMemcpy(ht, t, (warm + iters) * NT * sizeof *t, cudaMemcpyDeviceToHost);
  for (j = warm; j < warm + iters; j++) {
    printf("%" PRId64 " ", ht[j * NT]);
    for (i = 2; i < NT; i++)
      if (ht[j * NT + i] != -1)
	printf("%" PRId64 " ", ht[j * NT + i] - ht[j * NT + 1]);
    printf("\n");
  }
  free(ht);
  cudaFree(t);
  cudaFree(a);
  return 0;
}
