#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "tick.h"
enum { NUM_THREADS = 256 };

__global__ static void kernel(const float *input, float *output,
                              uint64_t *timer) {
  extern __shared__ float shared[];
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  if (tid == 0)
    tick_clock64(tid, &timer[bid]);
  shared[tid] = input[tid];
  shared[tid + blockDim.x] = input[tid + blockDim.x];
  for (int d = blockDim.x; d > 0; d /= 2) {
    __syncthreads();
    if (tid < d) {
      float f0 = shared[tid];
      float f1 = shared[tid + d];
      if (f1 < f0) {
        shared[tid] = f1;
      }
    }
  }
  if (tid == 0)
    output[bid] = shared[0];
  __syncthreads();
  if (tid == 0)
    tick_clock64((uint32_t)shared[0], &timer[bid + gridDim.x]);
}

int main(int argc, char **argv) {
  int num_block;
  float *dinput, *doutput, *input;
  uint64_t *dtimer, *timer;
  if (argc != 2) {
    fprintf(stderr, "usage: %s num_block\n", argv[0]);
    exit(2);
  }
  num_block = atoi(argv[1]);
  if (cudaMallocHost(&timer, 2 * num_block * sizeof *timer) != cudaSuccess ||
      cudaMallocHost(&input, NUM_THREADS * 2 * sizeof *input) != cudaSuccess ||
      cudaMalloc(&dinput, NUM_THREADS * 2 * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&doutput, num_block * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&dtimer, num_block * 2 * sizeof *dtimer) != cudaSuccess) {
    fprintf(stderr, "clock: error: cudaMalloc failed\n");
    exit(2);
  }
  for (int i = 0; i < NUM_THREADS * 2; i++) {
    input[i] = i;
  }
  if (cudaMemcpy(dinput, input, NUM_THREADS * 2 * sizeof(float),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "clock: error: HtoD memcpy failed\n");
    exit(2);
  }
  kernel<<<num_block, NUM_THREADS, sizeof(float) * 2 * NUM_THREADS>>>(
      dinput, doutput, dtimer);
  if (cudaGetLastError() != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "clock: error: kernel launch failed\n");
    exit(2);
  }
  if (cudaMemcpy(timer, dtimer, sizeof *timer * num_block * 2,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "clock: error: DtoH memcpy failed\n");
    exit(2);
  }
  uint64_t avgElapsedClocks = 0;
  for (int i = 0; i < num_block; i++) {
    avgElapsedClocks += timer[i + num_block] - timer[i];
  }
  if (cudaFreeHost(timer) != cudaSuccess ||
      cudaFreeHost(input) != cudaSuccess || cudaFree(dinput) != cudaSuccess ||
      cudaFree(doutput) != cudaSuccess || cudaFree(dtimer) != cudaSuccess) {
    fprintf(stderr, "clock: error: cudaFree failed\n");
    exit(2);
  }
  printf("% 10d % 25llu\n", num_block,
         (unsigned long long)(avgElapsedClocks / num_block));
}
