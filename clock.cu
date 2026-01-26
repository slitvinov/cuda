#include <cuda_runtime.h>
#include <stdio.h>
enum { NUM_THREADS = 256 };

__global__ static void kernel(const float *input, float *output,
                              clock_t *timer) {
  extern __shared__ float shared[];
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  if (tid == 0)
    timer[bid] = clock();
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
    timer[bid + gridDim.x] = clock();
}

int main(int argc, char **argv) {
  int num_block;
  float *dinput, *doutput, *input;
  clock_t *dtimer, *timer;
  num_block = atoi(argv[1]);
  cudaMallocHost(&timer, 2 * num_block * sizeof *timer);
  cudaMallocHost(&input, 2 * num_block * sizeof *input);

  for (int i = 0; i < NUM_THREADS * 2; i++) {
    input[i] = i;
  }
  cudaMalloc(&dinput, NUM_THREADS * 2 * sizeof(float));
  cudaMalloc(&doutput, num_block * sizeof(float));
  cudaMalloc(&dtimer, num_block * 2 * sizeof(clock_t));
  cudaMemcpy(dinput, input, NUM_THREADS * 2 * sizeof(float),
             cudaMemcpyHostToDevice);
  kernel<<<num_block, NUM_THREADS, sizeof(float) * 2 * NUM_THREADS>>>(
      dinput, doutput, dtimer);
  cudaMemcpy(timer, dtimer, sizeof(clock_t) * num_block * 2,
             cudaMemcpyDeviceToHost);
  long double avgElapsedClocks = 0;
  for (int i = 0; i < num_block; i++) {
    avgElapsedClocks += timer[i + num_block] - timer[i];
  }

  cudaFreeHost(timer);
  cudaFreeHost(input);
  cudaFree(dinput);
  cudaFree(doutput);
  cudaFree(dtimer);
  avgElapsedClocks = avgElapsedClocks / num_block;
  printf("% 10d % 25.16Le\n", num_block, avgElapsedClocks);
}
