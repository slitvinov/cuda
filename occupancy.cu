#include <iostream>
enum { manualBlockSize = 32 };
__global__ void square(int *array, int arrayCount) {
  extern __shared__ int dynamicSmem[];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < arrayCount) {
    array[idx] *= array[idx];
  }
}
static double reportPotentialOccupancy(void *kernel, int blockSize,
                                       size_t dynamicSMem) {
  cudaDeviceProp prop;
  int numBlocks;
  int activeWarps;
  int maxWarps;
  double occupancy;
  cudaGetDeviceProperties(&prop, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, blockSize,
                                                dynamicSMem);
  activeWarps = numBlocks * blockSize / prop.warpSize;
  maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
  occupancy = (double)activeWarps / maxWarps;
  return occupancy;
}
static int launchConfig(int *array, int arrayCount, bool automatic) {
  int blockSize;
  int minGridSize;
  int gridSize;
  size_t dynamicSMemUsage = 0;
  cudaEvent_t start;
  cudaEvent_t end;
  float elapsedTime;
  double potentialOccupancy;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  if (automatic) {
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)square,
                                       dynamicSMemUsage, arrayCount);
    std::cout << "Suggested block size: " << blockSize << std::endl
              << "Minimum grid size for maximum occupancy: " << minGridSize
              << std::endl;
  } else {
    blockSize = manualBlockSize;
  }
  gridSize = (arrayCount + blockSize - 1) / blockSize;
  cudaEventRecord(start);
  square<<<gridSize, blockSize, dynamicSMemUsage>>>(array, arrayCount);
  cudaEventRecord(end);
  cudaDeviceSynchronize();
  potentialOccupancy =
      reportPotentialOccupancy((void *)square, blockSize, dynamicSMemUsage);
  std::cout << "Potential occupancy: " << potentialOccupancy * 100 << "%"
            << std::endl;
  cudaEventElapsedTime(&elapsedTime, start, end);
  std::cout << "Elapsed time: " << elapsedTime << "ms" << std::endl;
  return 0;
}
int main() {
  int count = 1000000;
  int *array, *dArray;
  int size = count * sizeof(int);
  cudaMallocHost(&array, size);
  for (int i = 0; i < count; i += 1) {
    array[i] = i;
  }
  cudaMalloc(&dArray, size);
  cudaMemcpy(dArray, array, size, cudaMemcpyHostToDevice);
  for (int i = 0; i < count; i += 1) {
    array[i] = 0;
  }
  launchConfig(dArray, count, false);
  launchConfig(dArray, count, true);
  cudaMemcpy(array, dArray, size, cudaMemcpyDeviceToHost);
  cudaFree(dArray);
  cudaFreeHost(array);
}
