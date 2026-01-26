#include <iostream>

enum { manualBlockSize = 32, count = 1000000, size = count * sizeof(int) };

__global__ void square(int *a, int n) {
  extern __shared__ int dynamicSmem[];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    a[i] *= a[i];
  }
}
int main() {
  cudaDeviceProp prop;
  int numBlocks, activeWarps, maxWarps, automatic, blockSize, minGridSize,
      gridSize, i, *array, *darray;
  size_t dynamicSMemUsage = 1 << 15;
  cudaEvent_t start, end;
  float elapsedTime;
  double potentialOccupancy;

  cudaGetDeviceProperties(&prop, 0);
  cudaMallocHost(&array, size);
  for (i = 0; i < count; i++) {
    array[i] = i;
  }
  cudaMalloc(&darray, size);
  cudaMemcpy(darray, array, size, cudaMemcpyHostToDevice);
  for (i = 0; i < count; i++) {
    array[i] = 0;
  }
  for (automatic = 0; automatic < 2; automatic++) {
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    if (automatic) {
      cudaOccupancyMaxPotentialBlockSize(
          &minGridSize, &blockSize, (void *)square, dynamicSMemUsage, count);
      std::cout << "Suggested block size: " << blockSize << std::endl
                << "Minimum grid size for maximum occupancy: " << minGridSize
                << std::endl;
    } else {
      blockSize = manualBlockSize;
    }
    gridSize = (count + blockSize - 1) / blockSize;
    cudaEventRecord(start);
    square<<<gridSize, blockSize, dynamicSMemUsage>>>(darray, count);
    cudaEventRecord(end);
    cudaDeviceSynchronize();
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, square, blockSize,
                                                  dynamicSMemUsage);
    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    potentialOccupancy = (double)activeWarps / maxWarps;

    std::cout << "Potential occupancy: " << potentialOccupancy * 100 << "%"
              << std::endl;
    cudaEventElapsedTime(&elapsedTime, start, end);
    std::cout << "Elapsed time: " << elapsedTime << "ms" << std::endl;
  }
  cudaFree(darray);
  cudaFreeHost(array);
}
