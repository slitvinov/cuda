#include <inttypes.h>
#include <reg.h>
#include <stdint.h>
#include <stdio.h>

__global__ void kernel() {
  uint32_t laneid, smid, warpid;
  uint64_t clock64;
  reg_laneid(&laneid);
  reg_smid(&smid);
  reg_clock64(&clock64);
  reg_warpid(&warpid);

  printf("%5" PRIu32 " %3" PRIu32 " %3" PRIu32 " %5" PRIu32 " %8" PRIu64 "\n",
         threadIdx.x, laneid, warpid, smid, clock64);
}

int main(int argc, char **argv) {
  enum { gridSize = 1, blockSize = 256 };
  kernel<<<gridSize, blockSize>>>();
  cudaDeviceSynchronize();
}
