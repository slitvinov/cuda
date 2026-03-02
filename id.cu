#include <cuda_runtime.h>
#include <nvml.h>
#include <stdio.h>

int main() {
  cudaDeviceProp prop;
  int count, device;
  cudaUUID_t u;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    fprintf(stderr, "cudaGetDeviceCount failed\n");
    return 1;
  }
  for (device = 0; device < count; device++) {
    cudaGetDeviceProperties(&prop, device);
    u = prop.uuid;
    printf("id.cu: %s (UUID: "
           "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%"
           "02x%02x%02x)\n",
           prop.name, (unsigned char)u.bytes[0], (unsigned char)u.bytes[1],
           (unsigned char)u.bytes[2], (unsigned char)u.bytes[3],
           (unsigned char)u.bytes[4], (unsigned char)u.bytes[5],
           (unsigned char)u.bytes[6], (unsigned char)u.bytes[7],
           (unsigned char)u.bytes[8], (unsigned char)u.bytes[9],
           (unsigned char)u.bytes[10], (unsigned char)u.bytes[11],
           (unsigned char)u.bytes[12], (unsigned char)u.bytes[13],
           (unsigned char)u.bytes[14], (unsigned char)u.bytes[15]);
  }
}
