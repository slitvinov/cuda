#include <stdio.h>

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    char uuid[40];
    cudaGetDeviceProperties(&prop, 0);
    printf("Device Name: %s\n", prop.name);
    printf("PCI Bus ID: %02x:%02x.%x\n",
           prop.pciBusID,
           prop.pciDeviceID,
           prop.pciDomainID);
    cudaDeviceGetUuid((cudaUUID_t*)uuid, device);
    printf("GPU UUID: ");
    for (int i = 0; i < 16; i++)
        printf("%02x", ((unsigned char*)uuid)[i]);
    printf("\n");
}
