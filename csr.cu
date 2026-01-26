#include <stdio.h>

enum { blocksize = 256 };
static float row(int size, int *Aj, float *Av, float *x) {
  float sum = 0;
  int i;
  for (i = 0; i < size; i++)
    sum += Av[i] * x[Aj[i]];
  return sum;
}

static void mul(int n, int *Ap, int *Aj, float *Av, float *x, float *y) {
  int i, j, k;
  for (i = 0; i < n; i++) {
    j = Ap[i];
    k = Ap[i + 1];
    y[i] = row(k - j, Aj + j, Av + j, x);
  }
}

__global__ void muld(int n, const int *Ap, const int *Aj, const float *Av,
                     const float *x, float *y) {
  __shared__ float cache[blocksize];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x < n)
    cache[threadIdx.x] = x[threadIdx.x];
  __syncthreads();
  if (i < n) {
    int j = Ap[i], k = Ap[i + 1];
    float sum = 0.0f;
    for (int p = j; p < k; ++p) {
      int col = Aj[p];
      float xv = (col < n) ? cache[col] : x[col];
      sum += Av[p] * xv;
    }
    y[i] = sum;
  }
}

int main() {
  enum { n = 4, nnz = 7 };
  int i;
  int h_Ap[n + 1] = {0, 2, 3, 5, 7};
  int h_Aj[nnz] = {0, 3, 2, 0, 2, 2, 3};
  float h_Av[nnz] = {3, 1, 2, 1, 4, 1, 1};
  float h_x[n] = {1, 2, 3, 4};
  float h_y[n];

  int *d_Ap, *d_Aj;
  float *d_Av, *d_x, *d_y;

  cudaMalloc(&d_Ap, (n + 1) * sizeof(int));
  cudaMalloc(&d_Aj, nnz * sizeof(int));
  cudaMalloc(&d_Av, nnz * sizeof(float));
  cudaMalloc(&d_x, n * sizeof(float));
  cudaMalloc(&d_y, n * sizeof(float));

  cudaMemcpy(d_Ap, h_Ap, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Av, h_Av, nnz * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

  mul(n, h_Ap, h_Aj, h_Av, h_x, h_y);
  for (i = 0; i < n; i++) {
    printf("%g ", h_y[i]);
  }
  printf("\n");

  int gridSize = (n + blocksize - 1) / blocksize;
  muld<<<gridSize, blocksize>>>(n, d_Ap, d_Aj, d_Av, d_x, d_y);
  cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
  for (i = 0; i < n; i++) {
    printf("%g ", h_y[i]);
  }
  printf("\n");
}
