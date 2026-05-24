/*
   Isolated Cholesky-solve kernel for performance analysis.

   No CGP context -- just the linear algebra.  Given a batch of G
   matrices J of shape [m, n] in fp32, for each individual:
     1. Build  H = JTJ + lambdaI       (n x n, fp64)
     2. Build  g = JT * 1          (n,    fp64)        synthetic RHS
     3. Cholesky factor  H = L LT
     4. Forward + back solve to produce delta

   Three kernels timed:
     full       -- steps 1..4 (matches chgp.cu)
     build_only -- steps 1 & 2 only      (the "parallel" portion)
     factor_only -- steps 3 & 4 only     (the "serial on tid 0" portion)

   Comparing ms(full) vs ms(build) + ms(factor) shows where time goes.
*/

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

__global__ void chol_full(const float *J,
                          double      *delta,
                          double       lam,
                          int m, int n)
{
    extern __shared__ double sm[];
    double *H  = sm;
    double *L  = H  + (size_t)n * n;
    double *gv = L  + (size_t)n * n;
    double *y  = gv + n;
    double *s  = y  + n;

    const int    g   = blockIdx.x;
    const int    tid = threadIdx.x;
    const int    B   = blockDim.x;
    const size_t Jb  = (size_t)g * m * n;

    for (int idx = tid; idx < n * n; idx += B) {
        int i = idx / n, j = idx % n;
        double v = 0.0;
        for (int k = 0; k < m; ++k) {
            v += (double)J[Jb + (size_t)k * n + i] *
                 (double)J[Jb + (size_t)k * n + j];
        }
        if (i == j) v += lam;
        H[i * n + j] = v;
    }
    for (int i = tid; i < n; i += B) {
        double v = 0.0;
        for (int k = 0; k < m; ++k) v += (double)J[Jb + (size_t)k * n + i];
        gv[i] = v;
    }
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                double v = H[i * n + j];
                for (int k = 0; k < j; ++k) v -= L[i*n + k] * L[j*n + k];
                L[i * n + j] = (i == j) ? sqrt(v) : v / L[j * n + j];
            }
        }
        for (int i = 0; i < n; ++i) {
            double v = gv[i];
            for (int k = 0; k < i; ++k) v -= L[i*n + k] * y[k];
            y[i] = v / L[i * n + i];
        }
        for (int i = n - 1; i >= 0; --i) {
            double v = y[i];
            for (int k = i + 1; k < n; ++k) v -= L[k*n + i] * s[k];
            s[i] = v / L[i * n + i];
        }
        for (int i = 0; i < n; ++i) delta[(size_t)g * n + i] = s[i];
    }
}

__global__ void chol_build_only(const float *J,
                                double      *H_out,
                                double      *g_out,
                                double       lam,
                                int m, int n)
{
    const int    g   = blockIdx.x;
    const int    tid = threadIdx.x;
    const int    B   = blockDim.x;
    const size_t Jb  = (size_t)g * m * n;
    const size_t Hb  = (size_t)g * n * n;
    const size_t gb  = (size_t)g * n;

    for (int idx = tid; idx < n * n; idx += B) {
        int i = idx / n, j = idx % n;
        double v = 0.0;
        for (int k = 0; k < m; ++k) {
            v += (double)J[Jb + (size_t)k * n + i] *
                 (double)J[Jb + (size_t)k * n + j];
        }
        if (i == j) v += lam;
        H_out[Hb + i * n + j] = v;
    }
    for (int i = tid; i < n; i += B) {
        double v = 0.0;
        for (int k = 0; k < m; ++k) v += (double)J[Jb + (size_t)k * n + i];
        g_out[gb + i] = v;
    }
}

__global__ void chol_factor_only(const double *H_in,
                                 const double *g_in,
                                 double       *delta,
                                 int n)
{
    extern __shared__ double sm[];
    double *L  = sm;
    double *gv = L + (size_t)n * n;
    double *y  = gv + n;
    double *s  = y + n;

    const int    g   = blockIdx.x;
    const int    tid = threadIdx.x;
    const int    B   = blockDim.x;
    const size_t Hb  = (size_t)g * n * n;
    const size_t gb  = (size_t)g * n;

    for (int idx = tid; idx < n * n; idx += B) L[idx] = H_in[Hb + idx];
    for (int i   = tid; i   < n;     i   += B) gv[i] = g_in[gb + i];
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                double v = L[i * n + j];
                for (int k = 0; k < j; ++k) v -= L[i*n + k] * L[j*n + k];
                L[i * n + j] = (i == j) ? sqrt(v) : v / L[j * n + j];
            }
        }
        for (int i = 0; i < n; ++i) {
            double v = gv[i];
            for (int k = 0; k < i; ++k) v -= L[i*n + k] * y[k];
            y[i] = v / L[i * n + i];
        }
        for (int i = n - 1; i >= 0; --i) {
            double v = y[i];
            for (int k = i + 1; k < n; ++k) v -= L[k*n + i] * s[k];
            s[i] = v / L[i * n + i];
        }
        for (int i = 0; i < n; ++i) delta[(size_t)g * n + i] = s[i];
    }
}

int main(void) {
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int clockKHz = 0;
    cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, 0);
    printf("GPU: %s (%d SMs, clock %.2f GHz)\n",
           prop.name, prop.multiProcessorCount, clockKHz * 1e-6);
    double fp64_fma_per_sec =
        prop.multiProcessorCount * 32.0 * (double)clockKHz * 1e3;
    printf("Estimated FP64 FMA throughput: %.1f GFMA/s  (= %.1f TFLOPs)\n\n",
           fp64_fma_per_sec * 1e-9, fp64_fma_per_sec * 2e-12);

    int G = 65536;
    int m = 256;
    int ns[] = {8, 16, 32, 48, 64};
    int reps = 5;

    printf("G = %d, m = %d, reps = %d\n\n", G, m, reps);
    printf("%4s  %10s  %10s  %10s  %10s  %12s  %10s\n",
           "n", "full_ms", "build_ms", "factor_ms",
           "compute_lb", "occupancy",  "smem KB");
    printf("%4s  %10s  %10s  %10s  %10s  %12s  %10s\n",
           "----", "----------", "----------", "----------", "----------",
           "------------", "----------");

    for (size_t ni = 0; ni < sizeof(ns)/sizeof(ns[0]); ++ni) {
        int n = ns[ni];
        size_t smem_full   = (size_t)(2 * n * n + 3 * n) * sizeof(double);
        size_t smem_factor = (size_t)(    n * n + 3 * n) * sizeof(double);

        if (smem_full > 100 * 1024) {
            printf("%4d  (smem over cap)\n", n);
            continue;
        }

        size_t J_bytes     = (size_t)G * m * n * sizeof(float);
        size_t H_bytes     = (size_t)G * n * n * sizeof(double);
        size_t g_bytes     = (size_t)G * n * sizeof(double);
        size_t delta_bytes = (size_t)G * n * sizeof(double);

        float  *d_J;
        double *d_delta, *d_H, *d_g;
        cudaMalloc(&d_J,     J_bytes);
        cudaMalloc(&d_delta, delta_bytes);
        cudaMalloc(&d_H,     H_bytes);
        cudaMalloc(&d_g,     g_bytes);

        std::vector<float> h_J(G * m * n);
        for (size_t i = 0; i < h_J.size(); ++i)
            h_J[i] = 0.5f - (float)(rand() & 0xffff) / 65535.0f;
        cudaMemcpy(d_J, h_J.data(), J_bytes, cudaMemcpyHostToDevice);

        cudaFuncSetAttribute(chol_full,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_full);
        cudaFuncSetAttribute(chol_factor_only,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_factor);

        const int    B   = 128;
        const double lam = 1e-3;

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        /* full */
        chol_full<<<G, B, smem_full>>>(d_J, d_delta, lam, m, n);
        cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for (int r = 0; r < reps; ++r)
            chol_full<<<G, B, smem_full>>>(d_J, d_delta, lam, m, n);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms_full = 0.f; cudaEventElapsedTime(&ms_full, t0, t1);
        ms_full /= reps;

        /* build only */
        chol_build_only<<<G, B>>>(d_J, d_H, d_g, lam, m, n);
        cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for (int r = 0; r < reps; ++r)
            chol_build_only<<<G, B>>>(d_J, d_H, d_g, lam, m, n);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms_build = 0.f; cudaEventElapsedTime(&ms_build, t0, t1);
        ms_build /= reps;

        /* factor only -- H and g were just built above by the last build call */
        chol_factor_only<<<G, B, smem_factor>>>(d_H, d_g, d_delta, n);
        cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for (int r = 0; r < reps; ++r)
            chol_factor_only<<<G, B, smem_factor>>>(d_H, d_g, d_delta, n);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms_factor = 0.f; cudaEventElapsedTime(&ms_factor, t0, t1);
        ms_factor /= reps;

        cudaEventDestroy(t0); cudaEventDestroy(t1);

        double ops_per_ind = 2.0 * n * n * m + (double)n * n * n / 3.0;
        double total_fmas  = (double)G * ops_per_ind;
        double compute_lb_ms = total_fmas / fp64_fma_per_sec * 1e3;

        /* Occupancy: how many blocks fit per SM given smem? */
        int max_blocks_per_sm;
        cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            &max_blocks_per_sm, chol_full, B, (int)smem_full, 0);
        char occ[24];
        snprintf(occ, sizeof(occ), "%d blk * 4 wp/SM",
                 max_blocks_per_sm);

        printf("%4d  %10.3f  %10.3f  %10.3f  %10.3f  %12s  %10.1f\n",
               n, ms_full, ms_build, ms_factor, compute_lb_ms,
               occ, smem_full / 1024.0);

        cudaFree(d_J); cudaFree(d_delta); cudaFree(d_H); cudaFree(d_g);
    }

    return 0;
}
