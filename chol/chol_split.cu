/*
   Optimized Cholesky variants:
     - Symmetric H build (only compute lower triangle, ~0.5 work and 0.5 HBM)
     - Packed L storage in shared memory (n(n+1)/2 doubles instead of n^2)
     - Split build/factor into two kernels so the build phase runs at
       full occupancy (zero shared memory)

   Compared against the baseline chol_full kernel from chol.cu.

   Storage convention for packed lower triangle:
       L[i, j]  with  j <= i  lives at offset  i*(i+1)/2 + j
       Total slots:  T = n*(n+1)/2.
*/

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

/* ---- Baseline: identical to chol.cu's chol_full. ---- */
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

/* ---- Optimized: split + symmetric + packed lower triangle. ---- */

/* Map a packed-triangle linear index to (i, j) with j <= i. */
__device__ __forceinline__
void decode_pack(int idx, int *i_out, int *j_out) {
    double t = sqrt(1.0 + 8.0 * (double)idx);
    int i = (int)((-1.0 + t) * 0.5);
    while (i * (i + 1) / 2 > idx) --i;
    while ((i + 1) * (i + 2) / 2 <= idx) ++i;
    *i_out = i;
    *j_out = idx - i * (i + 1) / 2;
}

/* Build only the lower triangle of H (= JTJ + lambdaI), packed.
   Zero shared memory -> full occupancy.  0.5 the cells of the baseline. */
__global__ void chol_build_packed(const float *J,
                                  double      *H_packed,
                                  double      *g_out,
                                  double       lam,
                                  int m, int n)
{
    const int    g   = blockIdx.x;
    const int    tid = threadIdx.x;
    const int    B   = blockDim.x;
    const size_t Jb  = (size_t)g * m * n;
    const size_t T   = (size_t)n * (n + 1) / 2;
    const size_t Hb  = (size_t)g * T;
    const size_t gb  = (size_t)g * n;

    for (int idx = tid; idx < (int)T; idx += B) {
        int i, j;
        decode_pack(idx, &i, &j);
        double v = 0.0;
        for (int k = 0; k < m; ++k) {
            v += (double)J[Jb + (size_t)k * n + i] *
                 (double)J[Jb + (size_t)k * n + j];
        }
        if (i == j) v += lam;
        H_packed[Hb + idx] = v;
    }
    for (int i = tid; i < n; i += B) {
        double v = 0.0;
        for (int k = 0; k < m; ++k) v += (double)J[Jb + (size_t)k * n + i];
        g_out[gb + i] = v;
    }
}

/* Load packed H into shared, factor in-place, forward + back solve. */
__global__ void chol_factor_packed(const double *H_packed,
                                   const double *g_in,
                                   double       *delta,
                                   int n)
{
    extern __shared__ double sm[];
    const size_t T  = (size_t)n * (n + 1) / 2;
    double *L  = sm;            /* T doubles, holds H then L in-place */
    double *gv = L  + T;        /* n */
    double *y  = gv + n;        /* n */
    double *s  = y  + n;        /* n */

    const int    g   = blockIdx.x;
    const int    tid = threadIdx.x;
    const int    B   = blockDim.x;
    const size_t Hb  = (size_t)g * T;
    const size_t gb  = (size_t)g * n;

    for (size_t idx = tid; idx < T; idx += B) L[idx] = H_packed[Hb + idx];
    for (int    i   = tid; i   < n; i   += B) gv[i] = g_in[gb + i];
    __syncthreads();

    if (tid == 0) {
        /* Cholesky factor in-place. */
        for (int i = 0; i < n; ++i) {
            int i_off = i * (i + 1) / 2;
            for (int j = 0; j <= i; ++j) {
                int j_off = j * (j + 1) / 2;
                double v = L[i_off + j];
                for (int k = 0; k < j; ++k) v -= L[i_off + k] * L[j_off + k];
                L[i_off + j] = (i == j) ? sqrt(v) : v / L[j_off + j];
            }
        }
        /* Forward solve L y = g. */
        for (int i = 0; i < n; ++i) {
            int i_off = i * (i + 1) / 2;
            double v = gv[i];
            for (int k = 0; k < i; ++k) v -= L[i_off + k] * y[k];
            y[i] = v / L[i_off + i];
        }
        /* Back solve LT s = y. */
        for (int i = n - 1; i >= 0; --i) {
            double v = y[i];
            for (int k = i + 1; k < n; ++k) {
                int k_off = k * (k + 1) / 2;
                v -= L[k_off + i] * s[k];
            }
            int i_off = i * (i + 1) / 2;
            s[i] = v / L[i_off + i];
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
    printf("GPU: %s (%d SMs, clock %.2f GHz)\n\n",
           prop.name, prop.multiProcessorCount, clockKHz * 1e-6);

    int G    = 65536;
    int m    = 256;
    int ns[] = {8, 16, 32, 48, 64, 80, 96};
    int reps = 5;

    printf("G = %d, m = %d, reps = %d\n\n", G, m, reps);
    printf("%4s   %12s   %12s   %12s   %8s   %8s\n",
           "n", "baseline_ms", "build_ms", "factor_ms", "opt_ms", "speedup");
    printf("%4s   %12s   %12s   %12s   %8s   %8s\n",
           "----", "------------", "------------", "------------",
           "--------", "--------");

    for (size_t ni = 0; ni < sizeof(ns)/sizeof(ns[0]); ++ni) {
        int n = ns[ni];
        size_t T = (size_t)n * (n + 1) / 2;

        size_t smem_base = (size_t)(2 * n * n + 3 * n) * sizeof(double);
        size_t smem_opt  = (T + 3 * n) * sizeof(double);

        size_t J_bytes      = (size_t)G * m * n * sizeof(float);
        size_t H_pack_bytes = (size_t)G * T * sizeof(double);
        size_t g_bytes      = (size_t)G * n * sizeof(double);
        size_t delta_bytes  = (size_t)G * n * sizeof(double);

        float  *d_J;
        double *d_delta, *d_H_pack, *d_g;
        cudaMalloc(&d_J,      J_bytes);
        cudaMalloc(&d_delta,  delta_bytes);
        cudaMalloc(&d_H_pack, H_pack_bytes);
        cudaMalloc(&d_g,      g_bytes);

        std::vector<float> h_J(G * m * n);
        for (size_t i = 0; i < h_J.size(); ++i)
            h_J[i] = 0.5f - (float)(rand() & 0xffff) / 65535.0f;
        cudaMemcpy(d_J, h_J.data(), J_bytes, cudaMemcpyHostToDevice);

        const int    B   = 128;
        const double lam = 1e-3;

        /* Time baseline (chol_full). */
        double ms_base = -1.0;
        if (smem_base <= 100 * 1024) {
            cudaFuncSetAttribute(chol_full,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smem_base);
            cudaEvent_t t0, t1;
            cudaEventCreate(&t0); cudaEventCreate(&t1);
            chol_full<<<G, B, smem_base>>>(d_J, d_delta, lam, m, n);
            cudaDeviceSynchronize();
            cudaEventRecord(t0);
            for (int r = 0; r < reps; ++r)
                chol_full<<<G, B, smem_base>>>(d_J, d_delta, lam, m, n);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            float ms = 0.f; cudaEventElapsedTime(&ms, t0, t1);
            ms_base = (double)ms / reps;
            cudaEventDestroy(t0); cudaEventDestroy(t1);
        }

        /* Time optimized build + factor. */
        double ms_build = -1.0, ms_factor = -1.0;
        if (smem_opt <= 100 * 1024) {
            cudaFuncSetAttribute(chol_factor_packed,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smem_opt);
            cudaEvent_t t0, t1;
            cudaEventCreate(&t0); cudaEventCreate(&t1);

            chol_build_packed<<<G, B>>>(d_J, d_H_pack, d_g, lam, m, n);
            cudaDeviceSynchronize();
            cudaEventRecord(t0);
            for (int r = 0; r < reps; ++r)
                chol_build_packed<<<G, B>>>(d_J, d_H_pack, d_g, lam, m, n);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            float ms = 0.f; cudaEventElapsedTime(&ms, t0, t1);
            ms_build = (double)ms / reps;

            chol_factor_packed<<<G, B, smem_opt>>>(d_H_pack, d_g, d_delta, n);
            cudaDeviceSynchronize();
            cudaEventRecord(t0);
            for (int r = 0; r < reps; ++r)
                chol_factor_packed<<<G, B, smem_opt>>>(d_H_pack, d_g, d_delta, n);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms, t0, t1);
            ms_factor = (double)ms / reps;

            cudaEventDestroy(t0); cudaEventDestroy(t1);
        }

        double ms_opt = (ms_build < 0 || ms_factor < 0) ? -1.0 : ms_build + ms_factor;
        double speedup = (ms_base > 0 && ms_opt > 0) ? ms_base / ms_opt : 0.0;

        printf("%4d   ", n);
        if (ms_base   < 0) printf("%12s   ", "(smem cap)");
        else               printf("%12.3f   ", ms_base);
        if (ms_build  < 0) printf("%12s   ", "(skip)");
        else               printf("%12.3f   ", ms_build);
        if (ms_factor < 0) printf("%12s   ", "(skip)");
        else               printf("%12.3f   ", ms_factor);
        if (ms_opt    < 0) printf("%8s   ", "(skip)");
        else               printf("%8.3f   ", ms_opt);
        if (speedup   > 0) printf("%7.2fx", speedup);
        printf("\n");

        cudaFree(d_J); cudaFree(d_delta); cudaFree(d_H_pack); cudaFree(d_g);
    }

    printf("\nsmem layout:\n");
    printf("  baseline: H[n^2] + L[n^2] + g + y + s = (2n^2 + 3n) doubles\n");
    printf("  optimized: L[n(n+1)/2] + g + y + s = (n(n+1)/2 + 3n) doubles\n");
    return 0;
}
