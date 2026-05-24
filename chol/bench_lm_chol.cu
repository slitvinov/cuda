/*
   Compare LM Cholesky-step backends with a common interface.

   Every backend implements one "LM step":
       (J [G,m,n], r [G,m], lam [G])
         ->  (delta [G,n], Js [G,m], fnorm_lin [G])

   where delta solves (JTJ + lambdaI) delta = JTr, Js = J*delta, and
   fnorm_lin = ||r - Js||.

   Backends:
       v1_naive    -- single packed-smem kernel, all four steps fused
       v2_split    -- packed lower triangle, separate build / factor / Js;
                     factor body is serial on tid==0
       v3_cusolver -- full-storage H, cusolverDnDpotrfBatched + Dpotrs
       v4_warp     -- like v2 but the Cholesky inner dot product is a
                     warp-shuffle reduction across 32 lanes (the only
                     serial part is the cross-column dependency chain)
       v5_blocked  -- LAPACK-DPOTRF-style blocked Cholesky.  Splits the
                     matrix into NBxNB panels; each iteration does a
                     small diagonal-block factor (warp-coop), a panel
                     TRSM, and a trailing SYRK -- all parallelized
                     across the block's threads.

   Same input data; same output across backends (fp64 inside, fp32 outside).
*/

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)
#define CK_CUS(x) do { cusolverStatus_t s = (x); if (s != CUSOLVER_STATUS_SUCCESS) { \
    fprintf(stderr, "cuSOLVER err %d at %s:%d\n", (int)s, __FILE__, __LINE__); \
    exit(1); } } while (0)

/* =========================================================================
   Shared kernels: J/r staging is the same for all backends.
   ========================================================================= */

/* Build H = JTJ + lambdaI as a FULL nxn column-major matrix per individual,
   and g = JTr.   Used by the cuSOLVER backend.   No shared memory. */
__global__ void build_H_g_full(const float *J, const float *r, const float *lam,
                               double *H, double *g, int m, int n)
{
    const int    b   = blockIdx.x;
    const int    tid = threadIdx.x;
    const int    B   = blockDim.x;
    const size_t Jb  = (size_t)b * m * n;
    const size_t rb  = (size_t)b * m;
    const size_t Hb  = (size_t)b * n * n;
    const size_t gb  = (size_t)b * n;

    for (int idx = tid; idx < n * n; idx += B) {
        int i = idx / n, j = idx % n;
        double v = 0.0;
        for (int k = 0; k < m; ++k) {
            v += (double)J[Jb + (size_t)k * n + i] *
                 (double)J[Jb + (size_t)k * n + j];
        }
        if (i == j) v += (double)lam[b];
        /* Column-major for cuSOLVER: H[j*n + i] = element (i,j). */
        H[Hb + (size_t)j * n + i] = v;
    }
    for (int i = tid; i < n; i += B) {
        double v = 0.0;
        for (int k = 0; k < m; ++k)
            v += (double)J[Jb + (size_t)k * n + i] * (double)r[rb + k];
        g[gb + i] = v;
    }
}

/* Same as above but writes only the packed lower triangle of H.
   Used by v2_split.  ~0.5 the work and HBM traffic. */
__device__ __forceinline__
void decode_pack(int idx, int *i_out, int *j_out) {
    double t = sqrt(1.0 + 8.0 * (double)idx);
    int i = (int)((-1.0 + t) * 0.5);
    while (i * (i + 1) / 2 > idx) --i;
    while ((i + 1) * (i + 2) / 2 <= idx) ++i;
    *i_out = i;
    *j_out = idx - i * (i + 1) / 2;
}

__global__ void build_H_g_packed(const float *J, const float *r, const float *lam,
                                 double *H_pack, double *g, int m, int n)
{
    const int    b   = blockIdx.x;
    const int    tid = threadIdx.x;
    const int    B   = blockDim.x;
    const size_t Jb  = (size_t)b * m * n;
    const size_t rb  = (size_t)b * m;
    const int    T   = n * (n + 1) / 2;
    const size_t Hb  = (size_t)b * T;
    const size_t gb  = (size_t)b * n;

    for (int idx = tid; idx < T; idx += B) {
        int i, j; decode_pack(idx, &i, &j);
        double v = 0.0;
        for (int k = 0; k < m; ++k) {
            v += (double)J[Jb + (size_t)k * n + i] *
                 (double)J[Jb + (size_t)k * n + j];
        }
        if (i == j) v += (double)lam[b];
        H_pack[Hb + idx] = v;
    }
    for (int i = tid; i < n; i += B) {
        double v = 0.0;
        for (int k = 0; k < m; ++k)
            v += (double)J[Jb + (size_t)k * n + i] * (double)r[rb + k];
        g[gb + i] = v;
    }
}

/* Js = J * delta,   per individual, per sample.   Parallel across m. */
__global__ void compute_Js(const float *J, const double *delta,
                           float *Js, int m, int n)
{
    const int    b   = blockIdx.x;
    const int    tid = threadIdx.x;
    const int    B   = blockDim.x;
    const size_t Jb  = (size_t)b * m * n;
    const size_t db  = (size_t)b * n;
    const size_t Jsb = (size_t)b * m;
    for (int k = tid; k < m; k += B) {
        double v = 0.0;
        for (int i = 0; i < n; ++i)
            v += (double)J[Jb + (size_t)k * n + i] * delta[db + i];
        Js[Jsb + k] = (float)v;
    }
}

/* fnorm_lin = ||r - Js||.  One thread per individual. */
__global__ void compute_fnorm_lin(const float *r, const float *Js,
                                  float *fnorm_lin, int m, int Gn)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= Gn) return;
    double s = 0.0;
    for (int k = 0; k < m; ++k) {
        double d = (double)r[(size_t)b * m + k] - (double)Js[(size_t)b * m + k];
        s += d * d;
    }
    fnorm_lin[b] = (float)sqrt(s);
}

/* =========================================================================
   V1: naive -- single packed-smem kernel, mirrors lmgp.cu's lm_step_kernel.
   ========================================================================= */

__global__ void v1_naive(const float *J, const float *r, const float *lam,
                         double *delta, float *Js, float *fnorm_lin,
                         int m, int n)
{
    extern __shared__ double sm[];
    double *H  = sm;
    double *L  = H  + (size_t)n * n;
    double *gv = L  + (size_t)n * n;
    double *y  = gv + n;
    double *s  = y  + n;

    const int    b   = blockIdx.x;
    const int    tid = threadIdx.x;
    const int    B   = blockDim.x;
    const size_t Jb  = (size_t)b * m * n;
    const size_t rb  = (size_t)b * m;

    for (int idx = tid; idx < n * n; idx += B) {
        int i = idx / n, j = idx % n;
        double v = 0.0;
        for (int k = 0; k < m; ++k) {
            v += (double)J[Jb + (size_t)k * n + i] *
                 (double)J[Jb + (size_t)k * n + j];
        }
        if (i == j) v += (double)lam[b];
        H[i * n + j] = v;
    }
    for (int i = tid; i < n; i += B) {
        double v = 0.0;
        for (int k = 0; k < m; ++k)
            v += (double)J[Jb + (size_t)k * n + i] * (double)r[rb + k];
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
        for (int i = 0; i < n; ++i) delta[(size_t)b * n + i] = s[i];
    }
    __syncthreads();

    for (int k = tid; k < m; k += B) {
        double v = 0.0;
        for (int i = 0; i < n; ++i)
            v += (double)J[Jb + (size_t)k * n + i] * s[i];
        Js[(size_t)b * m + k] = (float)v;
    }
    __syncthreads();

    if (tid == 0) {
        double ss = 0.0;
        for (int k = 0; k < m; ++k) {
            double d = (double)r[rb + k] - (double)Js[(size_t)b * m + k];
            ss += d * d;
        }
        fnorm_lin[b] = (float)sqrt(ss);
    }
}

/* =========================================================================
   V2: split + packed lower triangle.  Build kernel has 0 smem (full
   occupancy); factor kernel runs on packed half-size H in smem.
   ========================================================================= */

__global__ void v2_factor_packed(const double *H_pack, const double *g_in,
                                 double *delta, int n)
{
    extern __shared__ double sm[];
    const size_t T  = (size_t)n * (n + 1) / 2;
    double *L  = sm;
    double *gv = L  + T;
    double *y  = gv + n;
    double *s  = y  + n;

    const int    b   = blockIdx.x;
    const int    tid = threadIdx.x;
    const int    B   = blockDim.x;
    const size_t Hb  = (size_t)b * T;
    const size_t gb  = (size_t)b * n;

    for (size_t idx = tid; idx < T; idx += B) L[idx] = H_pack[Hb + idx];
    for (int    i   = tid; i   < n; i   += B) gv[i] = g_in[gb + i];
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < n; ++i) {
            int i_off = i * (i + 1) / 2;
            for (int j = 0; j <= i; ++j) {
                int j_off = j * (j + 1) / 2;
                double v = L[i_off + j];
                for (int k = 0; k < j; ++k) v -= L[i_off + k] * L[j_off + k];
                L[i_off + j] = (i == j) ? sqrt(v) : v / L[j_off + j];
            }
        }
        for (int i = 0; i < n; ++i) {
            int i_off = i * (i + 1) / 2;
            double v = gv[i];
            for (int k = 0; k < i; ++k) v -= L[i_off + k] * y[k];
            y[i] = v / L[i_off + i];
        }
        for (int i = n - 1; i >= 0; --i) {
            double v = y[i];
            for (int k = i + 1; k < n; ++k) {
                int k_off = k * (k + 1) / 2;
                v -= L[k_off + i] * s[k];
            }
            int i_off = i * (i + 1) / 2;
            s[i] = v / L[i_off + i];
        }
        for (int i = 0; i < n; ++i) delta[(size_t)b * n + i] = s[i];
    }
}

/* =========================================================================
   V3: cuSOLVER backend.  H in full storage column-major;
   cusolverDnDpotrfBatched factors, cusolverDnDpotrsBatched solves.
   ========================================================================= */

struct CusolverCtx {
    cusolverDnHandle_t handle;
    int        n;
    int        G;
    double   **d_H_ptrs;       /* [G] device pointers into d_H */
    double   **d_g_ptrs;       /* [G] device pointers into d_g */
    int       *d_info;
};

static void cusolver_ctx_init(CusolverCtx *c, int G, int n,
                              double *d_H, double *d_g) {
    CK_CUS( cusolverDnCreate(&c->handle) );
    c->G = G; c->n = n;
    CK_CUDA( cudaMalloc(&c->d_H_ptrs, G * sizeof(double*)) );
    CK_CUDA( cudaMalloc(&c->d_g_ptrs, G * sizeof(double*)) );
    CK_CUDA( cudaMalloc(&c->d_info,   G * sizeof(int)) );
    std::vector<double*> h_Hp(G), h_gp(G);
    for (int i = 0; i < G; ++i) {
        h_Hp[i] = d_H + (size_t)i * n * n;
        h_gp[i] = d_g + (size_t)i * n;
    }
    CK_CUDA( cudaMemcpy(c->d_H_ptrs, h_Hp.data(), G * sizeof(double*), cudaMemcpyHostToDevice) );
    CK_CUDA( cudaMemcpy(c->d_g_ptrs, h_gp.data(), G * sizeof(double*), cudaMemcpyHostToDevice) );
}

static void cusolver_ctx_destroy(CusolverCtx *c) {
    cudaFree(c->d_H_ptrs);
    cudaFree(c->d_g_ptrs);
    cudaFree(c->d_info);
    cusolverDnDestroy(c->handle);
}

/* =========================================================================
   V4: warp-cooperative serial-column Cholesky.

   Same outer structure as v2 (packed lower triangle, in-place factor),
   but the inner dot product  sum L[i,k] * L[j,k]  for k=0..j-1  is done
   by 32 lanes in parallel via __shfl_xor_sync reduction.

   The cross-column dependency chain (cell L[i,j] needs L[j,j]) remains
   serial -- so only the first warp does the factor, but its per-cell
   cost drops from O(j) to O(j/32) + 5 shuffles.
   ========================================================================= */

__device__ __forceinline__ double warp_sum(double v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}

__global__ void v4_factor_warp_packed(const double *H_pack, const double *g_in,
                                      double *delta, int n)
{
    extern __shared__ double sm[];
    const size_t T  = (size_t)n * (n + 1) / 2;
    double *L  = sm;
    double *gv = L  + T;
    double *y  = gv + n;
    double *s  = y  + n;

    const int    b    = blockIdx.x;
    const int    tid  = threadIdx.x;
    const int    lane = tid & 31;
    const int    warp = tid >> 5;
    const int    B    = blockDim.x;
    const size_t Hb   = (size_t)b * T;
    const size_t gb   = (size_t)b * n;

    for (size_t idx = tid; idx < T; idx += B) L[idx] = H_pack[Hb + idx];
    for (int    i   = tid; i   < n; i   += B) gv[i] = g_in[gb + i];
    __syncthreads();

    if (warp == 0) {
        /* Cholesky: serial across columns, parallel within each cell. */
        for (int i = 0; i < n; ++i) {
            const int i_off = i * (i + 1) / 2;
            for (int j = 0; j <= i; ++j) {
                const int j_off = j * (j + 1) / 2;
                double partial = 0.0;
                for (int k = lane; k < j; k += 32)
                    partial += L[i_off + k] * L[j_off + k];
                partial = warp_sum(partial);
                if (lane == 0) {
                    double v = L[i_off + j] - partial;
                    L[i_off + j] = (i == j) ? sqrt(v) : v / L[j_off + j];
                }
                __syncwarp();
            }
        }
        /* Forward solve L y = g. */
        for (int i = 0; i < n; ++i) {
            const int i_off = i * (i + 1) / 2;
            double partial = 0.0;
            for (int k = lane; k < i; k += 32)
                partial += L[i_off + k] * y[k];
            partial = warp_sum(partial);
            if (lane == 0) y[i] = (gv[i] - partial) / L[i_off + i];
            __syncwarp();
        }
        /* Back solve LT s = y. */
        for (int i = n - 1; i >= 0; --i) {
            double partial = 0.0;
            for (int k = i + 1 + lane; k < n; k += 32) {
                const int k_off = k * (k + 1) / 2;
                partial += L[k_off + i] * s[k];
            }
            partial = warp_sum(partial);
            if (lane == 0) {
                const int i_off = i * (i + 1) / 2;
                s[i] = (y[i] - partial) / L[i_off + i];
            }
            __syncwarp();
        }
        for (int i = lane; i < n; i += 32) delta[(size_t)b * n + i] = s[i];
    }
}

/* =========================================================================
   V5: LAPACK-DPOTRF-style blocked Cholesky.

   Splits the nxn matrix into NBxNB panels.  For each block-column j:
     1. SYRK: update A[j:n, j:j+NB] -= A[j:n, 0:j] * A[j:n, 0:j]T
     2. Cholesky factor the NBxNB diagonal block (warp-coop, small)
     3. TRSM: solve L_diag * A[j+NB:n, j:j+NB]T = trailing panel

   Each SYRK and TRSM operation is parallelized across all the block's
   threads.  The serial portion shrinks from O(n^3/3) to O(n*NB^2).
   ========================================================================= */

#define V5_NB 16

/* Helper: full-storage cell access L[i, j] in shared memory (nxn). */
__device__ __forceinline__ double& L_at(double *L, int n, int i, int j) {
    return L[(size_t)i * n + j];
}

/* Serial Cholesky on an NBxNB block, single warp.  Uses full storage. */
__device__ __forceinline__
void chol_diag_block(double *L, int n, int j0, int sz, int lane) {
    /* L is full nxn in shared; we factor the block at [j0..j0+sz, j0..j0+sz]. */
    for (int i = 0; i < sz; ++i) {
        for (int j2 = 0; j2 <= i; ++j2) {
            double partial = 0.0;
            for (int k = lane; k < j2; k += 32)
                partial += L_at(L, n, j0+i, j0+k) * L_at(L, n, j0+j2, j0+k);
            partial = warp_sum(partial);
            if (lane == 0) {
                double v = L_at(L, n, j0+i, j0+j2) - partial;
                L_at(L, n, j0+i, j0+j2) = (i == j2) ? sqrt(v)
                                          : v / L_at(L, n, j0+j2, j0+j2);
            }
            __syncwarp();
        }
    }
}

__global__ void v5_factor_blocked(const double *H_in, const double *g_in,
                                  double *delta, int n)
{
    /* Full nxn storage in shared memory (since blocking needs random access). */
    extern __shared__ double sm[];
    double *L  = sm;
    double *gv = L  + (size_t)n * n;
    double *y  = gv + n;
    double *s  = y  + n;

    const int    b    = blockIdx.x;
    const int    tid  = threadIdx.x;
    const int    lane = tid & 31;
    const int    warp = tid >> 5;
    const int    nwarps = (blockDim.x + 31) / 32;
    const int    B    = blockDim.x;
    const size_t Hb   = (size_t)b * n * n;
    const size_t gb   = (size_t)b * n;

    /* Load full H into shared (cuSOLVER-style column-major in HBM). */
    for (int idx = tid; idx < n * n; idx += B) {
        int i = idx / n, j = idx % n;
        L_at(L, n, i, j) = H_in[Hb + (size_t)j * n + i];
    }
    for (int i = tid; i < n; i += B) gv[i] = g_in[gb + i];
    __syncthreads();

    /* Blocked Cholesky: process NB-sized diagonal blocks. */
    for (int j0 = 0; j0 < n; j0 += V5_NB) {
        const int sz = (j0 + V5_NB > n) ? (n - j0) : V5_NB;

        /* 1. SYRK: update diagonal block from previously-computed columns.
              A[j0:j0+sz, j0:j0+sz] -= A[j0:j0+sz, 0:j0] * A[j0:j0+sz, 0:j0]T
           Lower triangle only.  Parallel across (i, k) cells of the
           diagonal block. */
        for (int idx = tid; idx < sz * sz; idx += B) {
            int ii = idx / sz, kk = idx % sz;
            if (ii < kk) continue;
            double v = 0.0;
            for (int p = 0; p < j0; ++p)
                v += L_at(L, n, j0+ii, p) * L_at(L, n, j0+kk, p);
            L_at(L, n, j0+ii, j0+kk) -= v;
        }
        __syncthreads();

        /* 2. Cholesky factor of the diagonal block (small, warp 0). */
        if (warp == 0) chol_diag_block(L, n, j0, sz, lane);
        __syncthreads();

        /* 3. TRSM: solve L_diag * X = panel_below.
              For row i in [j0+sz, n), column c in [0, sz):
                X[i, c] = (A[i, j0+c] - sum_{p<c} X[i, p] * L_diag[c, p]) / L_diag[c, c]
              Each row processed sequentially in c (dependence); rows
              themselves are independent -> parallel across rows. */
        const int n_below = n - (j0 + sz);
        if (n_below > 0) {
            for (int row_idx = tid; row_idx < n_below; row_idx += B) {
                int i = j0 + sz + row_idx;
                for (int c = 0; c < sz; ++c) {
                    double v = L_at(L, n, i, j0+c);
                    /* Subtract contribution from columns 0..j0-1 (rank-k update). */
                    for (int p = 0; p < j0; ++p)
                        v -= L_at(L, n, i, p) * L_at(L, n, j0+c, p);
                    /* Within-block triangular solve for col c. */
                    for (int p = 0; p < c; ++p)
                        v -= L_at(L, n, i, j0+p) * L_at(L, n, j0+c, j0+p);
                    L_at(L, n, i, j0+c) = v / L_at(L, n, j0+c, j0+c);
                }
            }
        }
        __syncthreads();
    }

    /* Forward solve L y = g  (full triangular solve, lane 0 of warp 0). */
    if (warp == 0 && lane == 0) {
        for (int i = 0; i < n; ++i) {
            double v = gv[i];
            for (int k = 0; k < i; ++k) v -= L_at(L, n, i, k) * y[k];
            y[i] = v / L_at(L, n, i, i);
        }
        for (int i = n - 1; i >= 0; --i) {
            double v = y[i];
            for (int k = i + 1; k < n; ++k) v -= L_at(L, n, k, i) * s[k];
            s[i] = v / L_at(L, n, i, i);
        }
        for (int i = 0; i < n; ++i) delta[(size_t)b * n + i] = s[i];
    }
}

/* =========================================================================
   V6: fp32 throughout.

   Same structure as v4 (warp-cooperative packed Cholesky) but H, L,
   g, y, s are all float instead of double.  Two benefits:
     - fp32 throughput on A100/H100 is 2x fp64, so DFMA -> FFMA halves
       per-instruction latency
     - smem footprint per block halves -> ~2x the blocks/SM at the same n

   Cost: numerical stability.  fp32 Cholesky breaks down when
   kappa(H) > 10^4 or so; the trust region rejects bad steps but the
   max residual will be ~1e-7 (fp32 unit roundoff) rather than 1e-15.
   For LM that's usually fine -- the iteration self-corrects.
   ========================================================================= */

__device__ __forceinline__ float warp_sum_f(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}

__global__ void build_H_g_packed_f32(const float *J, const float *r, const float *lam,
                                     float *H_pack, float *g_out, int m, int n)
{
    const int    b   = blockIdx.x;
    const int    tid = threadIdx.x;
    const int    B   = blockDim.x;
    const size_t Jb  = (size_t)b * m * n;
    const size_t rb  = (size_t)b * m;
    const int    T   = n * (n + 1) / 2;
    const size_t Hb  = (size_t)b * T;
    const size_t gb  = (size_t)b * n;

    for (int idx = tid; idx < T; idx += B) {
        int i, j; decode_pack(idx, &i, &j);
        float v = 0.0f;
        for (int k = 0; k < m; ++k)
            v += J[Jb + (size_t)k * n + i] * J[Jb + (size_t)k * n + j];
        if (i == j) v += lam[b];
        H_pack[Hb + idx] = v;
    }
    for (int i = tid; i < n; i += B) {
        float v = 0.0f;
        for (int k = 0; k < m; ++k)
            v += J[Jb + (size_t)k * n + i] * r[rb + k];
        g_out[gb + i] = v;
    }
}

__global__ void v6_factor_warp_packed_f32(const float *H_pack, const float *g_in,
                                          double *delta, int n)
{
    extern __shared__ float smf[];
    const size_t T  = (size_t)n * (n + 1) / 2;
    float *L  = smf;
    float *gv = L  + T;
    float *y  = gv + n;
    float *s  = y  + n;

    const int    b    = blockIdx.x;
    const int    tid  = threadIdx.x;
    const int    lane = tid & 31;
    const int    warp = tid >> 5;
    const int    B    = blockDim.x;
    const size_t Hb   = (size_t)b * T;
    const size_t gb   = (size_t)b * n;

    for (size_t idx = tid; idx < T; idx += B) L[idx] = H_pack[Hb + idx];
    for (int    i   = tid; i   < n; i   += B) gv[i] = g_in[gb + i];
    __syncthreads();

    if (warp == 0) {
        for (int i = 0; i < n; ++i) {
            const int i_off = i * (i + 1) / 2;
            for (int j = 0; j <= i; ++j) {
                const int j_off = j * (j + 1) / 2;
                float partial = 0.0f;
                for (int k = lane; k < j; k += 32)
                    partial += L[i_off + k] * L[j_off + k];
                partial = warp_sum_f(partial);
                if (lane == 0) {
                    float v = L[i_off + j] - partial;
                    L[i_off + j] = (i == j) ? sqrtf(v) : v / L[j_off + j];
                }
                __syncwarp();
            }
        }
        for (int i = 0; i < n; ++i) {
            const int i_off = i * (i + 1) / 2;
            float partial = 0.0f;
            for (int k = lane; k < i; k += 32) partial += L[i_off + k] * y[k];
            partial = warp_sum_f(partial);
            if (lane == 0) y[i] = (gv[i] - partial) / L[i_off + i];
            __syncwarp();
        }
        for (int i = n - 1; i >= 0; --i) {
            float partial = 0.0f;
            for (int k = i + 1 + lane; k < n; k += 32) {
                const int k_off = k * (k + 1) / 2;
                partial += L[k_off + i] * s[k];
            }
            partial = warp_sum_f(partial);
            if (lane == 0) {
                const int i_off = i * (i + 1) / 2;
                s[i] = (y[i] - partial) / L[i_off + i];
            }
            __syncwarp();
        }
        /* Promote to fp64 on output so the rest of the pipeline matches. */
        for (int i = lane; i < n; i += 32)
            delta[(size_t)b * n + i] = (double)s[i];
    }
}

/* =========================================================================
   Cholesky-property test: verify that H * delta = g per individual.

   Each backend produces delta.  We separately have H (packed lower triangle)
   and g from build_H_g_packed.  We compute the absolute residual
       ||H delta - g||_inf   per individual,
   and reduce to the global max.

   For a correct Cholesky solve, the residual should be tiny -- on the
   order of eps * kappa(H) * ||g||.  Anything larger means the solve is broken
   (algorithmic bug, NaN propagation, or numerical instability).
   ========================================================================= */
__global__ void verify_Hd_eq_g(const double *H_pack, const double *delta,
                               const double *g, double *max_resid_per_g,
                               int n)
{
    extern __shared__ double sm[];
    const int    b   = blockIdx.x;
    const int    tid = threadIdx.x;
    const int    B   = blockDim.x;
    const int    T   = n * (n + 1) / 2;
    const size_t Hb  = (size_t)b * T;
    const size_t db  = (size_t)b * n;
    const size_t gb  = (size_t)b * n;

    double local_max = 0.0;
    for (int i = tid; i < n; i += B) {
        double v = 0.0;
        for (int j = 0; j < n; ++j) {
            /* H is symmetric, packed lower triangle:
               H[i, j] = H_pack[i*(i+1)/2 + j]  if j <= i
                       = H_pack[j*(j+1)/2 + i]  otherwise (by symmetry). */
            int idx = (j <= i) ? (i * (i + 1) / 2 + j)
                                : (j * (j + 1) / 2 + i);
            v += H_pack[Hb + idx] * delta[db + j];
        }
        double r = fabs(v - g[gb + i]);
        if (r > local_max) local_max = r;
    }
    sm[tid] = local_max;
    __syncthreads();
    for (int s = B / 2; s > 0; s /= 2) {
        if (tid < s) sm[tid] = fmax(sm[tid], sm[tid + s]);
        __syncthreads();
    }
    if (tid == 0) max_resid_per_g[b] = sm[0];
}

/* =========================================================================
   Bench harness: each backend does one LM step end-to-end.
   ========================================================================= */

static double bench_v1(int G, int m, int n, int reps,
                       const float *d_J, const float *d_r, const float *d_lam,
                       double *d_delta, float *d_Js, float *d_fnorm_lin)
{
    size_t smem = (size_t)(2 * n * n + 3 * n) * sizeof(double);
    if (smem > 100 * 1024) return -1;
    cudaFuncSetAttribute(v1_naive,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    v1_naive<<<G, 128, smem>>>(d_J, d_r, d_lam, d_delta, d_Js, d_fnorm_lin, m, n);
    CK_CUDA( cudaDeviceSynchronize() );

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < reps; ++r)
        v1_naive<<<G, 128, smem>>>(d_J, d_r, d_lam, d_delta, d_Js, d_fnorm_lin, m, n);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0.f; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (double)ms / reps;
}

static double bench_v2(int G, int m, int n, int reps,
                       const float *d_J, const float *d_r, const float *d_lam,
                       double *d_delta, float *d_Js, float *d_fnorm_lin,
                       double *d_H_pack, double *d_g)
{
    int T = n * (n + 1) / 2;
    size_t smem = (size_t)(T + 3 * n) * sizeof(double);
    if (smem > 100 * 1024) return -1;
    cudaFuncSetAttribute(v2_factor_packed,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    /* warmup */
    build_H_g_packed<<<G, 128>>>(d_J, d_r, d_lam, d_H_pack, d_g, m, n);
    v2_factor_packed<<<G, 128, smem>>>(d_H_pack, d_g, d_delta, n);
    compute_Js<<<G, 128>>>(d_J, d_delta, d_Js, m, n);
    compute_fnorm_lin<<<(G + 31) / 32, 32>>>(d_r, d_Js, d_fnorm_lin, m, G);
    CK_CUDA( cudaDeviceSynchronize() );

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < reps; ++r) {
        build_H_g_packed<<<G, 128>>>(d_J, d_r, d_lam, d_H_pack, d_g, m, n);
        v2_factor_packed<<<G, 128, smem>>>(d_H_pack, d_g, d_delta, n);
        compute_Js<<<G, 128>>>(d_J, d_delta, d_Js, m, n);
        compute_fnorm_lin<<<(G + 31) / 32, 32>>>(d_r, d_Js, d_fnorm_lin, m, G);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0.f; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (double)ms / reps;
}

static double bench_v6(int G, int m, int n, int reps,
                       const float *d_J, const float *d_r, const float *d_lam,
                       double *d_delta, float *d_Js, float *d_fnorm_lin,
                       float *d_H_pack_f32, float *d_g_f32)
{
    int T = n * (n + 1) / 2;
    size_t smem = (size_t)(T + 3 * n) * sizeof(float);
    if (smem > 100 * 1024) return -1;
    cudaFuncSetAttribute(v6_factor_warp_packed_f32,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    build_H_g_packed_f32<<<G, 128>>>(d_J, d_r, d_lam, d_H_pack_f32, d_g_f32, m, n);
    v6_factor_warp_packed_f32<<<G, 128, smem>>>(d_H_pack_f32, d_g_f32, d_delta, n);
    compute_Js<<<G, 128>>>(d_J, d_delta, d_Js, m, n);
    compute_fnorm_lin<<<(G + 31) / 32, 32>>>(d_r, d_Js, d_fnorm_lin, m, G);
    CK_CUDA( cudaDeviceSynchronize() );

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < reps; ++r) {
        build_H_g_packed_f32<<<G, 128>>>(d_J, d_r, d_lam, d_H_pack_f32, d_g_f32, m, n);
        v6_factor_warp_packed_f32<<<G, 128, smem>>>(d_H_pack_f32, d_g_f32, d_delta, n);
        compute_Js<<<G, 128>>>(d_J, d_delta, d_Js, m, n);
        compute_fnorm_lin<<<(G + 31) / 32, 32>>>(d_r, d_Js, d_fnorm_lin, m, G);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0.f; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (double)ms / reps;
}

static double bench_v4(int G, int m, int n, int reps,
                       const float *d_J, const float *d_r, const float *d_lam,
                       double *d_delta, float *d_Js, float *d_fnorm_lin,
                       double *d_H_pack, double *d_g)
{
    int T = n * (n + 1) / 2;
    size_t smem = (size_t)(T + 3 * n) * sizeof(double);
    if (smem > 100 * 1024) return -1;
    cudaFuncSetAttribute(v4_factor_warp_packed,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    build_H_g_packed<<<G, 128>>>(d_J, d_r, d_lam, d_H_pack, d_g, m, n);
    v4_factor_warp_packed<<<G, 128, smem>>>(d_H_pack, d_g, d_delta, n);
    compute_Js<<<G, 128>>>(d_J, d_delta, d_Js, m, n);
    compute_fnorm_lin<<<(G + 31) / 32, 32>>>(d_r, d_Js, d_fnorm_lin, m, G);
    CK_CUDA( cudaDeviceSynchronize() );

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < reps; ++r) {
        build_H_g_packed<<<G, 128>>>(d_J, d_r, d_lam, d_H_pack, d_g, m, n);
        v4_factor_warp_packed<<<G, 128, smem>>>(d_H_pack, d_g, d_delta, n);
        compute_Js<<<G, 128>>>(d_J, d_delta, d_Js, m, n);
        compute_fnorm_lin<<<(G + 31) / 32, 32>>>(d_r, d_Js, d_fnorm_lin, m, G);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0.f; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (double)ms / reps;
}

static double bench_v5(int G, int m, int n, int reps,
                       const float *d_J, const float *d_r, const float *d_lam,
                       double *d_delta, float *d_Js, float *d_fnorm_lin,
                       double *d_H_full, double *d_g)
{
    size_t smem = (size_t)(n * n + 3 * n) * sizeof(double);
    if (smem > 100 * 1024) return -1;
    cudaFuncSetAttribute(v5_factor_blocked,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    /* v5 reads H in cuSOLVER's full column-major layout. */
    build_H_g_full<<<G, 128>>>(d_J, d_r, d_lam, d_H_full, d_g, m, n);
    v5_factor_blocked<<<G, 128, smem>>>(d_H_full, d_g, d_delta, n);
    compute_Js<<<G, 128>>>(d_J, d_delta, d_Js, m, n);
    compute_fnorm_lin<<<(G + 31) / 32, 32>>>(d_r, d_Js, d_fnorm_lin, m, G);
    CK_CUDA( cudaDeviceSynchronize() );

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < reps; ++r) {
        build_H_g_full<<<G, 128>>>(d_J, d_r, d_lam, d_H_full, d_g, m, n);
        v5_factor_blocked<<<G, 128, smem>>>(d_H_full, d_g, d_delta, n);
        compute_Js<<<G, 128>>>(d_J, d_delta, d_Js, m, n);
        compute_fnorm_lin<<<(G + 31) / 32, 32>>>(d_r, d_Js, d_fnorm_lin, m, G);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0.f; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (double)ms / reps;
}

static double bench_v3(int G, int m, int n, int reps,
                       const float *d_J, const float *d_r, const float *d_lam,
                       double *d_delta, float *d_Js, float *d_fnorm_lin,
                       double *d_H_full, double *d_g, CusolverCtx *ctx)
{
    /* warmup */
    build_H_g_full<<<G, 128>>>(d_J, d_r, d_lam, d_H_full, d_g, m, n);
    CK_CUS( cusolverDnDpotrfBatched(ctx->handle, CUBLAS_FILL_MODE_LOWER, n,
                                    ctx->d_H_ptrs, n, ctx->d_info, G) );
    CK_CUS( cusolverDnDpotrsBatched(ctx->handle, CUBLAS_FILL_MODE_LOWER, n, 1,
                                    ctx->d_H_ptrs, n, ctx->d_g_ptrs, n,
                                    ctx->d_info, G) );
    /* g now holds delta in fp64.  Copy to d_delta (same buffer here: d_g == d_delta? no,
       cuSOLVER overwrites d_g in place).  Reuse d_g as delta. */
    CK_CUDA( cudaMemcpyAsync(d_delta, d_g, (size_t)G * n * sizeof(double),
                             cudaMemcpyDeviceToDevice) );
    compute_Js<<<G, 128>>>(d_J, d_delta, d_Js, m, n);
    compute_fnorm_lin<<<(G + 31) / 32, 32>>>(d_r, d_Js, d_fnorm_lin, m, G);
    CK_CUDA( cudaDeviceSynchronize() );

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < reps; ++r) {
        build_H_g_full<<<G, 128>>>(d_J, d_r, d_lam, d_H_full, d_g, m, n);
        cusolverDnDpotrfBatched(ctx->handle, CUBLAS_FILL_MODE_LOWER, n,
                                ctx->d_H_ptrs, n, ctx->d_info, G);
        cusolverDnDpotrsBatched(ctx->handle, CUBLAS_FILL_MODE_LOWER, n, 1,
                                ctx->d_H_ptrs, n, ctx->d_g_ptrs, n,
                                ctx->d_info, G);
        cudaMemcpyAsync(d_delta, d_g, (size_t)G * n * sizeof(double),
                        cudaMemcpyDeviceToDevice);
        compute_Js<<<G, 128>>>(d_J, d_delta, d_Js, m, n);
        compute_fnorm_lin<<<(G + 31) / 32, 32>>>(d_r, d_Js, d_fnorm_lin, m, G);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0.f; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (double)ms / reps;
}

int main(void) {
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs)\n\n", prop.name, prop.multiProcessorCount);

    int G    = 65536;
    int m    = 256;
    int ns[] = {8, 16, 32, 48, 64, 80, 96};
    int reps = 5;

    printf("G = %d, m = %d, reps = %d\n\n", G, m, reps);
    printf("%4s  %10s %10s %10s %10s %10s %10s\n",
           "n", "v1_naive", "v2_split", "v4_warp", "v5_blocked", "v3_cusolver", "v6_fp32");
    printf("%4s  %10s %10s %10s %10s %10s %10s\n",
           "----", "----------", "----------", "----------", "----------", "----------", "----------");

    for (size_t ni = 0; ni < sizeof(ns)/sizeof(ns[0]); ++ni) {
        int n = ns[ni];
        int T = n * (n + 1) / 2;

        size_t J_bytes      = (size_t)G * m * n * sizeof(float);
        size_t r_bytes      = (size_t)G * m * sizeof(float);
        size_t lam_bytes    = (size_t)G * sizeof(float);
        size_t delta_bytes  = (size_t)G * n * sizeof(double);
        size_t Js_bytes     = (size_t)G * m * sizeof(float);
        size_t fnl_bytes    = (size_t)G * sizeof(float);
        size_t H_pack_bytes = (size_t)G * T * sizeof(double);
        size_t H_full_bytes = (size_t)G * n * n * sizeof(double);
        size_t g_bytes      = (size_t)G * n * sizeof(double);

        float  *d_J, *d_r, *d_lam, *d_Js, *d_fnorm_lin;
        double *d_delta, *d_H_pack, *d_H_full, *d_g;
        float  *d_H_pack_f32, *d_g_f32;
        CK_CUDA( cudaMalloc(&d_J,         J_bytes) );
        CK_CUDA( cudaMalloc(&d_r,         r_bytes) );
        CK_CUDA( cudaMalloc(&d_lam,       lam_bytes) );
        CK_CUDA( cudaMalloc(&d_delta,     delta_bytes) );
        CK_CUDA( cudaMalloc(&d_Js,        Js_bytes) );
        CK_CUDA( cudaMalloc(&d_fnorm_lin, fnl_bytes) );
        CK_CUDA( cudaMalloc(&d_H_pack,    H_pack_bytes) );
        CK_CUDA( cudaMalloc(&d_H_full,    H_full_bytes) );
        CK_CUDA( cudaMalloc(&d_g,         g_bytes) );
        CK_CUDA( cudaMalloc(&d_H_pack_f32, (size_t)G * T * sizeof(float)) );
        CK_CUDA( cudaMalloc(&d_g_f32,      (size_t)G * n * sizeof(float)) );

        std::vector<float> h_J(G * m * n);
        for (size_t i = 0; i < h_J.size(); ++i)
            h_J[i] = 0.5f - (float)(rand() & 0xffff) / 65535.0f;
        std::vector<float> h_r(G * m);
        for (size_t i = 0; i < h_r.size(); ++i)
            h_r[i] = 0.5f - (float)(rand() & 0xffff) / 65535.0f;
        std::vector<float> h_lam(G, 1e-3f);
        CK_CUDA( cudaMemcpy(d_J,   h_J.data(),   J_bytes,   cudaMemcpyHostToDevice) );
        CK_CUDA( cudaMemcpy(d_r,   h_r.data(),   r_bytes,   cudaMemcpyHostToDevice) );
        CK_CUDA( cudaMemcpy(d_lam, h_lam.data(), lam_bytes, cudaMemcpyHostToDevice) );

        CusolverCtx ctx;
        cusolver_ctx_init(&ctx, G, n, d_H_full, d_g);

        double ms_v1 = bench_v1(G, m, n, reps, d_J, d_r, d_lam,
                                d_delta, d_Js, d_fnorm_lin);
        double ms_v2 = bench_v2(G, m, n, reps, d_J, d_r, d_lam,
                                d_delta, d_Js, d_fnorm_lin, d_H_pack, d_g);
        double ms_v4 = bench_v4(G, m, n, reps, d_J, d_r, d_lam,
                                d_delta, d_Js, d_fnorm_lin, d_H_pack, d_g);
        double ms_v5 = bench_v5(G, m, n, reps, d_J, d_r, d_lam,
                                d_delta, d_Js, d_fnorm_lin, d_H_full, d_g);
        double ms_v3 = bench_v3(G, m, n, reps, d_J, d_r, d_lam,
                                d_delta, d_Js, d_fnorm_lin, d_H_full, d_g, &ctx);
        double ms_v6 = bench_v6(G, m, n, reps, d_J, d_r, d_lam,
                                d_delta, d_Js, d_fnorm_lin, d_H_pack_f32, d_g_f32);

        /* Verification: two layers.
           Layer A (cross-backend): all backends should produce delta
                  identical to fp64 rounding.
           Layer B (Cholesky property): delta must satisfy H delta = g (the
                  actual linear system).  Computed via verify_Hd_eq_g,
                  which doesn't trust any backend's internal factor. */

        /* Build the reference H_pack and g (used by both layers). */
        build_H_g_packed<<<G, 128>>>(d_J, d_r, d_lam, d_H_pack, d_g, m, n);
        CK_CUDA( cudaDeviceSynchronize() );

        std::vector<double> ref(G * n), tst(G * n);
        bench_v2(G, m, n, 1, d_J, d_r, d_lam, d_delta, d_Js, d_fnorm_lin,
                 d_H_pack, d_g);
        cudaMemcpy(ref.data(), d_delta, ref.size() * sizeof(double),
                   cudaMemcpyDeviceToHost);
        double ref_max_abs = 0.0;
        for (double v : ref) if (std::abs(v) > ref_max_abs) ref_max_abs = std::abs(v);

        /* Layer B requires H_pack and g -- rebuild after v2 because its
           cholesky_solve_kernel doesn't modify H (it's read-only).
           But cuSOLVER and v5 destroy H_full; v3 also overwrites d_g
           with delta in place.  Re-build before each run that needs them. */
        double *d_resid_per_g;
        CK_CUDA( cudaMalloc(&d_resid_per_g, G * sizeof(double)) );
        std::vector<double> resid_per_g(G);

        auto residual_of_delta = [&]() {
            /* H_pack and d_g must hold the original H and g.  Caller
               must have re-built them just before calling this. */
            verify_Hd_eq_g<<<G, 128, 128 * sizeof(double)>>>
                (d_H_pack, d_delta, d_g, d_resid_per_g, n);
            cudaMemcpy(resid_per_g.data(), d_resid_per_g,
                       G * sizeof(double), cudaMemcpyDeviceToHost);
            double mx = 0.0;
            for (double v : resid_per_g) if (v > mx) mx = v;
            return mx;
        };
        auto err_vs_ref = [&]() {
            cudaMemcpy(tst.data(), d_delta, tst.size() * sizeof(double),
                       cudaMemcpyDeviceToHost);
            double e = 0.0;
            for (size_t i = 0; i < tst.size(); ++i) {
                double d = std::abs(tst[i] - ref[i]);
                if (d > e) e = d;
            }
            return e;
        };

        double e1 = -1, e2 = 0.0, e3 = -1, e4 = -1, e5 = -1, e6 = -1;
        double r1 = -1, r2 = -1, r3 = -1, r4 = -1, r5 = -1, r6 = -1;

        /* For each test, rebuild H_pack and g (since some backends mutate). */
        #define REBUILD() do { \
            build_H_g_packed<<<G, 128>>>(d_J, d_r, d_lam, d_H_pack, d_g, m, n); \
            CK_CUDA( cudaDeviceSynchronize() ); \
        } while (0)

        /* v2 reference delta already in d_delta from above. */
        r2 = residual_of_delta();

        if (ms_v1 >= 0) {
            REBUILD();
            bench_v1(G, m, n, 1, d_J, d_r, d_lam, d_delta, d_Js, d_fnorm_lin);
            e1 = err_vs_ref();
            r1 = residual_of_delta();
        }
        if (ms_v4 >= 0) {
            REBUILD();
            bench_v4(G, m, n, 1, d_J, d_r, d_lam,
                     d_delta, d_Js, d_fnorm_lin, d_H_pack, d_g);
            e4 = err_vs_ref();
            /* v4 doesn't mutate H_pack/g, but rebuild for safety. */
            REBUILD(); r4 = residual_of_delta();
        }
        if (ms_v5 >= 0) {
            REBUILD();
            bench_v5(G, m, n, 1, d_J, d_r, d_lam,
                     d_delta, d_Js, d_fnorm_lin, d_H_full, d_g);
            e5 = err_vs_ref();
            REBUILD(); r5 = residual_of_delta();
        }
        if (ms_v3 >= 0) {
            REBUILD();
            bench_v3(G, m, n, 1, d_J, d_r, d_lam,
                     d_delta, d_Js, d_fnorm_lin, d_H_full, d_g, &ctx);
            e3 = err_vs_ref();
            REBUILD(); r3 = residual_of_delta();
        }
        if (ms_v6 >= 0) {
            REBUILD();
            bench_v6(G, m, n, 1, d_J, d_r, d_lam,
                     d_delta, d_Js, d_fnorm_lin, d_H_pack_f32, d_g_f32);
            e6 = err_vs_ref();
            REBUILD(); r6 = residual_of_delta();
        }
        #undef REBUILD
        cudaFree(d_resid_per_g);

        cusolver_ctx_destroy(&ctx);

        printf("%4d ", n);
        auto fmt = [](double ms) {
            if (ms < 0) printf(" %10s", "(skip)"); else printf(" %10.3f", ms);
        };
        fmt(ms_v1); fmt(ms_v2); fmt(ms_v4); fmt(ms_v5); fmt(ms_v3); fmt(ms_v6);
        printf("\n");

        auto efmt = [](double e) {
            if (e < 0) printf(" %10s", "");
            else       printf(" %10.2e", e);
        };
        printf("      max|delta - delta_ref|:");
        efmt(e1); efmt(e2); efmt(e4); efmt(e5); efmt(e3); efmt(e6);
        printf("    (ref=v2, max|delta_ref|=%.2e)\n", ref_max_abs);

        printf("       max|H*delta - g|:");
        efmt(r1); efmt(r2); efmt(r4); efmt(r5); efmt(r3); efmt(r6);
        printf("\n");

        cudaFree(d_J);      cudaFree(d_r);     cudaFree(d_lam);
        cudaFree(d_delta);  cudaFree(d_Js);    cudaFree(d_fnorm_lin);
        cudaFree(d_H_pack); cudaFree(d_H_full); cudaFree(d_g);
        cudaFree(d_H_pack_f32); cudaFree(d_g_f32);
    }
    return 0;
}
