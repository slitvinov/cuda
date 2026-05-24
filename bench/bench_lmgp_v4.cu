/*
   Scaling sweep for one Levenberg-Marquardt iteration with two
   Cholesky backends benchmarked side-by-side.

   v1 backend (original lm_step_kernel from lmgp.cu): one fused kernel
   doing H build + Cholesky + Js + fnorm_lin in packed shared memory,
   single-threaded factor on tid==0.

   v4 backend: split into four kernels -- build_H_g_packed (zero smem,
   full occupancy) + v4_factor_warp_packed (warp-cooperative dot
   products via SHFL.BFLY) + compute_Js + compute_fnorm_lin.

   Same outer LM iteration: forward + J at current params -> step ->
   trial point -> trust region -> accept.  Same inputs, same outputs
   (verified to fp64 epsilon in bench_lm_chol.cu).  The only thing
   that differs is the Cholesky step in the middle.
*/

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

constexpr int gi = 1;
constexpr int go = 1;
constexpr int p  = 1;

__device__ __forceinline__ void apply_op_jvp(uint8_t op,
                                             float v0, float v1,
                                             float t0, float t1,
                                             float par, float par_t,
                                             float *v, float *t) {
    switch (op) {
        case 0: *v = v0 + v1;     *t = t0 + t1;                break;
        case 1: *v = v0 - v1;     *t = t0 - t1;                break;
        case 2: *v = v0 * v1;     *t = t0 * v1 + v0 * t1;      break;
        case 3: {
            float inv = 1.0f / v1;
            *v = v0 * inv;
            *t = (t0 - (*v) * t1) * inv;
            break;
        }
        case 4: *v = sinf(v0);    *t =  cosf(v0) * t0;         break;
        case 5: *v = cosf(v0);    *t = -sinf(v0) * t0;         break;
        case 6: *v = par * v0;    *t = par_t * v0 + par * t0;  break;
        case 7: *v = par;         *t = par_t;                  break;
        default: *v = 0.0f;       *t = 0.0f;
    }
}

__global__ void forward_jac_kernel(const float   *params,
                                   const float   *inputs,
                                   const uint8_t *genome,
                                   float         *state_v,
                                   float         *state_t,
                                   float         *out_v,
                                   float         *J_out,
                                   int gn, int N)
{
    const int    g          = blockIdx.x;
    const int    tid        = threadIdx.x;
    const int    B          = blockDim.x;
    const int    ng_total   = gi + gn + go;
    const int    n_p        = gn * p;
    const size_t state_base = (size_t)g * (gi + gn) * N;

    for (int i = 0; i < gi; ++i) {
        const size_t dst = state_base + (size_t)i * N;
        const size_t src = (size_t)g * gi * N + (size_t)i * N;
        for (int k = tid; k < N; k += B) state_v[dst + k] = inputs[src + k];
    }
    __syncthreads();

    for (int j = 0; j < gn; ++j) {
        const size_t  row  = (size_t)g * ng_total + (gi + j);
        const uint8_t op   = genome[row * 3 + 0];
        const uint8_t ptr0 = genome[row * 3 + 1];
        const uint8_t ptr1 = genome[row * 3 + 2];
        const float   par  = params[(size_t)g * n_p + j];

        const size_t in0 = state_base + (size_t)ptr0 * N;
        const size_t in1 = state_base + (size_t)ptr1 * N;
        const size_t dst = state_base + (size_t)(gi + j) * N;

        for (int k = tid; k < N; k += B) {
            float v, t_;
            apply_op_jvp(op,
                         state_v[in0 + k], state_v[in1 + k],
                         0.0f, 0.0f, par, 0.0f, &v, &t_);
            state_v[dst + k] = v;
        }
        __syncthreads();
    }

    for (int o = 0; o < go; ++o) {
        const size_t  out_row = (size_t)g * ng_total + (gi + gn + o);
        const uint8_t src_row = genome[out_row * 3 + 1];
        const size_t  src     = state_base + (size_t)src_row * N;
        const size_t  dst     = (size_t)(g * go + o) * N;
        for (int k = tid; k < N; k += B) out_v[dst + k] = state_v[src + k];
    }

    for (int q = 0; q < n_p; ++q) {
        for (int i = 0; i < gi; ++i) {
            const size_t dst = state_base + (size_t)i * N;
            for (int k = tid; k < N; k += B) state_t[dst + k] = 0.0f;
        }
        __syncthreads();

        for (int j = 0; j < gn; ++j) {
            const size_t  row  = (size_t)g * ng_total + (gi + j);
            const uint8_t op   = genome[row * 3 + 0];
            const uint8_t ptr0 = genome[row * 3 + 1];
            const uint8_t ptr1 = genome[row * 3 + 2];
            const float   par   = params[(size_t)g * n_p + j];
            const float   par_t = (j == q) ? 1.0f : 0.0f;

            const size_t in0 = state_base + (size_t)ptr0 * N;
            const size_t in1 = state_base + (size_t)ptr1 * N;
            const size_t dst = state_base + (size_t)(gi + j) * N;

            for (int k = tid; k < N; k += B) {
                float v_, t;
                apply_op_jvp(op,
                             state_v[in0 + k], state_v[in1 + k],
                             state_t[in0 + k], state_t[in1 + k],
                             par, par_t, &v_, &t);
                state_t[dst + k] = t;
            }
            __syncthreads();
        }

        for (int o = 0; o < go; ++o) {
            const size_t  out_row = (size_t)g * ng_total + (gi + gn + o);
            const uint8_t src_row = genome[out_row * 3 + 1];
            const size_t  src     = state_base + (size_t)src_row * N;
            const size_t  J_base  = ((size_t)g * go + (size_t)o) * N;
            for (int k = tid; k < N; k += B) {
                J_out[(J_base + k) * n_p + q] = state_t[src + k];
            }
        }
        __syncthreads();
    }
}

__global__ void forward_value_kernel(const float   *params,
                                     const float   *inputs,
                                     const uint8_t *genome,
                                     float         *state_v,
                                     float         *out_v,
                                     int gn, int N)
{
    const int    g          = blockIdx.x;
    const int    tid        = threadIdx.x;
    const int    B          = blockDim.x;
    const int    ng_total   = gi + gn + go;
    const int    n_p        = gn * p;
    const size_t state_base = (size_t)g * (gi + gn) * N;

    for (int i = 0; i < gi; ++i) {
        const size_t dst = state_base + (size_t)i * N;
        const size_t src = (size_t)g * gi * N + (size_t)i * N;
        for (int k = tid; k < N; k += B) state_v[dst + k] = inputs[src + k];
    }
    __syncthreads();

    for (int j = 0; j < gn; ++j) {
        const size_t  row  = (size_t)g * ng_total + (gi + j);
        const uint8_t op   = genome[row * 3 + 0];
        const uint8_t ptr0 = genome[row * 3 + 1];
        const uint8_t ptr1 = genome[row * 3 + 2];
        const float   par  = params[(size_t)g * n_p + j];

        const size_t in0 = state_base + (size_t)ptr0 * N;
        const size_t in1 = state_base + (size_t)ptr1 * N;
        const size_t dst = state_base + (size_t)(gi + j) * N;

        for (int k = tid; k < N; k += B) {
            float v, t_;
            apply_op_jvp(op,
                         state_v[in0 + k], state_v[in1 + k],
                         0.0f, 0.0f, par, 0.0f, &v, &t_);
            state_v[dst + k] = v;
        }
        __syncthreads();
    }

    for (int o = 0; o < go; ++o) {
        const size_t  out_row = (size_t)g * ng_total + (gi + gn + o);
        const uint8_t src_row = genome[out_row * 3 + 1];
        const size_t  src     = state_base + (size_t)src_row * N;
        const size_t  dst     = (size_t)(g * go + o) * N;
        for (int k = tid; k < N; k += B) out_v[dst + k] = state_v[src + k];
    }
}

__global__ void residual_kernel(const float *out, const float *target,
                                float *r, int mr) {
    const int g   = blockIdx.x;
    const int tid = threadIdx.x;
    const int B   = blockDim.x;
    for (int k = tid; k < mr; k += B)
        r[(size_t)g * mr + k] = out[(size_t)g * mr + k]
                              - target[(size_t)g * mr + k];
}

__global__ void norm_kernel(const float *v, float *out_norm, int mr, int Gn) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= Gn) return;
    double s = 0.0;
    for (int k = 0; k < mr; ++k) {
        double a = v[(size_t)g * mr + k];
        s += a * a;
    }
    out_norm[g] = (float)sqrt(s);
}

__global__ void apply_step_kernel(const float  *params,
                                  const double *delta,
                                  float        *params_trial,
                                  int n) {
    const int g   = blockIdx.x;
    const int tid = threadIdx.x;
    const int B   = blockDim.x;
    for (int i = tid; i < n; i += B) {
        params_trial[(size_t)g * n + i] = params[(size_t)g * n + i]
                                        - (float)delta[(size_t)g * n + i];
    }
}

__global__ void lm_step_kernel(const float *J,
                               const float *r,
                               const float *lam,
                               double      *delta,
                               float       *Js,
                               float       *fnorm_lin,
                               int mr, int n)
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
    const size_t Jb  = (size_t)g * mr * n;
    const size_t rb  = (size_t)g * mr;

    for (int idx = tid; idx < n * n; idx += B) {
        int i = idx / n, j = idx % n;
        double v = 0.0;
        for (int k = 0; k < mr; ++k) {
            v += (double)J[Jb + (size_t)k * n + i] *
                 (double)J[Jb + (size_t)k * n + j];
        }
        if (i == j) v += (double)lam[g];
        H[i * n + j] = v;
    }
    for (int i = tid; i < n; i += B) {
        double v = 0.0;
        for (int k = 0; k < mr; ++k)
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
        for (int i = 0; i < n; ++i) delta[(size_t)g * n + i] = s[i];
    }
    __syncthreads();

    for (int k = tid; k < mr; k += B) {
        double v = 0.0;
        for (int i = 0; i < n; ++i)
            v += (double)J[Jb + (size_t)k * n + i] * s[i];
        Js[(size_t)g * mr + k] = (float)v;
    }
    __syncthreads();

    if (tid == 0) {
        double ss = 0.0;
        for (int k = 0; k < mr; ++k) {
            double d = (double)r[rb + k] - (double)Js[(size_t)g * mr + k];
            ss += d * d;
        }
        fnorm_lin[g] = (float)sqrt(ss);
    }
}

__global__ void trust_region_kernel(const float *fnorm,
                                    const float *fnorm_trial,
                                    const float *fnorm_lin,
                                    float       *lam,
                                    int         *accept,
                                    int          Gn)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= Gn) return;
    float ssq   = fnorm[g]       * fnorm[g];
    float ssq_t = fnorm_trial[g] * fnorm_trial[g];
    float ssq_l = fnorm_lin[g]   * fnorm_lin[g];
    float actred = ssq - ssq_t;
    float prered = ssq - ssq_l;
    float safe   = prered > 0.0f ? prered : 1.0f;
    float ratio  = actred / safe;
    int   acc    = ratio > 1.0e-4f ? 1 : 0;
    accept[g]    = acc;
    float L = lam[g];
    if      (ratio > 0.75f && acc) L *= 0.5f;
    else if (ratio < 0.25f)        L *= 2.0f;
    if (L < 1.0e-12f) L = 1.0e-12f;
    lam[g] = L;
}

__global__ void accept_kernel(const float *params_trial, float *params,
                              const float *r_trial,      float *r,
                              const float *fnorm_trial,  float *fnorm,
                              const int   *accept,
                              int n_par, int mr) {
    const int g   = blockIdx.x;
    const int tid = threadIdx.x;
    const int B   = blockDim.x;
    if (!accept[g]) return;
    for (int i = tid; i < n_par; i += B)
        params[(size_t)g * n_par + i] = params_trial[(size_t)g * n_par + i];
    for (int k = tid; k < mr; k += B)
        r[(size_t)g * mr + k] = r_trial[(size_t)g * mr + k];
    if (tid == 0) fnorm[g] = fnorm_trial[g];
}

/* ---- v4 backend: split Cholesky pipeline ---- */

__device__ __forceinline__ void decode_pack(int idx, int *i_out, int *j_out) {
    double t = sqrt(1.0 + 8.0 * (double)idx);
    int i = (int)((-1.0 + t) * 0.5);
    while (i * (i + 1) / 2 > idx) --i;
    while ((i + 1) * (i + 2) / 2 <= idx) ++i;
    *i_out = i;
    *j_out = idx - i * (i + 1) / 2;
}

__device__ __forceinline__ double warp_sum(double v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}

__global__ void build_H_g_packed(const float *J, const float *r, const float *lam,
                                 double *H_pack, double *g_out, int mr, int n)
{
    const int    b   = blockIdx.x;
    const int    tid = threadIdx.x;
    const int    B   = blockDim.x;
    const size_t Jb  = (size_t)b * mr * n;
    const size_t rb  = (size_t)b * mr;
    const int    T   = n * (n + 1) / 2;
    const size_t Hb  = (size_t)b * T;
    const size_t gb  = (size_t)b * n;

    for (int idx = tid; idx < T; idx += B) {
        int i, j; decode_pack(idx, &i, &j);
        double v = 0.0;
        for (int k = 0; k < mr; ++k) {
            v += (double)J[Jb + (size_t)k * n + i] *
                 (double)J[Jb + (size_t)k * n + j];
        }
        if (i == j) v += (double)lam[b];
        H_pack[Hb + idx] = v;
    }
    for (int i = tid; i < n; i += B) {
        double v = 0.0;
        for (int k = 0; k < mr; ++k)
            v += (double)J[Jb + (size_t)k * n + i] * (double)r[rb + k];
        g_out[gb + i] = v;
    }
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
        for (int i = 0; i < n; ++i) {
            const int i_off = i * (i + 1) / 2;
            double partial = 0.0;
            for (int k = lane; k < i; k += 32) partial += L[i_off + k] * y[k];
            partial = warp_sum(partial);
            if (lane == 0) y[i] = (gv[i] - partial) / L[i_off + i];
            __syncwarp();
        }
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

__global__ void compute_Js(const float *J, const double *delta,
                           float *Js, int mr, int n)
{
    const int    b   = blockIdx.x;
    const int    tid = threadIdx.x;
    const int    B   = blockDim.x;
    const size_t Jb  = (size_t)b * mr * n;
    const size_t db  = (size_t)b * n;
    const size_t Jsb = (size_t)b * mr;
    for (int k = tid; k < mr; k += B) {
        double v = 0.0;
        for (int i = 0; i < n; ++i)
            v += (double)J[Jb + (size_t)k * n + i] * delta[db + i];
        Js[Jsb + k] = (float)v;
    }
}

__global__ void compute_fnorm_lin(const float *r, const float *Js,
                                  float *fnorm_lin, int mr, int Gn)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= Gn) return;
    double s = 0.0;
    for (int k = 0; k < mr; ++k) {
        double d = (double)r[(size_t)b * mr + k] - (double)Js[(size_t)b * mr + k];
        s += d * d;
    }
    fnorm_lin[b] = (float)sqrt(s);
}

static void make_random_genomes(uint8_t *gen, int G, int gn) {
    const int ng_total = gi + gn + go;
    for (int g = 0; g < G; ++g) {
        for (int j = 0; j < gn; ++j) {
            int row = g * ng_total + (gi + j);
            int max_ptr = gi + j;
            gen[row*3 + 0] = (uint8_t)(rand() % 8);
            gen[row*3 + 1] = (uint8_t)(rand() % max_ptr);
            gen[row*3 + 2] = (uint8_t)(rand() % max_ptr);
        }
        for (int o = 0; o < go; ++o) {
            int row = g * ng_total + (gi + gn + o);
            gen[row*3 + 1] = (uint8_t)(rand() % (gi + gn));
        }
    }
}

static double bench_v1(int G, int gn, int N, int reps) {
    const int    ng_total = gi + gn + go;
    const int    n_p      = gn * p;
    const int    m        = go * N;
    const size_t inputs_bytes  = (size_t)G * gi * N * sizeof(float);
    const size_t targets_bytes = (size_t)G * go * N * sizeof(float);
    const size_t genome_bytes  = (size_t)G * ng_total * 3;
    const size_t state_bytes   = (size_t)G * (gi + gn) * N * sizeof(float);
    const size_t out_bytes     = (size_t)G * go * N * sizeof(float);
    const size_t r_bytes       = (size_t)G * m * sizeof(float);
    const size_t params_bytes  = (size_t)G * n_p * sizeof(float);
    const size_t J_bytes       = (size_t)G * m * n_p * sizeof(float);
    const size_t delta_bytes   = (size_t)G * n_p * sizeof(double);
    const size_t scalar_bytes  = (size_t)G * sizeof(float);
    const size_t accept_bytes  = (size_t)G * sizeof(int);
    const size_t smem_bytes    = (size_t)(2 * n_p * n_p + 3 * n_p) * sizeof(double);

    if (smem_bytes > 100 * 1024) return -3;
    cudaFuncSetAttribute(lm_step_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);

    float   *d_params, *d_params_trial, *d_inputs, *d_targets,
            *d_state_v, *d_state_t, *d_out_v, *d_out_trial,
            *d_J, *d_r, *d_r_trial, *d_Js,
            *d_fnorm, *d_fnorm_trial, *d_fnorm_lin, *d_lam;
    int     *d_accept;
    uint8_t *d_genome;
    double  *d_delta;

#define MALLOC(p, sz) if (cudaMalloc(&p, sz) != cudaSuccess) goto oom
    MALLOC(d_params,       params_bytes);
    MALLOC(d_params_trial, params_bytes);
    MALLOC(d_inputs,       inputs_bytes);
    MALLOC(d_targets,      targets_bytes);
    MALLOC(d_genome,       genome_bytes);
    MALLOC(d_state_v,      state_bytes);
    MALLOC(d_state_t,      state_bytes);
    MALLOC(d_out_v,        out_bytes);
    MALLOC(d_out_trial,    out_bytes);
    MALLOC(d_J,            J_bytes);
    MALLOC(d_r,            r_bytes);
    MALLOC(d_r_trial,      r_bytes);
    MALLOC(d_Js,           r_bytes);
    MALLOC(d_fnorm,        scalar_bytes);
    MALLOC(d_fnorm_trial,  scalar_bytes);
    MALLOC(d_fnorm_lin,    scalar_bytes);
    MALLOC(d_lam,          scalar_bytes);
    MALLOC(d_accept,       accept_bytes);
    MALLOC(d_delta,        delta_bytes);
#undef MALLOC

    {
        std::vector<float>   h_inputs(G * gi * N);
        std::vector<float>   h_targets(G * go * N, 0.0f);
        std::vector<float>   h_params(G * n_p, 1.0f);
        std::vector<float>   h_lam(G, 1e-3f);
        std::vector<uint8_t> h_genome(genome_bytes, 0);
        for (size_t i = 0; i < h_inputs.size(); ++i)
            h_inputs[i] = -1.0f + 2.0f * (float)(i % 1024) / 1024.0f;
        make_random_genomes(h_genome.data(), G, gn);

        cudaMemcpy(d_inputs,  h_inputs.data(),  inputs_bytes,  cudaMemcpyHostToDevice);
        cudaMemcpy(d_targets, h_targets.data(), targets_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_params,  h_params.data(),  params_bytes,  cudaMemcpyHostToDevice);
        cudaMemcpy(d_lam,     h_lam.data(),     scalar_bytes,  cudaMemcpyHostToDevice);
        cudaMemcpy(d_genome,  h_genome.data(),  genome_bytes,  cudaMemcpyHostToDevice);
    }

    {
        int B = (N >= 128) ? 128 : ((N >= 32) ? 32 : N);

        /* Initial r, fnorm so the LM iteration is well-defined. */
        forward_value_kernel<<<G, B>>>(d_params, d_inputs, d_genome,
                                       d_state_v, d_out_v, gn, N);
        residual_kernel<<<G, B>>>(d_out_v, d_targets, d_r, m);
        norm_kernel<<<(G + 31) / 32, 32>>>(d_r, d_fnorm, m, G);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        /* Warmup: one full iteration body. */
        forward_jac_kernel<<<G, B>>>(d_params, d_inputs, d_genome,
                                     d_state_v, d_state_t, d_out_v, d_J, gn, N);
        lm_step_kernel<<<G, 128, smem_bytes>>>(d_J, d_r, d_lam, d_delta,
                                                d_Js, d_fnorm_lin, m, n_p);
        apply_step_kernel<<<G, B>>>(d_params, d_delta, d_params_trial, n_p);
        forward_value_kernel<<<G, B>>>(d_params_trial, d_inputs, d_genome,
                                       d_state_v, d_out_trial, gn, N);
        residual_kernel<<<G, B>>>(d_out_trial, d_targets, d_r_trial, m);
        norm_kernel<<<(G + 31) / 32, 32>>>(d_r_trial, d_fnorm_trial, m, G);
        trust_region_kernel<<<(G + 31) / 32, 32>>>(d_fnorm, d_fnorm_trial,
                                                   d_fnorm_lin, d_lam, d_accept, G);
        accept_kernel<<<G, B>>>(d_params_trial, d_params, d_r_trial, d_r,
                                d_fnorm_trial, d_fnorm, d_accept, n_p, m);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "warmup err G=%d gn=%d N=%d: %s\n",
                    G, gn, N, cudaGetErrorString(err));
            cudaEventDestroy(t0); cudaEventDestroy(t1);
            goto launch_err;
        }

        cudaEventRecord(t0);
        for (int r = 0; r < reps; ++r) {
            forward_jac_kernel<<<G, B>>>(d_params, d_inputs, d_genome,
                                         d_state_v, d_state_t, d_out_v, d_J, gn, N);
            lm_step_kernel<<<G, 128, smem_bytes>>>(d_J, d_r, d_lam, d_delta,
                                                    d_Js, d_fnorm_lin, m, n_p);
            apply_step_kernel<<<G, B>>>(d_params, d_delta, d_params_trial, n_p);
            forward_value_kernel<<<G, B>>>(d_params_trial, d_inputs, d_genome,
                                           d_state_v, d_out_trial, gn, N);
            residual_kernel<<<G, B>>>(d_out_trial, d_targets, d_r_trial, m);
            norm_kernel<<<(G + 31) / 32, 32>>>(d_r_trial, d_fnorm_trial, m, G);
            trust_region_kernel<<<(G + 31) / 32, 32>>>(d_fnorm, d_fnorm_trial,
                                                       d_fnorm_lin, d_lam, d_accept, G);
            accept_kernel<<<G, B>>>(d_params_trial, d_params, d_r_trial, d_r,
                                    d_fnorm_trial, d_fnorm, d_accept, n_p, m);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, t0, t1);

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(d_params);       cudaFree(d_params_trial); cudaFree(d_inputs);
        cudaFree(d_targets);      cudaFree(d_genome);       cudaFree(d_state_v);
        cudaFree(d_state_t);      cudaFree(d_out_v);        cudaFree(d_out_trial);
        cudaFree(d_J);            cudaFree(d_r);            cudaFree(d_r_trial);
        cudaFree(d_Js);           cudaFree(d_fnorm);        cudaFree(d_fnorm_trial);
        cudaFree(d_fnorm_lin);    cudaFree(d_lam);          cudaFree(d_accept);
        cudaFree(d_delta);
        return (double)ms / reps;
    }

oom:
    return -1;
launch_err:
    cudaFree(d_params);       cudaFree(d_params_trial); cudaFree(d_inputs);
    cudaFree(d_targets);      cudaFree(d_genome);       cudaFree(d_state_v);
    cudaFree(d_state_t);      cudaFree(d_out_v);        cudaFree(d_out_trial);
    cudaFree(d_J);            cudaFree(d_r);            cudaFree(d_r_trial);
    cudaFree(d_Js);           cudaFree(d_fnorm);        cudaFree(d_fnorm_trial);
    cudaFree(d_fnorm_lin);    cudaFree(d_lam);          cudaFree(d_accept);
    cudaFree(d_delta);
    return -2;
}

/* ---- v4 variant: split Cholesky pipeline ---- */
static double bench_v4(int G, int gn, int N, int reps) {
    const int    ng_total = gi + gn + go;
    const int    n_p      = gn * p;
    const int    m        = go * N;
    const int    T        = n_p * (n_p + 1) / 2;
    const size_t inputs_bytes  = (size_t)G * gi * N * sizeof(float);
    const size_t targets_bytes = (size_t)G * go * N * sizeof(float);
    const size_t genome_bytes  = (size_t)G * ng_total * 3;
    const size_t state_bytes   = (size_t)G * (gi + gn) * N * sizeof(float);
    const size_t out_bytes     = (size_t)G * go * N * sizeof(float);
    const size_t r_bytes       = (size_t)G * m * sizeof(float);
    const size_t params_bytes  = (size_t)G * n_p * sizeof(float);
    const size_t J_bytes       = (size_t)G * m * n_p * sizeof(float);
    const size_t delta_bytes   = (size_t)G * n_p * sizeof(double);
    const size_t scalar_bytes  = (size_t)G * sizeof(float);
    const size_t accept_bytes  = (size_t)G * sizeof(int);
    const size_t H_pack_bytes  = (size_t)G * T * sizeof(double);
    const size_t g_dbl_bytes   = (size_t)G * n_p * sizeof(double);
    const size_t smem_bytes    = (size_t)(T + 3 * n_p) * sizeof(double);

    if (smem_bytes > 100 * 1024) return -3;
    cudaFuncSetAttribute(v4_factor_warp_packed,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);

    float   *d_params, *d_params_trial, *d_inputs, *d_targets,
            *d_state_v, *d_state_t, *d_out_v, *d_out_trial,
            *d_J, *d_r, *d_r_trial, *d_Js,
            *d_fnorm, *d_fnorm_trial, *d_fnorm_lin, *d_lam;
    int     *d_accept;
    uint8_t *d_genome;
    double  *d_delta, *d_H_pack, *d_g_dbl;

#define MALLOC(p, sz) if (cudaMalloc(&p, sz) != cudaSuccess) goto oom
    MALLOC(d_params,       params_bytes);
    MALLOC(d_params_trial, params_bytes);
    MALLOC(d_inputs,       inputs_bytes);
    MALLOC(d_targets,      targets_bytes);
    MALLOC(d_genome,       genome_bytes);
    MALLOC(d_state_v,      state_bytes);
    MALLOC(d_state_t,      state_bytes);
    MALLOC(d_out_v,        out_bytes);
    MALLOC(d_out_trial,    out_bytes);
    MALLOC(d_J,            J_bytes);
    MALLOC(d_r,            r_bytes);
    MALLOC(d_r_trial,      r_bytes);
    MALLOC(d_Js,           r_bytes);
    MALLOC(d_fnorm,        scalar_bytes);
    MALLOC(d_fnorm_trial,  scalar_bytes);
    MALLOC(d_fnorm_lin,    scalar_bytes);
    MALLOC(d_lam,          scalar_bytes);
    MALLOC(d_accept,       accept_bytes);
    MALLOC(d_delta,        delta_bytes);
    MALLOC(d_H_pack,       H_pack_bytes);
    MALLOC(d_g_dbl,        g_dbl_bytes);
#undef MALLOC

    {
        std::vector<float>   h_inputs(G * gi * N);
        std::vector<float>   h_targets(G * go * N, 0.0f);
        std::vector<float>   h_params(G * n_p, 1.0f);
        std::vector<float>   h_lam(G, 1e-3f);
        std::vector<uint8_t> h_genome(genome_bytes, 0);
        for (size_t i = 0; i < h_inputs.size(); ++i)
            h_inputs[i] = -1.0f + 2.0f * (float)(i % 1024) / 1024.0f;
        make_random_genomes(h_genome.data(), G, gn);

        cudaMemcpy(d_inputs,  h_inputs.data(),  inputs_bytes,  cudaMemcpyHostToDevice);
        cudaMemcpy(d_targets, h_targets.data(), targets_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_params,  h_params.data(),  params_bytes,  cudaMemcpyHostToDevice);
        cudaMemcpy(d_lam,     h_lam.data(),     scalar_bytes,  cudaMemcpyHostToDevice);
        cudaMemcpy(d_genome,  h_genome.data(),  genome_bytes,  cudaMemcpyHostToDevice);
    }

    {
        int B = (N >= 128) ? 128 : ((N >= 32) ? 32 : N);

        forward_value_kernel<<<G, B>>>(d_params, d_inputs, d_genome,
                                       d_state_v, d_out_v, gn, N);
        residual_kernel<<<G, B>>>(d_out_v, d_targets, d_r, m);
        norm_kernel<<<(G + 31) / 32, 32>>>(d_r, d_fnorm, m, G);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        /* Warmup. */
        for (int once = 0; once < 1; ++once) {
            forward_jac_kernel<<<G, B>>>(d_params, d_inputs, d_genome,
                                         d_state_v, d_state_t, d_out_v, d_J, gn, N);
            build_H_g_packed<<<G, 128>>>(d_J, d_r, d_lam, d_H_pack, d_g_dbl, m, n_p);
            v4_factor_warp_packed<<<G, 128, smem_bytes>>>(d_H_pack, d_g_dbl, d_delta, n_p);
            compute_Js<<<G, 128>>>(d_J, d_delta, d_Js, m, n_p);
            compute_fnorm_lin<<<(G + 31) / 32, 32>>>(d_r, d_Js, d_fnorm_lin, m, G);
            apply_step_kernel<<<G, B>>>(d_params, d_delta, d_params_trial, n_p);
            forward_value_kernel<<<G, B>>>(d_params_trial, d_inputs, d_genome,
                                           d_state_v, d_out_trial, gn, N);
            residual_kernel<<<G, B>>>(d_out_trial, d_targets, d_r_trial, m);
            norm_kernel<<<(G + 31) / 32, 32>>>(d_r_trial, d_fnorm_trial, m, G);
            trust_region_kernel<<<(G + 31) / 32, 32>>>(d_fnorm, d_fnorm_trial,
                                                       d_fnorm_lin, d_lam, d_accept, G);
            accept_kernel<<<G, B>>>(d_params_trial, d_params, d_r_trial, d_r,
                                    d_fnorm_trial, d_fnorm, d_accept, n_p, m);
        }
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "warmup err G=%d gn=%d N=%d: %s\n",
                    G, gn, N, cudaGetErrorString(err));
            cudaEventDestroy(t0); cudaEventDestroy(t1);
            goto launch_err4;
        }

        cudaEventRecord(t0);
        for (int r = 0; r < reps; ++r) {
            forward_jac_kernel<<<G, B>>>(d_params, d_inputs, d_genome,
                                         d_state_v, d_state_t, d_out_v, d_J, gn, N);
            build_H_g_packed<<<G, 128>>>(d_J, d_r, d_lam, d_H_pack, d_g_dbl, m, n_p);
            v4_factor_warp_packed<<<G, 128, smem_bytes>>>(d_H_pack, d_g_dbl, d_delta, n_p);
            compute_Js<<<G, 128>>>(d_J, d_delta, d_Js, m, n_p);
            compute_fnorm_lin<<<(G + 31) / 32, 32>>>(d_r, d_Js, d_fnorm_lin, m, G);
            apply_step_kernel<<<G, B>>>(d_params, d_delta, d_params_trial, n_p);
            forward_value_kernel<<<G, B>>>(d_params_trial, d_inputs, d_genome,
                                           d_state_v, d_out_trial, gn, N);
            residual_kernel<<<G, B>>>(d_out_trial, d_targets, d_r_trial, m);
            norm_kernel<<<(G + 31) / 32, 32>>>(d_r_trial, d_fnorm_trial, m, G);
            trust_region_kernel<<<(G + 31) / 32, 32>>>(d_fnorm, d_fnorm_trial,
                                                       d_fnorm_lin, d_lam, d_accept, G);
            accept_kernel<<<G, B>>>(d_params_trial, d_params, d_r_trial, d_r,
                                    d_fnorm_trial, d_fnorm, d_accept, n_p, m);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, t0, t1);

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(d_params);       cudaFree(d_params_trial); cudaFree(d_inputs);
        cudaFree(d_targets);      cudaFree(d_genome);       cudaFree(d_state_v);
        cudaFree(d_state_t);      cudaFree(d_out_v);        cudaFree(d_out_trial);
        cudaFree(d_J);            cudaFree(d_r);            cudaFree(d_r_trial);
        cudaFree(d_Js);           cudaFree(d_fnorm);        cudaFree(d_fnorm_trial);
        cudaFree(d_fnorm_lin);    cudaFree(d_lam);          cudaFree(d_accept);
        cudaFree(d_delta);        cudaFree(d_H_pack);       cudaFree(d_g_dbl);
        return (double)ms / reps;
    }

oom:
    return -1;
launch_err4:
    cudaFree(d_params);       cudaFree(d_params_trial); cudaFree(d_inputs);
    cudaFree(d_targets);      cudaFree(d_genome);       cudaFree(d_state_v);
    cudaFree(d_state_t);      cudaFree(d_out_v);        cudaFree(d_out_trial);
    cudaFree(d_J);            cudaFree(d_r);            cudaFree(d_r_trial);
    cudaFree(d_Js);           cudaFree(d_fnorm);        cudaFree(d_fnorm_trial);
    cudaFree(d_fnorm_lin);    cudaFree(d_lam);          cudaFree(d_accept);
    cudaFree(d_delta);        cudaFree(d_H_pack);       cudaFree(d_g_dbl);
    return -2;
}

int main(void) {
    int Gs[]   = {64, 1024, 16384, 65536};
    int gns[]  = {16, 64, 256};
    int Ns[]   = {64, 256, 1024};

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs, %ld MB)\n", prop.name, prop.multiProcessorCount,
           (long)(prop.totalGlobalMem >> 20));
    printf("Full LM iteration with v1 (fused) vs v4 (split warp-coop Cholesky).\n\n");

    printf("%7s %5s %5s  %10s %10s  %12s %12s  %s\n",
           "G", "gn", "N", "v1 ms", "v4 ms", "v1 ind/s", "v4 ind/s", "speedup");
    printf("%7s %5s %5s  %10s %10s  %12s %12s  %s\n",
           "-------", "-----", "-----", "----------", "----------",
           "------------", "------------", "-------");

    for (int gi_idx = 0; gi_idx < (int)(sizeof(Gs)/sizeof(Gs[0])); ++gi_idx)
    for (int gni    = 0; gni    < (int)(sizeof(gns)/sizeof(gns[0])); ++gni)
    for (int ni     = 0; ni     < (int)(sizeof(Ns)/sizeof(Ns[0]));   ++ni) {
        int G  = Gs[gi_idx];
        int gn = gns[gni];
        int N  = Ns[ni];

        size_t big = (size_t)G * (gi + gn) * N * sizeof(float) * 3;
        if (big > (size_t)40 * (1ULL << 30)) {
            printf("%7d %5d %5d   %10s   3*state>40GB\n", G, gn, N, "(skip)");
            continue;
        }

        double work = (double)G * gn * gn * N;
        int reps = work > 1e10 ? 2 : (work > 1e8 ? 5 : 30);
        double ms_v1 = bench_v1(G, gn, N, reps);
        double ms_v4 = bench_v4(G, gn, N, reps);

        auto fmt_ms = [](double ms) {
            if (ms == -1) printf(" %10s", "(OOM)");
            else if (ms == -2) printf(" %10s", "(launch)");
            else if (ms == -3) printf(" %10s", "(smem)");
            else printf(" %10.3f", ms);
        };
        auto fmt_ips = [](double ms, int G) {
            if (ms <= 0) { printf(" %12s", ""); return; }
            printf(" %12.3e", G / (ms * 1e-3));
        };

        printf("%7d %5d %5d ", G, gn, N);
        fmt_ms(ms_v1); fmt_ms(ms_v4);
        printf(" ");
        fmt_ips(ms_v1, G); fmt_ips(ms_v4, G);
        if (ms_v1 > 0 && ms_v4 > 0)
            printf("  %5.2fx\n", ms_v1 / ms_v4);
        else
            printf("\n");
    }
    return 0;
}
