/*
   Levenberg-Marquardt CGP fitting with trust-region damping and
   accept/reject of each step.

   Per individual, per step:

       1. forward + J at current params           (out, J)
       2. r = out - target,   fnorm = ‖r‖
       3. solve  (JᵀJ + λI) δ = Jᵀr     (Cholesky, fp64)
          also compute  Js = J·δ  and  fnorm_lin = ‖r - Js‖
       4. params_trial = params - δ
       5. forward at trial → out_trial, r_trial, fnorm_trial
       6. trust region:
            ratio = (‖r‖² - ‖r_trial‖²) / (‖r‖² - ‖r_lin‖²)
            accept iff ratio > 1e-4
            ratio > 0.75 → λ *= 0.5   (step worked great; trust more)
            ratio < 0.25 → λ *= 2.0   (step was poor; trust less)
       7. if accept: params = params_trial,  r = r_trial,  fnorm = fnorm_trial
          else:      keep old values; λ has been bumped up

   Each individual carries its own λ that adapts independently.
*/

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

constexpr int G  = 4;
constexpr int gi = 1;
constexpr int gn = 6;
constexpr int go = 1;
constexpr int p  = 1;
constexpr int N  = 16;
constexpr int ng_total = gi + gn + go;
constexpr int n_p      = gn * p;
constexpr int m        = go * N;


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
                                   float         *J_out)
{
    const int    g          = blockIdx.x;
    const int    tid        = threadIdx.x;
    const int    B          = blockDim.x;
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

__global__ void residual_kernel(const float *out, const float *target,
                                float *r, int mr) {
    const int g   = blockIdx.x;
    const int tid = threadIdx.x;
    const int B   = blockDim.x;
    for (int k = tid; k < mr; k += B) {
        r[(size_t)g * mr + k] = out[(size_t)g * mr + k]
                              - target[(size_t)g * mr + k];
    }
}

/*
   Value-only forward (used at the trial point — we don't need the
   Jacobian there).  Pass 1 of the forward+J kernel, no pass 2.
*/
__global__ void forward_value_kernel(const float   *params,
                                     const float   *inputs,
                                     const uint8_t *genome,
                                     float         *state_v,
                                     float         *out_v)
{
    const int    g          = blockIdx.x;
    const int    tid        = threadIdx.x;
    const int    B          = blockDim.x;
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

/* ‖v‖₂ per individual.  Serial (one thread per individual). */
__global__ void norm_kernel(const float *v, float *out_norm, int mr) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= G) return;
    double s = 0.0;
    for (int k = 0; k < mr; ++k) {
        double a = v[(size_t)g * mr + k];
        s += a * a;
    }
    out_norm[g] = (float)sqrt(s);
}

/* params_trial = params - δ. */
__global__ void apply_step_kernel(const float  *params,
                                  const double *delta,
                                  float        *params_trial,
                                  int n)
{
    const int g   = blockIdx.x;
    const int tid = threadIdx.x;
    const int B   = blockDim.x;
    for (int i = tid; i < n; i += B) {
        params_trial[(size_t)g * n + i] = params[(size_t)g * n + i]
                                        - (float)delta[(size_t)g * n + i];
    }
}

/*
   LM step kernel — Cholesky solve plus the linearized residual norm.
     - per-individual λ from lam[G]
     - produces δ from (JᵀJ + λI)δ = Jᵀr
     - also produces Js = J·δ and fnorm_lin = ‖r - Js‖, used by the
       trust-region kernel to compare predicted vs actual reduction
*/
__global__ void lm_step_kernel(const float *J,        // [G, m, n_p]
                               const float *r,        // [G, m]
                               const float *lam,      // [G]
                               double      *delta,    // [G, n_p]
                               float       *Js,       // [G, m]
                               float       *fnorm_lin,// [G]
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

    /* H = JᵀJ + λI  (parallel). */
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
    /* g = Jᵀr  (parallel). */
    for (int i = tid; i < n; i += B) {
        double v = 0.0;
        for (int k = 0; k < mr; ++k) {
            v += (double)J[Jb + (size_t)k * n + i] * (double)r[rb + k];
        }
        gv[i] = v;
    }
    __syncthreads();

    /* Cholesky + solves  (serial). */
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

    /* Js = J · δ  (parallel, used by the trust-region predicted-reduction). */
    for (int k = tid; k < mr; k += B) {
        double v = 0.0;
        for (int i = 0; i < n; ++i) {
            v += (double)J[Jb + (size_t)k * n + i] * s[i];
        }
        Js[(size_t)g * mr + k] = (float)v;
    }
    __syncthreads();

    /* fnorm_lin = ‖r - Js‖   (serial, n_lin entries is tiny). */
    if (tid == 0) {
        double ss = 0.0;
        for (int k = 0; k < mr; ++k) {
            double d = (double)r[rb + k] - (double)Js[(size_t)g * mr + k];
            ss += d * d;
        }
        fnorm_lin[g] = (float)sqrt(ss);
    }
}

/*
   Trust region: compare actual vs predicted reduction in ‖r‖², adapt λ,
   set the accept flag.  One thread per individual.
*/
__global__ void trust_region_kernel(const float *fnorm,
                                    const float *fnorm_trial,
                                    const float *fnorm_lin,
                                    float       *lam,
                                    int         *accept)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= G) return;
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

/*
   Commit the trial step iff accepted.  Rejected individuals keep
   their old params and r; their λ has already been bumped up by
   trust_region_kernel for the next attempt.
*/
__global__ void accept_kernel(const float *params_trial, float *params,
                              const float *r_trial,      float *r,
                              const float *fnorm_trial,  float *fnorm,
                              const int   *accept,
                              int n_par, int mr)
{
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


static void set_node(uint8_t *gen, int g, int row, uint8_t op,
                     uint8_t p0, uint8_t p1) {
    const int b = (g * ng_total + row) * 3;
    gen[b + 0] = op;
    gen[b + 1] = p0;
    gen[b + 2] = p1;
}

static float target_fn(int ind, float x) {
    switch (ind) {
        case 0: return sinf(x) + x * x;
        case 1: return x * x;
        case 2: return sinf(x);
        case 3: return sinf(x);
    }
    return 0.0f;
}

int main(void) {
    float h_inputs[G * gi * N];
    for (int g = 0; g < G; ++g)
        for (int k = 0; k < N; ++k)
            h_inputs[g * gi * N + k] = -1.0f + 2.0f * (float)k / (N - 1);

    float h_targets[G * go * N];
    for (int g = 0; g < G; ++g)
        for (int k = 0; k < N; ++k)
            h_targets[(g * go + 0) * N + k] = target_fn(g, h_inputs[g * gi * N + k]);

    uint8_t h_genome[G * ng_total * 3] = {};

    set_node(h_genome, 0, 1, 6, 0, 0);
    set_node(h_genome, 0, 2, 4, 1, 0);
    set_node(h_genome, 0, 3, 6, 0, 0);
    set_node(h_genome, 0, 4, 2, 3, 3);
    set_node(h_genome, 0, 5, 0, 2, 4);
    set_node(h_genome, 0, 7, 0, 5, 0);

    set_node(h_genome, 1, 1, 6, 0, 0);
    set_node(h_genome, 1, 2, 2, 1, 1);
    set_node(h_genome, 1, 7, 0, 2, 0);

    set_node(h_genome, 2, 1, 6, 0, 0);
    set_node(h_genome, 2, 2, 4, 1, 0);
    set_node(h_genome, 2, 7, 0, 2, 0);

    set_node(h_genome, 3, 1, 6, 0, 0);
    set_node(h_genome, 3, 2, 4, 1, 0);
    set_node(h_genome, 3, 3, 6, 2, 0);
    set_node(h_genome, 3, 7, 0, 3, 0);

    float h_params[G * n_p] = {};
    h_params[0 * n_p + 0] = 2.0f;
    h_params[0 * n_p + 2] = 1.5f;
    h_params[1 * n_p + 0] = 1.5f;
    h_params[2 * n_p + 0] = 0.7f;
    h_params[3 * n_p + 0] = 1.5f;
    h_params[3 * n_p + 2] = 2.0f;

    /* GPU buffers. */
    float   *d_params, *d_params_trial, *d_inputs, *d_targets,
            *d_state_v, *d_state_t, *d_out_v, *d_out_trial,
            *d_J, *d_r, *d_r_trial, *d_Js,
            *d_fnorm, *d_fnorm_trial, *d_fnorm_lin, *d_lam;
    int     *d_accept;
    uint8_t *d_genome;
    double  *d_delta;
    cudaMalloc(&d_params,       sizeof(h_params));
    cudaMalloc(&d_params_trial, sizeof(h_params));
    cudaMalloc(&d_inputs,       sizeof(h_inputs));
    cudaMalloc(&d_targets,      sizeof(h_targets));
    cudaMalloc(&d_genome,       sizeof(h_genome));
    cudaMalloc(&d_state_v,      sizeof(float) * G * (gi + gn) * N);
    cudaMalloc(&d_state_t,      sizeof(float) * G * (gi + gn) * N);
    cudaMalloc(&d_out_v,        sizeof(float) * G * go * N);
    cudaMalloc(&d_out_trial,    sizeof(float) * G * go * N);
    cudaMalloc(&d_J,            sizeof(float) * G * m * n_p);
    cudaMalloc(&d_r,            sizeof(float) * G * m);
    cudaMalloc(&d_r_trial,      sizeof(float) * G * m);
    cudaMalloc(&d_Js,           sizeof(float) * G * m);
    cudaMalloc(&d_fnorm,        sizeof(float) * G);
    cudaMalloc(&d_fnorm_trial,  sizeof(float) * G);
    cudaMalloc(&d_fnorm_lin,    sizeof(float) * G);
    cudaMalloc(&d_lam,          sizeof(float) * G);
    cudaMalloc(&d_accept,       sizeof(int)   * G);
    cudaMalloc(&d_delta,        sizeof(double) * G * n_p);

    cudaMemcpy(d_params,  h_params,  sizeof(h_params),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs,  h_inputs,  sizeof(h_inputs),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, sizeof(h_targets), cudaMemcpyHostToDevice);
    cudaMemcpy(d_genome,  h_genome,  sizeof(h_genome),  cudaMemcpyHostToDevice);

    const int    max_iter   = 8;
    const float  lam0       = 1e-3f;
    const size_t smem_bytes = (size_t)(2 * n_p * n_p + 3 * n_p) * sizeof(double);

    /* Initial λ — same for all individuals. */
    float h_lam[G]; for (int g = 0; g < G; ++g) h_lam[g] = lam0;
    cudaMemcpy(d_lam, h_lam, sizeof(h_lam), cudaMemcpyHostToDevice);

    /* Initial forward + residual + fnorm. */
    forward_value_kernel<<<G, 32>>>(d_params, d_inputs, d_genome,
                                    d_state_v, d_out_v);
    residual_kernel<<<G, 32>>>(d_out_v, d_targets, d_r, m);
    norm_kernel<<<(G + 31) / 32, 32>>>(d_r, d_fnorm, m);

    printf("\nLevenberg-Marquardt iteration  (initial λ = %.0e):\n\n", lam0);
    printf("iter   i0 loss        i1 loss        i2 loss        i3 loss      acc\n");

    float h_fnorm[G];
    int   h_accept[G];

    /* Print initial. */
    cudaMemcpy(h_fnorm, d_fnorm, sizeof(h_fnorm), cudaMemcpyDeviceToHost);
    printf("%4d ", 0);
    for (int g = 0; g < G; ++g) {
        double loss = (double)h_fnorm[g] * h_fnorm[g] / m;
        printf("  %.6e", loss);
    }
    printf("    -\n");

    for (int it = 1; it <= max_iter; ++it) {
        /* J at current params. */
        forward_jac_kernel<<<G, 32>>>(d_params, d_inputs, d_genome,
                                      d_state_v, d_state_t, d_out_v, d_J);
        /* Step direction. */
        lm_step_kernel<<<G, 32, smem_bytes>>>(d_J, d_r, d_lam, d_delta,
                                              d_Js, d_fnorm_lin, m, n_p);
        /* Trial point. */
        apply_step_kernel<<<G, 32>>>(d_params, d_delta, d_params_trial, n_p);
        forward_value_kernel<<<G, 32>>>(d_params_trial, d_inputs, d_genome,
                                        d_state_v, d_out_trial);
        residual_kernel<<<G, 32>>>(d_out_trial, d_targets, d_r_trial, m);
        norm_kernel<<<(G + 31) / 32, 32>>>(d_r_trial, d_fnorm_trial, m);
        /* Decide. */
        trust_region_kernel<<<(G + 31) / 32, 32>>>(d_fnorm, d_fnorm_trial,
                                                   d_fnorm_lin, d_lam, d_accept);
        accept_kernel<<<G, 32>>>(d_params_trial, d_params, d_r_trial, d_r,
                                 d_fnorm_trial, d_fnorm, d_accept, n_p, m);

        /* Print progress. */
        cudaMemcpy(h_fnorm,  d_fnorm,  sizeof(h_fnorm),  cudaMemcpyDeviceToHost);
        cudaMemcpy(h_accept, d_accept, sizeof(h_accept), cudaMemcpyDeviceToHost);
        printf("%4d ", it);
        for (int g = 0; g < G; ++g) {
            double loss = (double)h_fnorm[g] * h_fnorm[g] / m;
            printf("  %.6e", loss);
        }
        int n_acc = 0;
        for (int g = 0; g < G; ++g) n_acc += h_accept[g];
        printf("    %d/%d\n", n_acc, G);
    }

    /* Final report. */
    cudaMemcpy(h_params, d_params, sizeof(h_params), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lam,    d_lam,    sizeof(h_lam),    cudaMemcpyDeviceToHost);

    printf("\nFinal parameters vs optimal (all targets are at a = b = 1):\n");
    printf("  i0:  a = %+9.6f  (target 1.0)    b = %+9.6f  (target 1.0)    λ = %.2e\n",
           h_params[0 * n_p + 0], h_params[0 * n_p + 2], h_lam[0]);
    printf("  i1:  a = %+9.6f  (target 1.0)                                  λ = %.2e\n",
           h_params[1 * n_p + 0], h_lam[1]);
    printf("  i2:  a = %+9.6f  (target 1.0)                                  λ = %.2e\n",
           h_params[2 * n_p + 0], h_lam[2]);
    printf("  i3:  b = %+9.6f  (target 1.0)    a = %+9.6f  (target 1.0)    λ = %.2e\n",
           h_params[3 * n_p + 0], h_params[3 * n_p + 2], h_lam[3]);

    cudaFree(d_params);       cudaFree(d_params_trial); cudaFree(d_inputs);
    cudaFree(d_targets);      cudaFree(d_genome);       cudaFree(d_state_v);
    cudaFree(d_state_t);      cudaFree(d_out_v);        cudaFree(d_out_trial);
    cudaFree(d_J);            cudaFree(d_r);            cudaFree(d_r_trial);
    cudaFree(d_Js);           cudaFree(d_fnorm);        cudaFree(d_fnorm_trial);
    cudaFree(d_fnorm_lin);    cudaFree(d_lam);          cudaFree(d_accept);
    cudaFree(d_delta);
    return 0;
}
