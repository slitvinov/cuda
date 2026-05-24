/*
   Gauss-Newton CGP fitting.

       for it = 0 .. max_iter - 1:
           1. forward + Jacobian   (out, J)  at current params
           2. r = out - target
           3. solve  (JTJ + lambdaI) delta = JTr        (Cholesky, fp64)
           4. params -= delta

   No trust region, no accept/reject, no lambda adaptation -- every step is
   taken blindly with a small fixed lambda.  Converges quadratically near
   the optimum for well-conditioned problems and can blow up far from
   it; full Levenberg-Marquardt adds the damping that makes it robust.

   Targets are generated on the host from the known optimal params
   (a = b = 1 for every individual).
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
constexpr int m        = go * N;        /* residual length per individual */


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

/* r = out - target,  per individual, per sample. */
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
   params -= delta,  per individual, per parameter.
   delta is fp64 (Cholesky output); params stay fp32.
*/
__global__ void apply_step_kernel(float *params, const double *delta, int n) {
    const int g   = blockIdx.x;
    const int tid = threadIdx.x;
    const int B   = blockDim.x;
    for (int i = tid; i < n; i += B) {
        params[(size_t)g * n + i] -= (float)delta[(size_t)g * n + i];
    }
}

/*
   Cholesky solve:  build H = JTJ + lambdaI and g = JTr, factor H = LLT,
   forward solve Ly = g, back solve LTdelta = y.
*/
__global__ void cholesky_solve_kernel(const float *J,        // [G, m, n_p]
                                      const float *r,        // [G, m]
                                      double      *delta,    // [G, n_p]
                                      double       lam,
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

    /* H = JTJ + lambdaI  (parallel). */
    for (int idx = tid; idx < n * n; idx += B) {
        int i = idx / n, j = idx % n;
        double v = 0.0;
        for (int k = 0; k < mr; ++k) {
            v += (double)J[Jb + (size_t)k * n + i] *
                 (double)J[Jb + (size_t)k * n + j];
        }
        if (i == j) v += lam;
        H[i * n + j] = v;
    }
    /* g = JTr  (parallel). */
    for (int i = tid; i < n; i += B) {
        double v = 0.0;
        for (int k = 0; k < mr; ++k) {
            v += (double)J[Jb + (size_t)k * n + i] * (double)r[rb + k];
        }
        gv[i] = v;
    }
    __syncthreads();

    /* Cholesky + forward + back  (serial on tid 0). */
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


static void set_node(uint8_t *gen, int g, int row, uint8_t op,
                     uint8_t p0, uint8_t p1) {
    const int b = (g * ng_total + row) * 3;
    gen[b + 0] = op;
    gen[b + 1] = p0;
    gen[b + 2] = p1;
}

/*
   The "true" function each individual is supposed to fit, at the
   optimal parameter values a = 1, b = 1.
*/
static float target_fn(int ind, float x) {
    switch (ind) {
        case 0: return sinf(x) + x * x;     /* sin(a*x) + (b*x)^2 at a=b=1 */
        case 1: return x * x;                /* (a*x)^2 at a=1 */
        case 2: return sinf(x);              /* sin(a*x) at a=1 */
        case 3: return sinf(x);              /* a*sin(b*x) at a=b=1 */
    }
    return 0.0f;
}

int main(void) {
    float h_inputs[G * gi * N];
    for (int g = 0; g < G; ++g)
        for (int k = 0; k < N; ++k)
            h_inputs[g * gi * N + k] = -1.0f + 2.0f * (float)k / (N - 1);

    /* Generate target curves at the optimal params (a=b=1). */
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

    /* Initial params, perturbed away from the optimum. */
    float h_params[G * n_p] = {};
    h_params[0 * n_p + 0] = 2.0f;   // i0: a
    h_params[0 * n_p + 2] = 1.5f;   // i0: b
    h_params[1 * n_p + 0] = 1.5f;   // i1: a
    h_params[2 * n_p + 0] = 0.7f;   // i2: a
    h_params[3 * n_p + 0] = 1.5f;   // i3: b
    h_params[3 * n_p + 2] = 2.0f;   // i3: a

    /* GPU buffers. */
    float   *d_params, *d_inputs, *d_targets, *d_state_v, *d_state_t,
            *d_out_v, *d_J, *d_r;
    uint8_t *d_genome;
    double  *d_delta;
    cudaMalloc(&d_params,  sizeof(h_params));
    cudaMalloc(&d_inputs,  sizeof(h_inputs));
    cudaMalloc(&d_targets, sizeof(h_targets));
    cudaMalloc(&d_genome,  sizeof(h_genome));
    cudaMalloc(&d_state_v, sizeof(float) * G * (gi + gn) * N);
    cudaMalloc(&d_state_t, sizeof(float) * G * (gi + gn) * N);
    cudaMalloc(&d_out_v,   sizeof(float) * G * go * N);
    cudaMalloc(&d_J,       sizeof(float) * G * m * n_p);
    cudaMalloc(&d_r,       sizeof(float) * G * m);
    cudaMalloc(&d_delta,   sizeof(double) * G * n_p);

    cudaMemcpy(d_params,  h_params,  sizeof(h_params),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs,  h_inputs,  sizeof(h_inputs),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, sizeof(h_targets), cudaMemcpyHostToDevice);
    cudaMemcpy(d_genome,  h_genome,  sizeof(h_genome),  cudaMemcpyHostToDevice);

    const int    max_iter   = 8;
    const double lam        = 1e-3;
    const size_t smem_bytes = (size_t)(2 * n_p * n_p + 3 * n_p) * sizeof(double);

    printf("\nGauss-Newton iteration (fixed lambda = %.0e):\n\n", lam);
    printf("iter   i0 loss        i1 loss        i2 loss        i3 loss\n");

    float h_r[G * m];

    for (int it = 0; it <= max_iter; ++it) {
        forward_jac_kernel<<<G, 32>>>(d_params, d_inputs, d_genome,
                                      d_state_v, d_state_t, d_out_v, d_J);
        residual_kernel<<<G, 32>>>(d_out_v, d_targets, d_r, m);

        cudaMemcpy(h_r, d_r, sizeof(h_r), cudaMemcpyDeviceToHost);
        printf("%4d ", it);
        for (int g = 0; g < G; ++g) {
            double loss = 0.0;
            for (int k = 0; k < m; ++k) {
                double e = h_r[g * m + k];
                loss += e * e;
            }
            loss /= m;
            printf("  %.6e", loss);
        }
        printf("\n");

        if (it == max_iter) break;

        cholesky_solve_kernel<<<G, 32, smem_bytes>>>(d_J, d_r, d_delta,
                                                     lam, m, n_p);
        apply_step_kernel<<<G, 32>>>(d_params, d_delta, n_p);
    }

    /* Final report: where did the params converge to? */
    cudaMemcpy(h_params, d_params, sizeof(h_params), cudaMemcpyDeviceToHost);

    printf("\nFinal parameters vs optimal (all targets are at a = b = 1):\n");
    printf("  i0:  a = %+9.6f  (target 1.0)    b = %+9.6f  (target 1.0)\n",
           h_params[0 * n_p + 0], h_params[0 * n_p + 2]);
    printf("  i1:  a = %+9.6f  (target 1.0)\n",
           h_params[1 * n_p + 0]);
    printf("  i2:  a = %+9.6f  (target 1.0)\n",
           h_params[2 * n_p + 0]);
    printf("  i3:  b = %+9.6f  (target 1.0)    a = %+9.6f  (target 1.0)\n",
           h_params[3 * n_p + 0], h_params[3 * n_p + 2]);

    cudaFree(d_params); cudaFree(d_inputs); cudaFree(d_targets);
    cudaFree(d_genome); cudaFree(d_state_v); cudaFree(d_state_t);
    cudaFree(d_out_v);  cudaFree(d_J); cudaFree(d_r);  cudaFree(d_delta);
    return 0;
}
