/*
   Cholesky solve for one CGP step.  Per individual:
     1. Build  H = JTJ + lambdaI   (n_p x n_p, symmetric positive definite)
     2. Build  g = JT * r     (here r = all-ones, self-contained demo)
     3. Cholesky factor  H = L LT
     4. Forward solve   L y = g
     5. Back solve      LT delta = y
     6. Verify by recomputing  H * delta  and comparing against g.

   The kernel uses double precision for H, L, g, delta.  Single precision
   is fine for the forward pass and the Jacobian, but JTJ is often
   ill-conditioned and FP32 Cholesky breaks down.

   Inside the kernel, H/g build and the H*delta verify are block-parallel;
   the Cholesky and the two triangular solves run single-threaded on
   tid == 0 because n_p is small.

   No iteration, no damping, no residual from a target -- just the small
   dense linear algebra in isolation, verified against the equation it
   was supposed to solve.
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

/* ---- new in chgp.cu ---- */

/*
   Cholesky-solve kernel.  One block per individual.  Each block holds
   H, L, g, y, delta in shared memory (double precision).

   Parallel:  H = JTJ + lambdaI  build         (one thread per (i,j) cell)
              g = JT * 1     build         (one thread per i)
              H * delta          verify       (one thread per i)
   Serial:    Cholesky factor              (tid == 0 only)
              forward + back substitution  (tid == 0 only)

   J has shape [G, go, N, n_p]; we treat it as [G, m, n] with m = go*N.
*/
__global__ void cholesky_solve_kernel(const float *J,        // [G, m, n_p]
                                      double      *delta,    // [G, n_p]
                                      double      *max_err,  // [G]
                                      double       lam,
                                      int m, int n)
{
    extern __shared__ double sm[];
    double *H = sm;                 // n x n  (symmetric, stored full)
    double *L = H + (size_t)n * n;  // n x n  (lower triangular)
    double *gv = L + (size_t)n * n; // n      (the RHS)
    double *y  = gv + n;            // n      (intermediate)
    double *s  = y + n;             // n      (solution delta)

    const int    g   = blockIdx.x;
    const int    tid = threadIdx.x;
    const int    B   = blockDim.x;
    const size_t Jb  = (size_t)g * m * n;

    /* 1. H = JTJ + lambdaI   (parallel). */
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

    /* 2. g = JT * 1  (residual vector = all ones, for a self-contained
          demo without targets).  Parallel: one thread per i. */
    for (int i = tid; i < n; i += B) {
        double v = 0.0;
        for (int k = 0; k < m; ++k) {
            v += (double)J[Jb + (size_t)k * n + i];
        }
        gv[i] = v;
    }
    __syncthreads();

    /* 3, 4, 5. Cholesky factor + forward + back solve -- serial, since
       n_p is small (<= ~100 in practice) and Cholesky has tight data
       dependencies that don't parallelize cheaply. */
    if (tid == 0) {
        /* H = L LT  (lower triangle of L). */
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                double v = H[i * n + j];
                for (int k = 0; k < j; ++k) v -= L[i*n + k] * L[j*n + k];
                L[i * n + j] = (i == j) ? sqrt(v) : v / L[j * n + j];
            }
        }
        /* L y = g   (forward). */
        for (int i = 0; i < n; ++i) {
            double v = gv[i];
            for (int k = 0; k < i; ++k) v -= L[i*n + k] * y[k];
            y[i] = v / L[i * n + i];
        }
        /* LT s = y   (back).  L_{k,i} for k > i lives in lower triangle of L. */
        for (int i = n - 1; i >= 0; --i) {
            double v = y[i];
            for (int k = i + 1; k < n; ++k) v -= L[k*n + i] * s[k];
            s[i] = v / L[i * n + i];
        }
        for (int i = 0; i < n; ++i) delta[(size_t)g * n + i] = s[i];
    }
    __syncthreads();

    /* 6. Verify: max_i |sum_j H_{ij} delta_j - g_i|   (parallel rows; reduce
          across rows via shared scratch -- but n is small enough that
          we do it single-threaded for clarity). */
    if (tid == 0) {
        double max_e = 0.0;
        for (int i = 0; i < n; ++i) {
            double v = 0.0;
            for (int j = 0; j < n; ++j) v += H[i * n + j] * s[j];
            double e = fabs(v - gv[i]);
            if (e > max_e) max_e = e;
        }
        max_err[g] = max_e;
    }
}


static void set_node(uint8_t *gen, int g, int row, uint8_t op,
                     uint8_t p0, uint8_t p1) {
    const int b = (g * ng_total + row) * 3;
    gen[b + 0] = op;
    gen[b + 1] = p0;
    gen[b + 2] = p1;
}

int main(void) {
    float h_inputs[G * gi * N];
    for (int g = 0; g < G; ++g)
        for (int k = 0; k < N; ++k)
            h_inputs[g * gi * N + k] = -1.0f + 2.0f * (float)k / (N - 1);

    uint8_t h_genome[G * ng_total * 3] = {};

    /*
       Four non-linear-in-parameter individuals:
         i0:  y = sin(a*x) + (b*x)^2
         i1:  y = (a*x)^2
         i2:  y = sin(a*x)
         i3:  y = a * sin(b*x)
    */
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
    h_params[0 * n_p + 0] = 2.0f;   // i0: a
    h_params[0 * n_p + 2] = 1.5f;   // i0: b
    h_params[1 * n_p + 0] = 1.5f;   // i1: a
    h_params[2 * n_p + 0] = 0.7f;   // i2: a
    h_params[3 * n_p + 0] = 1.5f;   // i3: b
    h_params[3 * n_p + 2] = 2.0f;   // i3: a

    /* GPU. */
    float   *d_params, *d_inputs, *d_state_v, *d_state_t, *d_out_v, *d_J;
    uint8_t *d_genome;
    constexpr int m = go * N;          // residual length
    size_t bytes_J     = (size_t)G * m * n_p * sizeof(float);
    size_t bytes_delta = (size_t)G * n_p * sizeof(double);
    size_t bytes_err   = (size_t)G * sizeof(double);

    cudaMalloc(&d_params,  sizeof(h_params));
    cudaMalloc(&d_inputs,  sizeof(h_inputs));
    cudaMalloc(&d_genome,  sizeof(h_genome));
    cudaMalloc(&d_state_v, sizeof(float) * G * (gi + gn) * N);
    cudaMalloc(&d_state_t, sizeof(float) * G * (gi + gn) * N);
    cudaMalloc(&d_out_v,   sizeof(float) * G * go * N);
    cudaMalloc(&d_J,       bytes_J);

    double *d_delta, *d_err;
    cudaMalloc(&d_delta, bytes_delta);
    cudaMalloc(&d_err,   bytes_err);

    cudaMemcpy(d_params, h_params, sizeof(h_params), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs, h_inputs, sizeof(h_inputs), cudaMemcpyHostToDevice);
    cudaMemcpy(d_genome, h_genome, sizeof(h_genome), cudaMemcpyHostToDevice);

    /* Stage 1: produce J (same as jgp.cu). */
    forward_jac_kernel<<<G, 32>>>(d_params, d_inputs, d_genome,
                                  d_state_v, d_state_t, d_out_v, d_J);

    /* Stage 2: build H = JTJ + lambdaI, solve Hdelta = g, verify. */
    const double lam = 1e-3;
    size_t smem_bytes = (size_t)(2 * n_p * n_p + 3 * n_p) * sizeof(double);
    cholesky_solve_kernel<<<G, 32, smem_bytes>>>(d_J, d_delta, d_err,
                                                  lam, m, n_p);
    cudaDeviceSynchronize();

    double  h_delta[G * n_p];
    double  h_err  [G];
    cudaMemcpy(h_delta, d_delta, bytes_delta, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_err,   d_err,   bytes_err,   cudaMemcpyDeviceToHost);

    /* Report. */
    printf("\nCholesky solve verification.  lambda = %.1e,  m = %d,  n_p = %d.\n",
           lam, m, n_p);
    printf("\ndelta vector per individual (one row per individual, n_p=6 columns):\n");
    printf("           q=0          q=1          q=2          q=3          q=4          q=5\n");
    for (int g = 0; g < G; ++g) {
        printf("  i%d  ", g);
        for (int q = 0; q < n_p; ++q) {
            printf("  %+10.4e", h_delta[g * n_p + q]);
        }
        printf("\n");
    }

    printf("\nVerification: max |H*delta - g| per individual\n");
    const char *labels[G] = {
        "i0  sin(a*x) + (b*x)^2   active q = {0, 2}",
        "i1  (a*x)^2              active q = {0}",
        "i2  sin(a*x)            active q = {0}",
        "i3  a*sin(b*x)          active q = {0, 2}",
    };
    for (int g = 0; g < G; ++g) {
        printf("  %-40s   %.3e\n", labels[g], h_err[g]);
    }

    cudaFree(d_params); cudaFree(d_inputs); cudaFree(d_genome);
    cudaFree(d_state_v); cudaFree(d_state_t);
    cudaFree(d_out_v); cudaFree(d_J);
    cudaFree(d_delta); cudaFree(d_err);
    return 0;
}
