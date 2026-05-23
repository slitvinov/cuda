/*
   Scaling sweep for one-shot Cholesky solve.  Pre-computes J once via
   the forward+Jacobian kernel, then times the Cholesky solve alone.

   Configurations where the required dynamic shared memory exceeds the
   opt-in cap are skipped.
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

/* Builds H = JᵀJ + λI, synthetic g = Jᵀ·1, Cholesky factor, forward
   and back solves.  No verification pass. */
__global__ void cholesky_solve_kernel(const float *J,
                                      double      *delta,
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
    for (int i = tid; i < n; i += B) {
        double v = 0.0;
        for (int k = 0; k < mr; ++k) v += (double)J[Jb + (size_t)k * n + i];
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

static double bench(int G, int gn, int N, int reps) {
    const int    ng_total = gi + gn + go;
    const int    n_p      = gn * p;
    const int    m        = go * N;
    const size_t inputs_bytes = (size_t)G * gi * N * sizeof(float);
    const size_t genome_bytes = (size_t)G * ng_total * 3;
    const size_t state_bytes  = (size_t)G * (gi + gn) * N * sizeof(float);
    const size_t out_bytes    = (size_t)G * go * N * sizeof(float);
    const size_t params_bytes = (size_t)G * n_p * sizeof(float);
    const size_t J_bytes      = (size_t)G * m * n_p * sizeof(float);
    const size_t delta_bytes  = (size_t)G * n_p * sizeof(double);
    const size_t smem_bytes   = (size_t)(2 * n_p * n_p + 3 * n_p) * sizeof(double);

    if (smem_bytes > 100 * 1024) return -3;
    cudaFuncSetAttribute(cholesky_solve_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);

    float   *d_params, *d_inputs, *d_state_v, *d_state_t, *d_out_v, *d_J;
    uint8_t *d_genome;
    double  *d_delta;
    if (cudaMalloc(&d_params,  params_bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_inputs,  inputs_bytes) != cudaSuccess) { cudaFree(d_params); return -1; }
    if (cudaMalloc(&d_genome,  genome_bytes) != cudaSuccess) { cudaFree(d_params); cudaFree(d_inputs); return -1; }
    if (cudaMalloc(&d_state_v, state_bytes)  != cudaSuccess) { cudaFree(d_params); cudaFree(d_inputs); cudaFree(d_genome); return -1; }
    if (cudaMalloc(&d_state_t, state_bytes)  != cudaSuccess) { cudaFree(d_params); cudaFree(d_inputs); cudaFree(d_genome); cudaFree(d_state_v); return -1; }
    if (cudaMalloc(&d_out_v,   out_bytes)    != cudaSuccess) { cudaFree(d_params); cudaFree(d_inputs); cudaFree(d_genome); cudaFree(d_state_v); cudaFree(d_state_t); return -1; }
    if (cudaMalloc(&d_J,       J_bytes)      != cudaSuccess) { cudaFree(d_params); cudaFree(d_inputs); cudaFree(d_genome); cudaFree(d_state_v); cudaFree(d_state_t); cudaFree(d_out_v); return -1; }
    if (cudaMalloc(&d_delta,   delta_bytes)  != cudaSuccess) { cudaFree(d_params); cudaFree(d_inputs); cudaFree(d_genome); cudaFree(d_state_v); cudaFree(d_state_t); cudaFree(d_out_v); cudaFree(d_J); return -1; }

    std::vector<float>   h_inputs(G * gi * N);
    std::vector<float>   h_params(G * n_p, 1.0f);
    std::vector<uint8_t> h_genome(genome_bytes, 0);
    for (size_t i = 0; i < h_inputs.size(); ++i)
        h_inputs[i] = -1.0f + 2.0f * (float)(i % 1024) / 1024.0f;
    make_random_genomes(h_genome.data(), G, gn);

    cudaMemcpy(d_inputs, h_inputs.data(), inputs_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, h_params.data(), params_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_genome, h_genome.data(), genome_bytes, cudaMemcpyHostToDevice);

    int B = (N >= 128) ? 128 : ((N >= 32) ? 32 : N);

    /* Pre-compute J once (not timed). */
    forward_jac_kernel<<<G, B>>>(d_params, d_inputs, d_genome,
                                 d_state_v, d_state_t, d_out_v, d_J, gn, N);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "forward_jac err G=%d gn=%d N=%d: %s\n",
                G, gn, N, cudaGetErrorString(err));
        cudaFree(d_params); cudaFree(d_inputs); cudaFree(d_genome);
        cudaFree(d_state_v); cudaFree(d_state_t);
        cudaFree(d_out_v); cudaFree(d_J); cudaFree(d_delta);
        return -2;
    }

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    /* Warmup. */
    cholesky_solve_kernel<<<G, 128, smem_bytes>>>(d_J, d_delta, 1e-3, m, n_p);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cholesky err G=%d gn=%d N=%d: %s\n",
                G, gn, N, cudaGetErrorString(err));
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(d_params); cudaFree(d_inputs); cudaFree(d_genome);
        cudaFree(d_state_v); cudaFree(d_state_t);
        cudaFree(d_out_v); cudaFree(d_J); cudaFree(d_delta);
        return -2;
    }

    cudaEventRecord(t0);
    for (int r = 0; r < reps; ++r) {
        cholesky_solve_kernel<<<G, 128, smem_bytes>>>(d_J, d_delta, 1e-3, m, n_p);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_params); cudaFree(d_inputs); cudaFree(d_genome);
    cudaFree(d_state_v); cudaFree(d_state_t);
    cudaFree(d_out_v); cudaFree(d_J); cudaFree(d_delta);
    return (double)ms / reps;
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
    printf("One-shot Cholesky solve (J pre-computed, untimed).\n\n");

    printf("%7s %5s %5s   %9s   %11s    %s\n",
           "G", "gn", "N", "ms/call", "solves/sec", "note");
    printf("%7s %5s %5s   %9s   %11s    %s\n",
           "-------", "-----", "-----", "---------", "-----------", "----");

    for (int gi_idx = 0; gi_idx < (int)(sizeof(Gs)/sizeof(Gs[0])); ++gi_idx)
    for (int gni    = 0; gni    < (int)(sizeof(gns)/sizeof(gns[0])); ++gni)
    for (int ni     = 0; ni     < (int)(sizeof(Ns)/sizeof(Ns[0]));   ++ni) {
        int G  = Gs[gi_idx];
        int gn = gns[gni];
        int N  = Ns[ni];
        int n_p = gn * p;

        size_t big = (size_t)G * (gi + gn) * N * sizeof(float) * 3;
        if (big > (size_t)40 * (1ULL << 30)) {
            printf("%7d %5d %5d   %9s    state>40GB\n", G, gn, N, "(skip)");
            continue;
        }

        int reps = ((double)G * n_p * n_p > 1e7) ? 5 : 30;
        double ms = bench(G, gn, N, reps);
        if (ms == -1) { printf("%7d %5d %5d   (OOM)\n", G, gn, N); continue; }
        if (ms == -2) { printf("%7d %5d %5d   (launch err)\n", G, gn, N); continue; }
        if (ms == -3) {
            printf("%7d %5d %5d   %9s    smem over cap\n", G, gn, N, "(skip)");
            continue;
        }

        double solves_per_s = G / (ms * 1e-3);
        printf("%7d %5d %5d   %9.3f   %11.3e\n", G, gn, N, ms, solves_per_s);
    }
    return 0;
}
