/*
   bench_jgp.cu -- scaling sweep for the parameterized-Jacobian kernel.

   Same (G, gn, N) grid as bench_gp.cu and bench_dgp.cu so numbers are
   directly comparable.  Forward-mode Jacobian: one value pass followed
   by n_p = gn tangent passes (p=1), each producing one column of J.

   Build:  nvcc -arch=sm_80 -O2 bench_jgp.cu -o bench_jgp
   Run:    ./bench_jgp
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

    /* Pass 1: forward values only. */
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

    /* Pass 2: one tangent sweep per parameter q. */
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

static void make_random_genomes(uint8_t *gen, int G, int gn) {
    /* Use all 8 ops including par-ops. */
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
    size_t inputs_bytes = (size_t)G * gi * N * sizeof(float);
    size_t genome_bytes = (size_t)G * ng_total * 3;
    size_t state_bytes  = (size_t)G * (gi + gn) * N * sizeof(float);
    size_t out_bytes    = (size_t)G * go * N * sizeof(float);
    size_t params_bytes = (size_t)G * n_p * sizeof(float);
    size_t J_bytes      = (size_t)G * go * N * n_p * sizeof(float);

    float   *d_params, *d_inputs, *d_state_v, *d_state_t, *d_out_v, *d_J;
    uint8_t *d_genome;
    if (cudaMalloc(&d_params,  params_bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_inputs,  inputs_bytes) != cudaSuccess) { cudaFree(d_params); return -1; }
    if (cudaMalloc(&d_genome,  genome_bytes) != cudaSuccess) { cudaFree(d_params); cudaFree(d_inputs); return -1; }
    if (cudaMalloc(&d_state_v, state_bytes)  != cudaSuccess) { cudaFree(d_params); cudaFree(d_inputs); cudaFree(d_genome); return -1; }
    if (cudaMalloc(&d_state_t, state_bytes)  != cudaSuccess) { cudaFree(d_params); cudaFree(d_inputs); cudaFree(d_genome); cudaFree(d_state_v); return -1; }
    if (cudaMalloc(&d_out_v,   out_bytes)    != cudaSuccess) { cudaFree(d_params); cudaFree(d_inputs); cudaFree(d_genome); cudaFree(d_state_v); cudaFree(d_state_t); return -1; }
    if (cudaMalloc(&d_J,       J_bytes)      != cudaSuccess) { cudaFree(d_params); cudaFree(d_inputs); cudaFree(d_genome); cudaFree(d_state_v); cudaFree(d_state_t); cudaFree(d_out_v); return -1; }

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

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    forward_jac_kernel<<<G, B>>>(d_params, d_inputs, d_genome,
                                 d_state_v, d_state_t, d_out_v, d_J, gn, N);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "launch/sync err at G=%d gn=%d N=%d: %s\n",
                G, gn, N, cudaGetErrorString(err));
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(d_params); cudaFree(d_inputs); cudaFree(d_genome);
        cudaFree(d_state_v); cudaFree(d_state_t);
        cudaFree(d_out_v);  cudaFree(d_J);
        return -2;
    }

    cudaEventRecord(t0);
    for (int r = 0; r < reps; ++r)
        forward_jac_kernel<<<G, B>>>(d_params, d_inputs, d_genome,
                                     d_state_v, d_state_t, d_out_v, d_J, gn, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_params); cudaFree(d_inputs); cudaFree(d_genome);
    cudaFree(d_state_v); cudaFree(d_state_t);
    cudaFree(d_out_v);  cudaFree(d_J);

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
    printf("Parameterized forward + full Jacobian (n_p = gn, p=1).\n\n");

    printf("%7s %5s %5s   %10s   %8s   %11s   %12s\n",
           "G", "gn", "N", "ms/call", "GB/s", "ind/s", "ns/ind/node");
    printf("%7s %5s %5s   %10s   %8s   %11s   %12s\n",
           "-------", "-----", "-----", "----------", "--------", "-----------", "------------");

    for (int gi_idx = 0; gi_idx < (int)(sizeof(Gs)/sizeof(Gs[0])); ++gi_idx)
    for (int gni    = 0; gni    < (int)(sizeof(gns)/sizeof(gns[0])); ++gni)
    for (int ni     = 0; ni     < (int)(sizeof(Ns)/sizeof(Ns[0]));   ++ni) {
        int G  = Gs[gi_idx];
        int gn = gns[gni];
        int N  = Ns[ni];

        /* state_v + state_t + J  ~ 3*G*gn*N*4 bytes once you ignore the
           inputs/out fluff.  Skip configs that wouldn't fit on 80 GB. */
        size_t big_bytes = 3 * (size_t)G * (gi + gn) * N * sizeof(float);
        if (big_bytes > (size_t)40 * (1ULL << 30)) {
            printf("%7d %5d %5d   %10s   3*state~%.1f GB\n",
                   G, gn, N, "(skipped)", big_bytes / 1e9);
            continue;
        }

        /* Jacobian kernels are slow (n_p+1 sweeps).  Use fewer reps for big jobs. */
        int reps;
        double work = (double)G * gn * gn * N;
        if      (work > 1e10) reps = 2;
        else if (work > 1e8)  reps = 5;
        else                  reps = 30;

        double ms = bench(G, gn, N, reps);
        if (ms == -1) { printf("%7d %5d %5d   (OOM)\n", G, gn, N); continue; }
        if (ms == -2) { printf("%7d %5d %5d   (launch err)\n", G, gn, N); continue; }

        /* Memory accounting per call:
             inputs   : G * gi*N * 4
             genome   : G * ng_total * 3
             state_v  : Pass 1 R/W = G*gn*3*N*4
             state_t  : Pass 2 R/W = n_p * G*gn*3*N*4
             J        : G * go * N * n_p * 4 (write)                       */
        int n_p = gn * p;
        double bytes = (double)G * ((double)gi*N*4
                                    + (double)gn*3*N*4
                                    + (double)n_p*gn*3*N*4
                                    + (double)go*N*4
                                    + (double)go*N*n_p*4);
        double gbs   = bytes / (ms * 1e-3) / 1e9;
        double ind_s = G / (ms * 1e-3);
        double ns_per_indnode = ms * 1e6 / ((double)G * gn);

        printf("%7d %5d %5d   %10.3f   %8.1f   %11.3e   %12.2f\n",
               G, gn, N, ms, gbs, ind_s, ns_per_indnode);
    }
    return 0;
}
