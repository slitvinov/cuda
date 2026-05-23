/*
   bench_gp.cu — scaling sweep for forward_kernel.

   Sweeps over (G, gn, N) on one GPU and reports kernel-only time,
   effective DRAM bandwidth, and individuals/second.

   Build:  nvcc -arch=sm_80 -O2 bench_gp.cu -o bench_gp
   Run:    ./bench_gp
*/

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

constexpr int gi = 1;
constexpr int go = 1;

__device__ __forceinline__ float apply_op(uint8_t op, float v0, float v1) {
    switch (op) {
        case 0: return v0 + v1;
        case 1: return v0 - v1;
        case 2: return v0 * v1;
        case 3: return v0 / v1;
        case 4: return sinf(v0);
        case 5: return cosf(v0);
        default: return 0.0f;
    }
}

__global__ void forward_kernel(const float   *inputs,
                               const uint8_t *genome,
                               float         *state,
                               float         *out,
                               int gn, int N)
{
    const int    g          = blockIdx.x;
    const int    tid        = threadIdx.x;
    const int    B          = blockDim.x;
    const int    ng_total   = gi + gn + go;
    const size_t state_base = (size_t)g * (gi + gn) * N;

    for (int i = 0; i < gi; ++i) {
        const size_t dst = state_base + (size_t)i * N;
        const size_t src = (size_t)g * gi * N + (size_t)i * N;
        for (int k = tid; k < N; k += B) state[dst + k] = inputs[src + k];
    }
    __syncthreads();

    for (int j = 0; j < gn; ++j) {
        const size_t  row  = (size_t)g * ng_total + (gi + j);
        const uint8_t op   = genome[row * 3 + 0];
        const uint8_t ptr0 = genome[row * 3 + 1];
        const uint8_t ptr1 = genome[row * 3 + 2];

        const size_t in0_base = state_base + (size_t)ptr0 * N;
        const size_t in1_base = state_base + (size_t)ptr1 * N;
        const size_t dst_base = state_base + (size_t)(gi + j) * N;

        for (int k = tid; k < N; k += B) {
            float v0 = state[in0_base + k];
            float v1 = state[in1_base + k];
            state[dst_base + k] = apply_op(op, v0, v1);
        }
        __syncthreads();
    }

    for (int o = 0; o < go; ++o) {
        const size_t  out_row = (size_t)g * ng_total + (gi + gn + o);
        const uint8_t src_row = genome[out_row * 3 + 1];
        const size_t  src     = state_base + (size_t)src_row * N;
        const size_t  dst     = (size_t)(g * go + o) * N;
        for (int k = tid; k < N; k += B) out[dst + k] = state[src + k];
    }
}

static void make_random_genomes(uint8_t *gen, int G, int gn) {
    /* Per node: random op in [0,6), random ptrs in [0, gi+j). */
    const int ng_total = gi + gn + go;
    for (int g = 0; g < G; ++g) {
        for (int j = 0; j < gn; ++j) {
            int row = g * ng_total + (gi + j);
            int max_ptr = gi + j;
            gen[row*3 + 0] = (uint8_t)(rand() % 6);
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
    const int ng_total = gi + gn + go;
    size_t inputs_bytes = (size_t)G * gi * N * sizeof(float);
    size_t genome_bytes = (size_t)G * ng_total * 3;
    size_t state_bytes  = (size_t)G * (gi + gn) * N * sizeof(float);
    size_t out_bytes    = (size_t)G * go * N * sizeof(float);

    float   *d_inputs, *d_state, *d_out;
    uint8_t *d_genome;
    if (cudaMalloc(&d_inputs, inputs_bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_genome, genome_bytes) != cudaSuccess) { cudaFree(d_inputs); return -1; }
    if (cudaMalloc(&d_state,  state_bytes)  != cudaSuccess) { cudaFree(d_inputs); cudaFree(d_genome); return -1; }
    if (cudaMalloc(&d_out,    out_bytes)    != cudaSuccess) { cudaFree(d_inputs); cudaFree(d_genome); cudaFree(d_state); return -1; }

    std::vector<float> h_inputs(G * gi * N);
    for (size_t i = 0; i < h_inputs.size(); ++i)
        h_inputs[i] = -1.0f + 2.0f * (float)(i % 1024) / 1024.0f;
    std::vector<uint8_t> h_genome(genome_bytes, 0);
    make_random_genomes(h_genome.data(), G, gn);

    cudaMemcpy(d_inputs, h_inputs.data(),  inputs_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_genome, h_genome.data(),  genome_bytes, cudaMemcpyHostToDevice);

    int B = (N >= 128) ? 128 : ((N >= 32) ? 32 : N);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    forward_kernel<<<G, B>>>(d_inputs, d_genome, d_state, d_out, gn, N);  // warmup
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "launch/sync err at G=%d gn=%d N=%d: %s\n",
                G, gn, N, cudaGetErrorString(err));
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(d_inputs); cudaFree(d_genome); cudaFree(d_state); cudaFree(d_out);
        return -2;
    }

    cudaEventRecord(t0);
    for (int r = 0; r < reps; ++r)
        forward_kernel<<<G, B>>>(d_inputs, d_genome, d_state, d_out, gn, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_inputs);
    cudaFree(d_genome);
    cudaFree(d_state);
    cudaFree(d_out);

    return (double)ms / reps;
}

int main(void) {
    int Gs[]   = {64, 1024, 16384, 65536};
    int gns[]  = {16, 64, 256};
    int Ns[]   = {64, 256, 1024};

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs, %ld MB total)\n\n",
           prop.name, prop.multiProcessorCount, (long)(prop.totalGlobalMem >> 20));

    printf("%7s %5s %5s   %9s   %8s   %11s   %12s\n",
           "G", "gn", "N", "ms/call", "GB/s", "ind/s", "ns/ind/node");
    printf("%7s %5s %5s   %9s   %8s   %11s   %12s\n",
           "-------", "-----", "-----", "---------", "--------", "-----------", "------------");

    for (int gi_idx = 0; gi_idx < (int)(sizeof(Gs)/sizeof(Gs[0])); ++gi_idx)
    for (int gni    = 0; gni    < (int)(sizeof(gns)/sizeof(gns[0])); ++gni)
    for (int ni     = 0; ni     < (int)(sizeof(Ns)/sizeof(Ns[0]));   ++ni) {
        int G  = Gs[gi_idx];
        int gn = gns[gni];
        int N  = Ns[ni];

        size_t state_bytes = (size_t)G * (gi + gn) * N * sizeof(float);
        if (state_bytes > (size_t)20 * (1ULL << 30)) {
            printf("%7d %5d %5d   %9s   state=%.1f GB\n",
                   G, gn, N, "(skipped)", state_bytes / 1e9);
            continue;
        }

        int reps = ((double)G * gn * N > 1e7) ? 10 : 50;
        double ms = bench(G, gn, N, reps);
        if (ms == -1) { printf("%7d %5d %5d   (OOM)\n", G, gn, N); continue; }
        if (ms == -2) { printf("%7d %5d %5d   (launch err)\n", G, gn, N); continue; }

        double bytes = (double)G * ((double)gi*N*4 + (double)gn*3*N*4 + (double)go*N*4);
        double gbs   = bytes / (ms * 1e-3) / 1e9;
        double ind_s = G / (ms * 1e-3);
        double ns_per_indnode = ms * 1e6 / ((double)G * gn);

        printf("%7d %5d %5d   %9.3f   %8.1f   %11.3e   %12.2f\n",
               G, gn, N, ms, gbs, ind_s, ns_per_indnode);
    }
    return 0;
}
