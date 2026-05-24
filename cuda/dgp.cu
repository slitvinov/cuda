/*
   Differentiable CGP via forward-mode automatic differentiation.

   Alongside every node's value we propagate a tangent  d(node)/dx,
   seeded with tangent = 1 on the input row (since dx/dx = 1) and
   chain-ruled through every op.  The kernel returns both  y(x)  and
   dy/dx  at each sample.

   Verification: each of the four hand-crafted individuals has a known
   analytic gradient, which we compare against the GPU result.
       i0:  y = sin(x) + x^2    dy/dx = cos(x) + 2x
       i1:  y = x^2             dy/dx = 2x
       i2:  y = sin(x)         dy/dx = cos(x)
       i3:  y = x              dy/dx = 1
*/

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

/* ---- identical to gp.cu ---- */

constexpr int G  = 4;
constexpr int gi = 1;
constexpr int gn = 6;
constexpr int go = 1;
constexpr int N  = 16;
constexpr int ng_total = gi + gn + go;

/*
   AD-aware op: same six ops as gp.cu, but each one now returns both the
   value and the tangent (chain-rule applied per op).
*/
__device__ __forceinline__ void apply_op_ad(uint8_t op,
                                            float v0, float v1,
                                            float d0, float d1,
                                            float *v, float *d) {
    switch (op) {
        case 0: *v = v0 + v1;     *d = d0 + d1;                break;  // add
        case 1: *v = v0 - v1;     *d = d0 - d1;                break;  // sub
        case 2: *v = v0 * v1;     *d = d0 * v1 + v0 * d1;      break;  // mul
        case 3: {                                                       // div
            float inv = 1.0f / v1;
            *v = v0 * inv;
            *d = (d0 - (*v) * d1) * inv;
            break;
        }
        case 4: *v = sinf(v0);    *d =  cosf(v0) * d0;         break;  // sin
        case 5: *v = cosf(v0);    *d = -sinf(v0) * d0;         break;  // cos
        default: *v = 0.0f;       *d = 0.0f;
    }
}

/*
   Same kernel structure as gp.cu's forward_kernel.  Extension: parallel
   tangent buffer  state_d  flows alongside  state_v.
*/
__global__ void forward_kernel_ad(const float   *inputs,    // [G, gi, N]
                                  const uint8_t *genome,    // [G, ng_total, 3]
                                  float         *state_v,   // [G, gi+gn, N]
                                  float         *state_d,   // [G, gi+gn, N]
                                  float         *out_v,     // [G, go, N]
                                  float         *out_d)     // [G, go, N]
{
    const int    g          = blockIdx.x;
    const int    tid        = threadIdx.x;
    const int    B          = blockDim.x;
    const size_t state_base = (size_t)g * (gi + gn) * N;

    /* 1. Stage inputs.  Tangent seed: dx/dx = 1 on the input row. */
    for (int i = 0; i < gi; ++i) {
        const size_t dst = state_base + (size_t)i * N;
        const size_t src = (size_t)g * gi * N + (size_t)i * N;
        for (int k = tid; k < N; k += B) {
            state_v[dst + k] = inputs[src + k];
            state_d[dst + k] = 1.0f;
        }
    }
    __syncthreads();

    /* 2. Evaluate nodes, propagating (value, tangent) together. */
    for (int j = 0; j < gn; ++j) {
        const size_t  row  = (size_t)g * ng_total + (gi + j);
        const uint8_t op   = genome[row * 3 + 0];
        const uint8_t ptr0 = genome[row * 3 + 1];
        const uint8_t ptr1 = genome[row * 3 + 2];

        const size_t in0_base = state_base + (size_t)ptr0 * N;
        const size_t in1_base = state_base + (size_t)ptr1 * N;
        const size_t dst_base = state_base + (size_t)(gi + j) * N;

        for (int k = tid; k < N; k += B) {
            float v, d;
            apply_op_ad(op,
                        state_v[in0_base + k], state_v[in1_base + k],
                        state_d[in0_base + k], state_d[in1_base + k],
                        &v, &d);
            state_v[dst_base + k] = v;
            state_d[dst_base + k] = d;
        }
        __syncthreads();
    }

    /* 3. Materialize outputs (value AND tangent). */
    for (int o = 0; o < go; ++o) {
        const size_t  out_row = (size_t)g * ng_total + (gi + gn + o);
        const uint8_t src_row = genome[out_row * 3 + 1];
        const size_t  src     = state_base + (size_t)src_row * N;
        const size_t  dst     = (size_t)(g * go + o) * N;
        for (int k = tid; k < N; k += B) {
            out_v[dst + k] = state_v[src + k];
            out_d[dst + k] = state_d[src + k];
        }
    }
}

/* ---- host driver: identical genome construction to gp.cu ---- */

static void set_node(uint8_t *gen, int g, int row, uint8_t op,
                     uint8_t p0, uint8_t p1) {
    const int b = (g * ng_total + row) * 3;
    gen[b + 0] = op;
    gen[b + 1] = p0;
    gen[b + 2] = p1;
}

/* Analytic dy/dx for the four hand-crafted individuals. */
static float analytic_grad(int ind, float x) {
    switch (ind) {
        case 0: return cosf(x) + 2.0f * x;   // sin(x) + x^2
        case 1: return 2.0f * x;             // x^2
        case 2: return cosf(x);              // sin(x)
        case 3: return 1.0f;                 // x
    }
    return 0.0f;
}

int main(void) {
    float h_inputs[G * gi * N];
    for (int g = 0; g < G; ++g)
        for (int k = 0; k < N; ++k)
            h_inputs[g * gi * N + k] = -1.0f + 2.0f * (float)k / (N - 1);

    uint8_t h_genome[G * ng_total * 3] = {};

    /* Same four genomes as gp.cu. */

    /* Individual 0 -- y = sin(x) + x^2. */
    set_node(h_genome, 0, 1, 2, 0, 0);
    set_node(h_genome, 0, 2, 4, 0, 0);
    set_node(h_genome, 0, 3, 0, 1, 2);
    set_node(h_genome, 0, 7, 0, 3, 0);

    /* Individual 1 -- y = x^2. */
    set_node(h_genome, 1, 1, 2, 0, 0);
    set_node(h_genome, 1, 7, 0, 1, 0);

    /* Individual 2 -- y = sin(x). */
    set_node(h_genome, 2, 1, 4, 0, 0);
    set_node(h_genome, 2, 7, 0, 1, 0);

    /* Individual 3 -- y = x. */
    set_node(h_genome, 3, 7, 0, 0, 0);

    /* GPU side -- same as gp.cu, plus the tangent buffers. */
    float   *d_inputs, *d_state_v, *d_state_d, *d_out_v, *d_out_d;
    uint8_t *d_genome;
    cudaMalloc(&d_inputs,  sizeof(h_inputs));
    cudaMalloc(&d_genome,  sizeof(h_genome));
    cudaMalloc(&d_state_v, sizeof(float) * G * (gi + gn) * N);
    cudaMalloc(&d_state_d, sizeof(float) * G * (gi + gn) * N);
    cudaMalloc(&d_out_v,   sizeof(float) * G * go * N);
    cudaMalloc(&d_out_d,   sizeof(float) * G * go * N);
    cudaMemcpy(d_inputs, h_inputs, sizeof(h_inputs), cudaMemcpyHostToDevice);
    cudaMemcpy(d_genome, h_genome, sizeof(h_genome), cudaMemcpyHostToDevice);

    forward_kernel_ad<<<G, 32>>>(d_inputs, d_genome,
                                 d_state_v, d_state_d,
                                 d_out_v,   d_out_d);
    cudaDeviceSynchronize();

    float h_out_v[G * go * N];
    float h_out_d[G * go * N];
    cudaMemcpy(h_out_v, d_out_v, sizeof(h_out_v), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_d, d_out_d, sizeof(h_out_d), cudaMemcpyDeviceToHost);

    /* GPU dy/dx vs analytic dy/dx, per individual. */
    printf("\n   x          i0 (cos x + 2x)       i1 (2x)             i2 (cos x)          i3 (1)\n");
    printf("          GPU        ref      GPU        ref      GPU        ref      GPU      ref\n");
    for (int k = 0; k < N; ++k) {
        printf("%+6.3f ", h_inputs[k]);
        for (int g = 0; g < G; ++g) {
            float gpu = h_out_d[g * N + k];
            float ref = analytic_grad(g, h_inputs[k]);
            printf("  %+8.4f  %+8.4f", gpu, ref);
        }
        printf("\n");
    }

    printf("\nMax |GPU - analytic| per individual:\n");
    const char *names[G] = {
        "i0 (cos x + 2x)",
        "i1 (2x)",
        "i2 (cos x)",
        "i3 (1)",
    };
    for (int g = 0; g < G; ++g) {
        float max_err = 0.0f;
        for (int k = 0; k < N; ++k) {
            float gpu = h_out_d[g * N + k];
            float ref = analytic_grad(g, h_inputs[k]);
            float e = fabsf(gpu - ref);
            if (e > max_err) max_err = e;
        }
        printf("  %-20s  %.3e\n", names[g], max_err);
    }

    cudaFree(d_inputs); cudaFree(d_genome);
    cudaFree(d_state_v); cudaFree(d_state_d);
    cudaFree(d_out_v); cudaFree(d_out_d);
    return 0;
}
