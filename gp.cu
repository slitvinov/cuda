/*
   gp.cu — Cartesian Genetic Programming, GPU forward pass.

   One self-contained file.  Build:
       nvcc -arch=sm_80 -O2 gp.cu -o gp
   Run:
       ./gp

   What this demonstrates:
     - The CGP genome layout (G individuals × ng_total rows × 3 bytes/row).
     - The forward-evaluation kernel: one block per individual, threads
       cooperate on the N sample points.
     - The "state" activation buffer that holds every node's value at
       every sample point.

   Four hand-crafted individuals all evaluate the same N=16 inputs in
   [-1, 1].  We then print each individual's prediction next to the
   target sin(x)+x*x and report MSE.  One individual is the correct
   model; the others are wrong in different ways.  There is no
   mutation, no selection, no fitness loop here — only the forward pass.
*/

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

/*
   Problem dimensions (compile-time for educational clarity; in a real
   implementation these are kernel arguments).
*/
constexpr int G  = 4;                // individuals (population size)
constexpr int gi = 1;                // input nodes: one variable, x
constexpr int gn = 6;                // internal computational nodes
constexpr int go = 1;                // output nodes
constexpr int N  = 16;               // sample points per individual
constexpr int ng_total = gi + gn + go;

/*
   Op set.  Each op consumes up to two operands (v0, v1) and returns a
   float.  Ops that use only one operand simply ignore v1.
*/
__device__ __forceinline__ float apply_op(uint8_t op, float v0, float v1) {
    switch (op) {
        case 0: return v0 + v1;     // add
        case 1: return v0 - v1;     // sub
        case 2: return v0 * v1;     // mul
        case 3: return v0 / v1;     // div  (no zero-guard: IEEE inf/NaN propagate)
        case 4: return sinf(v0);    // sin
        case 5: return cosf(v0);    // cos
        default: return 0.0f;
    }
}

/*
   Forward kernel — one block per individual, blockDim.x threads stride
   over N.

   Genome layout (uint8, shape [G, ng_total, 3]):
       row 0..gi-1                   : input rows.  Fields ignored.
       row gi..gi+gn-1               : internal nodes.  (op, ptr0, ptr1)
       row gi+gn..gi+gn+go-1         : output rows.   (_, src_row, _)
                                       "ptr0" doubles as "which state row
                                        to copy into out[o]".

   State (float, shape [G, gi+gn, N]):
       row 0..gi-1                   : holds the input samples
       row gi+j                      : holds the output of internal node j
*/
__global__ void forward_kernel(const float   *inputs,   // [G, gi, N]
                               const uint8_t *genome,   // [G, ng_total, 3]
                               float         *state,    // [G, gi+gn, N]
                               float         *out)      // [G, go, N]
{
    const int    g          = blockIdx.x;
    const int    tid        = threadIdx.x;
    const int    B          = blockDim.x;
    const size_t state_base = (size_t)g * (gi + gn) * N;

    /* 1. Stage inputs into the first gi rows of state. */
    for (int i = 0; i < gi; ++i) {
        const size_t dst = state_base + (size_t)i * N;
        const size_t src = (size_t)g * gi * N + (size_t)i * N;
        for (int k = tid; k < N; k += B) state[dst + k] = inputs[src + k];
    }
    __syncthreads();

    /*
       2. Evaluate the gn computational nodes in genome order.  Each
          node reads two earlier rows of state and writes its own row.
    */
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
        __syncthreads();          // later nodes may depend on this row
    }

    /*
       3. Materialize outputs: each output row's ptr0 names the state
          row to copy into out[g, o, :].
    */
    for (int o = 0; o < go; ++o) {
        const size_t  out_row = (size_t)g * ng_total + (gi + gn + o);
        const uint8_t src_row = genome[out_row * 3 + 1];
        const size_t  src     = state_base + (size_t)src_row * N;
        const size_t  dst     = (size_t)(g * go + o) * N;
        for (int k = tid; k < N; k += B) out[dst + k] = state[src + k];
    }
}

/* Host driver. */

static void set_node(uint8_t *gen, int g, int row, uint8_t op,
                     uint8_t p0, uint8_t p1) {
    const int b = (g * ng_total + row) * 3;
    gen[b + 0] = op;
    gen[b + 1] = p0;
    gen[b + 2] = p1;
}

int main(void) {
    /* Inputs: x in [-1, 1] at N points, identical across individuals. */
    float h_inputs[G * gi * N];
    for (int g = 0; g < G; ++g)
        for (int k = 0; k < N; ++k)
            h_inputs[g * gi * N + k] = -1.0f + 2.0f * (float)k / (N - 1);

    /* Target: y = sin(x) + x*x. */
    float h_target[N];
    for (int k = 0; k < N; ++k) {
        const float x = h_inputs[k];
        h_target[k] = sinf(x) + x * x;
    }

    /*
       Hand-craft four genomes.  In all cases:
         state row 0 = x        (input)
         state rows 1..6 = node outputs (we have gn = 6 internal nodes)
         row 7 = output row, ptr0 = which state row goes to out[0]
    */
    uint8_t h_genome[G * ng_total * 3] = {};

    /*
       Individual 0 — the true model: y = sin(x) + x*x.
       Nodes 3..5 (rows 4..6) are inactive: still evaluated, but their
       outputs are never referenced.  This is normal in CGP.
    */
    set_node(h_genome, 0, 1, 2, 0, 0);   // node 0: mul → x*x        in row 1
    set_node(h_genome, 0, 2, 4, 0, 0);   // node 1: sin → sin(x)     in row 2
    set_node(h_genome, 0, 3, 0, 1, 2);   // node 2: add → x*x+sin(x) in row 3
    set_node(h_genome, 0, 7, 0, 3, 0);   // output ← row 3

    /* Individual 1 — wrong, predicts just x*x. */
    set_node(h_genome, 1, 1, 2, 0, 0);
    set_node(h_genome, 1, 7, 0, 1, 0);   // output ← row 1

    /* Individual 2 — wrong, predicts just sin(x). */
    set_node(h_genome, 2, 1, 4, 0, 0);
    set_node(h_genome, 2, 7, 0, 1, 0);   // output ← row 1

    /* Individual 3 — very wrong, predicts x itself. */
    set_node(h_genome, 3, 7, 0, 0, 0);   // output ← row 0 (the input)

    /* GPU side. */
    float   *d_inputs, *d_state, *d_out;
    uint8_t *d_genome;
    cudaMalloc(&d_inputs, sizeof(h_inputs));
    cudaMalloc(&d_genome, sizeof(h_genome));
    cudaMalloc(&d_state,  sizeof(float) * G * (gi + gn) * N);
    cudaMalloc(&d_out,    sizeof(float) * G * go * N);
    cudaMemcpy(d_inputs, h_inputs, sizeof(h_inputs), cudaMemcpyHostToDevice);
    cudaMemcpy(d_genome, h_genome, sizeof(h_genome), cudaMemcpyHostToDevice);

    forward_kernel<<<G, 32>>>(d_inputs, d_genome, d_state, d_out);
    cudaDeviceSynchronize();

    float h_out[G * go * N];
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);

    /* Pretty-print. */
    const char *names[G] = {
        "i0 = sin(x)+x*x  (true)",
        "i1 = x*x",
        "i2 = sin(x)",
        "i3 = x",
    };
    printf("\n   x      target       i0          i1          i2          i3\n");
    for (int k = 0; k < N; ++k) {
        printf("%+6.3f   %+9.5f", h_inputs[k], h_target[k]);
        for (int g = 0; g < G; ++g) printf("  %+9.5f", h_out[g * N + k]);
        printf("\n");
    }
    printf("\nMSE per individual:\n");
    for (int g = 0; g < G; ++g) {
        float s = 0.0f;
        for (int k = 0; k < N; ++k) {
            const float d = h_out[g * N + k] - h_target[k];
            s += d * d;
        }
        printf("  %-25s  %.6e\n", names[g], s / N);
    }

    cudaFree(d_inputs); cudaFree(d_genome); cudaFree(d_state); cudaFree(d_out);
    return 0;
}
