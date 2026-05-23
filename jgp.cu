/*
   Parameterized CGP: full Jacobian via forward-mode AD over per-node
   continuous parameters.

   Op set includes  6 = par·v0  and  7 = par  for parameterized nodes,
   so each individual carries a parameter array  params[G, n_p]  with
   p parameters per node (n_p = gn·p).  Output is the Jacobian
   J[G, go, N, n_p] = ∂(out) / ∂(every parameter).

   Two-pass kernel:
     Pass 1 — forward values only.
     Pass 2 — for each q ∈ [0, n_p):  seed par_t = 1 at node q,
              propagate tangents, store column q of J.

   No solver, no fitting, no iteration — just compute the Jacobian and
   verify it column-by-column against analytic gradients.
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
constexpr int p  = 1;          // parameters per node
constexpr int N  = 16;
constexpr int ng_total = gi + gn + go;
constexpr int n_p      = gn * p;

/*
   JVP op: like apply_op_ad from dgp.cu, but also takes a per-node
   parameter `par` and its tangent `par_t`.  Two new ops (6, 7) use them.
*/
__device__ __forceinline__ void apply_op_jvp(uint8_t op,
                                             float v0, float v1,
                                             float t0, float t1,
                                             float par, float par_t,
                                             float *v, float *t) {
    switch (op) {
        case 0: *v = v0 + v1;     *t = t0 + t1;                break;  // add
        case 1: *v = v0 - v1;     *t = t0 - t1;                break;  // sub
        case 2: *v = v0 * v1;     *t = t0 * v1 + v0 * t1;      break;  // mul
        case 3: {                                                       // div
            float inv = 1.0f / v1;
            *v = v0 * inv;
            *t = (t0 - (*v) * t1) * inv;
            break;
        }
        case 4: *v = sinf(v0);    *t =  cosf(v0) * t0;         break;  // sin
        case 5: *v = cosf(v0);    *t = -sinf(v0) * t0;         break;  // cos
        case 6: *v = par * v0;    *t = par_t * v0 + par * t0;  break;  // par·v0
        case 7: *v = par;         *t = par_t;                  break;  // par
        default: *v = 0.0f;       *t = 0.0f;
    }
}

/*
   Two-pass forward+Jacobian kernel.  Pass 1 fills state_v with the
   value-only forward sweep.  Pass 2 reuses state_t as scratch: for
   each parameter q, seeds par_t = 1 at node q, propagates the
   tangent, and stores the output tangent into column q of J.
*/
__global__ void forward_jac_kernel(const float   *params,    // [G, n_p]
                                   const float   *inputs,    // [G, gi, N]
                                   const uint8_t *genome,    // [G, ng_total, 3]
                                   float         *state_v,   // [G, gi+gn, N]
                                   float         *state_t,   // [G, gi+gn, N]
                                   float         *out_v,     // [G, go, N]
                                   float         *J_out)     // [G, go, N, n_p]
{
    const int    g          = blockIdx.x;
    const int    tid        = threadIdx.x;
    const int    B          = blockDim.x;
    const size_t state_base = (size_t)g * (gi + gn) * N;

    /* Stage inputs into state_v. */
    for (int i = 0; i < gi; ++i) {
        const size_t dst = state_base + (size_t)i * N;
        const size_t src = (size_t)g * gi * N + (size_t)i * N;
        for (int k = tid; k < N; k += B) state_v[dst + k] = inputs[src + k];
    }
    __syncthreads();

    /* Pass 1: forward values only (tangents irrelevant). */
    for (int j = 0; j < gn; ++j) {
        const size_t  row  = (size_t)g * ng_total + (gi + j);
        const uint8_t op   = genome[row * 3 + 0];
        const uint8_t ptr0 = genome[row * 3 + 1];
        const uint8_t ptr1 = genome[row * 3 + 2];
        const float   par  = params[(size_t)g * n_p + j];

        const size_t in0_base = state_base + (size_t)ptr0 * N;
        const size_t in1_base = state_base + (size_t)ptr1 * N;
        const size_t dst_base = state_base + (size_t)(gi + j) * N;

        for (int k = tid; k < N; k += B) {
            float v, t_unused;
            apply_op_jvp(op,
                         state_v[in0_base + k], state_v[in1_base + k],
                         0.0f, 0.0f,
                         par, 0.0f,
                         &v, &t_unused);
            state_v[dst_base + k] = v;
        }
        __syncthreads();
    }

    /* Output values. */
    for (int o = 0; o < go; ++o) {
        const size_t  out_row = (size_t)g * ng_total + (gi + gn + o);
        const uint8_t src_row = genome[out_row * 3 + 1];
        const size_t  src     = state_base + (size_t)src_row * N;
        const size_t  dst     = (size_t)(g * go + o) * N;
        for (int k = tid; k < N; k += B) out_v[dst + k] = state_v[src + k];
    }

    /* Pass 2: column-by-column Jacobian. */
    for (int q = 0; q < n_p; ++q) {

        /* Inputs are independent of every parameter. */
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

            const size_t in0_base = state_base + (size_t)ptr0 * N;
            const size_t in1_base = state_base + (size_t)ptr1 * N;
            const size_t dst_base = state_base + (size_t)(gi + j) * N;

            for (int k = tid; k < N; k += B) {
                float v_unused, t;
                apply_op_jvp(op,
                             state_v[in0_base + k], state_v[in1_base + k],
                             state_t[in0_base + k], state_t[in1_base + k],
                             par, par_t,
                             &v_unused, &t);
                state_t[dst_base + k] = t;
            }
            __syncthreads();
        }

        /* Write tangent column q of J for each output. */
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

static void set_node(uint8_t *gen, int g, int row, uint8_t op,
                     uint8_t p0, uint8_t p1) {
    const int b = (g * ng_total + row) * 3;
    gen[b + 0] = op;
    gen[b + 1] = p0;
    gen[b + 2] = p1;
}

/*
   Analytic gradients for the four individuals' active params.  Each
   takes the per-individual parameter vector so the result can depend
   on parameter values (these are non-linear in the parameter).
*/
static float grad_i0_q0(float x, const float *p) { return cosf(p[0] * x) * x; }       /* dy/da of sin(a·x) + (b·x)² */
static float grad_i0_q2(float x, const float *p) { return 2.0f * p[2] * x * x; }      /* dy/db */
static float grad_i1_q0(float x, const float *p) { return 2.0f * p[0] * x * x; }      /* dy/da of (a·x)² */
static float grad_i2_q0(float x, const float *p) { return cosf(p[0] * x) * x; }       /* dy/da of sin(a·x) */
static float grad_i3_q0(float x, const float *p) { return p[2] * cosf(p[0]*x) * x; }  /* dy/db of a·sin(b·x), b at q=0 */
static float grad_i3_q2(float x, const float *p) { return sinf(p[0] * x); }            /* dy/da, a at q=2 */

int main(void) {
    /* Same input grid as gp.cu / dgp.cu. */
    float h_inputs[G * gi * N];
    for (int g = 0; g < G; ++g)
        for (int k = 0; k < N; ++k)
            h_inputs[g * gi * N + k] = -1.0f + 2.0f * (float)k / (N - 1);

    /*
       Four individuals, all with at least one parameter that enters
       non-linearly (i.e. ∂y/∂θ depends on θ):

         i0:  y = sin(a·x) + (b·x)²      params a (q=0), b (q=2)
         i1:  y = (a·x)²                  param  a (q=0)
         i2:  y = sin(a·x)                param  a (q=0)
         i3:  y = a · sin(b·x)            params b (q=0), a (q=2)
                                          — a is linear, b is non-linear
    */
    uint8_t h_genome[G * ng_total * 3] = {};

    /* i0:  a·x, sin(a·x), b·x, (b·x)², sum, output. */
    set_node(h_genome, 0, 1, 6, 0, 0);   // par·x       = a·x      | par at q=0
    set_node(h_genome, 0, 2, 4, 1, 0);   // sin(row1)   = sin(a·x)
    set_node(h_genome, 0, 3, 6, 0, 0);   // par·x       = b·x      | par at q=2
    set_node(h_genome, 0, 4, 2, 3, 3);   // row3·row3   = (b·x)²
    set_node(h_genome, 0, 5, 0, 2, 4);   // row2 + row4
    set_node(h_genome, 0, 7, 0, 5, 0);   // output ← row 5

    /* i1:  a·x, (a·x)², output. */
    set_node(h_genome, 1, 1, 6, 0, 0);   // par·x       = a·x      | par at q=0
    set_node(h_genome, 1, 2, 2, 1, 1);   // row1·row1   = (a·x)²
    set_node(h_genome, 1, 7, 0, 2, 0);

    /* i2:  a·x, sin(a·x), output. */
    set_node(h_genome, 2, 1, 6, 0, 0);   // par·x       = a·x      | par at q=0
    set_node(h_genome, 2, 2, 4, 1, 0);   // sin(row1)   = sin(a·x)
    set_node(h_genome, 2, 7, 0, 2, 0);

    /* i3:  b·x, sin(b·x), a·sin(b·x), output. */
    set_node(h_genome, 3, 1, 6, 0, 0);   // par·x       = b·x      | par at q=0 (b)
    set_node(h_genome, 3, 2, 4, 1, 0);   // sin(row1)   = sin(b·x)
    set_node(h_genome, 3, 3, 6, 2, 0);   // par·row2    = a·sin(b·x)| par at q=2 (a)
    set_node(h_genome, 3, 7, 0, 3, 0);

    /* Non-trivial parameter values so the non-linear dependence shows. */
    float h_params[G * n_p] = {};
    h_params[0 * n_p + 0] = 2.0f;   // i0: a
    h_params[0 * n_p + 2] = 1.5f;   // i0: b
    h_params[1 * n_p + 0] = 1.5f;   // i1: a
    h_params[2 * n_p + 0] = 0.7f;   // i2: a
    h_params[3 * n_p + 0] = 1.5f;   // i3: b  (param at node 0)
    h_params[3 * n_p + 2] = 2.0f;   // i3: a  (param at node 2)

    /* GPU side. */
    float   *d_params, *d_inputs, *d_state_v, *d_state_t, *d_out_v, *d_J;
    uint8_t *d_genome;
    size_t bytes_J = (size_t)G * go * N * n_p * sizeof(float);
    cudaMalloc(&d_params,  sizeof(h_params));
    cudaMalloc(&d_inputs,  sizeof(h_inputs));
    cudaMalloc(&d_genome,  sizeof(h_genome));
    cudaMalloc(&d_state_v, sizeof(float) * G * (gi + gn) * N);
    cudaMalloc(&d_state_t, sizeof(float) * G * (gi + gn) * N);
    cudaMalloc(&d_out_v,   sizeof(float) * G * go * N);
    cudaMalloc(&d_J,       bytes_J);

    cudaMemcpy(d_params, h_params, sizeof(h_params), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs, h_inputs, sizeof(h_inputs), cudaMemcpyHostToDevice);
    cudaMemcpy(d_genome, h_genome, sizeof(h_genome), cudaMemcpyHostToDevice);

    forward_jac_kernel<<<G, 32>>>(d_params, d_inputs, d_genome,
                                  d_state_v, d_state_t, d_out_v, d_J);
    cudaDeviceSynchronize();

    float  h_out_v[G * go * N];
    float *h_J = (float*)malloc(bytes_J);
    cudaMemcpy(h_out_v, d_out_v, sizeof(h_out_v), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_J,     d_J,     bytes_J,         cudaMemcpyDeviceToHost);

    /* Active params per individual: which q's are used, what they
       represent, and the analytic gradient (which now depends on the
       parameter vector). */
    struct Active {
        int q;
        const char *label;     /* gradient expression for the printout */
        float (*grad)(float, const float *);
    };
    Active active[G][2] = {
        {{0, "cos(a·x)·x",   grad_i0_q0}, {2, "2·b·x²",      grad_i0_q2}},
        {{0, "2·a·x²",       grad_i1_q0}, {-1, nullptr,       nullptr}},
        {{0, "cos(a·x)·x",   grad_i2_q0}, {-1, nullptr,       nullptr}},
        {{0, "a·cos(b·x)·x", grad_i3_q0}, {2, "sin(b·x)",    grad_i3_q2}},
    };

    /* Side-by-side table: primary active gradient (column 0 of `active`). */
    const char *primary[G] = {
        "i0 dy/da = cos(a·x)·x", "i1 dy/da = 2·a·x²",
        "i2 dy/da = cos(a·x)·x", "i3 dy/db = a·cos(b·x)·x",
    };
    printf("\n   x      %-22s   %-22s   %-22s   %-22s\n",
           primary[0], primary[1], primary[2], primary[3]);
    printf("           GPU       ref          GPU       ref          GPU       ref          GPU       ref\n");
    for (int k = 0; k < N; ++k) {
        printf("%+6.3f ", h_inputs[k]);
        for (int g = 0; g < G; ++g) {
            int   q = active[g][0].q;
            float v = h_J[(((size_t)g * go + 0) * N + k) * n_p + q];
            float r = active[g][0].grad(h_inputs[k], &h_params[g * n_p]);
            printf("  %+8.4f %+8.4f", v, r);
        }
        printf("\n");
    }

    /* Verify every active column. */
    printf("\nActive-column verification (max |GPU - analytic|):\n");
    for (int g = 0; g < G; ++g) {
        for (int a_i = 0; a_i < 2; ++a_i) {
            if (active[g][a_i].q < 0) continue;
            int q = active[g][a_i].q;
            float max_err = 0.0f;
            for (int k = 0; k < N; ++k) {
                float v = h_J[(((size_t)g * go + 0) * N + k) * n_p + q];
                float r = active[g][a_i].grad(h_inputs[k], &h_params[g * n_p]);
                float e = fabsf(v - r);
                if (e > max_err) max_err = e;
            }
            printf("  i%d  q=%d  ∂y/∂%-18s  %.3e\n",
                   g, q, active[g][a_i].label, max_err);
        }
    }

    /* Verify inactive columns are 0. */
    printf("\nInactive-column sanity (should all be 0):\n");
    float max_inactive = 0.0f;
    for (int g = 0; g < G; ++g) {
        for (int q = 0; q < n_p; ++q) {
            bool is_active = false;
            for (int a_i = 0; a_i < 2; ++a_i)
                if (active[g][a_i].q == q) { is_active = true; break; }
            if (is_active) continue;
            for (int k = 0; k < N; ++k) {
                float v = h_J[(((size_t)g * go + 0) * N + k) * n_p + q];
                float a = fabsf(v);
                if (a > max_inactive) max_inactive = a;
            }
        }
    }
    printf("  max |J[g, 0, k, q_inactive]| across all individuals  =  %.3e\n",
           max_inactive);

    free(h_J);
    cudaFree(d_params); cudaFree(d_inputs); cudaFree(d_genome);
    cudaFree(d_state_v); cudaFree(d_state_t);
    cudaFree(d_out_v); cudaFree(d_J);
    return 0;
}
