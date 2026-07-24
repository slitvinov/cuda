#define SREGS                                                                  \
  V3(tid)                                                                      \
  V3(ntid)                                                                     \
  V3(ctaid)                                                                    \
  V3(nctaid)                                                                   \
  V3(clusterid)                                                                \
  V3(nclusterid)                                                               \
  V3(cluster_ctaid)                                                            \
  V3(cluster_nctaid)                                                           \
  U32(laneid)                                                                  \
  U32(warpid)                                                                  \
  U32(nwarpid)                                                                 \
  U32(smid)                                                                    \
  U32(nsmid)                                                                   \
  U32(cluster_ctarank)                                                         \
  U32(cluster_nctarank)                                                        \
  U32(lanemask_eq)                                                             \
  U32(lanemask_le)                                                             \
  U32(lanemask_lt)                                                             \
  U32(lanemask_ge)                                                             \
  U32(lanemask_gt)                                                             \
  U32(clock)                                                                   \
  U32(clock_hi)                                                                \
  U64(clock64)                                                                 \
  U64(globaltimer)                                                             \
  U32(globaltimer_lo)                                                          \
  U32(globaltimer_hi)                                                          \
  U64(gridid)                                                                  \
  U32(total_smem_size)                                                         \
  U32(aggr_smem_size)                                                          \
  U32(dynamic_smem_size)                                                       \
  U64(current_graph_exec)                                                      \
  U32(pm0)                                                                     \
  U32(pm1)                                                                     \
  U32(pm2)                                                                     \
  U32(pm3)                                                                     \
  U32(pm4)                                                                     \
  U32(pm5)                                                                     \
  U32(pm6)                                                                     \
  U32(pm7)                                                                     \
  U64(pm0_64)                                                                  \
  U64(pm1_64)                                                                  \
  U64(pm2_64)                                                                  \
  U64(pm3_64)                                                                  \
  U64(pm4_64)                                                                  \
  U64(pm5_64)                                                                  \
  U64(pm6_64)                                                                  \
  U64(pm7_64)                                                                  \
  U32(envreg0)                                                                 \
  U32(envreg1)                                                                 \
  U32(envreg2)                                                                 \
  U32(envreg3)                                                                 \
  U32(envreg4)                                                                 \
  U32(envreg5)                                                                 \
  U32(envreg6)                                                                 \
  U32(envreg7)                                                                 \
  U32(envreg8)                                                                 \
  U32(envreg9)                                                                 \
  U32(envreg10)                                                                \
  U32(envreg11)                                                                \
  U32(envreg12)                                                                \
  U32(envreg13)                                                                \
  U32(envreg14)                                                                \
  U32(envreg15)                                                                \
  U32(envreg16)                                                                \
  U32(envreg17)                                                                \
  U32(envreg18)                                                                \
  U32(envreg19)                                                                \
  U32(envreg20)                                                                \
  U32(envreg21)                                                                \
  U32(envreg22)                                                                \
  U32(envreg23)                                                                \
  U32(envreg24)                                                                \
  U32(envreg25)                                                                \
  U32(envreg26)                                                                \
  U32(envreg27)                                                                \
  U32(envreg28)                                                                \
  U32(envreg29)                                                                \
  U32(envreg30)                                                                \
  U32(envreg31)

#define U32(n)                                                                 \
  static __device__ __forceinline__ void reg_##n(uint32_t *p) {                \
    asm volatile("mov.u32 %0, %%" #n ";" : "=r"(*p));                          \
  }
#define U64(n)                                                                 \
  static __device__ __forceinline__ void reg_##n(uint64_t *p) {                \
    asm volatile("mov.u64 %0, %%" #n ";" : "=l"(*p));                          \
  }
#define V3(n)                                                                  \
  static __device__ __forceinline__ void reg_##n(uint32_t p[3]) {              \
    asm volatile("mov.u32 %0, %%" #n ".x;" : "=r"(p[0]));                      \
    asm volatile("mov.u32 %0, %%" #n ".y;" : "=r"(p[1]));                      \
    asm volatile("mov.u32 %0, %%" #n ".z;" : "=r"(p[2]));                      \
  }
SREGS
#undef U32
#undef U64
#undef V3

static __device__ __forceinline__ void reg_is_explicit_cluster(uint32_t *p) {
  asm volatile("{ .reg .pred %%q;                      \n\t"
               "  mov.pred %%q, %%is_explicit_cluster; \n\t"
               "  selp.u32 %0, 1, 0, %%q;              \n\t"
               "}"
               : "=r"(*p));
}
