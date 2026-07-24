#define SREGS                                                                  \
  V3(tid)                                                                      \
  V3(ntid)                                                                     \
  V3(ctaid)                                                                    \
  V3(nctaid)                                                                   \
  R32(laneid)                                                                  \
  R32(warpid)                                                                  \
  R32(nwarpid)                                                                 \
  R32(smid)                                                                    \
  R32(nsmid)                                                                   \
  MASK(lanemask_eq)                                                            \
  MASK(lanemask_lt)                                                            \
  MASK(lanemask_le)                                                            \
  MASK(lanemask_ge)                                                            \
  MASK(lanemask_gt)                                                            \
  R32(dynamic_smem_size)                                                       \
  R32(total_smem_size)                                                         \
  R32(clock)                                                                   \
  CNT(clock64)                                                                 \
  R64(gridid)                                                                  \
  TNS(globaltimer)

#define R32(n)                                                                 \
  static __device__ __forceinline__ void reg_##n(uint32_t *p) {                \
    asm volatile("mov.u32 %0, %%" #n ";" : "=r"(*p));                          \
  }
#define R64(n)                                                                 \
  static __device__ __forceinline__ void reg_##n(uint64_t *p) {                \
    asm volatile("mov.u64 %0, %%" #n ";" : "=l"(*p));                          \
  }
#define MASK(n) R32(n)
#define CNT(n) R64(n)
#define TNS(n) R64(n)
#define V3(n)                                                                  \
  static __device__ __forceinline__ void reg_##n(uint32_t p[3]) {              \
    asm volatile("mov.u32 %0, %%" #n ".x;" : "=r"(p[0]));                      \
    asm volatile("mov.u32 %0, %%" #n ".y;" : "=r"(p[1]));                      \
    asm volatile("mov.u32 %0, %%" #n ".z;" : "=r"(p[2]));                      \
  }
SREGS
#undef R32
#undef R64
#undef V3
#undef MASK
#undef CNT
#undef TNS
