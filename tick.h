#define TICKS                                                                  \
  U32(clock)                                                                   \
  U32(clock_hi)                                                                \
  U64(clock64)                                                                 \
  U64(globaltimer)                                                             \
  U32(globaltimer_lo)                                                          \
  U32(globaltimer_hi)

#define U32(n)                                                                 \
  static __device__ __forceinline__ void tick_##n(uint64_t fake,               \
                                                   uint32_t *p) {              \
    asm volatile("{ .reg .pred %%q;              \n\t"                          \
                 "  setp.ge.u64 %%q, %1, 0;      \n\t"                          \
                 "  @%%q mov.u32 %0, %%" #n ";   \n\t"                          \
                 "}"                                                           \
                 : "=r"(*p)                                                    \
                 : "l"(fake));                                                 \
  }
#define U64(n)                                                                 \
  static __device__ __forceinline__ void tick_##n(uint64_t fake,               \
                                                   uint64_t *p) {              \
    asm volatile("{ .reg .pred %%q;              \n\t"                          \
                 "  setp.ge.u64 %%q, %1, 0;      \n\t"                          \
                 "  @%%q mov.u64 %0, %%" #n ";   \n\t"                          \
                 "}"                                                           \
                 : "=l"(*p)                                                    \
                 : "l"(fake));                                                 \
  }
TICKS
#undef U32
#undef U64
