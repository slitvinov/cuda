#include <stdio.h>
#include <stddef.h>
#include <string.h>

enum kind { K_INT, K_UINT, K_SIZE, K_STR, K_IARR, K_BYTES };

enum cat { CAT_ID, CAT_COMPUTE, CAT_MEMORY, CAT_CAP, CAT_TEXTURE, CAT_N };
static const char *CATNAME[CAT_N] = { "id", "compute", "memory", "cap", "texture" };

struct field;
typedef void (*fmt_fn)(const struct field *f, const void *q);
struct field { const char *name; size_t off, size; enum kind kind; fmt_fn fmt; enum cat cat; };

static unsigned long long uval(const void *q, size_t sz) {
    switch (sz) {
        case 1:  return *(const unsigned char *)q;
        case 4:  return *(const unsigned int *)q;
        case 8:  return *(const unsigned long long *)q;
        default: return 0;
    }
}

static void fmt_bytes(const struct field *f, const void *q) {
    double v = (double)uval(q, f->size);
    const char *u = "B";
    if      (v >= (1ull<<30)) { v /= (1u<<30); u = "GiB"; }
    else if (v >= (1ull<<20)) { v /= (1u<<20); u = "MiB"; }
    else if (v >= (1ull<<10)) { v /= (1u<<10); u = "KiB"; }
    printf("%.2f %s", v, u);
}
static void fmt_clk(const struct field *f, const void *q) {
    double khz = (double)uval(q, f->size);
    if (khz >= 1e6) printf("%.2f GHz", khz / 1e6);
    else            printf("%.0f MHz", khz / 1e3);
}
static void fmt_bool(const struct field *f, const void *q) { printf("%s", uval(q, f->size) ? "yes" : "no"); }
static void fmt_bits(const struct field *f, const void *q) { printf("%llu bits", uval(q, f->size)); }

#include "info.h"   /* generated: #define FIELDS X(...) X(...) ... */

static const struct field TABLE[] = {
#define X(f, k, ff, c) { #f, offsetof(cudaDeviceProp, f), \
                         sizeof(((cudaDeviceProp *)0)->f), k, ff, c },
    FIELDS
#undef X
};

static void print_field(const cudaDeviceProp *p, const struct field *f) {
    const void *q = (const char *)p + f->off;
    printf("  %-42s ", f->name);
    if (f->fmt) { f->fmt(f, q); }
    else switch (f->kind) {
        case K_INT:  printf("%d",  *(const int *)q);      break;
        case K_UINT: printf("%u",  *(const unsigned *)q); break;
        case K_SIZE: printf("%zu", *(const size_t *)q);   break;
        case K_STR:  printf("%s",  (const char *)q);      break;
        case K_IARR: { size_t n = f->size / sizeof(int); const int *v = (const int *)q;
                       for (size_t i = 0; i < n; i++) printf("%s%d", i ? " " : "", v[i]); break; }
        case K_BYTES:{ const unsigned char *b = (const unsigned char *)q;
                       for (size_t i = 0; i < f->size; i++) printf("%02x", b[i]); break; }
    }
    putchar('\n');
}

static unsigned parse_cats(int argc, char **argv) {
    unsigned mask = 0;
    for (int i = 1; i < argc; i++)
        for (int c = 0; c < CAT_N; c++)
            if (strcmp(argv[i], CATNAME[c]) == 0) mask |= 1u << c;
    return mask;
}

int main(int argc, char **argv) {
  unsigned want = parse_cats(argc, argv);
  int i, c;
  cudaDeviceProp prop;
  cudaError_t err;
  
  err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    fprintf(stderr, "info: error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  for (c = 0; c < CAT_N; c++) {
    if (want && !(want & (1u << c))) continue;
    printf("-- %s --\n", CATNAME[c]);
    for (i = 0; i < sizeof(TABLE) / sizeof(TABLE[0]); i++)
      if (TABLE[i].cat == (enum cat)c)
	print_field(&prop, &TABLE[i]);
  }
  return 0;
}
