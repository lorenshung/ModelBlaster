/* B (streaming BN / add / activation unit) driver: soc/phase2/b_unit/CONTRACT.md v1.
 *
 * Header-only. Include it in ONE translation unit, the one that runs on the hart that owns B
 * (recommended: hart 1, which also drives A + C). The entry points take the same arguments as the
 * deployed kernels (soc/phase2/dronet/sw/sw_kernels.c), so lowering only swaps the name:
 *   kernel_bn_nhwc_s8      -> b_bn_nhwc
 *   kernel_add_q_s8        -> b_add
 *   kernel_relu_s8_dronet  -> b_relu
 * Call b_init() once before the first op.
 *
 * Freestanding: no float, no libc beyond stdint/stddef. Builds with -march=rv64imac_zve64x -mabi=lp64.
 * -DB_HOST_MODEL: the same code drives b_model.c's register-level emulation of b_mmio + C's DMA instead
 * of MMIO (host pointers are the DMA addresses). That is the off-board verification path.
 *
 * Usage model per op (CONTRACT.md section 5):
 *   BN:   [first use of a layer: pack {M,B,S}, DMA to COEF, stays resident]
 *         CFG = {LO,HI,LOG2C,CBASE}; MODE = BN; DMA in -> DATA 0; DMA out DATA 0 -> output
 *   ADD:  MODE = RAW; DMA a -> DATA 0; CFG = {LO,HI,S}; MUL = {ma,mb}; MODE = ADD; DMA b -> DATA 0;
 *         DMA out
 *   RELU: CFG = {0,127,S=0}; MUL = {1,0}; BIAS = 0; MODE = AFFINE; DMA in; DMA out
 * Unaligned buffers, odd sizes or out-of-range constants take the scalar fallback (same numerics),
 * counted in b_prof.fallbacks. */
#ifndef B_DRIVER_H
#define B_DRIVER_H

#include <stddef.h>
#include <stdint.h>

#ifndef B_BASE
#define B_BASE 0x11010000UL
#endif

/* ---- window map (CONTRACT.md section 2) ---- */
#define B_DATA        0x0000u
#define B_DATA_BYTES  32768u
#define B_COEF        0xE000u
#define B_COEF_ENTRIES 256u
#define B_REG_ID      0xF000u
#define B_REG_MODE    0xF008u
#define B_REG_CFG     0xF010u
#define B_REG_MUL     0xF018u
#define B_REG_BIAS    0xF020u
#define B_REG_STATUS  0xF028u
#define B_ID_VALUE    0x00000001424E5244ULL   /* "DRNB" v1 */
#define B_MODE_RAW    0u
#define B_MODE_BN     1u
#define B_MODE_ADD    2u
#define B_MODE_AFFINE 3u
#define B_MUL_LIMIT   (1L << 26)              /* |M|, |MA|, |MB| < 2^26 */
/* C's DMA (DronetConv0.scala), same layout in B's wrapper */
#define B_DMA_EXT     0xF100u
#define B_DMA_LOC     0xF108u
#define B_DMA_LEN     0xF110u
#define B_DMA_CTRL    0xF118u                 /* W: bit 0 go, bit 1 local->memory. R: STATUS */
#define B_DMA_CYCLES  0xF120u
#define B_DMA_ID      0xF128u
#define B_DMA_ID_VALUE 0x00000001414D4443ULL  /* "CDMA" v1 */
#define B_DMA_DONE    2u
#define B_DMA_ERR     4u

static inline uint64_t b_cfg_word(int lo, int hi, unsigned log2c, unsigned cbase, unsigned s)
{
    return (uint64_t)(uint8_t)lo | ((uint64_t)(uint8_t)hi << 8) | ((uint64_t)(log2c & 15) << 16) |
           ((uint64_t)(cbase & 255) << 24) | ((uint64_t)(s & 31) << 32);
}

/* ---- the numerics (CONTRACT.md section 3); used by the fallback and by b_model.c ---- */
static inline int8_t b_rq(int64_t acc, int s, int lo, int hi)
{
    int64_t v = s == 0 ? acc : (acc + ((int64_t)1 << (s - 1)) - (acc < 0 ? 1 : 0)) >> s;
    return (int8_t)(v < lo ? lo : (v > hi ? hi : v));
}
static inline int b_clamp8(int v) { return v < -128 ? -128 : (v > 127 ? 127 : v); }

/* ---- register access ---- */
#ifdef B_HOST_MODEL
uint64_t b_model_r64(unsigned off);
void b_model_w64(unsigned off, uint64_t v);
#define B_R64(off)    b_model_r64(off)
#define B_W64(off, v) b_model_w64((off), (v))
#define B_FENCE()     ((void)0)
#else
#define B_R64(off)    (*(volatile uint64_t *)(B_BASE + (off)))
#define B_W64(off, v) (*(volatile uint64_t *)(B_BASE + (off)) = (uint64_t)(v))
#define B_FENCE()     __asm__ volatile("fence" ::: "memory")
#endif

/* ---- driver state (one owner hart) ---- */
typedef struct {
    uint64_t dma_in_cycles, dma_in2_cycles, dma_out_cycles; /* device counters of the last op */
    uint64_t dma_status;
    uint32_t calls, fallbacks, dma_errors, coef_loads;
} b_prof_t;
static b_prof_t b_prof;

#define B_MAX_LAYERS 16
static struct {
    int present;
    uint64_t mode, cfg, mul, bias;        /* cached register values */
    int mode_ok, cfg_ok, mul_ok, bias_ok;
    unsigned next_cbase;
    int n_layers;
    struct { const int32_t *key; int C; unsigned cbase; } layer[B_MAX_LAYERS];
} b_st;
static uint64_t b_coef_stage_buf[2 * B_COEF_ENTRIES] __attribute__((aligned(64)));
static uint64_t *b_coef_stage = b_coef_stage_buf;   /* staging table in memory (DMA source) */

static inline void b_set(unsigned off, uint64_t v, uint64_t *cache, int *ok)
{
    if (*ok && *cache == v) return;
    B_W64(off, v);
    *cache = v; *ok = 1;
}
#define B_SET_MODE(v) b_set(B_REG_MODE, (v), &b_st.mode, &b_st.mode_ok)
#define B_SET_CFG(v)  b_set(B_REG_CFG, (v), &b_st.cfg, &b_st.cfg_ok)
#define B_SET_MUL(v)  b_set(B_REG_MUL, (v), &b_st.mul, &b_st.mul_ok)
#define B_SET_BIAS(v) b_set(B_REG_BIAS, (v), &b_st.bias, &b_st.bias_ok)

static inline uint64_t b_ptr(const void *p) { return (uint64_t)(uintptr_t)p; }

/* One DMA transfer: EXT, LOC, LEN, then CTRL go; poll done. Returns 0 on success. */
static inline int b_dma(const void *ext, unsigned loc, unsigned len, int to_mem, uint64_t *cycles)
{
    uint64_t st;
    B_FENCE();
    B_W64(B_DMA_EXT, b_ptr(ext));
    B_W64(B_DMA_LOC, loc);
    B_W64(B_DMA_LEN, len);
    B_W64(B_DMA_CTRL, 1u | (to_mem ? 2u : 0u));
    while (!((st = B_R64(B_DMA_CTRL)) & B_DMA_DONE)) { }
#ifndef B_NO_PROFILE
    if (cycles) *cycles = B_R64(B_DMA_CYCLES);
#else
    (void)cycles;
#endif
    B_FENCE();
    b_prof.dma_status = st;
    if (st & B_DMA_ERR) { b_prof.dma_errors++; return -1; }
    return 0;
}

static inline int b_init(void)
{
    b_st.present = B_R64(B_REG_ID) == B_ID_VALUE && B_R64(B_DMA_ID) == B_DMA_ID_VALUE;
    b_st.mode_ok = b_st.cfg_ok = b_st.mul_ok = b_st.bias_ok = 0;
    b_st.next_cbase = 0;
    b_st.n_layers = 0;
    return b_st.present ? 0 : -1;
}

static inline int b_aligned(const void *p) { return ((uintptr_t)p & 7) == 0; }
static inline int b_log2_pow2(int C)    /* log2 C if C is a power of two in [8, 256], else -1 */
{
    for (int l = 3; l <= 8; l++) if (C == (1 << l)) return l;
    return -1;
}

/* ---- scalar fallbacks (same numerics as the deployed kernels) ---- */
static inline void b_bn_scalar(const int8_t *in, int8_t *out, int HW, int C, const int32_t *M,
                               const int32_t *B, const int16_t *S, int lo, int hi)
{
    for (int p = 0; p < HW; p++)
        for (int c = 0; c < C; c++)
            out[(size_t)p * C + c] = b_rq((int64_t)in[(size_t)p * C + c] * M[c] + B[c], S[c], lo, hi);
}
static inline void b_add_scalar(const int8_t *a, const int8_t *b, int8_t *out, int n, int32_t ma,
                                int32_t mb, int S, int lo, int hi)
{
    for (int i = 0; i < n; i++) out[i] = b_rq((int64_t)a[i] * ma + (int64_t)b[i] * mb, S, lo, hi);
}

/* Find or load the layer's coefficients; returns CBASE, or -1 if they cannot be made resident. */
static inline int b_coef_slot(int C, const int32_t *M, const int32_t *B, const int16_t *S)
{
    for (int i = 0; i < b_st.n_layers; i++)
        if (b_st.layer[i].key == M && b_st.layer[i].C == C) return (int)b_st.layer[i].cbase;
    for (int c = 0; c < C; c++)
        if (M[c] >= B_MUL_LIMIT || M[c] <= -B_MUL_LIMIT || S[c] < 0 || S[c] > 31) return -1;
    if (b_st.n_layers == B_MAX_LAYERS || b_st.next_cbase + (unsigned)C > B_COEF_ENTRIES) {
        b_st.n_layers = 0;                                  /* table full: start over */
        b_st.next_cbase = 0;
    }
    unsigned cbase = b_st.next_cbase;
    for (int c = 0; c < C; c++) {
        b_coef_stage[2 * c] = (uint64_t)(uint32_t)M[c] | ((uint64_t)(uint32_t)B[c] << 32);
        b_coef_stage[2 * c + 1] = (uint64_t)(uint16_t)S[c] & 31;
    }
    B_SET_MODE(B_MODE_RAW);                                  /* COEF ignores MODE; kept for clarity */
    if (b_dma(b_coef_stage, B_COEF + 16u * cbase, 16u * (unsigned)C, 0, 0)) return -1;
    b_prof.coef_loads++;
    b_st.layer[b_st.n_layers].key = M;
    b_st.layer[b_st.n_layers].C = C;
    b_st.layer[b_st.n_layers].cbase = cbase;
    b_st.n_layers++;
    b_st.next_cbase = cbase + (unsigned)C;                  /* C is a multiple of 8 */
    return (int)cbase;
}

/* = kernel_bn_nhwc_s8 (BP is accepted for signature compatibility and ignored). */
static inline void b_bn_nhwc(const int8_t *input, int8_t *output, int HW, int C, const int32_t *M,
                             const int32_t *B, const int32_t *BP, const int16_t *S,
                             int activation_min, int activation_max)
{
    (void)BP;
    const int lo = b_clamp8(activation_min), hi = b_clamp8(activation_max);
    const int l2 = b_log2_pow2(C);
    b_prof.calls++;
    int cbase = -1;
    if (b_st.present && l2 >= 0 && b_aligned(input) && b_aligned(output))
        cbase = b_coef_slot(C, M, B, S);
    if (cbase < 0) { b_prof.fallbacks++; b_bn_scalar(input, output, HW, C, M, B, S, lo, hi); return; }
    B_SET_CFG(b_cfg_word(lo, hi, (unsigned)l2, (unsigned)cbase, 0));
    B_SET_MODE(B_MODE_BN);
    const size_t total = (size_t)HW * (size_t)C, tile = (B_DATA_BYTES / (unsigned)C) * (unsigned)C;
    for (size_t off = 0; off < total; off += tile) {
        const unsigned len = (unsigned)(total - off < tile ? total - off : tile);
        if (b_dma(input + off, B_DATA, len, 0, &b_prof.dma_in_cycles) ||
            b_dma(output + off, B_DATA, len, 1, &b_prof.dma_out_cycles)) {
            b_prof.fallbacks++;
            b_bn_scalar(input + off, output + off, (int)(len / (unsigned)C), C, M, B, S, lo, hi);
        }
    }
}

/* = kernel_add_q_s8: out = rq(a*ma + b*mb, S), a stored first, b streamed. */
static inline void b_add(const int8_t *a, const int8_t *b, int8_t *output, int n, int32_t ma,
                         int32_t mb, int S, int activation_min, int activation_max)
{
    const int lo = b_clamp8(activation_min), hi = b_clamp8(activation_max);
    b_prof.calls++;
    if (!b_st.present || (n & 7) || !b_aligned(a) || !b_aligned(b) || !b_aligned(output) ||
        ma >= B_MUL_LIMIT || ma <= -B_MUL_LIMIT || mb >= B_MUL_LIMIT || mb <= -B_MUL_LIMIT ||
        S < 0 || S > 31) {
        b_prof.fallbacks++;
        b_add_scalar(a, b, output, n, ma, mb, S, lo, hi);
        return;
    }
    for (int off = 0; off < n; off += (int)B_DATA_BYTES) {
        const unsigned len = (unsigned)(n - off) < B_DATA_BYTES ? (unsigned)(n - off) : B_DATA_BYTES;
        B_SET_MODE(B_MODE_RAW);
        if (b_dma(a + off, B_DATA, len, 0, &b_prof.dma_in_cycles)) goto fail;
        B_SET_CFG(b_cfg_word(lo, hi, 3, 0, (unsigned)S));
        B_SET_MUL((uint64_t)(uint32_t)ma | ((uint64_t)(uint32_t)mb << 32));
        B_SET_MODE(B_MODE_ADD);
        if (b_dma(b + off, B_DATA, len, 0, &b_prof.dma_in2_cycles)) goto fail;
        if (b_dma(output + off, B_DATA, len, 1, &b_prof.dma_out_cycles)) goto fail;
        continue;
    fail:
        b_prof.fallbacks++;
        b_add_scalar(a + off, b + off, output + off, (int)len, ma, mb, S, lo, hi);
    }
}

/* = kernel_relu_s8_dronet: out = max(x, 0), as AFFINE with MA = 1, BG = 0, S = 0, [0, 127]. */
static inline void b_relu(const int8_t *input, int8_t *output, int n)
{
    b_prof.calls++;
    if (!b_st.present || (n & 7) || !b_aligned(input) || !b_aligned(output)) {
        b_prof.fallbacks++;
        for (int i = 0; i < n; i++) output[i] = input[i] < 0 ? 0 : input[i];
        return;
    }
    B_SET_CFG(b_cfg_word(0, 127, 3, 0, 0));
    B_SET_MUL(1);
    B_SET_BIAS(0);
    B_SET_MODE(B_MODE_AFFINE);
    for (int off = 0; off < n; off += (int)B_DATA_BYTES) {
        const unsigned len = (unsigned)(n - off) < B_DATA_BYTES ? (unsigned)(n - off) : B_DATA_BYTES;
        if (b_dma(input + off, B_DATA, len, 0, &b_prof.dma_in_cycles) ||
            b_dma(output + off, B_DATA, len, 1, &b_prof.dma_out_cycles)) {
            b_prof.fallbacks++;
            for (unsigned i = 0; i < len; i++) output[off + i] = input[off + i] < 0 ? 0 : input[off + i];
        }
    }
}

#endif /* B_DRIVER_H */
