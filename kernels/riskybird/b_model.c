/* Host functional model of B on the SoC: b_mmio (CONTRACT.md v1) plus C's DMA, at register level.
 *
 * Build the driver with -DB_HOST_MODEL and link this file: b_bn_nhwc / b_add / b_relu then run the
 * exact register sequence they run on the board, against this emulation, with host pointers as the
 * DMA's memory addresses. Numerics come from b_rq() in b_driver.h (CONTRACT.md section 3).
 *
 * Optional trace (used to generate the RTL testbench's stimulus): b_model_trace_open() makes every
 * B register write, B register read and DMA go emit one line, with memory addresses as offsets into
 * a caller-supplied arena:
 *   W <off> <value> <wmask>     CPU write to the window
 *   R <off> <value> <mask>      CPU read of a B register, expected value (bits under mask)
 *   I <ext> <loc> <len>         DMA memory -> local
 *   O <ext> <loc> <len>         DMA local -> memory
 *   M <ext> <word>              memory word the CPU wrote since the arena snapshot (emitted before
 *                               the DMA that reads it, e.g. the packed BN coefficient table)
 * DMA status polls and counter reads are not traced (the testbench runs each DMA to completion). */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "b_driver.h"

static uint8_t m_data[B_DATA_BYTES];
static int32_t m_M[B_COEF_ENTRIES], m_B[B_COEF_ENTRIES];
static uint8_t m_S[B_COEF_ENTRIES];
static uint64_t m_mode, m_cfg = 0x7F80u | (3u << 16), m_mul, m_bias, m_beats;
static uint64_t d_ext, d_loc, d_len, d_status, d_cycles;
static FILE *m_tr;
static const uint8_t *m_arena;
static size_t m_arena_len;
static uint8_t *m_shadow;                  /* what the testbench's memory holds */

void b_model_reset(void)
{
    memset(m_data, 0, sizeof m_data);
    memset(m_M, 0, sizeof m_M); memset(m_B, 0, sizeof m_B); memset(m_S, 0, sizeof m_S);
    m_mode = m_mul = m_bias = m_beats = 0;
    m_cfg = 0x7F80u | (3u << 16);
    d_ext = d_loc = d_len = d_status = d_cycles = 0;
}

/* shadow: a writable copy of the arena as it was when the testbench's memory image was taken */
void b_model_trace_open(FILE *f, const void *arena, size_t len, uint8_t *shadow)
{
    m_tr = f; m_arena = (const uint8_t *)arena; m_arena_len = len; m_shadow = shadow;
}
void b_model_trace_close(void) { m_tr = 0; }
const uint8_t *b_model_data(void) { return m_data; }

static uint64_t merge(uint64_t old, uint64_t v, unsigned mask)
{
    for (int b = 0; b < 8; b++)
        if (mask & (1u << b)) old = (old & ~(0xFFULL << (8 * b))) | (v & (0xFFULL << (8 * b)));
    return old;
}

/* One write to the window as b_mmio sees it (not the DMA page). */
static void window_write(unsigned off, uint64_t v, unsigned mask)
{
    off &= ~7u;
    if (off < B_DATA_BYTES) {
        const int lo = (int8_t)(m_cfg & 0xFF), hi = (int8_t)((m_cfg >> 8) & 0xFF);
        const unsigned l2 = (m_cfg >> 16) & 15, cbase = (m_cfg >> 24) & 255, s = (m_cfg >> 32) & 31;
        const int32_t ma = (int32_t)(m_mul & 0xFFFFFFFFu), mb = (int32_t)(m_mul >> 32);
        const int32_t bg = (int32_t)(m_bias & 0xFFFFFFFFu);
        for (int k = 0; k < 8; k++) {
            if (!(mask & (1u << k))) continue;
            const unsigned o = off + (unsigned)k;
            const int8_t xn = (int8_t)(v >> (8 * k)), xo = (int8_t)m_data[o];
            int8_t r;
            switch (m_mode & 3) {
            case B_MODE_BN: {
                const unsigned c = ((o & ((1u << l2) - 1)) + cbase) & 255;
                r = b_rq((int64_t)xn * m_M[c] + m_B[c], m_S[c], lo, hi);
                break;
            }
            case B_MODE_ADD: r = b_rq((int64_t)xo * ma + (int64_t)xn * mb, (int)s, lo, hi); break;
            case B_MODE_AFFINE: r = b_rq((int64_t)xn * ma + bg, (int)s, lo, hi); break;
            default: r = xn;
            }
            m_data[o] = (uint8_t)r;
        }
        m_beats++;
        return;
    }
    if (off >= B_COEF && off < B_COEF + 16 * B_COEF_ENTRIES) {
        const unsigned e = (off - B_COEF) >> 4;
        if (off & 8) {
            if (mask & 1) m_S[e] = (uint8_t)(v & 31);
        } else {
            uint64_t w = merge((uint64_t)(uint32_t)m_M[e] | ((uint64_t)(uint32_t)m_B[e] << 32), v, mask);
            m_M[e] = (int32_t)(w & 0xFFFFFFFFu); m_B[e] = (int32_t)(w >> 32);
        }
        return;
    }
    switch (off) {
    case B_REG_MODE: m_mode = merge(m_mode, v, mask) & 3; m_beats = 0; break;
    case B_REG_CFG:  m_cfg = merge(m_cfg, v, mask) & 0x1FFF0FFFFFULL; break;
    case B_REG_MUL:  m_mul = merge(m_mul, v, mask); break;
    case B_REG_BIAS: m_bias = merge(m_bias, v, mask) & 0xFFFFFFFFu; break;
    default: break;
    }
}

static uint64_t window_read(unsigned off)
{
    off &= ~7u;
    if (off < B_DATA_BYTES) { uint64_t v; memcpy(&v, m_data + off, 8); return v; }
    switch (off) {
    case B_REG_ID:     return B_ID_VALUE;
    case B_REG_MODE:   return m_mode;
    case B_REG_CFG:    return m_cfg;
    case B_REG_MUL:    return m_mul;
    case B_REG_BIAS:   return m_bias;
    case B_REG_STATUS: return (m_beats & 0xFFFFFFFFu) << 32;   /* BUSY is 0 between accesses */
    default:           return 0;
    }
}

static uint64_t arena_off(uint64_t p)
{
    const uint8_t *q = (const uint8_t *)(uintptr_t)p;
    if (!m_arena || q < m_arena || q >= m_arena + m_arena_len) {
        fprintf(stderr, "b_model trace: DMA address %p outside the arena\n", (const void *)q);
        return ~0ULL;
    }
    return (uint64_t)(q - m_arena);
}

static void dma_go(uint64_t ctrl)
{
    const int to_mem = (ctrl >> 1) & 1;
    d_cycles = 0;
    if ((d_ext & 7) || (d_loc & 7) || (d_len & 7) || d_len == 0 || d_loc + d_len > 0xF000) {
        d_status = B_DMA_DONE | B_DMA_ERR;
        return;
    }
    uint8_t *mem = (uint8_t *)(uintptr_t)d_ext;
    const uint64_t aoff = m_tr ? arena_off(d_ext) : ~0ULL;
    if (m_tr && m_shadow && !to_mem && aoff != ~0ULL)
        for (uint64_t i = 0; i < d_len; i += 8)
            if (memcmp(m_shadow + aoff + i, mem + i, 8)) {
                uint64_t w; memcpy(&w, mem + i, 8);
                fprintf(m_tr, "M %llx %llx\n", (unsigned long long)(aoff + i), (unsigned long long)w);
                memcpy(m_shadow + aoff + i, mem + i, 8);
            }
    if (m_tr) fprintf(m_tr, "%c %llx %llx %llx\n", to_mem ? 'O' : 'I', (unsigned long long)aoff,
                      (unsigned long long)d_loc, (unsigned long long)d_len);
    for (uint64_t i = 0; i < d_len; i += 8) {
        uint64_t w;
        if (to_mem) { w = window_read((unsigned)(d_loc + i)); memcpy(mem + i, &w, 8); }
        else { memcpy(&w, mem + i, 8); window_write((unsigned)(d_loc + i), w, 0xFF); }
    }
    if (m_tr && m_shadow && to_mem && aoff != ~0ULL) memcpy(m_shadow + aoff, mem, d_len);
    d_cycles = d_len / 8;
    d_status = B_DMA_DONE;
}

void b_model_w64_mask(unsigned off, uint64_t v, unsigned mask)
{
    off &= 0xFFF8u;
    if ((off & 0xFF00u) == 0xF100u) {
        switch (off) {
        case B_DMA_EXT: d_ext = v; break;
        case B_DMA_LOC: d_loc = v & 0xFFFF; break;
        case B_DMA_LEN: d_len = v & 0xFFFFFFFFu; break;
        case B_DMA_CTRL: if (v & 1) dma_go(v); break;
        default: break;
        }
        return;
    }
    if (m_tr) fprintf(m_tr, "W %x %llx %x\n", off, (unsigned long long)v, mask);
    window_write(off, v, mask);
}

void b_model_w64(unsigned off, uint64_t v) { b_model_w64_mask(off, v, 0xFF); }

uint64_t b_model_r64(unsigned off)
{
    off &= 0xFFF8u;
    if ((off & 0xFF00u) == 0xF100u) {
        switch (off) {
        case B_DMA_EXT: return d_ext;
        case B_DMA_LOC: return d_loc;
        case B_DMA_LEN: return d_len;
        case B_DMA_CTRL: return d_status;
        case B_DMA_CYCLES: return d_cycles;
        case B_DMA_ID: return B_DMA_ID_VALUE;
        default: return 0;
        }
    }
    const uint64_t v = window_read(off);
    if (m_tr) fprintf(m_tr, "R %x %llx %llx\n", off, (unsigned long long)v, 0xFFFFFFFFFFFFFFFFULL);
    return v;
}
