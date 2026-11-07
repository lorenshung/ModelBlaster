/* Engine D (whole-network DroNet accelerator): register map, formats and bare-metal driver API.
 * Contract: soc/phase2/d_engine/SPEC.md sections 5-7. Freestanding C99; no libc beyond <stdint.h>/<stddef.h>.
 *
 * D sits behind a 1 MiB memory-mapped window (one 64-bit access per cycle, the conv0_mmio port protocol)
 * with a C-style DMA whose registers sit in the window's 0xF1000 page. Boot: d_load_blob() copies the blob
 * produced by pack_d.py (weights, per-channel records, descriptor program, engine-A weights) into the window
 * once. Per frame: d_run_frame() moves the 12,544-byte image in, runs the program, and moves the 16-byte head
 * result (linear1, linear2) out; optionally the 2,048-byte relu_modules_6 tensor (NHWC) as well.
 */
#ifndef D_DRIVER_H
#define D_DRIVER_H
#include <stddef.h>
#include <stdint.h>

/* ---------------- window map (byte offsets from the window base) ---------------- */
#define D_WIN_BYTES        0x100000u
#define D_WIN_WMEM         0x00000u   /* weight rows, write-only, row r at r * P_OC * P_IC        */
#define D_WIN_PMEM         0x80000u   /* descriptor program, 512 x 64-bit words, RW               */
#define D_WIN_AW           0x88000u   /* engine A weights, 36 words (conv0_mmio 0xE000 encoding)  */
#define D_WIN_AB           0x88200u   /* engine A bias, 16 words (conv0_mmio 0xE200 encoding)     */
#define D_WIN_IMG          0x90000u   /* input image, 112 x 112 int8, write-only                  */
#define D_WIN_BANK(b)      (0xA0000u + 0x10000u * (uint32_t)(b))   /* activation bank b, RW while idle */
#define D_WIN_REGS         0xF0000u
#define D_WIN_DMA          0xF1000u   /* the integrator's DMA register page (C's 0xF100 page)     */
#define D_WIN_LOCAL_LIMIT  0xF0000u   /* DMA: LOC + LEN <= this                                    */

/* ---------------- registers (64-bit, offsets from D_WIN_REGS) ---------------- */
#define D_REG_ID      0x00   /* R  0x0000_0001_444E_5244 = "DRND" v1                                        */
#define D_REG_CTRL    0x08   /* W  bit0 start (ignored while busy), bit1 clear done/irq/error                */
#define D_REG_STATUS  0x08   /* R  bit0 busy, bit1 done (sticky), bit2 error (sticky), bit3 irq             */
#define D_REG_CYCLES  0x10   /* R  cycles of the last run, start -> done                                     */
#define D_REG_PROG    0x18   /* RW [8:0] first descriptor index                                              */
#define D_REG_CAPS    0x20   /* R  [7:0] P_OC [15:8] P_IC [23:16] N_Q [31:24] banks [47:32] WROWS           */
#define D_REG_IRQEN   0x28   /* RW bit0: irq = done && IRQEN                                                 */
#define D_REG_PC      0x30   /* R  [8:0] index of the descriptor running (or last run)                        */
#define D_REG_BANKW   0x38   /* R  [16b x 4] words per activation bank                                       */
#define D_ID_VALUE    0x00000001444E5244ull
#define D_ST_BUSY  1u
#define D_ST_DONE  2u
#define D_ST_ERROR 4u
#define D_ST_IRQ   8u

/* ---------------- descriptor (8 x 64-bit words in PMEM; SPEC 6.2) ---------------- */
#define D_OP_END   0
#define D_OP_CONV0 1
#define D_OP_CONV  2
#define D_F_RES    1
#define D_F_OUT1   2
#define D_F_OUT2   4
#define D_DESC_WORDS 8
#define D_FLD(w, lo, n) ((uint32_t)(((w) >> (lo)) & ((1ull << (n)) - 1)))
#define D_SFLD8(w, lo)  ((int32_t)(int8_t)D_FLD(w, lo, 8))
/* word 0 */
#define D_DESC_OP(d)     D_FLD((d)[0], 0, 8)
#define D_DESC_FLAGS(d)  D_FLD((d)[0], 8, 8)
#define D_DESC_BIN(d)    D_FLD((d)[0], 16, 2)
#define D_DESC_BRES(d)   D_FLD((d)[0], 18, 2)
#define D_DESC_BO1(d)    D_FLD((d)[0], 20, 2)
#define D_DESC_BO2(d)    D_FLD((d)[0], 22, 2)
#define D_DESC_K(d)      D_FLD((d)[0], 24, 4)
#define D_DESC_S(d)      D_FLD((d)[0], 28, 2)
#define D_DESC_P(d)      D_FLD((d)[0], 30, 2)
#define D_DESC_IC(d)     D_FLD((d)[0], 32, 16)
#define D_DESC_OC(d)     D_FLD((d)[0], 48, 16)
/* word 1 */
#define D_DESC_IH(d)     D_FLD((d)[1], 0, 8)
#define D_DESC_IW(d)     D_FLD((d)[1], 8, 8)
#define D_DESC_OH(d)     D_FLD((d)[1], 16, 8)
#define D_DESC_OW(d)     D_FLD((d)[1], 24, 8)
#define D_DESC_WBASE(d)  D_FLD((d)[1], 32, 16)
#define D_DESC_PBASE(d)  D_FLD((d)[1], 48, 16)
/* word 2: bank word addresses */
#define D_DESC_AIN(d)    D_FLD((d)[2], 0, 16)
#define D_DESC_ARES(d)   D_FLD((d)[2], 16, 16)
#define D_DESC_AO1(d)    D_FLD((d)[2], 32, 16)
#define D_DESC_AO2(d)    D_FLD((d)[2], 48, 16)
/* word 3 */
#define D_DESC_AMULT(d)  D_FLD((d)[3], 0, 32)   /* engine A multiplier (CONV0 only) */
#define D_DESC_SHIFT(d)  D_FLD((d)[3], 32, 8)   /* requant rounding shift, 0..31 */
#define D_DESC_LO1(d)    D_SFLD8((d)[3], 40)
#define D_DESC_HI1(d)    D_SFLD8((d)[3], 48)
#define D_DESC_SA(d)     D_FLD((d)[3], 56, 8)   /* add rounding shift, 0..31 */
/* word 4 */
#define D_DESC_MA(d)     ((int32_t)D_FLD((d)[4], 0, 32))
#define D_DESC_MB(d)     ((int32_t)D_FLD((d)[4], 32, 32))
/* word 5 */
#define D_DESC_LO2(d)    D_SFLD8((d)[5], 0)
#define D_DESC_HI2(d)    D_SFLD8((d)[5], 8)
#define D_DESC_LO3(d)    D_SFLD8((d)[5], 16)
#define D_DESC_HI3(d)    D_SFLD8((d)[5], 24)
/* word 7: written by the engine when the descriptor finishes: [31:0] cycles */
#define D_DESC_CYCLES(d) D_FLD((d)[7], 0, 32)

/* ---------------- per-channel record (16 bytes in a weight-memory row; SPEC 6.3) ---------------- */
#define D_REC_BYTES 16   /* bias i32 | Mc i24 + Sc u8 | Bc i32 | M i32 */

/* ---------------- blob (pack_d.py output; SPEC 6.4) ---------------- */
#define D_BLOB_MAGIC "DRNDBLB1"
#define D_SEG_WINDOW 0
#define D_SEG_HOST_SIGMOID_LUT 1
typedef struct {
    char     magic[8];
    uint32_t version, hdr_bytes;
    uint8_t  p_oc, p_ic, n_q, n_banks;
    uint32_t wrows;
    uint16_t bank_words[4];
    uint32_t n_seg, n_desc, prog_start;
    uint32_t img_off, img_len;
    uint32_t relu6_off, relu6_len;
    uint32_t head_off, head_len, head_count;
    uint32_t wmem_rows, window_bytes;
    uint8_t  reserved[128 - 80];
} d_blob_hdr_t;
typedef struct { uint32_t win_off, len, file_off, kind; } d_blob_seg_t;

/* ---------------- driver ---------------- */
typedef struct {
    volatile uint8_t *win;      /* window base (uncached) */
    int has_dma;                /* set by d_probe() */
} d_dev_t;

static inline volatile uint64_t *d_reg(d_dev_t *d, uint32_t off) {
    return (volatile uint64_t *)(d->win + D_WIN_REGS + off);
}
static inline volatile uint64_t *d_dma_reg(d_dev_t *d, uint32_t off) {
    return (volatile uint64_t *)(d->win + D_WIN_DMA + off);
}

/* C-style DMA registers (DronetConv0.scala; page moved to D_WIN_DMA, semantics unchanged). */
#define D_DMA_EXT 0x00
#define D_DMA_LOC 0x08
#define D_DMA_LEN 0x10
#define D_DMA_CTRL 0x18      /* W bit0 go, bit1 dir (0 memory -> local, 1 local -> memory); R status */
#define D_DMA_CYCLES 0x20
#define D_DMA_ID 0x28
#define D_DMA_ID_VALUE 0x00000001414D4443ull

/* Returns 0 if the engine answers and matches the blob's array shape. */
static inline int d_probe(d_dev_t *d, const void *blob) {
    const d_blob_hdr_t *h = (const d_blob_hdr_t *)blob;
    if (*d_reg(d, D_REG_ID) != D_ID_VALUE) return -1;
    uint64_t caps = *d_reg(d, D_REG_CAPS);
    if (h && ((caps & 0xFF) != h->p_oc || ((caps >> 8) & 0xFF) != h->p_ic || ((caps >> 16) & 0xFF) != h->n_q ||
              ((caps >> 32) & 0xFFFF) < h->wmem_rows))
        return -2;
    d->has_dma = *d_dma_reg(d, D_DMA_ID) == D_DMA_ID_VALUE;
    return 0;
}

/* Copy len bytes (multiple of 8, 8-byte aligned both sides) between memory and the window.
 * dir 0: memory -> window, 1: window -> memory. Uses the DMA when present, else 64-bit MMIO. */
static inline int d_copy(d_dev_t *d, void *mem, uint32_t win_off, uint32_t len, int dir) {
    if (d->has_dma) {
        __asm__ volatile("fence" ::: "memory");
        *d_dma_reg(d, D_DMA_EXT) = (uint64_t)(uintptr_t)mem;
        *d_dma_reg(d, D_DMA_LOC) = win_off;
        *d_dma_reg(d, D_DMA_LEN) = len;
        *d_dma_reg(d, D_DMA_CTRL) = 1u | ((uint64_t)dir << 1);
        uint64_t st;
        while ((st = *d_dma_reg(d, D_DMA_CTRL)) & 1u) {}
        __asm__ volatile("fence" ::: "memory");
        return (st & 4u) ? -1 : 0;
    }
    volatile uint64_t *w = (volatile uint64_t *)(d->win + win_off);
    uint64_t *m = (uint64_t *)mem;
    for (uint32_t i = 0; i < len / 8; i++) {
        if (dir) m[i] = w[i]; else w[i] = m[i];
    }
    return 0;
}

/* Boot: copy every window segment of the blob into the engine. The blob must be 8-byte aligned. */
static inline int d_load_blob(d_dev_t *d, const void *blob) {
    const d_blob_hdr_t *h = (const d_blob_hdr_t *)blob;
    const d_blob_seg_t *s = (const d_blob_seg_t *)((const uint8_t *)blob + h->hdr_bytes);
    for (uint32_t i = 0; i < h->n_seg; i++)
        if (s[i].kind == D_SEG_WINDOW &&
            d_copy(d, (void *)((const uint8_t *)blob + s[i].file_off), s[i].win_off, s[i].len, 0))
            return -1;
    *d_reg(d, D_REG_PROG) = h->prog_start;
    return 0;
}

/* Host sigmoid table (-128..127 -> int8) from the blob, or NULL. */
static inline const int8_t *d_blob_sigmoid_lut(const void *blob) {
    const d_blob_hdr_t *h = (const d_blob_hdr_t *)blob;
    const d_blob_seg_t *s = (const d_blob_seg_t *)((const uint8_t *)blob + h->hdr_bytes);
    for (uint32_t i = 0; i < h->n_seg; i++)
        if (s[i].kind == D_SEG_HOST_SIGMOID_LUT) return (const int8_t *)blob + s[i].file_off;
    return 0;
}

/* One frame. img: 12,544 bytes, 8-byte aligned. head16: 16-byte buffer, receives the head word
 * (byte n = output n of the head; DroNet: [0] linear1 = steering, [1] linear2 = collision logit).
 * relu6: NULL or a 2,048-byte buffer for relu_modules_6 (NHWC). out[2] = {steer, sigmoid(linear2)}.
 * Returns 0, or <0 on a DMA or engine error. */
static inline int d_run_frame(d_dev_t *d, const void *blob, const void *img, void *head16, void *relu6,
                              int8_t out[2]) {
    const d_blob_hdr_t *h = (const d_blob_hdr_t *)blob;
    if (d_copy(d, (void *)img, h->img_off, h->img_len, 0)) return -1;
    *d_reg(d, D_REG_CTRL) = 2u;                 /* clear done */
    *d_reg(d, D_REG_CTRL) = 1u;                 /* start */
    uint64_t st;
    while (((st = *d_reg(d, D_REG_STATUS)) & D_ST_DONE) == 0) {}
    if (st & D_ST_ERROR) return -2;
    if (d_copy(d, head16, h->head_off, 16, 1)) return -3;
    if (relu6 && h->relu6_len && d_copy(d, relu6, h->relu6_off, h->relu6_len, 1)) return -4;
    const int8_t *hd = (const int8_t *)head16, *lut = d_blob_sigmoid_lut(blob);
    out[0] = hd[0];
    out[1] = lut ? lut[(int)hd[1] + 128] : hd[1];
    return 0;
}

#endif /* D_DRIVER_H */
