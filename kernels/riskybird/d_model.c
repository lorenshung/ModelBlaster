/* Host functional model of engine D (SPEC.md): executes the descriptor program of a pack_d.py blob on an
 * image with the engine's exact integer arithmetic and memory layout, and dumps every tensor each
 * descriptor writes. Bit-exactness target: the golden tensors in /scratch2/loren/soc-cache/p2goal/golden.
 *
 *   cc -O2 -std=c99 -o d_model d_model.c
 *   ./d_model d_blob.bin image.bin OUTDIR      -> OUTDIR/dNN_o1.bin, dNN_o2.bin (compact NHWC), cycles.txt
 *
 * The model follows the RTL's memory images: weight rows, per-channel records, 4 activation banks of
 * P_IC-byte words, engine A's weight/bias registers, and the descriptor words in PMEM.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "d_driver.h"

typedef struct {
    int p_oc, p_ic, n_q, rowb, wrows;
    int bank_words[4];
    uint8_t *wmem;                 /* wrows * rowb */
    uint64_t pmem[512];
    int8_t aw[9][32];              /* engine A weights [tap][lane] */
    int32_t ab[32];
    int8_t img[112 * 112];
    int8_t *bank[4];               /* bank_words[b] * p_ic bytes */
} d_state_t;

static int64_t clampi(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : v > hi ? hi : v; }

/* round half away from zero, divide by 2^S (S = 0: identity) */
static int64_t rha(int64_t v, int S) {
    if (S == 0) return v;
    return (v + ((int64_t)1 << (S - 1)) - (v < 0)) >> S;
}

/* Q0.31 multiply, half-up, then rounding (half-up) shift */
static int64_t requant(int64_t t, int64_t M, int shift) {
    int64_t q = (t * M + ((int64_t)1 << 30)) >> 31;
    if (shift > 0) q = (q + ((int64_t)1 << (shift - 1))) >> shift;
    return q;
}

typedef struct { int32_t bias, mc, bc, m; int sc; } rec_t;

static rec_t record(const d_state_t *s, int pbase, int c) {
    const uint8_t *p = s->wmem + (size_t)(pbase + c / s->n_q) * s->rowb + D_REC_BYTES * (c % s->n_q);
    uint32_t w[4];
    memcpy(w, p, 16);
    rec_t r;
    r.bias = (int32_t)w[0];
    r.mc = ((int32_t)(w[1] << 8)) >> 8;
    r.sc = (int)(w[1] >> 24);
    r.bc = (int32_t)w[2];
    r.m = (int32_t)w[3];
    return r;
}

static int8_t *bank_byte(d_state_t *s, int b, int word, int byte) {
    if (b < 0 || b > 3 || word < 0 || word >= s->bank_words[b] || byte < 0 || byte >= s->p_ic) {
        fprintf(stderr, "bank access out of range: bank %d word %d byte %d\n", b, word, byte);
        exit(2);
    }
    return s->bank[b] + (size_t)word * s->p_ic + byte;
}

static int t32_overflow;

/* The post unit for one channel: c1 is the requantized conv value (or engine A's byte). */
static void post(d_state_t *s, const uint64_t *d, int c, int8_t c1, int pix, int OC) {
    int flags = D_DESC_FLAGS(d), Q = s->p_ic, ocw = (OC + Q - 1) / Q;
    rec_t r = record(s, D_DESC_PBASE(d), c);
    int word = pix * ocw + c / Q, byte = c % Q;
    int64_t sv = c1;
    if (flags & D_F_RES) {
        int8_t res = *bank_byte(s, D_DESC_BRES(d), D_DESC_ARES(d) + word, byte);
        sv = clampi(rha((int64_t)res * D_DESC_MA(d) + (int64_t)c1 * D_DESC_MB(d), D_DESC_SA(d)),
                    D_DESC_LO2(d), D_DESC_HI2(d));
    }
    int64_t u = clampi(rha(sv * r.mc + r.bc, r.sc), D_DESC_LO3(d), D_DESC_HI3(d));
    if (flags & D_F_OUT1) *bank_byte(s, D_DESC_BO1(d), D_DESC_AO1(d) + word, byte) = (int8_t)sv;
    if (flags & D_F_OUT2) *bank_byte(s, D_DESC_BO2(d), D_DESC_AO2(d) + word, byte) = (int8_t)u;
}

static void run_conv0(d_state_t *s, const uint64_t *d) {
    /* engine A: conv 3x3/s2/p1 on 112x112 -> conv pixels 0..54, maxpool 3x3/s2 on raw accumulators,
     * requantize the 27x27x32 pooled values (exact: requant is monotonic for M > 0). */
    static int32_t acc[55][55][32];
    for (int r = 0; r < 55; r++)
        for (int c = 0; c < 55; c++)
            for (int l = 0; l < 32; l++) {
                int32_t a = s->ab[l];
                for (int t = 0; t < 9; t++) {
                    int iy = 2 * r - 1 + t / 3, ix = 2 * c - 1 + t % 3;
                    int x = (iy < 0 || ix < 0) ? 0 : s->img[iy * 112 + ix];
                    a += x * s->aw[t][l];
                }
                acc[r][c][l] = a;
            }
    for (int p = 0; p < 27; p++)
        for (int q = 0; q < 27; q++)
            for (int l = 0; l < 32; l++) {
                int32_t m = INT32_MIN;
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        if (acc[2 * p + i][2 * q + j][l] > m) m = acc[2 * p + i][2 * q + j][l];
                int64_t y = clampi(requant(m, D_DESC_AMULT(d), D_DESC_SHIFT(d)), D_DESC_LO1(d), D_DESC_HI1(d));
                post(s, d, l, (int8_t)y, p * 27 + q, 32);
            }
}

static void run_conv(d_state_t *s, const uint64_t *d) {
    int K = D_DESC_K(d), S = D_DESC_S(d), P = D_DESC_P(d), IC = D_DESC_IC(d), OC = D_DESC_OC(d);
    int IH = D_DESC_IH(d), IW = D_DESC_IW(d), OH = D_DESC_OH(d), OW = D_DESC_OW(d);
    int PO = s->p_oc, Q = s->p_ic, icc_n = (IC + Q - 1) / Q, og_n = (OC + PO - 1) / PO;
    int wbase = D_DESC_WBASE(d), bin = D_DESC_BIN(d), ain = D_DESC_AIN(d);
    for (int og = 0; og < og_n; og++)
        for (int oy = 0; oy < OH; oy++)
            for (int ox = 0; ox < OW; ox++) {
                int64_t acc[32] = {0};
                for (int kh = 0; kh < K; kh++) {
                    int iy = oy * S - P + kh;
                    if (iy < 0 || iy >= IH) continue;      /* padded taps are skipped */
                    for (int kw = 0; kw < K; kw++) {
                        int ix = ox * S - P + kw;
                        if (ix < 0 || ix >= IW) continue;
                        for (int icc = 0; icc < icc_n; icc++) {
                            const int8_t *x = bank_byte(s, bin, ain + (iy * IW + ix) * icc_n + icc, 0);
                            const uint8_t *row = s->wmem + (size_t)(wbase + ((og * K + kh) * K + kw) * icc_n + icc) * s->rowb;
                            for (int o = 0; o < PO; o++) {
                                int32_t dot = 0;
                                for (int j = 0; j < Q; j++) dot += x[j] * (int8_t)row[j * PO + o];
                                acc[o] += dot;
                            }
                        }
                    }
                }
                for (int o = 0; o < PO; o++) {
                    int c = og * PO + o;
                    if (c >= OC) continue;
                    rec_t r = record(s, D_DESC_PBASE(d), c);
                    int64_t t = acc[o] + r.bias;
                    if (t != (int32_t)t) t32_overflow++;
                    int64_t y = clampi(requant(t, r.m, D_DESC_SHIFT(d)), D_DESC_LO1(d), D_DESC_HI1(d));
                    post(s, d, c, (int8_t)y, oy * OW + ox, OC);
                }
            }
}

/* Dump an output tensor (compact NHWC, H*W*C bytes) from its bank. */
static void dump(d_state_t *s, const char *dir, int idx, const char *slot, int b, int base, int H, int W, int C) {
    char path[1024];
    snprintf(path, sizeof path, "%s/d%02d_%s.bin", dir, idx, slot);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    int Q = s->p_ic, cw = (C + Q - 1) / Q;
    for (int p = 0; p < H * W; p++)
        for (int c = 0; c < C; c++) fputc((uint8_t)*bank_byte(s, b, base + p * cw + c / Q, c % Q), f);
    fclose(f);
}

static uint8_t *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(*n + 8);
    if (fread(b, 1, *n, f) != *n) { perror(path); exit(1); }
    fclose(f);
    return b;
}

/* Copy a window write (as the driver's d_load_blob does) into the model's memories. */
static void window_write(d_state_t *s, uint32_t off, const uint8_t *src, uint32_t len) {
    if (off + len <= D_WIN_PMEM) {
        if (off + len > (uint32_t)s->wrows * s->rowb) { fprintf(stderr, "WMEM overflow\n"); exit(2); }
        memcpy(s->wmem + off, src, len);
    } else if (off >= D_WIN_PMEM && off + len <= D_WIN_PMEM + 8 * 512) {
        memcpy((uint8_t *)s->pmem + (off - D_WIN_PMEM), src, len);
    } else if (off == D_WIN_AW && len == 36 * 8) {
        for (int w = 0; w < 36; w++)
            for (int i = 0; i < 8; i++) s->aw[w / 4][8 * (w % 4) + i] = (int8_t)src[8 * w + i];
    } else if (off == D_WIN_AB && len == 16 * 8) {
        memcpy(s->ab, src, 128);
    } else if (off == D_WIN_IMG && len == 112 * 112) {
        memcpy(s->img, src, len);
    } else {
        fprintf(stderr, "unsupported window write 0x%x + %u\n", off, len);
        exit(2);
    }
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s d_blob.bin image.bin outdir\n", argv[0]); return 1; }
    size_t nb, ni;
    uint8_t *blob = read_file(argv[1], &nb), *img = read_file(argv[2], &ni);
    const d_blob_hdr_t *h = (const d_blob_hdr_t *)blob;
    if (memcmp(h->magic, D_BLOB_MAGIC, 8) || h->version != 1) { fprintf(stderr, "bad blob\n"); return 1; }
    if (ni != 112 * 112) { fprintf(stderr, "image must be 12544 bytes\n"); return 1; }
    d_state_t s;
    memset(&s, 0, sizeof s);
    s.p_oc = h->p_oc; s.p_ic = h->p_ic; s.n_q = h->n_q; s.rowb = s.p_oc * s.p_ic; s.wrows = (int)h->wrows;
    s.wmem = calloc((size_t)s.wrows, (size_t)s.rowb);
    for (int b = 0; b < 4; b++) {
        s.bank_words[b] = h->bank_words[b];
        s.bank[b] = calloc((size_t)s.bank_words[b], (size_t)s.p_ic);
    }
    const d_blob_seg_t *seg = (const d_blob_seg_t *)(blob + h->hdr_bytes);
    for (uint32_t i = 0; i < h->n_seg; i++)
        if (seg[i].kind == D_SEG_WINDOW) window_write(&s, seg[i].win_off, blob + seg[i].file_off, seg[i].len);
    window_write(&s, h->img_off, img, (uint32_t)ni);

    char path[1024];
    snprintf(path, sizeof path, "%s/outputs.txt", argv[3]);
    FILE *lf = fopen(path, "w");
    if (!lf) { perror(path); return 1; }
    for (int pc = (int)h->prog_start; pc < 64; pc++) {
        const uint64_t *d = &s.pmem[pc * D_DESC_WORDS];
        int op = D_DESC_OP(d);
        if (op == D_OP_END) break;
        int OH = op == D_OP_CONV0 ? 27 : (int)D_DESC_OH(d), OW = op == D_OP_CONV0 ? 27 : (int)D_DESC_OW(d);
        int OC = op == D_OP_CONV0 ? 32 : (int)D_DESC_OC(d);
        if (op == D_OP_CONV0) run_conv0(&s, d);
        else if (op == D_OP_CONV) run_conv(&s, d);
        else { fprintf(stderr, "bad opcode %d at %d\n", op, pc); return 1; }
        if (D_DESC_FLAGS(d) & D_F_OUT1) {
            dump(&s, argv[3], pc, "o1", D_DESC_BO1(d), D_DESC_AO1(d), OH, OW, OC);
            fprintf(lf, "%d o1 %d %d %d\n", pc, OH, OW, OC);
        }
        if (D_DESC_FLAGS(d) & D_F_OUT2) {
            dump(&s, argv[3], pc, "o2", D_DESC_BO2(d), D_DESC_AO2(d), OH, OW, OC);
            fprintf(lf, "%d o2 %d %d %d\n", pc, OH, OW, OC);
        }
    }
    fclose(lf);
    if (t32_overflow) fprintf(stderr, "warning: %d accumulators + bias overflow int32\n", t32_overflow);
    return 0;
}
