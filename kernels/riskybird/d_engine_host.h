/* Off-board stand-in for engine D: runs a pack_d blob's descriptor program on one image with the
 * imported functional model (d_model.c, the D team's bit-exact model of the RTL) and returns what the
 * driver's d_run_frame() would read back: the 16-byte head word and relu_modules_6 (NHWC).
 * d_model.c is included with its CLI main() renamed; nothing in it is modified. */
#ifndef D_ENGINE_HOST_H
#define D_ENGINE_HOST_H
#define main d_model_cli_main_
#include "d_model.c"
#undef main

static d_state_t dh_s;
static const void *dh_loaded;
/* Static backing (window maxima, SPEC §5) rather than calloc: bare-metal heaps are small. */
static uint8_t dh_wmem[D_WIN_PMEM];
static int8_t dh_bank[4][0x10000];

/* Bytes [off, off+len) of the window's bank region, from the model's banks. */
static void dh_bank_read(uint32_t off, uint8_t *dst, uint32_t len) {
    for (uint32_t i = 0; i < len; i++) {
        uint32_t o = off + i, b = (o - D_WIN_BANK(0)) / 0x10000u, r = o - D_WIN_BANK(b);
        dst[i] = (uint8_t)dh_s.bank[b][r];
    }
}

/* Load a blob into the model, as d_load_blob() does into the engine. Returns 0 or <0. */
static int d_model_load(const void *blob) {
    const d_blob_hdr_t *h = (const d_blob_hdr_t *)blob;
    const uint8_t *bb = (const uint8_t *)blob;
    if (memcmp(h->magic, D_BLOB_MAGIC, 8) || h->version != 1) return -1;
    memset(&dh_s, 0, sizeof dh_s);
    dh_s.p_oc = h->p_oc; dh_s.p_ic = h->p_ic; dh_s.n_q = h->n_q;
    dh_s.rowb = dh_s.p_oc * dh_s.p_ic; dh_s.wrows = (int)h->wrows;
    if ((size_t)dh_s.wrows * (size_t)dh_s.rowb > sizeof dh_wmem) return -3;
    memset(dh_wmem, 0, sizeof dh_wmem);
    dh_s.wmem = dh_wmem;
    for (int b = 0; b < 4; b++) {
        dh_s.bank_words[b] = h->bank_words[b];
        if ((size_t)dh_s.bank_words[b] * (size_t)dh_s.p_ic > sizeof dh_bank[b]) return -3;
        memset(dh_bank[b], 0, sizeof dh_bank[b]);
        dh_s.bank[b] = dh_bank[b];
    }
    const d_blob_seg_t *seg = (const d_blob_seg_t *)(bb + h->hdr_bytes);
    for (uint32_t i = 0; i < h->n_seg; i++)
        if (seg[i].kind == D_SEG_WINDOW) window_write(&dh_s, seg[i].win_off, bb + seg[i].file_off, seg[i].len);
    dh_loaded = blob;
    return 0;
}

/* One frame through the model; same outputs as d_run_frame(). Loads the blob first if it is not the
 * one loaded. The banks keep the previous frame's contents, as the engine's do. Returns 0 or <0. */
static int d_model_frame(const void *blob, const void *img, uint8_t head16[16], int8_t *relu6, int8_t out[2]) {
    const d_blob_hdr_t *h = (const d_blob_hdr_t *)blob;
    if (dh_loaded != blob) { int rc = d_model_load(blob); if (rc) return rc; }
    window_write(&dh_s, h->img_off, (const uint8_t *)img, h->img_len);
    for (int pc = (int)h->prog_start; pc < 64; pc++) {
        const uint64_t *d = &dh_s.pmem[pc * D_DESC_WORDS];
        int op = D_DESC_OP(d);
        if (op == D_OP_END) break;
        if (op == D_OP_CONV0) run_conv0(&dh_s, d);
        else if (op == D_OP_CONV) run_conv(&dh_s, d);
        else return -2;
    }
    dh_bank_read(h->head_off, head16, 16);
    if (relu6 && h->relu6_len) dh_bank_read(h->relu6_off, (uint8_t *)relu6, h->relu6_len);
    const int8_t *lut = d_blob_sigmoid_lut(blob);
    out[0] = (int8_t)head16[0];
    out[1] = lut ? lut[(int)(int8_t)head16[1] + 128] : (int8_t)head16[1];
    return 0;
}
#endif
