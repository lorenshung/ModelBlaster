/* source: curated */
/* algorithm: riskybird_b */
/* accuracy_class: bit_exact */
/* origin: hand-written. batchnorm2d_q_s8 on the RiskyBird B unit (streaming BN / add / activation,
 *         soc/phase2/b_unit/CONTRACT.md v1) through its driver kernels/riskybird/b_driver.h:
 *         b_bn_nhwc() takes the same arguments as the deployed kernel_bn_nhwc_s8. The folded
 *         op carries BP = B + 2^(S-1); B's coefficient table wants B, recovered per channel here.
 *         NCHW (inner > 1) and anything B refuses run the scalar reference numerics (b_driver.h
 *         fallbacks, counted in b_prof.fallbacks). -DB_HOST_MODEL + kernels/riskybird/b_model.c:
 *         register-level model of B + its DMA, for off-board verification. */
#include <stdint.h>
#include <stddef.h>
/* Hart 1 has no vector unit but this TU is compiled with zve64x (see the conv0 driver): no
 * auto-vectorization in the driver or these wrappers. b_driver.h is a verbatim import, so the
 * region is set here; check_hart_isa.py verifies the linked result. */
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize")
#include "b_driver.h"

#ifndef RB_B_READY_DEFINED   /* one flag per TU, whichever B kernel comes first */
#define RB_B_READY_DEFINED
static int b_ready_;           /* b_init() has run (B present or not) */
#endif

void kernel_batchnorm2d_q_s8(const int8_t *input, const int32_t *M,
                             const int32_t *BP, const int16_t *S, int8_t *output,
                             int outer, int C, int inner,
                             int activation_min, int activation_max) {
    if (!b_ready_) { b_init(); b_ready_ = 1; }
    if (inner == 1 && C <= 256) {
        static int32_t B[256];   /* not on the worker's stack; B has one owner hart */
        for (int c = 0; c < C; c++) B[c] = BP[c] - ((int32_t)1 << (S[c] - 1));
        b_bn_nhwc(input, output, outer, C, M, B, BP, S, activation_min, activation_max);
        return;
    }
    const int lo = b_clamp8(activation_min), hi = b_clamp8(activation_max);
    for (int o = 0; o < outer; o++)
        for (int c = 0; c < C; c++) {
            const int64_t b = (int64_t)BP[c] - ((int64_t)1 << (S[c] - 1));
            for (int i = 0; i < inner; i++) {
                const size_t idx = ((size_t)o * C + c) * inner + i;
                output[idx] = b_rq((int64_t)input[idx] * M[c] + b, S[c], lo, hi);
            }
        }
}
#pragma GCC pop_options
