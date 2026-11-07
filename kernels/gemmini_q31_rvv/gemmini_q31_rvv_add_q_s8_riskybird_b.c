/* source: curated */
/* algorithm: riskybird_b */
/* accuracy_class: bit_exact */
/* origin: hand-written. add_q_s8 on the RiskyBird B unit (CONTRACT.md v1): b_add() takes the same
 *         arguments as the deployed kernel_add_q_s8. See the batchnorm2d_q_s8 file for the model. */
#include <stdint.h>
#include <stddef.h>
#include "b_driver.h"

#ifndef RB_B_READY_DEFINED   /* one flag per TU, whichever B kernel comes first */
#define RB_B_READY_DEFINED
static int b_ready_;           /* b_init() has run (B present or not) */
#endif

void kernel_add_q_s8(const int8_t *a, const int8_t *b, int8_t *output, int n,
                     int ma, int mb, int S, int activation_min, int activation_max) {
    if (!b_ready_) { b_init(); b_ready_ = 1; }
    b_add(a, b, output, n, ma, mb, S, activation_min, activation_max);
}
