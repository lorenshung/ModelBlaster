/* Replay hook (-DREPLAY_AFTER_HOOK): reports whether the B model's data window changed during the
 * dispatch, i.e. whether the driver really drove B instead of taking its scalar fallback. */
#include <stdint.h>
#include <stdio.h>
const uint8_t *b_model_data(void);
static uint64_t last;
void replay_after(int d) {
    const uint8_t *m = b_model_data();
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < 32768; i++) h = (h ^ m[i]) * 1099511628211ull;
    printf("  [B model window %s after dispatch %d]\n", h != last ? "CHANGED" : "unchanged", d);
    last = h;
}
