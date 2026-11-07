/* Replay hook (-DREPLAY_AFTER_HOOK) for d_engine_s8: prints the kernel's setup/frame counters after
 * each dispatch, so a replay shows whether model_<mid>_init() did the blob load (d_lazy_boots == 0). */
#include <stdio.h>
extern long long d_boot_status, d_status;
extern unsigned long long d_boots, d_lazy_boots, d_frames;
void replay_after(int d) {
    printf("  [D after dispatch %d: boot_status %ld boots %lu lazy_boots %lu frames %lu status %ld]\n", d,
           (long)d_boot_status, (unsigned long)d_boots, (unsigned long)d_lazy_boots, (unsigned long)d_frames,
           (long)d_status);
}
