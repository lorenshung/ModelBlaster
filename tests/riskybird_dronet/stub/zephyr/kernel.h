/* Bare-metal stand-in for the one Zephyr symbol generated model.c uses (spike replay). */
#pragma once
#include <stdint.h>
static inline uint64_t k_cycle_get_64(void) { uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }
