#!/bin/bash
# Regenerate the two conv0-engine curated drivers from the one template.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); K=$H/..
gen() { sed -e "s|@ALGO@|$1|" -e "s|@USE_DMA@|$2|" -e "s|@MODE_DESC@|$3|" "$H/dronet_conv0_driver.c.in" \
        > "$K/gemmini_q31_rvv/gemmini_q31_rvv_conv2d_pool_s8_$1.c"; }
gen dronet_conv0_mmio 0 "the CPU copying the image in and the result out with 8-byte uncached MMIO accesses (README §9)."
gen dronet_conv0_dma 1 "the engine's DMA moving the data (candidate C, README §10); MMIO copies when the DMA ID does not read back or the device rejects a transfer."
