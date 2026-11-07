#!/usr/bin/env bash
# KU040 dual-hart DroNet, NHWC, with the RiskyBird lowerings selectable as recipe options.
# Same toolchain/env contract as build_ku040_dual_nhwc.sh (and the same variables), but the
# model and the schedule are ALWAYS regenerated, into a per-image directory, because the
# lowerings change the dispatch list and the committed ku040_dual_nhwc/ table hard-codes
# 28 dispatch ids (building a 20-dispatch graph against it would run the wrong functions).
#
# Recipe options (environment):
#   LOWER_SW=1     software lowerings:
#                    fold_quant_consts.py   batchnorm2d_s8 -> batchnorm2d_q_s8, add_s8 -> add_q_s8
#                                           (build-time integer constants, no per-frame soft-float)
#                    assign_layouts --alias-c1 --flatten-into-linear
#                                           BN joins the NHWC island (no relayouts around it), the
#                                           C=1 input relayout is an alias, and the linears take
#                                           NHWC-permuted weights (no tail relayout).
#                  0 = the baseline graph (identical to build_ku040_dual_nhwc.sh's).
#   ACCEL=none|mmio|dma
#                  conv2d_pool_s8 (conv0 + maxpool1) on hart 1:
#                    none  Gemmini tiled_conv_pool_nhwc (default)
#                    mmio  RiskyBird conv0 engine, CPU copies over MMIO (dronet_conv0_mmio)
#                    dma   conv0 engine + its DMA, MMIO fallback when no DMA (dronet_conv0_dma)
#   FOLD_REQUIRE_EXACT=1  refuse the fold if any folded op differs from its float reference
#                  anywhere in the int8 input domain (true for DroNet; see the fold log).
#
# Outputs: $GEN (graph, weights, both backends' sources) and $SCHED (schedule, dispatch table,
# main) next to $BUILD_DIR, so every image keeps its own sources.
set -u
: "${MB:=$(cd "$(dirname "$0")/../../.." && pwd)}"
: "${ZCS:=/scratch2/dima/misc_sw/XPU-RT/zephyr-chipyard-sw}"
: "${ZEPHYR_SDK:=$ZCS/tools-manual/zephyr-sdk-1.0.0-beta1}"
: "${CONDA_SH:=/scratch2/dima/miniforge3/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=zephyr}"
: "${GEMMINI_SPIKE:=/scratch2/dima/chipyard-fsim/.conda-env/riscv-tools/bin/spike}"
: "${GEMMINI_LIB:=/scratch2/dima/chipyard-fsim/.conda-env/riscv-tools/lib/libgemmini.so}"
: "${CONSOLE_OVERLAY:=/scratch2/dima/misc_sw/ku040_dronedual_console.overlay}"
: "${CONSOLE_CONF:=/scratch2/dima/misc_sw/ku040_dronedual_console.conf}"
: "${GEMMINI_CONFIG:=q31ws_32x32_acc}"
: "${BUILD_DIR:=$MB/examples/dronet/int8/build/ku040_dronet_nhwc_lowered}"
: "${LOWER_SW:=1}"
: "${ACCEL:=none}"
: "${FOLD_REQUIRE_EXACT:=1}"
: "${GEN:=${BUILD_DIR%/}.gen}"
: "${SCHED:=${BUILD_DIR%/}.sched}"
: "${BUILD_ONLY:=0}"     # 1 = skip regeneration, rebuild from existing $GEN/$SCHED

case "$ACCEL" in
  none) PIN=() ;;
  mmio) PIN=(--pin-algorithm conv2d_pool_s8=dronet_conv0_mmio) ;;
  dma)  PIN=(--pin-algorithm conv2d_pool_s8=dronet_conv0_dma) ;;
  *) echo "ACCEL must be none|mmio|dma, got '$ACCEL'"; exit 2 ;;
esac

source "$CONDA_SH"; conda activate "$CONDA_ENV"
export ZEPHYR_BASE="$ZCS/zephyr_ws/zephyr"
export ZEPHYR_SDK_INSTALL_DIR="$ZEPHYR_SDK"
export ZEPHYR_TOOLCHAIN_VARIANT=zephyr
export PATH="/usr/bin:$ZEPHYR_SDK/gnu/riscv64-zephyr-elf/bin:$PATH"
export PYTHONPATH="$(dirname "$MB")"
export MODELBLASTER_GEMMINI_CONFIG="$GEMMINI_CONFIG"
export MODELBLASTER_CURATED_VERIFY=0
export MODELBLASTER_DRONET_CHANNELS=1
export MB_ENABLE_FUSION=1
export MODELBLASTER_GEMMINI_SPIKE="$GEMMINI_SPIKE"
export MODELBLASTER_GEMMINI_LIB="$GEMMINI_LIB"
cd "$MB"

die() { echo "FAILED: $*"; exit 1; }

if [ "$BUILD_ONLY" != "1" ]; then
  rm -rf "$GEN" "$SCHED"; mkdir -p "$GEN" "$SCHED"
  echo "== recipe: LOWER_SW=$LOWER_SW ACCEL=$ACCEL GEN=$GEN SCHED=$SCHED =="
  echo "== extract (grayscale, fused) =="
  python -m modelblaster.pipeline.extract_graph --model dronet --out-dir "$GEN" \
      --quant int8 --num-calibration 1 --fusion-target gemmini_q31_rvv || die extract
  LAYOUT_FLAGS=()
  if [ "$LOWER_SW" = "1" ]; then
    echo "== fold_quant_consts (BN + add constants at build time) =="
    EXACT=(); [ "$FOLD_REQUIRE_EXACT" = "1" ] && EXACT=(--require-exact)
    python pipeline/fold_quant_consts.py "$GEN/graph.json" "$GEN/weights.npz" \
        --ops batchnorm2d_s8,add_s8 "${EXACT[@]}" || die fold
    LAYOUT_FLAGS=(--alias-c1 --flatten-into-linear)
  fi
  echo "== assign_layouts --policy islands ${LAYOUT_FLAGS[*]} =="
  python pipeline/assign_layouts.py "$GEN/graph.json" "$GEN/graph.json" --policy islands \
      --hint examples/dronet/int8/dronet_layout_hint.json --model dronet --report \
      "${LAYOUT_FLAGS[@]}" || die assign_layouts
  for BK in gemmini_q31_rvv rvv; do
    BKPIN=(); [ "$BK" = gemmini_q31_rvv ] && BKPIN=("${PIN[@]}")
    echo "== generate_skeleton [$BK] =="
    python -m modelblaster.pipeline.generate_skeleton --ir "$GEN/graph.json" \
        --weights "$GEN/weights.npz" --io "$GEN/io.npz" --out-dir "$GEN/$BK" --backend "$BK" \
        || die "skeleton $BK"
    echo "== generate_kernels [$BK] ${BKPIN[*]} =="
    python -m modelblaster.pipeline.generate_kernels --ir "$GEN/graph.json" \
        --out-dir "$GEN/$BK" --backend reference --target "$BK" --quant int8 \
        --io "$GEN/io.npz" --repo-root "$MB" --build-dir "$GEN/kbuild_$BK" \
        --harness-dir "$MB/harness" --cache-dir "$GEN/cache_$BK" \
        --algorithms all --global-curated-dir "$MB/kernels" "${BKPIN[@]}" || die "kernels $BK"
  done

  echo "== schedule (conv/pool + relayouts -> gemmini/hart1; rest -> rvv/hart0) =="
  python scripts/gen_hetero_schedule.py --ir "$GEN/graph.json" \
      --out "$SCHED/dronet_ku040_nhwc_sched.json" --job-name dronet --policy gemmini_main_opu_skip \
      || die schedule
  python - "$GEN/graph.json" "$SCHED/dronet_ku040_nhwc_sched.json" <<'PY' || die placement
import json,sys
g=json.load(open(sys.argv[1])); opk={o.get("dispatch_id"):o.get("op") for o in g["ops"] if o.get("dispatch_id") is not None}
S=json.load(open(sys.argv[2]))
# Same placement rule as build_ku040_dual_nhwc.sh: Gemmini-side kinds -> hart1, the rest -> hart0.
GEM={"conv2d_s8","conv2d_pool_s8","nchw_to_nhwc_s8","nhwc_to_nchw_s8"}
for k,v in S["dispatches"].items():
    v.pop("impl",None)
    v["hardware_target"]="CPU_P#0" if opk.get(v.get("id")) in GEM else "CPU_E#0"
json.dump(S,open(sys.argv[2],"w"),indent=2)
PY
  python -m modelblaster.pipeline.ingest_xpurt_schedule --schedule "$SCHED/dronet_ku040_nhwc_sched.json" \
      --registry cores/chipyard_ku040_dronedual_q31.json --ir dronet:"$GEN/graph.json" \
      --cpu-p-kind gemmini_q31 --cpu-e-kind rvv --name ku040_dronet_nhwc_dev \
      --out "$SCHED/ku040_dronet_nhwc_dev.c" || die ingest
  python -m modelblaster.pipeline.generate_xpurt_main --schedule "$SCHED/dronet_ku040_nhwc_sched.json" \
      --out "$SCHED/ku040_dronet_nhwc_dev_main.c" --name ku040_dronet_nhwc_dev \
      --dispatch-table-header ku040_dronet_nhwc_dev.h --platform zephyr \
      --core-kinds rvv,gemmini_q31 --backends rvv,gemmini_q31_rvv \
      --model-gen-dir dronet="$GEN/gemmini_q31_rvv" --networks dronet \
      --registry cores/chipyard_ku040_dronedual_q31.json || die main
fi

SOFTFLOAT_CONF=$(mktemp)
cat > "$SOFTFLOAT_CONF" <<CONF
CONFIG_FPU=n
CONFIG_RISCV_ISA_EXT_V=y
CONFIG_RISCV_ISA_EXT_V_LAZY=n
CONFIG_RISCV_V_KERNEL_ONLY=y
CONFIG_MP_MAX_NUM_CPUS=2
CONFIG_RV_BOOT_HART=0
CONF
PC=$MB/cores/gemmini/include/per_config/$GEMMINI_CONFIG
GCF="-march=rv64imac_zve64x;-mabi=lp64;-isystem$PC;-isystem$MB/cores/gemmini/include;-isystem$MB/cores/gemmini;-isystem$MB/kernels/gemmini_q31_rvv;-DGEMMINI_ROCC;-DBAREMETAL;-DMODELBLASTER_GEMMINI_HWIO_WEIGHTS=1;-DMODELBLASTER_GEMMINI_Q31_ACC_SCALE=1"
RCF="-march=rv64imac_zve64x;-mabi=lp64;-DMODELBLASTER_RVV_IHWOC_WEIGHTS=1"
echo "== west build -> $BUILD_DIR =="
west build -p always -b chipyard_riscv64 harness_xpurt --build-dir "$BUILD_DIR" -- \
  -DMODEL_NAMES=dronet -DMODEL_DIRS_BASE="$GEN" -DMODEL_BACKENDS=gemmini_q31_rvv,rvv \
  -DXPURT_SCHEDULE_C="$SCHED/ku040_dronet_nhwc_dev.c" -DXPURT_MAIN_C="$SCHED/ku040_dronet_nhwc_dev_main.c" \
  -DXPURT_INCLUDE_DIR="$SCHED" \
  -DMODELBLASTER_KERNEL_CFLAGS_GEMMINI_Q31_RVV="$GCF" -DMODELBLASTER_KERNEL_CFLAGS_RVV="$RCF" \
  -DEXTRA_CONF_FILE="$MB/harness_xpurt/backends/rvv.conf;$MB/harness/backends/firesim_chipyard_dual_gemmini.conf;$CONSOLE_CONF;$SOFTFLOAT_CONF" \
  -DEXTRA_DTC_OVERLAY_FILE="$CONSOLE_OVERLAY"
rc=$?
rm -f "$SOFTFLOAT_CONF"
echo "build rc=$rc  elf=$BUILD_DIR/zephyr/zephyr.elf"
exit $rc
