#!/usr/bin/env bash
# DroNet for the RiskyBird option-2 SoC (RocketKU040LeanConfig): ONE Rocket hart (rv64imafc, FP32 FPU,
# no vector, no Gemmini), optionally engine D. Same toolchain/env variables as build_ku040_dual_nhwc_lowered.sh.
#
#   MODE=cpu  all-CPU baseline: extract -> fold_quant_consts (integer BN/add, no per-frame soft-float)
#             -> NCHW graph (no layout islands: every kernel is the scalar backend's) -> scalar kernels.
#             CPU_NUMERICS=board (default) pins conv2d_s8=q31_acc_scale, the deployed Gemmini requantize, so
#             the tensors equal the board's (golden) and the D image's; =reference keeps ModelBlaster's
#             two-stage reference requantize (outputs [117, 127], io.npz).
#   MODE=d    engine D: extract -> lower_d_engine (generates D's descriptor program + packed weight blob
#             with the imported reference packer; the network becomes one d_engine_s8 dispatch) ->
#             scalar backend with kernels/scalar/scalar_d_engine_s8_riskybird_d.c (d_driver.h).
#             D_SHAPE (default 16,16,4) must match the engine built into the bitstream (d_probe checks).
#             D_BASE (default 0x11000000, the integrator's RocketKU040LeanDConfig) is the engine window base.
#
# Zephyr target: stock chipyard_riscv64 + the HW integrator's lean overlay/conf (single hart, soft-float
# kernel, -march=rv64imac -mabi=lp64). Outputs: $GEN and $SCHED next to $BUILD_DIR.
set -u
: "${MB:=$(cd "$(dirname "$0")/../../.." && pwd)}"
: "${ZCS:?set ZCS to the zephyr-chipyard-sw checkout}"
: "${ZEPHYR_SDK:=$ZCS/tools-manual/zephyr-sdk-1.0.0-beta1}"
: "${CONDA_SH:?set CONDA_SH to etc/profile.d/conda.sh of a conda install}"
: "${CONDA_ENV:=zephyr}"
: "${LEAN_OVERLAY:?set LEAN_OVERLAY to the single-hart SoC devicetree overlay}"
: "${LEAN_CONF:?set LEAN_CONF to the single-hart SoC Kconfig fragment}"
: "${MODE:=cpu}"
: "${D_SHAPE:=16,16,4}"
: "${D_BASE:=0x11000000}"
: "${BUILD_DIR:=$MB/examples/dronet/int8/build/ku040_lean_$MODE}"
: "${GEN:=${BUILD_DIR%/}.gen}"
: "${SCHED:=${BUILD_DIR%/}.sched}"
: "${BUILD_ONLY:=0}"
: "${CPU_NUMERICS:=board}"
case "$MODE" in cpu|d) ;; *) echo "MODE must be cpu|d"; exit 2 ;; esac

source "$CONDA_SH"; conda activate "$CONDA_ENV"
export ZEPHYR_BASE="$ZCS/zephyr_ws/zephyr"
export ZEPHYR_SDK_INSTALL_DIR="$ZEPHYR_SDK"
export ZEPHYR_TOOLCHAIN_VARIANT=zephyr
export PATH="/usr/bin:$ZEPHYR_SDK/gnu/riscv64-zephyr-elf/bin:$PATH"
export PYTHONPATH="$(dirname "$MB")"
export MODELBLASTER_CURATED_VERIFY=0
export MODELBLASTER_DRONET_CHANNELS=1
export MB_ENABLE_FUSION=1
export MODELBLASTER_GEMMINI_CONFIG=q31ws_32x32_acc
cd "$MB"
die() { echo "FAILED: $*"; exit 1; }
REG=cores/chipyard_ku040_lean.json

if [ "$BUILD_ONLY" != "1" ]; then
  rm -rf "$GEN" "$SCHED"; mkdir -p "$GEN" "$SCHED"
  echo "== recipe: MODE=$MODE GEN=$GEN =="
  # Same extraction as the dual build (fused conv0+pool, BN/relu fused where the dual build fuses).
  python -m modelblaster.pipeline.extract_graph --model dronet --out-dir "$GEN/extract" \
      --quant int8 --num-calibration 1 --fusion-target gemmini_q31_rvv || die extract
  PIN=()
  if [ "$MODE" = "d" ]; then
    python pipeline/lower_d_engine.py "$GEN/extract/graph.json" "$GEN/extract/weights.npz" \
        --out-dir "$GEN" --shape "$D_SHAPE" || die lower_d_engine
    PIN=(--pin-algorithm d_engine_s8=riskybird_d)
  else
    cp "$GEN/extract/graph.json" "$GEN/extract/weights.npz" "$GEN/extract/io.npz" "$GEN/"
    python pipeline/fold_quant_consts.py "$GEN/graph.json" "$GEN/weights.npz" \
        --ops batchnorm2d_s8,add_s8 --require-exact || die fold
    [ "$CPU_NUMERICS" = "board" ] && PIN=(--pin-algorithm conv2d_s8=q31_acc_scale)
  fi
  python -m modelblaster.pipeline.generate_skeleton --ir "$GEN/graph.json" --weights "$GEN/weights.npz" \
      --io "$GEN/io.npz" --out-dir "$GEN/scalar" --backend scalar || die skeleton
  python -m modelblaster.pipeline.generate_kernels --ir "$GEN/graph.json" --out-dir "$GEN/scalar" \
      --backend reference --target scalar --quant int8 --io "$GEN/io.npz" --repo-root "$MB" \
      --build-dir "$GEN/kbuild" --harness-dir "$MB/harness" --cache-dir "$GEN/kcache" \
      --algorithms all --global-curated-dir "$MB/kernels" "${PIN[@]}" || die kernels
  echo "== schedule: every dispatch on hart 0 =="
  python scripts/gen_hetero_schedule.py --ir "$GEN/graph.json" --out "$SCHED/dronet_lean_sched.json" \
      --job-name dronet --policy gemmini_main_opu_skip || die schedule
  python - "$SCHED/dronet_lean_sched.json" <<'PY' || die placement
import json,sys
S=json.load(open(sys.argv[1]))
for v in S["dispatches"].values():
    v.pop("impl",None); v["hardware_target"]="CPU_P#0"
json.dump(S,open(sys.argv[1],"w"),indent=2)
PY
  python -m modelblaster.pipeline.ingest_xpurt_schedule --schedule "$SCHED/dronet_lean_sched.json" \
      --registry "$REG" --ir dronet:"$GEN/graph.json" --cpu-p-kind scalar --cpu-e-kind scalar \
      --name ku040_dronet_lean --out "$SCHED/ku040_dronet_lean.c" || die ingest
  python -m modelblaster.pipeline.generate_xpurt_main --schedule "$SCHED/dronet_lean_sched.json" \
      --out "$SCHED/ku040_dronet_lean_main.c" --name ku040_dronet_lean \
      --dispatch-table-header ku040_dronet_lean.h --platform zephyr --core-kinds scalar --backends scalar \
      --model-gen-dir dronet="$GEN/scalar" --networks dronet --registry "$REG" || die main
fi

KCF="-march=rv64imac;-mabi=lp64;-isystem$MB/kernels/riskybird;-DD_BASE=$D_BASE"
echo "== west build -> $BUILD_DIR =="
west build -p always -b chipyard_riscv64 harness_xpurt --build-dir "$BUILD_DIR" -- \
  -DMODEL_NAMES=dronet -DMODEL_DIRS_BASE="$GEN" -DMODEL_BACKENDS=scalar \
  -DXPURT_SCHEDULE_C="$SCHED/ku040_dronet_lean.c" -DXPURT_MAIN_C="$SCHED/ku040_dronet_lean_main.c" \
  -DXPURT_INCLUDE_DIR="$SCHED" -DMODELBLASTER_KERNEL_CFLAGS_SCALAR="$KCF" \
  -DEXTRA_CONF_FILE="$MB/harness_xpurt/backends/scalar.conf;$LEAN_CONF" \
  -DEXTRA_DTC_OVERLAY_FILE="$LEAN_OVERLAY"
rc=$?
echo "build rc=$rc  elf=$BUILD_DIR/zephyr/zephyr.elf"
exit $rc
