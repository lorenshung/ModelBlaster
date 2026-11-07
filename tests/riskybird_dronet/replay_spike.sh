#!/bin/bash
# Replay a generated backend's dispatches on spike against golden tensors (see gen_replay.py).
# Usage: replay_spike.sh GEN BACKEND GOLDEN_DIR [OUT_DIR] [--ops ...] [EXTRA_CFLAGS via $CFLAGS_EXTRA]
# Toolchain: Zephyr SDK GCC 14 for the code, Chipyard conda newlib (htif) to link, spike VLEN=128.
set -euo pipefail
GEN=$(readlink -f "$1"); BK=$2; GOLD=$(readlink -f "$3"); O=${4:-$GEN/replay_$BK}; shift 3; [ $# -gt 0 ] && shift
H=$(dirname "$(readlink -f "$0")"); MB=$(readlink -f "$H/../..")
: "${RT:=/scratch2/loren/riskybird-dev/backends/chipyard/.conda-env/riscv-tools}"
: "${CC14:=/scratch2/loren/riskybird-dev/backends/zephyr-chipyard-sw/tools-manual/zephyr-sdk-1.0.0-beta1/gnu/riscv64-zephyr-elf/bin/riscv64-zephyr-elf-gcc}"
: "${PYTHON:=python3}"
mkdir -p "$O"
$PYTHON "$H/gen_replay.py" "$GEN" "$BK" "$GOLD" "$O/replay.c" "$@"
D=$GEN/$BK
case $BK in
  rvv) KF="-DMODELBLASTER_RVV_IHWOC_WEIGHTS=1" ;;
  gemmini_q31_rvv) KF="-DMODELBLASTER_GEMMINI_HWIO_WEIGHTS=1 -DMODELBLASTER_GEMMINI_Q31_ACC_SCALE=1 -DGEMMINI_ROCC -DBAREMETAL -isystem $MB/cores/gemmini/include/per_config/q31ws_32x32_acc -isystem $MB/cores/gemmini/include -isystem $MB/cores/gemmini -isystem $MB/kernels/gemmini_q31_rvv" ;;
  *) KF="" ;;
esac
F="-march=rv64gcv -mabi=lp64d -mcmodel=medany -O2 -ffp-contract=off -isystem $RT/riscv64-unknown-elf/include -I$H/stub -I$D $KF ${CFLAGS_EXTRA:-}"
F="$F -isystem $MB/kernels/riskybird"
for s in model kernels weights buffers; do $CC14 $F -c $D/$s.c -o $O/$s.o; done
for x in ${EXTRA_SRCS:-}; do $CC14 $F -c $x -o $O/extra_$(basename $x .c).o; done
$CC14 $F -c $O/replay.c -o $O/replay.o
$RT/bin/riscv64-unknown-elf-gcc -march=rv64gcv -mabi=lp64d -mcmodel=medany -specs=htif_nano.specs -static \
  -T /scratch2/loren/riskybird-dev/backends/chipyard/tests/htif.ld $O/*.o -lm -o $O/replay.riscv
$RT/bin/spike --isa=rv64gcv_zicntr_zvl128b ${SPIKE_EXTRA:-} $O/replay.riscv
