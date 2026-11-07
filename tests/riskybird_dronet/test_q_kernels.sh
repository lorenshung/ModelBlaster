#!/bin/bash
# Spike unit test: curated rvv kernels for the folded ops vs their KernelSpec reference_impl,
# over random constants (both clamp signs, NCHW and NHWC strides, C not a multiple of VLMAX).
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); MB=$(cd "$H/../.." && pwd)  # logical path: the package dir must be named modelblaster
O=${1:-/tmp/rb_q_kernels}; mkdir -p $O
: "${RT:=/scratch2/loren/riskybird-dev/backends/chipyard/.conda-env/riscv-tools}"
: "${CC14:=/scratch2/loren/riskybird-dev/backends/zephyr-chipyard-sw/tools-manual/zephyr-sdk-1.0.0-beta1/gnu/riscv64-zephyr-elf/bin/riscv64-zephyr-elf-gcc}"
: "${PYTHON:=python3}"
PYTHONPATH=$(dirname "$MB") $PYTHON - "$O" <<'PY'
import sys
from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
o = sys.argv[1]
with open(f"{o}/ref.c", "w") as f:
    f.write("#include <stdint.h>\n#include <stddef.h>\n")
    for op in ("batchnorm2d_q_s8", "add_q_s8"):
        f.write(KERNEL_SPECS[op].reference_impl.replace(f"kernel_{op}(", f"ref_{op}("))
PY
cat > $O/main.c <<'C'
#include <stdint.h>
#include <stdio.h>
#include <string.h>
void kernel_batchnorm2d_q_s8(const int8_t *, const int32_t *, const int32_t *, const int16_t *, int8_t *, int, int, int, int, int);
void ref_batchnorm2d_q_s8(const int8_t *, const int32_t *, const int32_t *, const int16_t *, int8_t *, int, int, int, int, int);
void kernel_add_q_s8(const int8_t *, const int8_t *, int8_t *, int, int, int, int, int, int);
void ref_add_q_s8(const int8_t *, const int8_t *, int8_t *, int, int, int, int, int, int);
static uint64_t st = 88172645463325252ull;
static uint32_t rnd(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (uint32_t)st; }
static int8_t in[32768], a8[32768], b8[32768], o1[32768], o2[32768];
static int32_t M[256], BP[256]; static int16_t S[256];
int main(void) {
    int fails = 0, tests = 0;
    static const int shapes[][3] = {{1,32,196},{729,32,1},{49,64,1},{7,40,1},{3,5,17},{60,128,1},{1,7,1}};
    for (int t = 0; t < 7; t++) for (int r = 0; r < 4; r++) {
        int outer = shapes[t][0], C = shapes[t][1], inner = shapes[t][2], n = outer * C * inner;
        for (int c = 0; c < C; c++) {
            int s = 2 + rnd() % 23;                       /* S in [2, 24] */
            int32_t m = (int32_t)(rnd() % (1u << 22)) - (1 << 21);
            int32_t b = (int32_t)(rnd() % (1u << 24)) - (1 << 23);
            M[c] = m; S[c] = (int16_t)s; BP[c] = b + (1 << (s - 1));
        }
        for (int i = 0; i < n; i++) in[i] = (int8_t)rnd();
        int lo = (r & 1) ? 0 : -128 + (int)(rnd() % 100), hi = 127 - (int)(rnd() % 50);
        memset(o1, 0x11, n); memset(o2, 0x22, n);
        kernel_batchnorm2d_q_s8(in, M, BP, S, o1, outer, C, inner, lo, hi);
        ref_batchnorm2d_q_s8(in, M, BP, S, o2, outer, C, inner, lo, hi);
        tests++; if (memcmp(o1, o2, n)) { fails++; printf("bn FAIL shape %d,%d,%d lo %d\n", outer, C, inner, lo); }
    }
    for (int r = 0; r < 40; r++) {
        int n = 1 + rnd() % 8000, S2 = 1 + rnd() % 24;
        int lim = (int)(((1u << 31) - (1u << (S2 - 1))) / 128u / 2u);
        int ma = (int)(rnd() % (unsigned)lim) * ((rnd() & 1) ? 1 : -1), mb = (int)(rnd() % (unsigned)lim);
        for (int i = 0; i < n; i++) { a8[i] = (int8_t)rnd(); b8[i] = (int8_t)rnd(); }
        int lo = (r & 1) ? 0 : -128, hi = 127 - (r % 3) * 20;
        kernel_add_q_s8(a8, b8, o1, n, ma, mb, S2, lo, hi);
        ref_add_q_s8(a8, b8, o2, n, ma, mb, S2, lo, hi);
        tests++; if (memcmp(o1, o2, n)) { fails++; printf("add FAIL n %d S %d\n", n, S2); }
    }
    printf("Q KERNELS %s: %d/%d cases byte-identical to reference_impl\n", fails ? "FAIL" : "PASS", tests - fails, tests);
    return fails != 0;
}
C
F="-march=rv64gcv -mabi=lp64d -mcmodel=medany -O2 -isystem $RT/riscv64-unknown-elf/include"
$CC14 $F -c $MB/kernels/rvv/rvv_batchnorm2d_q_s8_direct.c -o $O/bn.o
$CC14 $F -c $MB/kernels/rvv/rvv_add_q_s8_direct.c -o $O/add.o
$CC14 $F -c $O/ref.c -o $O/ref.o
$CC14 $F -c $O/main.c -o $O/main.o
$RT/bin/riscv64-unknown-elf-gcc -march=rv64gcv -mabi=lp64d -mcmodel=medany -specs=htif_nano.specs -static \
  -T /scratch2/loren/riskybird-dev/backends/chipyard/tests/htif.ld $O/*.o -o $O/t.riscv
$RT/bin/spike --isa=rv64gcv_zicntr_zvl128b $O/t.riscv
