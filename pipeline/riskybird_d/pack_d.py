#!/usr/bin/env python3
"""Reference generator of engine D's descriptor program and weight blob (SPEC.md sections 5-7).

    pack(graph, weights, shape=Shape(16, 16, 4)) -> (blob: bytes, listing: dict)

A pure function of ModelBlaster's `graph.json` (as a dict) and `weights.npz` (any mapping name -> ndarray).
It lowers the int8 DroNet graph to D's fused layers, folds every quantization constant to integers, allocates
the activation banks, packs the weights and per-channel records into weight-memory rows, and writes the blob
(segment table + data) that the driver copies into D's window once at boot.

Numerics (golden NUMERICS.md, the deployed KU040 board):
  conv2d_pool_s8 (engine A): two-stage requant, bias as is.
  conv2d_s8, numerics='gemmini' (default): Gemmini MODELBLASTER_GEMMINI_Q31_ACC_SCALE fold: M' = (M + 2^(s-1)) >> s,
      shift' = 0, bias' = bias + 1. Falls back to the two-stage form where that kernel does (shift outside [0, 30]).
  conv2d_s8, numerics='reference': ModelBlaster's reference two-stage requant, bias as is.
  linear_s8: two-stage requant (rvv kernel_linear_s8), run on D as a KxK "conv" over the NHWC input.
  batchnorm2d_s8 / add_s8: integer constants derived in IEEE float32 exactly as kernel_bn_nhwc_s8 /
      kernel_add_q_s8 (gen_sw_consts.c) do; relu_s8 = identity affine with clamp [0, 127].
  sigmoid_s8 stays on the host: a 256-entry table is emitted (mb_sigmoid_s8_fx).

CLI:  python3 pack_d.py --graph G --weights W [--shape 16,16,4] [--banks 1536,1536,512,512] [--wrows 1536]
                        [--numerics gemmini|reference] --out DIR
      writes DIR/d_blob.bin and DIR/d_program.json.
Only the standard library and numpy are used.
Importers: also call selfcheck(graph, weights) for the deployed DroNet (it asserts the constants of the board's
sw_consts.h; it is model-specific, so pack() does not call it). main() does.
"""
import argparse, json, math, os, struct, sys
from dataclasses import dataclass, field
import numpy as np

f32 = np.float32

# ----------------------------------------------------------------------------------------------------------
# Format constants (SPEC.md section 6). Changing any of these is a contract change.
# ----------------------------------------------------------------------------------------------------------
BLOB_MAGIC = b'DRNDBLB1'
BLOB_VERSION = 1
HDR_BYTES = 128
SEG_BYTES = 16
SEG_WINDOW, SEG_HOST_SIGMOID_LUT = 0, 1

WIN_WMEM = 0x00000      # weight memory rows (write-only window), up to 512 KiB
WIN_PMEM = 0x80000      # descriptor program, 64-bit words (RW)
WIN_AW = 0x88000        # engine A weights, 36 words (same encoding as conv0_mmio 0xE000)
WIN_AB = 0x88200        # engine A bias, 16 words (same encoding as conv0_mmio 0xE200)
WIN_IMG = 0x90000       # input image, 12,544 bytes (write-only)
WIN_BANK = 0xA0000      # activation bank b at WIN_BANK + b * WIN_BANK_STRIDE (RW while idle)
WIN_BANK_STRIDE = 0x10000
WIN_REGS = 0xF0000      # registers
WIN_DMA_PAGE = 0xF1000  # reserved for the integrator's DMA registers (C's page), not decoded by D
WINDOW_BYTES = 0x100000
PMEM_WORDS = 512
DESC_WORDS = 8

OP_END, OP_CONV0, OP_CONV = 0, 1, 2
F_RES, F_OUT1, F_OUT2 = 1, 2, 4
REC_BYTES = 16          # per-channel record: bias i32 | Mc i24 + Sc u8 | Bc i32 | M i32 (SPEC 6.3)

A_GEOM = dict(IC=1, IH=112, IW=112, OC=32, OH=56, OW=56, KH=3, KW=3, SH=2, SW=2, PH=1, PW=1,
              pool_KH=3, pool_KW=3, pool_SH=2, pool_SW=2, pool_PH=0, pool_PW=0)
A_OUT = (27, 27, 32)


@dataclass(frozen=True)
class Shape:
    p_oc: int = 16          # output-channel lanes (one dot-product chain each)
    p_ic: int = 16          # chain length = activation word bytes = weight banks
    n_q: int = 4            # post-unit lanes
    bank_words: tuple = (1536, 1536, 512, 512)
    wrows: int = 1536

    def check(self):
        assert self.p_oc in (8, 16, 32) and self.p_ic in (8, 16, 32), self
        assert self.p_oc <= self.p_ic, 'an oc group must fit in one activation word'
        assert self.p_oc % self.n_q == 0 and REC_BYTES * self.n_q <= self.p_oc * self.p_ic
        assert len(self.bank_words) == 4
        assert self.wrows * self.p_oc * self.p_ic <= WIN_PMEM, 'weight memory exceeds its window'
        assert all(w * self.p_ic <= WIN_BANK_STRIDE for w in self.bank_words)

    @property
    def rowb(self):
        return self.p_oc * self.p_ic


# ----------------------------------------------------------------------------------------------------------
# Integer constants, IEEE float32 exactly as the deployed kernels compute them
# ----------------------------------------------------------------------------------------------------------
def lrintf(x):
    """C lrintf on a float32 under round-to-nearest-even."""
    return int(np.rint(f32(x)))


def bn_consts(scale, bias, scale_in, scale_out):
    """kernel_bn_nhwc_s8 / gen_sw_consts.c bn(): per channel (M, B, S)."""
    inv_so = f32(1.0) / f32(scale_out)
    out = []
    for sc, bi in zip(np.asarray(scale, dtype=f32), np.asarray(bias, dtype=f32)):
        cs = f32(f32(sc * f32(scale_in)) * inv_so)
        cb = f32(bi * inv_so)
        S = 24
        while S > 1 and (f32(abs(cs) * f32(1 << S)) >= f32(4194304.0) or
                         f32(abs(cb) * f32(1 << S)) >= f32(1073741824.0)):
            S -= 1
        M = lrintf(f32(cs * f32(1 << S)))
        B = lrintf(f32(cb * f32(1 << S)))
        assert 128 * abs(M) + abs(B) + (1 << (S - 1)) < (1 << 31)
        out.append((M, B, S))
    return out


def add_consts(scale_a, scale_b, scale_out):
    """kernel_add_q_s8 / gen_sw_consts.c add(): (ma, mb, S)."""
    ar = f32(f32(scale_a) / f32(scale_out))
    br = f32(f32(scale_b) / f32(scale_out))
    mx = max(abs(ar), abs(br))
    S = 24
    while S > 1 and f32(mx * f32(1 << S)) >= f32(8388608.0):
        S -= 1
    ma, mb = lrintf(f32(ar * f32(1 << S))), lrintf(f32(br * f32(1 << S)))
    assert 128 * (abs(ma) + abs(mb)) + (1 << (S - 1)) < (1 << 31)
    return ma, mb, S


def conv_requant(q, bias, numerics):
    """(M', shift', bias') for the post unit's  ((acc + bias') * M' + 2^30) >> 31, then rounding >> shift'."""
    M, s = int(q['output_multiplier']), int(q['output_shift'])
    assert q.get('input_offset', 0) == 0 and q.get('filter_offset', 0) == 0 and q.get('output_offset', 0) == 0
    if numerics == 'gemmini' and 0 <= s <= 30:
        assert q['activation_min'] in (0, -128) and q['activation_max'] <= 127, \
            'Gemmini honours only act_min 0 (relu) or -128'
        sq = M if s == 0 else (M + (1 << (s - 1))) >> s
        return sq, 0, [int(b) + 1 for b in bias]
    assert 0 <= s <= 31, 'negative (left) shifts are not supported by the post unit'
    return M, s, [int(b) for b in bias]


def sigmoid_lut(scale_in, scale_out):
    """mb_sigmoid_s8_fx (golden NUMERICS.md sigmoid1) for x = -128..127."""
    si = lrintf(f32(f32(scale_in) * f32(65536.0)))
    inv = lrintf(f32(f32(f32(1.0) / f32(scale_out)) * f32(65536.0)))

    def exp_neg_q16(a):
        if a <= 0:
            return 65536
        t = (a * 94548) >> 16
        k, f = t >> 16, t & 0xFFFF
        if k >= 31:
            return 0
        f2 = (f * f) >> 16
        f3 = (f2 * f) >> 16
        p = 65536 - ((45426 * f) >> 16) + ((15743 * f2) >> 16) - ((3638 * f3) >> 16)
        if k > 0:
            p = (p + (1 << (k - 1))) >> k
        return min(max(p, 0), 65536)
    lut = []
    for x in range(-128, 128):
        z = x * si
        e = exp_neg_q16(abs(z))
        sig = (1 << 32) // (65536 + e) if z >= 0 else (e * 65536) // (65536 + e)
        lut.append(min(127, max(-128, (sig * inv + (1 << 31)) >> 32)))
    return lut


# ----------------------------------------------------------------------------------------------------------
# Graph lowering
# ----------------------------------------------------------------------------------------------------------
@dataclass
class Layer:
    op: int
    name: str
    inp: str = None             # tensor read by the array (conv) or None (conv0 reads the image)
    res: str = None             # residual tensor for a fused add
    o1: str = None              # post-add (or plain conv) tensor
    o2: str = None              # post-BN / post-relu tensor
    geom: dict = field(default_factory=dict)
    M: int = 0
    shift: int = 0
    lo1: int = -128
    hi1: int = 127
    ma: int = 0
    mb: int = 0
    Sa: int = 0
    lo2: int = -128
    hi2: int = 127
    lo3: int = -128
    hi3: int = 127
    bias: list = None           # per output channel (post-unit bias)
    bn: list = None             # per output channel (Mc, Bc, Sc); identity when None
    w: np.ndarray = None        # [K, K, IC, OC] int8 (HWIO)
    aw: np.ndarray = None       # conv0 weights [OC, 1, 3, 3]
    ab: np.ndarray = None       # conv0 bias
    o1_parts: list = None
    mch: list = None            # per output channel requant multiplier (record)       # for the head: [(graph tensor, channel)] instead of one tensor
    src_ops: list = None


ALIAS_OPS = ('nchw_to_nhwc_s8', 'nhwc_to_nchw_s8', 'view')


def lower(graph, weights, numerics='gemmini'):
    """Return (layers, tensor shapes {name: (H, W, C)}, host-side ops)."""
    ops = graph['ops']
    alias = {}

    def res_(t):
        while t in alias:
            t = alias[t]
        return t
    for o in ops:
        if o['op'] in ALIAS_OPS:
            alias[o['outputs'][0]] = o['inputs'][0]
    real = [o for o in ops if o['op'] not in ALIAS_OPS]
    producer, consumers = {}, {}
    for o in real:
        for t in o['outputs']:
            producer[res_(t)] = o
        for t in o['inputs']:
            consumers.setdefault(res_(t), []).append(o)
    shapes = {}
    host_ops = []
    layers = []
    fused = set()
    input_name = res_(graph['input']['tensor'])

    def take_bn_or_relu(t):
        """If exactly one consumer of t is a BN or a relu, fuse it: return (op, out tensor)."""
        cs = [c for c in consumers.get(t, []) if c['op'] in ('batchnorm2d_s8', 'relu_s8') and id(c) not in fused]
        if len(cs) == 1:
            return cs[0], res_(cs[0]['outputs'][0])
        return None, None

    def fuse_post(L, t_out):
        """Fuse add -> BN/relu after a conv whose output is t_out. Fills L.res/o1/o2 and constants."""
        cs = consumers.get(t_out, [])
        L.o1 = t_out
        if len(cs) == 1 and cs[0]['op'] == 'add_s8' and id(cs[0]) not in fused:
            a_op = cs[0]
            a, b = (res_(x) for x in a_op['inputs'])
            q = a_op['quant']
            ma, mb, S = add_consts(q['scale_a'], q['scale_b'], q['scale_out'])
            if b == t_out:       # residual = a
                L.res, L.ma, L.mb = a, ma, mb
            else:                # residual = b, swap the multipliers
                assert a == t_out
                L.res, L.ma, L.mb = b, mb, ma
            L.Sa, L.lo2, L.hi2 = S, int(q['activation_min']), int(q['activation_max'])
            L.o1 = res_(a_op['outputs'][0])
            fused.add(id(a_op)); L.src_ops.append(a_op['name'])
        bo, bt = take_bn_or_relu(L.o1)
        if bo is not None:
            fused.add(id(bo)); L.src_ops.append(bo['name'])
            if bo['op'] == 'batchnorm2d_s8':
                q = bo['quant']
                L.bn = bn_consts(weights[bo['weight']], weights[bo['bias']], q['scale_in'], q['scale_out'])
                L.lo3, L.hi3 = int(q['activation_min']), int(q['activation_max'])
            else:
                L.bn = None; L.lo3, L.hi3 = 0, 127
            L.o2 = bt
            # o1 is still written when anything other than the fused BN reads it, or for an add (debug value)
            if len(consumers.get(L.o1, [])) == 1 and L.res is None:
                L.o1 = None
        g = L.geom
        for t in (L.o1, L.o2):
            if t is not None:
                shapes[t] = (g['OH'], g['OW'], g['OC'])

    for o in real:
        if id(o) in fused:
            continue
        k = o['op']
        if k == 'conv2d_pool_s8':
            s = o['shape']
            for key, v in A_GEOM.items():
                assert s[key] == v, f'conv2d_pool_s8 {key}={s[key]} does not match engine A ({v})'
            assert res_(o['inputs'][0]) == input_name
            q = o['quant']
            L = Layer(OP_CONV0, o['name'], src_ops=[o['name']])
            L.M, L.shift = int(q['output_multiplier']), int(q['output_shift'])
            L.lo1, L.hi1 = int(q['activation_min']), int(q['activation_max'])
            L.aw = np.asarray(weights[o['weight']], dtype=np.int8)
            L.ab = np.asarray(weights[o['bias']], dtype=np.int64)
            L.bias = [0] * 32
            L.geom = dict(IC=1, OC=32, OH=27, OW=27, K=3, S=2, P=1, IH=112, IW=112)
            t = res_(o['outputs'][0]); shapes[t] = A_OUT
            fuse_post(L, t)
            layers.append(L)
        elif k == 'conv2d_s8':
            s = o['shape']
            assert s['KH'] == s['KW'] and s['SH'] == s['SW'] and s['PH'] == s['PW']
            L = Layer(OP_CONV, o['name'], inp=res_(o['inputs'][0]), src_ops=[o['name']])
            L.geom = dict(IC=s['IC'], IH=s['IH'], IW=s['IW'], OC=s['OC'], OH=s['OH'], OW=s['OW'],
                          K=s['KH'], S=s['SH'], P=s['PH'])
            q = o['quant']
            L.M, L.shift, L.bias = conv_requant(q, weights[o['bias']], numerics)
            L.mch = [L.M] * s['OC']
            L.lo1, L.hi1 = int(q['activation_min']), int(q['activation_max'])
            w = np.asarray(weights[o['weight']], dtype=np.int8)          # [OC, IC, K, K]
            L.w = w.transpose(2, 3, 1, 0).copy()                         # HWIO
            t = res_(o['outputs'][0]); shapes[t] = (s['OH'], s['OW'], s['OC'])
            fuse_post(L, t)
            layers.append(L)
        elif k == 'linear_s8':
            # All linears that read the same flattened tensor become one head layer.
            src = res_(o['inputs'][0])
            group = [x for x in real if x['op'] == 'linear_s8' and res_(x['inputs'][0]) == src
                     and id(x) not in fused]
            H, W_, C = shapes[src]
            L = Layer(OP_CONV, 'head:' + '+'.join(x['name'] for x in group), inp=src,
                      src_ops=[x['name'] for x in group])
            L.geom = dict(IC=C, IH=H, IW=W_, OC=len(group), OH=1, OW=1, K=H, S=1, P=0)
            assert H == W_
            qs = [x['quant'] for x in group]
            # M is per channel (record); shift and clamp are per layer: they must agree
            assert len({(int(q['output_shift']), q['activation_min'], q['activation_max']) for q in qs}) == 1
            q = qs[0]
            L.mch = [int(x['output_multiplier']) for x in qs]
            L.M, L.shift = L.mch[0], int(q['output_shift'])
            assert 0 <= L.shift <= 31
            L.lo1, L.hi1 = int(q['activation_min']), int(q['activation_max'])
            L.bias = [int(np.asarray(weights[x['bias']]).reshape(-1)[0]) for x in group]
            # torch flatten of the NCHW tensor: index c*H*W + h*W + w; D's input is NHWC
            w = np.zeros((H, W_, C, len(group)), dtype=np.int8)
            for n, x in enumerate(group):
                wt = np.asarray(weights[x['weight']], dtype=np.int8).reshape(C, H, W_)
                w[:, :, :, n] = wt.transpose(1, 2, 0)
            L.w = w
            L.o1 = None
            L.o1_parts = [(res_(x['outputs'][0]), n) for n, x in enumerate(group)]
            for x in group:
                fused.add(id(x))
            layers.append(L)
        elif k == 'sigmoid_s8':
            q = o['quant']
            host_ops.append(dict(op='sigmoid_s8', name=o['name'], input=res_(o['inputs'][0]),
                                 output=res_(o['outputs'][0]),
                                 lut=sigmoid_lut(q['scale_in'], q['scale_out'])))
        else:
            raise NotImplementedError(f"op {k} ({o['name']}) has no D lowering")
    # shapes of o1/o2 equal the conv output shape
    for L in layers:
        g = L.geom
        for t in (L.o1, L.o2):
            if t is not None:
                shapes[t] = (g['OH'], g['OW'], g['OC'])
        if L.o1_parts:
            shapes['head'] = (1, 1, g['OC'])
    return layers, shapes, host_ops


def schedule(layers):
    """Topological order (stable by graph order): a layer runs after the producers of its in and res."""
    prod = {}
    for i, L in enumerate(layers):
        for t in (L.o1, L.o2):
            if t:
                prod[t] = i
        if L.o1_parts:
            prod['head'] = i
    deps = []
    for L in layers:
        d = set()
        for t in (L.inp, L.res):
            if t is not None and t in prod:
                d.add(prod[t])
        deps.append(d)
    order, done = [], set()
    while len(order) < len(layers):
        for i in range(len(layers)):
            if i not in done and deps[i] <= done:
                order.append(i); done.add(i); break
        else:
            raise ValueError('cyclic layer dependencies')
    return [layers[i] for i in order]


# ----------------------------------------------------------------------------------------------------------
# Activation bank allocation
# ----------------------------------------------------------------------------------------------------------
def allocate(layers, shapes, shape: Shape, keep=()):
    """First-fit allocation of every tensor D writes, in program order. Returns {tensor: (bank, base word)}.
    Rules (SPEC 4.2): in and res of a layer in different banks; o1 and o2 of a layer in different banks;
    no live overlap. A tensor lives from the layer that writes it to its last reader (or the end, for keep)."""
    cw = lambda C: -(-C // shape.p_ic)
    words = {t: shapes[t][0] * shapes[t][1] * cw(shapes[t][2]) for t in shapes}
    last = {}
    for i, L in enumerate(layers):
        for t in (L.inp, L.res):
            if t is not None:
                last[t] = i
    n = len(layers)
    for t in keep:
        last[t] = n
    pairs = []     # (a, b) that must be in different banks
    for L in layers:
        if L.inp and L.res:
            pairs.append((L.inp, L.res))
        if L.o1 and L.o2:
            pairs.append((L.o1, L.o2))
    alloc, born = {}, {}

    def outs(L):
        r = [t for t in (L.o1, L.o2) if t]
        if L.o1_parts:
            r.append('head')
        return r
    for i, L in enumerate(layers):
        for t in outs(L):
            last.setdefault(t, i)          # written but never read: lives through its own layer
            forbid = set()
            for a, b in pairs:
                other = b if a == t else a if b == t else None
                if other in alloc:
                    forbid.add(alloc[other][0])
            placed = False
            for bank in range(4):
                if bank in forbid:
                    continue
                busy = sorted((alloc[u][1], alloc[u][1] + words[u]) for u in alloc
                              if alloc[u][0] == bank and last[u] >= i and born[u] <= i)
                base = 0
                for s, e in busy:
                    if base + words[t] <= s:
                        break
                    base = max(base, e)
                if base + words[t] <= shape.bank_words[bank]:
                    alloc[t] = (bank, base); born[t] = i; placed = True
                    break
            if not placed:
                raise ValueError(f'cannot place {t} ({words[t]} words) at layer {L.name}')
    return alloc, words


# ----------------------------------------------------------------------------------------------------------
# Packing
# ----------------------------------------------------------------------------------------------------------
def u64(fields):
    v = 0
    for val, lo, width in fields:
        v |= (int(val) & ((1 << width) - 1)) << lo
    return v


def descriptor(L, alloc, wbase, pbase):
    g = L.geom
    flags = 0
    bank = lambda t: alloc[t][0] if t else 0
    base = lambda t: alloc[t][1] if t else 0
    o1 = L.o1 if L.o1 else ('head' if L.o1_parts else None)
    if L.res: flags |= F_RES
    if o1: flags |= F_OUT1
    if L.o2: flags |= F_OUT2
    w0 = u64([(L.op, 0, 8), (flags, 8, 8), (bank(L.inp), 16, 2), (bank(L.res), 18, 2), (bank(o1), 20, 2),
              (bank(L.o2), 22, 2), (g['K'], 24, 4), (g['S'], 28, 2), (g['P'], 30, 2), (g['IC'], 32, 16),
              (g['OC'], 48, 16)])
    w1 = u64([(g['IH'], 0, 8), (g['IW'], 8, 8), (g['OH'], 16, 8), (g['OW'], 24, 8), (wbase, 32, 16),
              (pbase, 48, 16)])
    w2 = u64([(base(L.inp), 0, 16), (base(L.res), 16, 16), (base(o1), 32, 16), (base(L.o2), 48, 16)])
    w3 = u64([(L.M, 0, 32), (L.shift, 32, 8), (L.lo1, 40, 8), (L.hi1, 48, 8), (L.Sa, 56, 8)])
    w4 = u64([(L.ma, 0, 32), (L.mb, 32, 32)])
    w5 = u64([(L.lo2, 0, 8), (L.hi2, 8, 8), (L.lo3, 16, 8), (L.hi3, 24, 8)])
    return [w0, w1, w2, w3, w4, w5, 0, 0]


def pack(graph, weights, shape: Shape = Shape(), numerics='gemmini', keep_relu6=True):
    shape.check()
    layers, shapes, host_ops = lower(graph, weights, numerics)
    layers = schedule(layers)
    keep = ['head'] + ([t for t in shapes if t == 'relu_modules_6'] if keep_relu6 else [])
    alloc, words = allocate(layers, shapes, shape, keep)
    P, Q, R = shape.p_oc, shape.p_ic, shape.rowb

    rows = []          # list of bytearray(R)
    descs = []
    listing = dict(shape=dict(p_oc=P, p_ic=Q, n_q=shape.n_q, bank_words=list(shape.bank_words),
                              wrows=shape.wrows), numerics=numerics, layers=[], host_ops=[])

    def param_rows(L, OC):
        start = len(rows)
        bn = L.bn or [(1, 0, 0)] * OC
        for r0 in range(0, OC, shape.n_q):
            row = bytearray(R)
            for l in range(shape.n_q):
                c = r0 + l
                if c < OC:
                    Mc, Bc, Sc = bn[c]
                    assert -(1 << 23) <= Mc < (1 << 23) and 0 <= Sc <= 31
                    M = L.mch[c] if L.mch else 0
                    assert 0 <= M < (1 << 31)
                    row[REC_BYTES * l: REC_BYTES * (l + 1)] = struct.pack(
                        '<iIiI', L.bias[c], (Mc & 0xFFFFFF) | (Sc << 24), Bc, M)
            rows.append(row)
        return start

    for idx, L in enumerate(layers):
        g = L.geom
        pbase = param_rows(L, g['OC'])
        wbase = len(rows)
        if L.op == OP_CONV:
            K, IC, OC = g['K'], g['IC'], g['OC']
            icc_n, og_n = -(-IC // Q), -(-OC // P)
            for og in range(og_n):
                for kh in range(K):
                    for kw in range(K):
                        for icc in range(icc_n):
                            row = bytearray(R)
                            for j in range(Q):
                                ic = icc * Q + j
                                if ic >= IC:
                                    continue
                                for o in range(P):
                                    oc = og * P + o
                                    if oc < OC:
                                        row[j * P + o] = int(L.w[kh, kw, ic, oc]) & 0xFF
                            rows.append(row)
        descs.append(descriptor(L, alloc, wbase, pbase))
        ent = dict(index=idx, name=L.name, op=['END', 'CONV0', 'CONV'][L.op], src_ops=L.src_ops,
                   geom=g, wbase=wbase, pbase=pbase, M=L.M, shift=L.shift, lo1=L.lo1, hi1=L.hi1,
                   ma=L.ma, mb=L.mb, Sa=L.Sa, lo2=L.lo2, hi2=L.hi2, lo3=L.lo3, hi3=L.hi3,
                   bn_identity=L.bn is None or all(Mc == 1 << Sc and Bc == 0 for Mc, Bc, Sc in L.bn),
                   inp=L.inp, res=L.res, outputs=[])
        for slot, t in (('o1', L.o1), ('o2', L.o2)):
            if t:
                b, base = alloc[t]
                ent['outputs'].append(dict(slot=slot, tensor=t, bank=b, base=base, shape=list(shapes[t]),
                                           window=WIN_BANK + b * WIN_BANK_STRIDE + base * Q))
        if L.o1_parts:
            b, base = alloc['head']
            for t, ch in L.o1_parts:
                ent['outputs'].append(dict(slot='o1', tensor=t, bank=b, base=base, shape=[1, 1, 1],
                                           channel=ch, window=WIN_BANK + b * WIN_BANK_STRIDE + base * Q + ch))
        listing['layers'].append(ent)
    descs.append([OP_END, 0, 0, 0, 0, 0, 0, 0])
    assert len(rows) <= shape.wrows, f'{len(rows)} weight rows exceed WROWS={shape.wrows}'
    assert len(descs) * DESC_WORDS <= PMEM_WORDS

    # engine A weights/bias in conv0_mmio's encoding
    A = [L for L in layers if L.op == OP_CONV0]
    aw_bytes = ab_bytes = b''
    if A:
        L = A[0]
        aw = bytearray(36 * 8)
        for tap in range(9):
            kh, kw = divmod(tap, 3)
            for lane in range(32):
                aw[(tap * 4 + lane // 8) * 8 + lane % 8] = int(L.aw[lane, 0, kh, kw]) & 0xFF
        aw_bytes = bytes(aw)
        ab_bytes = struct.pack('<32i', *[int(b) for b in L.ab])

    wmem = b''.join(bytes(r) for r in rows)
    pmem = b''.join(struct.pack('<Q', w) for d in descs for w in d)
    segs = [(WIN_WMEM, wmem, SEG_WINDOW), (WIN_PMEM, pmem, SEG_WINDOW)]
    if A:
        segs += [(WIN_AW, aw_bytes, SEG_WINDOW), (WIN_AB, ab_bytes, SEG_WINDOW)]
    for h in host_ops:
        if h['op'] == 'sigmoid_s8':
            segs.append((0, bytes(v & 0xFF for v in h['lut']), SEG_HOST_SIGMOID_LUT))
            listing['host_ops'].append({k: v for k, v in h.items() if k != 'lut'})

    relu6 = alloc.get('relu_modules_6')
    head = alloc.get('head')
    hdr = bytearray(HDR_BYTES)
    hdr[0:8] = BLOB_MAGIC
    struct.pack_into('<II', hdr, 8, BLOB_VERSION, HDR_BYTES)
    struct.pack_into('<BBBB', hdr, 16, P, Q, shape.n_q, 4)
    struct.pack_into('<I4H', hdr, 20, shape.wrows, *shape.bank_words)
    struct.pack_into('<III', hdr, 32, len(segs), len(descs), 0)
    struct.pack_into('<II', hdr, 44, WIN_IMG, 112 * 112)
    struct.pack_into('<II', hdr, 52, (WIN_BANK + relu6[0] * WIN_BANK_STRIDE + relu6[1] * Q) if relu6 else 0,
                     words['relu_modules_6'] * Q if relu6 else 0)
    struct.pack_into('<III', hdr, 60, (WIN_BANK + head[0] * WIN_BANK_STRIDE + head[1] * Q) if head else 0,
                     Q if head else 0, shapes['head'][2] if head else 0)
    struct.pack_into('<II', hdr, 72, len(rows), WINDOW_BYTES)
    table = bytearray()
    data = bytearray()
    off = HDR_BYTES + SEG_BYTES * len(segs)
    for win, payload, kind in segs:
        pad = (-len(payload)) % 8
        table += struct.pack('<IIII', win, len(payload) + pad, off + len(data), kind)
        data += payload + b'\0' * pad
    blob = bytes(hdr) + bytes(table) + bytes(data)
    listing.update(wmem_rows=len(rows), descriptors=len(descs), blob_bytes=len(blob),
                   alloc={t: dict(bank=b, base=s, words=words[t]) for t, (b, s) in alloc.items()},
                   segments=[dict(window=w, bytes=len(p), kind=k) for w, p, k in segs],
                   relu6_window=(WIN_BANK + relu6[0] * WIN_BANK_STRIDE + relu6[1] * Q) if relu6 else None,
                   head_window=(WIN_BANK + head[0] * WIN_BANK_STRIDE + head[1] * Q) if head else None)
    return blob, listing


# ----------------------------------------------------------------------------------------------------------
def selfcheck(graph, weights):
    """The deployed constants (generated/rvv/sw_consts.h, golden layers.json) must come out exactly."""
    want_add = {'add': (3160474, 6409318, 23), 'add_1': (5660994, 7122066, 23), 'add_2': (3744491, 6287046, 23)}
    want_bn = {'bn_modules.0': (2097152, 0, 21), 'bn_modules.2': (2097152, 0, 21), 'bn_modules.4': (4194304, 0, 22)}
    for o in graph['ops']:
        if o['name'] in want_add:
            q = o['quant']
            got = add_consts(q['scale_a'], q['scale_b'], q['scale_out'])
            assert got == want_add[o['name']], (o['name'], got)
        if o['name'] in want_bn:
            q = o['quant']
            got = set(bn_consts(weights[o['weight']], weights[o['bias']], q['scale_in'], q['scale_out']))
            assert got == {want_bn[o['name']]}, (o['name'], got)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--graph', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--shape', default='16,16,4')
    ap.add_argument('--banks', default='1536,1536,512,512')
    ap.add_argument('--wrows', type=int, default=None)
    ap.add_argument('--numerics', default='gemmini', choices=('gemmini', 'reference'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    graph = json.load(open(a.graph))
    weights = dict(np.load(a.weights))
    p_oc, p_ic, n_q = (int(v) for v in a.shape.split(','))
    wrows = a.wrows or (1536 if p_oc * p_ic >= 256 else 3072)
    shape = Shape(p_oc, p_ic, n_q, tuple(int(v) for v in a.banks.split(',')), wrows)
    selfcheck(graph, weights)
    blob, listing = pack(graph, weights, shape, a.numerics)
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, 'd_blob.bin'), 'wb').write(blob)
    json.dump(listing, open(os.path.join(a.out, 'd_program.json'), 'w'), indent=1)
    print(f"d_blob.bin {len(blob)} B, {listing['wmem_rows']} weight rows of {shape.rowb} B, "
          f"{listing['descriptors']} descriptors -> {a.out}")
    for L in listing['layers']:
        print(f"  {L['index']:2d} {L['op']:5s} {L['name']:28s} in={L['inp']} res={L['res']} -> "
              + ', '.join(f"{o['slot']}:{o['tensor']}@B{o['bank']}+{o['base']}" for o in L['outputs']))


if __name__ == '__main__':
    main()
