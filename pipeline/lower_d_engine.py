#!/usr/bin/env python3
"""Lower a whole int8 DroNet graph onto RiskyBird engine D (the "riskybird_d" target).

  lower_d_engine.py GRAPH.json WEIGHTS.npz --out-dir DIR [--shape 16,16,4]
                    [--banks 1536,1536,512,512] [--wrows N] [--numerics gemmini|reference]

Engine D (riskybird soc/phase2/d_engine/SPEC.md, contract frozen 2026-11-07) runs the
network from the image through relu_modules_6 and the two linears as a descriptor
program over weights it holds in its own memory. This pass GENERATES that program and
the packed weight blob from ModelBlaster's graph.json / weights.npz, with the D team's
reference packer imported verbatim as pipeline/riskybird_d/pack_d.py (selfcheck() and
pack(); the importer contract asks for both), and rewrites the graph for the rest of the
pipeline:

  DIR/d_blob.bin        the blob (d_load_blob / d_model.c input; the D testbench's input)
  DIR/d_program.json    the packer's listing (descriptors, tensors, banks, offsets)
  DIR/graph.json        one op, d_engine_s8: input -> the model outputs (linear1, sigmoid1);
                        its weight "d_engine.blob" is the blob
  DIR/weights.npz       {"d_engine.blob": blob as int8}
  DIR/io.npz            copied from the input graph's directory when present
  DIR/d_engine.json     provenance: shape, numerics, sha256 of blob and packer

generate_skeleton / generate_kernels then build it like any other graph (backend
scalar, kernel kernels/scalar/scalar_d_engine_s8_riskybird_d.c via
--pin-algorithm d_engine_s8=riskybird_d). The sigmoid runs on the host from the blob's
table, inside the same kernel call, exactly as d_driver.h's d_run_frame() does.

The input graph is ModelBlaster's extract_graph output (with or without
assign_layouts); the packer resolves view/relayout aliases itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np

from modelblaster.pipeline.riskybird_d import pack_d


def lower(graph: dict, weights, shape: "pack_d.Shape", numerics: str = "gemmini"):
    pack_d.selfcheck(graph, weights)
    blob, listing = pack_d.pack(graph, weights, shape, numerics)
    outs = graph["output"].get("tensors") or [graph["output"]["tensor"]]
    names = {o.get("name") for o in graph["ops"]}
    if not {"linear1", "linear2", "sigmoid1"} <= names or list(outs) != ["linear1", "sigmoid1"]:
        sys.exit("[lower_d_engine] the D head contract is linear1 + sigmoid(linear2) as the model "
                 f"outputs; this graph's outputs are {outs}")
    inp = graph["input"].get("tensor") or graph["input"]["tensors"][0]
    keep = {inp, *outs}
    tensors = {t: v for t, v in graph["tensors"].items() if t in keep}
    op = {"op": "d_engine_s8", "name": "d_engine", "inputs": [inp], "outputs": list(outs),
          "weight": "d_engine.blob", "dispatch_id": 0, "depends_on": [],
          "hardware_target": "any",
          "shape": {"blob_bytes": len(blob), "descriptors": listing["descriptors"],
                    "p_oc": shape.p_oc, "p_ic": shape.p_ic, "n_q": shape.n_q}}
    g = {k: v for k, v in graph.items() if k not in ("ops", "tensors", "dispatches", "id_remap",
                                                    "_layout")}
    g["tensors"] = tensors
    g["ops"] = [op]
    g["dispatches"] = [0]
    g.setdefault("_rewrite", []).append({"pass": "lower_d_engine", "numerics": numerics,
                                         "shape": [shape.p_oc, shape.p_ic, shape.n_q]})
    return g, blob, listing


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("graph")
    ap.add_argument("weights")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shape", default="16,16,4")
    ap.add_argument("--banks", default="1536,1536,512,512")
    ap.add_argument("--wrows", type=int, default=None)
    ap.add_argument("--numerics", default="gemmini", choices=("gemmini", "reference"))
    a = ap.parse_args()
    graph = json.load(open(a.graph))
    weights = dict(np.load(a.weights))
    p_oc, p_ic, n_q = (int(v) for v in a.shape.split(","))
    wrows = a.wrows or (1536 if p_oc * p_ic >= 256 else 3072)
    shape = pack_d.Shape(p_oc, p_ic, n_q, tuple(int(v) for v in a.banks.split(",")), wrows)
    g, blob, listing = lower(graph, weights, shape, a.numerics)
    os.makedirs(a.out_dir, exist_ok=True)
    open(os.path.join(a.out_dir, "d_blob.bin"), "wb").write(blob)
    json.dump(listing, open(os.path.join(a.out_dir, "d_program.json"), "w"), indent=1)
    json.dump(g, open(os.path.join(a.out_dir, "graph.json"), "w"), indent=1)
    np.savez(os.path.join(a.out_dir, "weights.npz"),
             **{"d_engine.blob": np.frombuffer(blob, np.uint8).view(np.int8)})
    io = os.path.join(os.path.dirname(os.path.abspath(a.graph)), "io.npz")
    if os.path.exists(io) and os.path.abspath(os.path.dirname(io)) != os.path.abspath(a.out_dir):
        shutil.copy(io, os.path.join(a.out_dir, "io.npz"))
    prov = {"blob_bytes": len(blob), "blob_sha256": hashlib.sha256(blob).hexdigest(),
            "packer_sha256": hashlib.sha256(open(pack_d.__file__, "rb").read()).hexdigest(),
            "shape": [p_oc, p_ic, n_q], "banks": list(shape.bank_words), "wrows": wrows,
            "numerics": a.numerics, "descriptors": listing["descriptors"],
            "source_graph": os.path.abspath(a.graph)}
    json.dump(prov, open(os.path.join(a.out_dir, "d_engine.json"), "w"), indent=1)
    print(f"[lower_d_engine] d_blob.bin {len(blob)} B sha256 {prov['blob_sha256'][:16]}..., "
          f"{listing['descriptors']} descriptors, shape {p_oc}x{p_ic}x{n_q} -> {a.out_dir}")


if __name__ == "__main__":
    main()
