"""AlphaFold 2's native weights as a MAP onto the page's published bundle (af2-monomer-int5 /
af2-multimer-int5), so a machine loads that bundle as it is - its codes decoded on the device by
native/af3/src/common.cuh's loadBundle - and no float32 file is written.

    ~/.venv-lfjax/bin/python native/af2/make_map.py --model model_1_ptm --bundle model \
        --out native/af2/maps/model_1_ptm.map

The native weights are export_weights.py's: DeepMind's parameters, a monomer's converted onto the
multimer graph (alphafold3/af2/convert.py). The bundle holds the same parameters quantised under the
page's names. So:
  1. each original parameter is found in the decoded bundle - a whole tensor or a contiguous slice of a
     stacked one - by value (relRMS under 8%: int5's own error is ~4%), and must be found uniquely;
  2. every parameter is replaced by the IDS of the bundle elements it came from, and the converter and
     this exporter's own entry loop run on those ids - the converter only reshapes, transposes, slices
     and zero-pads - so each native element names its bundle element (or 0: a pad);
  3. each native tensor is written as strided PARTS of bundle tensors (common.cuh's `p` lines):
       p <name> <length> v <rank> <dims> <dst> <dst strides> <bundle tensor> <offset> <strides>
     one part where the converter only sliced, transposed or padded, more where it concatenated (the
     triangle multiplications' left and right projections, which the bundle keeps apart), nothing for a
     pad (a gathered tensor starts as zeros); the residue tables the structure module reads are the
     bundle's own (geometry*), an integer one as `i`.
A bundle may leave out what the page's AF2 never runs - the monomer's leaves out the templates' single
features (the page folds a monomer's template through the pair term alone, and native/af2 then appends
no template rows) and the two training heads - and only those are passed over where absent.
A parameter found nowhere, or a native tensor that is not one affine view, is an error.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def decode_bundle(bundle_dir, scratch, delta_dir=None):
    """the bundle decoded by the page's own reader: name -> float32 array (flat). With a delta bundle, the
    model it reconstructs, as shared/bundles/delta-tensor-store.js reconstructs it: `addTo` the base rounded
    to float16 plus the delta, `whole` the delta's own, `absent` gone, anything else the base's"""
    script = os.path.join(scratch, "decode_bundle.mjs")
    with open(script, "w") as f:
        f.write(f'''import {{ readFileSync, writeFileSync, openSync, writeSync, closeSync }} from "node:fs";
import {{ readTensor }} from "{REPO}/shared/weights/dtype.js";
const [dir, out, deltaDir] = process.argv.slice(2);
const read = (d) => {{
  const m = JSON.parse(readFileSync(`${{d}}/manifest.json`, "utf8")); const shards = new Map();
  return {{ m, get: (name) => {{
    const r = m.tensors[name];
    if (!shards.has(r.file)) {{ const b = readFileSync(`${{d}}/${{r.file}}`); shards.set(r.file, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }}
    return readTensor(r, shards.get(r.file), r.byteOffset ?? 0, true);
  }} }};
}};
const base = read(dir), delta = deltaDir ? read(deltaDir) : null;
const h = delta ? delta.m.delta : null;
const addTo = new Set(h?.addTo ?? []), whole = new Set(h?.whole ?? []), absent = new Set(h?.absent ?? []);
const idx = {{}}; let off = 0; const fd = openSync(`${{out}}.f32`, "w");
for (const name of Object.keys(base.m.tensors)) {{
  if (absent.has(name)) continue;
  let v;
  if (whole.has(name)) v = delta.get(name);
  else if (addTo.has(name)) {{
    const held = Float16Array.from(base.get(name)), added = delta.get(name);
    v = new Float32Array(held.length); for (let i = 0; i < v.length; i += 1) v[i] = held[i] + added[i];
  }} else v = base.get(name);
  writeSync(fd, Buffer.from(v.buffer, v.byteOffset, v.byteLength)); idx[name] = [off, v.length]; off += v.length;
}}
closeSync(fd); writeFileSync(`${{out}}.json`, JSON.stringify({{ idx, model: h?.model ?? null }}));
''')
    out = os.path.join(scratch, "bundle")
    subprocess.run(["node", "--js-float16array", "--max-old-space-size=24000", script, bundle_dir, out,
                    *([delta_dir] if delta_dir else [])], check=True)
    flat = np.fromfile(out + ".f32", dtype=np.float32)
    meta = json.load(open(out + ".json"))
    decode_bundle.model = meta["model"]
    return {name: flat[o:o + n] for name, (o, n) in meta["idx"].items()}


def rel(a, b):
    d = np.linalg.norm(b)
    return np.linalg.norm(a - b) / d if d > 0 else (0.0 if not np.any(a) else np.inf)


def within_step(t, v, first, group=32):
    """int5's own guarantee: each element within half a quantisation step of its original (the step its
    group's range / 31, the group the bundle tensor's own 32 elements from element 0)"""
    g = (first + np.arange(t.size)) // group
    lo = np.full(g.max() + 1, np.inf); hi = np.full(g.max() + 1, -np.inf)
    np.minimum.at(lo, g, t); np.maximum.at(hi, g, t)
    step = (hi - lo)[g] / 31
    return bool(np.all(np.abs(t - v) <= 0.51 * step + 1e-6 * (np.abs(t) + 1e-30)))


def manifest_names(bundle_dir):
    """bundle tensor -> the haiku paths (module/param, relative) the manifest's sections give it"""
    m = json.load(open(os.path.join(bundle_dir, "manifest.json")))
    names = {}

    def walk(node, path):
        for k, v in node.items():
            if isinstance(v, str):
                names.setdefault(v, []).append(f"{path}/{k}" if path else k)
            elif isinstance(v, dict):
                walk(v, f"{path}/{k}" if path else k)
    for section in m.values():
        if isinstance(section, dict) and isinstance(section.get("parameters"), dict):
            walk(section["parameters"], "")
    return names


STEP_CHECK = [True]


def locate(value, bundle, path="", names=None):
    """the (bundle tensor, first element) a parameter is stored at: the best match, which must be close
    and clearly better than the next"""
    t = value.ravel().astype(np.float32); n = t.size
    # 🔴 THE MANIFEST'S NAME FIRST, WHERE IT GIVES ONE: a bundle tensor whose name is the parameter's own
    # haiku path, of its size, is an identification, so the values only have to confirm it (relRMS under
    # 0.3) - a delta model's small biases sit at 0.15 (an int3 difference over a few dozen values), past
    # what a search by value alone can be trusted at
    if names:
        named = [b for b, ns in names.items() if b in bundle and bundle[b].size == n
                 and any(path.endswith("/" + x) for x in ns)]
        if len(named) == 1 and rel(bundle[named[0]], t) < 0.3:
            return (named[0], 0), rel(bundle[named[0]], t)
    scored = []
    for name, v in bundle.items():
        if v.size % n:
            continue
        rows = v.reshape(-1, n)
        cand = np.argsort(np.abs(rows[:, :min(16, n)] - t[:min(16, n)]).sum(1))[:4] if len(rows) > 4 else range(len(rows))
        for k in cand:
            scored.append((rel(rows[k], t), name, int(k) * n))
    scored.sort()
    if not scored or scored[0][0] > 0.08:
        return None, scored[0][0] if scored else np.inf
    # among the close ones, those int5's step bound admits (a float32 tensor matches exactly)
    # (a delta model's values are the base's f16 plus an int3 difference, which int5's step does not bound:
    # there the relRMS and the names decide)
    ok = [x for x in scored if x[0] < 0.08 and (x[0] == 0 or not STEP_CHECK[0]
                                                or within_step(t, bundle[x[1]][x[2]:x[2] + n], x[2]))]
    if not ok:
        return None, scored[0][0]
    scored = ok
    # two close candidates (a template stack's left and right gates can be nearly equal): the one whose
    # manifest name the parameter's haiku path ends with
    if names and len(scored) > 1 and scored[1][0] < 1.5 * scored[0][0]:
        named = [x for x in scored if any(path.endswith("/" + n) for n in names.get(x[1], []))]
        if named:
            scored = named
    if len(scored) > 1 and scored[1][0] < 0.08 and scored[1][0] < 1.5 * scored[0][0] and np.any(t):
        a = bundle[scored[0][1]][scored[0][2]:scored[0][2] + n]; b = bundle[scored[1][1]][scored[1][2]:scored[1][2] + n]
        if not np.array_equal(a, b):        # (the same values stored twice is no ambiguity)
            sys.exit(f"ambiguous: two different bundle slices match within 1.5x ({scored[0]} / {scored[1]})")
    return scored[0][1:], scored[0][0]


def parts(ids, shape):
    """ids (int64, 0 = pad) as strided parts: [(tensor, src offset, [(dim, src stride, dst stride)], dst)].
    A block that is one affine view of one bundle tensor is one part; otherwise it is split - where its
    source tensor changes along an axis, else in half along its longest - until every piece is."""
    ids = ids.reshape(shape)
    dst_strides = [int(np.prod(shape[k + 1:])) for k in range(len(shape))]
    out = []

    def go(block, origin):
        if not np.any(block):
            return                                  # a pad: the gathered tensor's zeros
        if np.all(block):
            src = block.astype(np.int64) - 1
            t = src.ravel() >> 40
            if np.all(t == t[0]):
                elem = src & ((1 << 40) - 1)
                first = int(elem.flat[0])
                strides = []
                for k, d in enumerate(block.shape):
                    idx = [0] * block.ndim
                    if d > 1:
                        idx[k] = 1
                    strides.append(int(elem[tuple(idx)]) - first if d > 1 else 0)
                grid = np.indices(block.shape).reshape(block.ndim, -1)
                if np.array_equal(first + (np.array(strides)[:, None] * grid).sum(0), elem.ravel()):
                    dst = int(sum(o * s for o, s in zip(origin, dst_strides)))
                    out.append((int(t[0]), first, [(d, st, dst_strides[k]) for k, (d, st) in enumerate(zip(block.shape, strides))], dst))
                    return
        # split: at the first change of source tensor (or of padding) along some axis, else halve the longest
        key = np.where(block > 0, (block - 1) >> 40, -1)
        for k in range(block.ndim):
            if block.shape[k] < 2:
                continue
            first_slice = np.take(key, [0], axis=k)
            change = np.nonzero(np.any(key != first_slice, axis=tuple(a for a in range(block.ndim) if a != k)))[0]
            if change.size:
                cut = int(change[0])
                break
        else:
            k = int(np.argmax(block.shape))
            if block.shape[k] < 2:
                sys.exit("a single element is not a view - impossible")
            cut = block.shape[k] // 2
        lo = [slice(None)] * block.ndim; hi = [slice(None)] * block.ndim
        lo[k] = slice(0, cut); hi[k] = slice(cut, None)
        go(block[tuple(lo)], origin)
        o2 = list(origin); o2[k] += cut
        go(block[tuple(hi)], o2)

    go(ids, [0] * len(shape))
    return out


def best_parts(ids):
    """parts(), and where that takes more than a few, the same with the last axis factored into two or
    three (a permutation inside one axis - IPA's point projections interleave xyz, heads and points -
    is affine only over its factors); the fewest parts win"""
    shape = list(ids.shape)
    best = parts(ids, shape)
    if len(best) <= 4 or not shape:
        return best
    n = shape[-1]
    divs = [d for d in range(2, n) if n % d == 0]
    cands = [[a, n // a] for a in divs] + [[a, b, n // (a * b)] for a in divs for b in divs if (n // a) % b == 0 and n // (a * b) > 1]
    for f in cands:
        got = parts(ids, shape[:-1] + f)
        if len(got) < len(best):
            best = got
    return best


def multimer_ids(args, bundle, names):
    """The multimer, exactly: every checkpoint element gets an id, and the ids go through BOTH sides'
    own code - tools/export_multimer_model.py's (convert_multimer_params, then its section layout: the
    page's bundle) and the reference loader's (flat_params_to_haiku fused, fit_to_multimer_graph: the
    native graph). Both only slice, concatenate, reshape and pad, so each native element names the
    checkpoint element it came from, and so the bundle element that holds it. The decoded bundle is then
    held to the checkpoint's values through that join (int5's error, ~4%), which a wrong join fails."""
    sys.path.insert(0, os.path.join(REPO, "tools"))
    from convert_multimer_params import load_params as raw_load, convert_multimer_params
    from export_multimer_model import SECTION_SCOPES, CONFIDENCE, A
    from alphafold3.af2.model.utils import flat_params_to_haiku
    from alphafold3.af2.runner import fit_to_multimer_graph
    raw = raw_load(os.path.join(args.params, f"params_{args.model}.npz"))
    counter, raw_ids, values = 1, {}, {}
    for module in sorted(raw):
        for leaf in sorted(raw[module]):
            v = np.asarray(raw[module][leaf]); n = v.size
            raw_ids.setdefault(module, {})[leaf] = np.arange(counter, counter + n, dtype=np.int64).reshape(v.shape)
            values[counter] = v.astype(np.float32).ravel(); counter += n
    flat_values = np.zeros(counter, np.float32)
    for start, v in values.items():
        flat_values[start:start + v.size] = v
    # the page's side: checkpoint id -> bundle element
    page = dict(raw_ids); page.update(convert_multimer_params(raw_ids))
    m = json.load(open(os.path.join(args.bundle, "manifest.json")))
    placed = []                                          # (bundle tensor, checkpoint ids)
    for section, scope in SECTION_SCOPES.items():
        for module, leaves in m[section]["parameters"].items():
            for leaf, bname in leaves.items():
                placed.append((bname, page[scope + module][leaf]))
    for head, modules in CONFIDENCE.items():
        for leaf_name, path in modules.items():
            for leaf, bname in m["confidenceHeads"]["parameters"][head][leaf_name].items():
                placed.append((bname, page[path][leaf]))
    for module, leaves in m["templateEmbedding"]["parameters"].items():
        for leaf, bname in leaves.items():
            placed.append((bname, page[A + "evoformer/" + module][leaf]))
    inv = np.zeros(counter, np.int64)
    for bname, cid in placed:
        cid = np.asarray(cid).ravel().astype(np.int64)
        if cid.size != bundle[bname].size:
            sys.exit(f"{bname}: the exporter's layout gives {cid.size} elements, the bundle holds {bundle[bname].size}")
        live = cid > 0
        got = flat_values[cid[live]]
        # (int5's error is ~4%; a delta model's small tensors, an int3 difference over a few hundred values,
        # sit at 0.12-0.15 - a wrong join is far past either)
        if live.any() and rel(bundle[bname][live], got) > (0.08 if STEP_CHECK[0] else 0.3):
            sys.exit(f"{bname}: the bundle does not hold the checkpoint where the join says (relRMS {rel(bundle[bname][live], got):.3f})")
        pos = np.nonzero(live)[0]
        fresh = inv[cid[live]] == 0
        inv[cid[live][fresh]] = (names.index(bname) << 40) + pos[fresh] + 1
    # the native side: the reference's own loader, on the ids
    flat = {f"{module}//{leaf}": a for module in raw_ids for leaf, a in raw_ids[module].items()}
    native = fit_to_multimer_graph(flat_params_to_haiku(flat, fuse=True))
    ids, missing = {}, []
    for module in native:
        for leaf, a in native[module].items():
            a = np.asarray(a).astype(np.int64)
            if a.max() >= counter or (a.dtype != np.int64):
                sys.exit(f"{module}/{leaf}: ids did not survive the loader")
            out = inv[a]
            if np.any((a > 0) & (out == 0)):
                missing.append(f"{module}/{leaf} {a.shape}")
            ids.setdefault(module, {})[leaf] = out
    unexpected = [x for x in missing if not any(f"alphafold_iteration/{y}" in x for y in NOT_IN_PAGE + ORACLE_ONLY)]
    if unexpected:
        sys.exit(f"{len(unexpected)} parameters in no tensor of {args.bundle}:\n  " + "\n  ".join(unexpected))
    for x in missing:          # (passed over: the page leaves them out, or native reads them for an oracle only)
        module, leaf = x.split(" ")[0].rsplit("/", 1)
        ids[module][leaf] = np.zeros_like(ids[module][leaf])
    print(f"{sum(len(v) for v in ids.values())} parameters joined exactly, {len(missing)} not in the bundle")
    return ids


# the distogram head: native/af2 reads it only to check against an oracle (which wants DeepMind's float32
# weights in any case); the page's multimer bundle has none
ORACLE_ONLY = ("distogram_head/",)


# what the page's bundles leave out, because the page's AF2 never runs it (see the docstring)
NOT_IN_PAGE = ("evoformer/template_single_embedding/", "evoformer/template_projection/",
               "experimentally_resolved_head/", "masked_msa_head/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_1_ptm")
    ap.add_argument("--params", default=os.path.expanduser("~/lfjax/af2_params"))
    ap.add_argument("--reference", default="/tmp/claude-1000/ref")
    ap.add_argument("--bundle", required=True, help="the bundle's directory (model, model-multimer, ...)")
    ap.add_argument("--delta", default=None, help="a delta bundle on it (model-mono-2-delta, ...): models 2-5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default="/tmp/claude-1000/af2map")
    args = ap.parse_args()
    sys.path[:0] = [os.path.join(args.reference, "dev", "oracles")]
    from alphafold3.af2.runner import load_params
    from alphafold3.af2.convert import convert_monomer_params
    from alphafold3.af2.common import residue_constants as rc
    from alphafold3.af2.model import all_atom

    os.makedirs(args.scratch, exist_ok=True)
    bundle = decode_bundle(args.bundle, args.scratch, args.delta)
    STEP_CHECK[0] = args.delta is None
    names = list(bundle)
    names_of = manifest_names(args.bundle)
    multimer = "multimer" in args.model
    with_templates = not multimer and any(k in args.model for k in ("model_1", "model_2"))
    params = load_params([args.model], args.params, with_templates, multimer)[0]

    # 1-2: every parameter's elements as bundle ids (+1, so 0 is a pad)
    ids = multimer_ids(args, bundle, names) if multimer else {}
    worst = 0.0
    missing = []
    for module in ([] if multimer else params):
        for pname, value in params[module].items():
            value = np.asarray(value)
            optional = any(f"alphafold_iteration/{x}" in f"{module}/" for x in NOT_IN_PAGE + ORACLE_ONLY)
            if optional and locate(value, bundle, f"{module}/{pname}", names_of)[0] is None:
                ids.setdefault(module, {})[pname] = np.zeros(value.shape, np.int64)   # (through the converter, then dropped)
                continue
            # the triangle multiplications' fused projection/gate: the bundle keeps left and right apart
            pieces = [value]
            if module.endswith(("triangle_multiplication_incoming/projection", "triangle_multiplication_incoming/gate",
                                "triangle_multiplication_outgoing/projection", "triangle_multiplication_outgoing/gate")):
                half = value.shape[-1] // 2
                pieces = [value[..., :half], value[..., half:]]
            got = []
            for side, piece in zip(("left_", "right_") if len(pieces) > 1 else ("",), pieces):
                head, tail = module.rsplit("/", 1)
                where, err = locate(piece, bundle, f"{head}/{side}{tail}/{pname}", names_of)
                if where is None:
                    missing.append(f"{module}/{pname} {piece.shape} (best relRMS {err:.3f})")
                    break
                worst = max(worst, err)
                source, first = where
                base = (names.index(source) << 40) + first + 1
                got.append((base + np.arange(piece.size, dtype=np.int64)).reshape(piece.shape))
            if len(got) == len(pieces):
                ids.setdefault(module, {})[pname] = np.concatenate(got, -1) if len(got) > 1 else got[0]
    if missing:
        sys.exit(f"{len(missing)} parameters in no tensor of {args.bundle}:\n  " + "\n  ".join(missing))
    if not multimer:
        print(f"{sum(len(v) for v in ids.values())} parameters located, worst relRMS {worst:.4f}")
    if not multimer:
        ids = convert_monomer_params(ids)

    # 3: export_weights.py's own entry loop, on ids
    lines = [f"D {decode_bundle.model}"] if args.delta else []     # (loaded only with that delta: common.cuh)
    prefix = "alphafold/alphafold_iteration/"
    for module in sorted(ids):
        for pname, value in sorted(ids[module].items()):
            value = np.asarray(value)
            key = "w/" + module.removeprefix(prefix) + "/" + pname
            if not np.any(value):
                if any(f"alphafold_iteration/{x}" in f"{module}/" for x in NOT_IN_PAGE + ORACLE_ONLY):
                    continue             # (a module this bundle leaves out: native/af2 then does without it)
                # zeros a loader makes (fit_to_multimer_graph's IPA scalar biases): a missing element was
                # refused above, so an all-pad tensor here is all zeros
                lines.append(f"z {key} {value.size}")
                lines.append(f"m {key}#r {value.ndim}")
                lines += [f"m {key}#{k} {d}" for k, d in enumerate(value.shape)]
                continue
            for t, offset, dims, dst in best_parts(value):
                lines.append(f"p {key} {value.size} v {len(dims)} " + " ".join(str(d) for d, _, _ in dims) + f" {dst} "
                             + " ".join(str(ds) for _, _, ds in dims) + f" {names[t]} {offset} " + " ".join(str(ss) for _, ss, _ in dims))
            lines.append(f"m {key}#r {value.ndim}")
            for k, d in enumerate(value.shape):
                lines.append(f"m {key}#{k} {d}")
    # the residue tables: the bundle's own (exactly - they are float32 there), an integer one as `i`
    tables = {
        "rigid_group_default_frame": (rc.restype_rigid_group_default_frame, "v"),
        "atom14_to_rigid_group": (rc.restype_atom14_to_rigid_group, "i"),
        "atom14_rigid_group_positions": (rc.restype_atom14_rigid_group_positions, "v"),
        "atom14_mask": (rc.restype_atom14_mask, "v"),
        "atom37_to_atom14": (all_atom.RESTYPE_ATOM37_TO_ATOM14, "i"),
        "atom37_mask": (all_atom.RESTYPE_ATOM37_MASK, "v"),
    }
    for tname, (value, op) in tables.items():
        value = np.asarray(value, np.float32).ravel()
        hits = [n for n, v in bundle.items() if n.startswith("geometry") and v.size == value.size and np.array_equal(v, value)]
        if len(hits) != 1:
            sys.exit(f"c/{tname}: {len(hits)} bundle tensors hold it exactly")
        lines.append(f"p c/{tname} {value.size} {op} 1 {value.size} 0 1 {hits[0]} 0 1")
    lines += [f"m meta/multimer {int(multimer)}", f"m meta/position_scale {20.0 if multimer else 10.0}",
              f"m meta/opm_first {int(multimer)}"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{args.out}: {len({l.split()[1] for l in lines if l[0] == 'p'})} tensors in {sum(1 for l in lines if l[0] == 'p')} parts")


if __name__ == "__main__":
    main()
