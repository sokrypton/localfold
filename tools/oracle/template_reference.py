#!/usr/bin/env python3
"""AF2-multimer's template embedder, in numpy.

🔴 ITS PAIR BLOCKS ARE WRONG. USE tools/oracle/dump_multimer_template.py.
Measured against AF2 itself, captured through hk.intercept_methods: this file
reproduces the module to relRMS 1.0e-2 with the templates masked and 2.5e-1
with a real template, while src/multimer/template.js reproduces it to 6.5e-5
and 3.0e-4. The disagreement is entirely after `construct_input` - the input
term here agrees with the GPU to 2.15e-7 - and is somewhere in the transcribed
triangle multiplication, triangle attention or transition below. It has never
been found, because for a long time nothing compared this file to anything: it
wrote oracle-dumps/toy-template.json and no reader existed.

What it is still good for is that input term, which is a second and
independently written reading of construct_input including the six geometry
features, and tools/gpu/check-multimer-template.js asserts exactly that much of
it and no more.

The original note follows.

AF2-multimer's template embedder, evaluated for a fully-masked template.

This is the term LocalFold omits. With every template masked off it is far
smaller than the module count suggests: the distogram, the unit vectors and the
backbone mask are all zero, so `construct_input` collapses to

    act = sum(biases) + W_aatype_row[0] + W_aatype_col[0]
        + W_query . LayerNorm(pair)

and the rest is two ordinary pair blocks, a layer norm, a relu and a projection.
Writing it here first gives the GPU implementation an exact target instead of a
plausible one.
"""
import sys, json, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from convert_multimer_params import load_params

T = ("alphafold/alphafold_iteration/evoformer/template_embedding/")
S = T + "single_template_embedding/"
IT = S + "template_embedding_iteration/"

sigmoid = lambda x: 1 / (1 + np.exp(-x))
def norm(p, x, block=None):
    scale = p["scale"] if block is None else p["scale"][block]
    offset = p["offset"] if block is None else p["offset"][block]
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * scale + offset
def linear(p, x, block=None):
    w = p["weights"] if block is None else p["weights"][block]
    b = p["bias"] if block is None else p["bias"][block]
    return x @ w + b

def triangle(P, base, act, mask, block, equation):
    C = P[base + "projection"]["weights"].shape[-1] // 2
    x = norm(P[base + "left_norm_input"], act, block)
    proj = mask[..., None] * linear(P[base + "projection"], x, block)
    proj = proj * sigmoid(linear(P[base + "gate"], x, block))
    out = np.einsum(equation, proj[:, :, :C], proj[:, :, C:])
    out = norm(P[base + "center_norm"], out, block)
    out = linear(P[base + "output_projection"], out, block)
    return out * sigmoid(linear(P[base + "gating_linear"], x, block))

def triangle_attention(P, base, act, mask, block, ending):
    a = P[base + "/attention"]
    x = norm(P[base + "/query_norm"], act, block)
    if ending:
        x = x.transpose(1, 0, 2)
        m = mask.T
    else:
        m = mask
    q = np.einsum("ijc,chd->ijhd", x, a["query_w"][block])
    k = np.einsum("ijc,chd->ijhd", x, a["key_w"][block])
    v = np.einsum("ijc,chd->ijhd", x, a["value_w"][block])
    bias = np.einsum("ijc,ch->ijh", norm(P[base + "/query_norm"], act, block)
                     if False else x, P[base]["feat_2d_weights"][block])
    logits = np.einsum("iqhd,ikhd->ihqk", q, k) / np.sqrt(q.shape[-1])
    logits = logits + 1e9 * (m[:, None, None, :] - 1.0)
    logits = logits + bias.transpose(0, 2, 1)[:, :, None, :]
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    weights = weights / weights.sum(-1, keepdims=True)
    out = np.einsum("ihqk,ikhd->iqhd", weights, v)
    gate = sigmoid(np.einsum("ijc,chd->ijhd", x, a["gating_w"][block]) + a["gating_b"][block])
    out = out * gate
    out = np.einsum("iqhd,hdc->iqc", out, a["output_w"][block]) + a["output_b"][block]
    return out.transpose(1, 0, 2) if ending else out

def transition(P, base, act, block):
    x = norm(P[base + "/input_layer_norm"], act, block)
    x = np.maximum(linear(P[base + "/transition1"], x, block), 0)
    return linear(P[base + "/transition2"], x, block)

def geometry_features(aatype, positions, atom_mask, multichain_mask_2d):
    """The six geometry features, from atom37 coordinates.

    Transcribed from `construct_input` in AF2's modules_multimer.py. The same
    six AF3 computes, in a different atom layout - see the note at the top of
    src/af3/template-features.js, which is the implementation these check.
    """
    L = len(aatype)
    N, CA, C, CB = 0, 1, 2, 3
    positions = positions * atom_mask[..., None]

    # Pseudo-beta: CB, or CA for glycine, which is restype_order 7.
    is_gly = aatype == 7
    beta = np.where(is_gly[:, None], positions[:, CA], positions[:, CB])
    beta_mask = np.where(is_gly, atom_mask[:, CA], atom_mask[:, CB])
    beta_2d = (beta_mask[:, None] * beta_mask[None, :]) * multichain_mask_2d

    # 🔴 (d2 > lower) * (d2 < upper), NOT a clamped bucketisation: a pair closer
    # than the first break is in NO bin. AF2 and AF3 share this and the bins.
    lower = np.linspace(3.25, 50.75, 39) ** 2
    upper = np.concatenate([lower[1:], [1e8]])
    d2 = ((beta[:, None] - beta[None, :]) ** 2).sum(-1)[..., None]
    dgram = ((d2 > lower) * (d2 < upper)).astype(np.float64) * beta_2d[..., None]

    # The frame: from_two_vectors(C - CA, N - CA), translated to CA.
    e1 = positions[:, C] - positions[:, CA]
    e1 = e1 / np.maximum(np.linalg.norm(e1, axis=-1, keepdims=True), 1e-6)
    v2 = positions[:, N] - positions[:, CA]
    e2 = v2 - (v2 * e1).sum(-1, keepdims=True) * e1
    e2 = e2 / np.maximum(np.linalg.norm(e2, axis=-1, keepdims=True), 1e-6)
    e3 = np.cross(e1, e2)
    rotation = np.stack([e1, e2, e3], axis=-1)          # columns are the axes
    frame_mask = atom_mask[:, N] * atom_mask[:, CA] * atom_mask[:, C]
    frame_2d = (frame_mask[:, None] * frame_mask[None, :]) * multichain_mask_2d

    # 🔴 R_i^-1 (t_j - t_i): the FRAME is the row and the POINT is the column.
    # `rigid[:, None].inverse().apply_to_point(points)` puts frames on axis 0.
    delta = positions[None, :, CA] - positions[:, None, CA]
    vector = np.einsum("iac,ija->ijc", rotation, delta)
    vector = vector / np.maximum(
        np.linalg.norm(vector, axis=-1, keepdims=True), 1e-6)
    vector = vector * frame_2d[..., None]
    return dgram, beta_2d, vector, frame_2d


def template_embedding(P, pair, mask, aatype_index=0, num_templates=1,
                       template=None, multichain_mask_2d=None):
    L, C = pair.shape[0], 64
    act = np.zeros((L, L, C))
    if multichain_mask_2d is None:
        multichain_mask_2d = np.ones((L, L))

    if template is None:
        # Every geometry feature is zero, so six of the nine Linears reduce to
        # their biases. That is why the masked module is far smaller than its
        # thirty-five modules suggest.
        for i in (0, 1, 4, 5, 6, 7):
            act = act + P[S + f"template_pair_embedding_{i}"]["bias"]
        aatype = np.full(L, aatype_index)
    else:
        aatype = np.asarray(template["aatype"])
        dgram, beta_2d, vector, frame_2d = geometry_features(
            aatype, np.asarray(template["positions"], np.float64),
            np.asarray(template["atom_mask"], np.float64), multichain_mask_2d)
        for i, feature in ((0, dgram), (1, beta_2d), (4, vector[..., 0]),
                           (5, vector[..., 1]), (6, vector[..., 2]), (7, frame_2d)):
            p = P[S + f"template_pair_embedding_{i}"]
            act = act + (feature @ p["weights"] if feature.ndim == 3
                         else feature[..., None] * p["weights"]) + p["bias"]

    # 🔴 22 CLASSES, IN RESTYPE ORDER. hhsearch writes HHBLITS order and AF2's
    # fix_templates_aatype converts before the model sees it; a template built
    # from a structure is already in restype order.
    onehot = np.zeros((L, 22))
    onehot[np.arange(L), np.clip(aatype, 0, 21)] = 1
    act = act + onehot[None, :, :] @ P[S + "template_pair_embedding_2"]["weights"] \
              + P[S + "template_pair_embedding_2"]["bias"]
    act = act + onehot[:, None, :] @ P[S + "template_pair_embedding_3"]["weights"] \
              + P[S + "template_pair_embedding_3"]["bias"]
    act = act + linear(P[S + "template_pair_embedding_8"],
                       norm(P[S + "query_embedding_norm"], pair))

    stages = [act.copy()]
    for block in range(P[IT + "pair_transition/transition1"]["weights"].shape[0]):
        act = act + triangle(P, IT + "triangle_multiplication_outgoing/", act, mask, block, "ikc,jkc->ijc")
        act = act + triangle(P, IT + "triangle_multiplication_incoming/", act, mask, block, "kjc,kic->ijc")
        act = act + triangle_attention(P, IT + "triangle_attention_starting_node", act, mask, block, False)
        act = act + triangle_attention(P, IT + "triangle_attention_ending_node", act, mask, block, True)
        act = act + transition(P, IT + "pair_transition", act, block)
        stages.append(act.copy())

    act = norm(P[S + "output_layer_norm"], act)
    act = np.maximum(act / num_templates, 0)
    template_embedding.stages = stages
    return linear(P[T + "output_linear"], act)


ONE_LETTER = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
              "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
              "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
              "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
RESTYPES = "ARNDCQEGHILKMFPSTWYV"
# atom37's order, which is what AF2 indexes template_all_atom_positions by.
ATOM37 = ("N CA C CB O CG CG1 CG2 OG OG1 SG CD CD1 CD2 ND1 ND2 OD1 OD2 SD CE"
          " CE1 CE2 CE3 NE NE1 NE2 OE1 OE2 CH2 NH1 NH2 OH CZ CZ2 CZ3 NZ OXT"
          ).split()


def read_template(path, chain, length):
    """A PDB chain as atom37 arrays, over the query's first `length` residues.

    🔴 KEYED ON THE RESIDUE NUMBER, never on position in the atom list: a
    structure missing residues has no lines for them, and grouping by position
    closes the hole up silently.
    """
    order, atoms = [], {}
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        if chain and line[21] != chain:
            continue
        if line[16] not in " A":
            continue
        key = line[22:27]
        if key not in atoms:
            atoms[key] = (line[17:20].strip(), {})
            order.append(key)
        name = line[12:16].strip()
        atoms[key][1].setdefault(name, (float(line[30:38]), float(line[38:46]),
                                        float(line[46:54])))
    order = order[:length]
    aatype = np.full(length, 21, np.int32)          # gap where uncovered
    positions = np.zeros((length, 37, 3))
    mask = np.zeros((length, 37))
    for index, key in enumerate(order):
        name3, found = atoms[key]
        code = ONE_LETTER.get(name3)
        aatype[index] = RESTYPES.index(code) if code in RESTYPES else 20
        for slot, atom in enumerate(ATOM37):
            if atom in found:
                positions[index, slot] = found[atom]
                mask[index, slot] = 1
    return {"aatype": aatype, "positions": positions, "atom_mask": mask,
            "covered": len(order)}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--template", default=None, metavar="PATH[:CHAIN]",
                        help="a PDB to embed as a REAL template. Without it"
                             " every template is masked, which is the case"
                             " LocalFold shipped and the one that cannot"
                             " check the six geometry features at all.")
    parser.add_argument("--chains", type=int, default=1,
                        help="split the query into this many chains, so the"
                             " cross-chain mask is exercised")
    parser.add_argument("--span-chains", action="store_true",
                        help="let one template speak across chains, which AF2"
                             " never does - see src/af3/template-features.js")
    parser.add_argument("--out", default="oracle-dumps/toy-template.json")
    arguments = parser.parse_args()

    P = load_params(pathlib.Path(
        "/Users/mini/Documents/GitHub/af-params/params_model_1_multimer_v3.npz"))
    stage = json.load(open("oracle-dumps/toy-oracle-stages.json"))
    L = 16
    pair = np.array(stage["embedder_pair"], np.float64).reshape(L, L, 128)

    asym = np.array([index * arguments.chains // L for index in range(L)])
    multichain = (asym[:, None] == asym[None, :]).astype(np.float64)

    template = None
    if arguments.template:
        path, _, chain = arguments.template.partition(":")
        template = read_template(path, chain, L)
        print(f"template {path}: {template['covered']} of {L} residues,"
              f" {int(template['atom_mask'].sum())} atoms")
        if arguments.span_chains:
            covered = template["atom_mask"].any(-1).astype(np.float64)
            multichain = np.maximum(multichain, covered[:, None] * covered[None, :])

    out = template_embedding(P, pair, np.ones((L, L)), template=template,
                             multichain_mask_2d=multichain)
    print(f"template term {out.shape}  |x| {np.abs(out).mean():.4f}  max {np.abs(out).max():.3f}")
    payload = {"template_term": out.ravel().tolist(), "length": L,
               "chains": asym.tolist(), "spanChains": arguments.span_chains,
               # 🔴 THE INPUT TERM AND EVERY BLOCK, because a single number at
               # the end says only THAT two implementations disagree. The
               # shipped GPU path and this reference differ by relRMS 1.1e-2
               # and neither had ever been compared to the other, let alone to
               # AF2 - so localising is the whole job.
               "stages": [stage.ravel().tolist()
                          for stage in template_embedding.stages]}
    if template is not None:
        payload["template"] = {
            "aatype": template["aatype"].tolist(),
            "positions": template["positions"].ravel().tolist(),
            "atomMask": template["atom_mask"].ravel().tolist()}
    json.dump(payload, open(arguments.out, "w"))
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
