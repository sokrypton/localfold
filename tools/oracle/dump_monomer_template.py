"""Capture AF2-MONOMER's template embedder by running AF2's own module.

    python3 tools/oracle/dump_monomer_template.py --template 1qys-crystal.pdb:A \
      --out toy-template-monomer-jax.json

🔴 COLABDESIGN2 CANNOT DO THIS ONE AND SAYS SO. Its AF2Runner puts the monomer
on the multimer graph and raises: "the monomer and multimer template embedders
differ and monomer template weights cannot be converted". That is true - see
below - so the harness the rest of this directory uses is no help here.

🔴 SO THE MODULE IS RUN ON ITS OWN, NOT THE MODEL. `TemplateEmbedding` is an
ordinary haiku module: transformed, given the slice of the checkpoint whose
names it owns, and called with a query pair and a template batch. That needs no
MSA, no featuriser, no structure module and no 200 MB of unrelated parameters,
and it is a stricter check than a whole-model capture because nothing upstream
can absorb an error.

WHAT MAKES THE MONOMER DIFFERENT, all of it visible in modules.py:

  * ONE Linear over a CONCATENATION - `embedding2d` on 88 channels - where
    multimer and AF3 sum nine separate projections. Same arithmetic, one
    weight.
  * the concatenation is masked by the BACKBONE mask (N, CA and C), the whole
    88 of it, where the other two mask the distogram by pseudo-beta and the
    unit vectors by backbone, separately.
  * its distogram is NOT pseudo-beta-masked at all; that mask appears only as
    its own feature column.
  * `use_template_unit_vector` is False in the monomer config, so three of the
    six geometry features are deliberately ZEROED.
  * the query pair is not a feature. It enters afterwards through
    TemplatePointwiseAttention, which the other two do not have.
"""
import argparse
import json
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
ALPHAFOLD = os.path.expanduser("~/Documents/GitHub/alphafold")
sys.path.insert(0, ALPHAFOLD)

import numpy as np                                             # noqa: E402
import jax                                                     # noqa: E402
import haiku as hk                                             # noqa: E402

from alphafold.model import config as af_config                # noqa: E402
from alphafold.model import modules as af_modules              # noqa: E402
from alphafold.common import residue_constants as rc           # noqa: E402

PARAMS = os.path.expanduser(
    "~/Documents/GitHub/af-params/oracle/params/params_model_1_ptm.npz")
PAIR_CHANNELS = 128


def read_template(path, chain, length):
    """A PDB chain as atom37 arrays, keyed on residue NUMBER not position."""
    order, atoms = [], {}
    for line in open(os.path.expanduser(path)):
        if not line.startswith("ATOM") or (chain and line[21] != chain):
            continue
        if line[16] not in " A":
            continue
        key = line[22:27]
        if key not in atoms:
            atoms[key] = (line[17:20].strip(), {})
            order.append(key)
        atoms[key][1].setdefault(line[12:16].strip(), (
            float(line[30:38]), float(line[38:46]), float(line[46:54])))
    order = order[:length]
    aatype = np.full(length, rc.restype_num, np.int32)          # unknown
    positions = np.zeros((length, 37, 3), np.float32)
    mask = np.zeros((length, 37), np.float32)
    for index, key in enumerate(order):
        name3, found = atoms[key]
        aatype[index] = rc.restype_order.get(
            rc.restype_3to1.get(name3, "X"), rc.restype_num)
        for slot, atom in enumerate(rc.atom_types):
            if atom in found:
                positions[index, slot] = found[atom]
                mask[index, slot] = 1
    return aatype, positions, mask, len(order)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--template", default=None, metavar="PATH[:CHAIN]")
    parser.add_argument("--masked-template", action="store_true",
                        help="a template that is PRESENT but carries no atoms."
                             " 🔴 THIS IS NOT THE SAME AS NO TEMPLATE. AF2's"
                             " TemplateEmbedding ends with"
                             " `embedding *= (sum(template_mask) > 0)`, so with"
                             " template_mask zero the whole term is EXACTLY"
                             " zero - while a present-but-empty template gives"
                             " embedding2d's bias through two pair blocks and a"
                             " projection, which is not small. Which of the two"
                             " a fold means is a question about the caller.")
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="toy-template-monomer-jax.json")
    arguments = parser.parse_args()

    length = arguments.length
    config = af_config.model_config("model_1_ptm")
    template_config = config.model.embeddings_and_evoformer.template
    global_config = config.model.global_config
    print(f"use_template_unit_vector = {template_config.use_template_unit_vector}")
    # 🔴 THE CHECKPOINT PREDATES THE FUSED TRIANGLE PROJECTION AND THE CONFIG
    # DOES NOT. This alphafold checkout sets `fuse_projection_weights: True`
    # everywhere - it is a multimer-v3-era fork - while model_1_ptm's weights
    # are the older `layer_norm_input` / `left_projection` / `right_projection`
    # / `center_layer_norm` names. Left alone, haiku asks for `left_norm_input`
    # and reports every parameter missing, which reads as a checkpoint with no
    # template weights rather than as two spellings of one module.
    for side in ("triangle_multiplication_outgoing", "triangle_multiplication_incoming"):
        template_config.template_pair_stack[side].fuse_projection_weights = False

    rng = np.random.default_rng(arguments.seed)
    query = rng.normal(size=(length, length, PAIR_CHANNELS)).astype(np.float32)
    # Symmetric, because a pair representation is - and an asymmetry here would
    # hide one in the module.
    query = ((query + query.transpose(1, 0, 2)) / 2).astype(np.float32)
    mask_2d = np.ones((length, length), np.float32)

    covered = 0
    if arguments.template:
        path, _, chain = arguments.template.partition(":")
        aatype, positions, atom_mask, covered = read_template(path, chain, length)
        template_mask = np.ones((1,), np.float32)
        print(f"template {path}: {covered} of {length} residues,"
              f" {int(atom_mask.sum())} atoms")
    else:
        aatype = np.zeros(length, np.int32)
        positions = np.zeros((length, 37, 3), np.float32)
        atom_mask = np.zeros((length, 37), np.float32)
        template_mask = np.ones((1,), np.float32) if arguments.masked_template \
            else np.zeros((1,), np.float32)

    # 🔴 pseudo_beta AND ITS MASK COME FROM THE DATA PIPELINE, not the module -
    # the monomer's SingleTemplateEmbedding reads `template_pseudo_beta`
    # straight out of the batch. CB for everything but glycine, which takes CA.
    is_glycine = aatype == rc.restype_order["G"]
    beta_index = np.where(is_glycine, rc.atom_order["CA"], rc.atom_order["CB"])
    pseudo_beta = positions[np.arange(length), beta_index]
    pseudo_beta_mask = atom_mask[np.arange(length), beta_index]

    batch = {
        "template_aatype": aatype[None],
        "template_all_atom_positions": positions[None],
        "template_all_atom_masks": atom_mask[None],
        "template_pseudo_beta": pseudo_beta[None],
        "template_pseudo_beta_mask": pseudo_beta_mask[None],
        "template_mask": template_mask,
    }

    def forward(query_embedding, template_batch, mask):
        return af_modules.TemplateEmbedding(template_config, global_config)(
            query_embedding, template_batch, mask, is_training=False)

    transformed = hk.transform(forward)
    raw = np.load(PARAMS, allow_pickle=True)
    # 🔴 ONLY THE MODULE'S OWN PARAMETERS, RENAMED TO ITS OWN ROOT. The
    # checkpoint scopes everything under alphafold/alphafold_iteration/
    # evoformer/, and a transformed module is its own root - so the prefix has
    # to come off or haiku reports every weight missing.
    prefix = "alphafold/alphafold_iteration/evoformer/template_embedding"
    params = {}
    for key in raw.files:
        module, _, leaf = key.rpartition("//")
        if not module.startswith(prefix):
            continue
        renamed = "template_embedding" + module[len(prefix):]
        params.setdefault(renamed, {})[leaf] = np.asarray(raw[key])
    print(f"{len(params)} parameter modules under {prefix}")

    out = np.asarray(transformed.apply(
        params, jax.random.PRNGKey(0), query, batch, mask_2d), np.float32)
    print(f"template term {out.shape}  |x| {np.abs(out).mean():.4f}"
          f"  std {out.std():.4f}  max {np.abs(out).max():.3f}")

    json.dump({
        "length": length, "covered": covered,
        "useTemplateUnitVector": bool(template_config.use_template_unit_vector),
        "templatePresent": bool(template_mask[0] > 0),
        "pair": query.ravel().tolist(),
        "pairMask": mask_2d.ravel().tolist(),
        "template_term": out.ravel().tolist(),
        "template": None if not arguments.template else {
            "aatype": aatype.tolist(),
            "positions": positions.ravel().tolist(),
            "atomMask": atom_mask.ravel().tolist()},
    }, open(arguments.out, "w"))
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
