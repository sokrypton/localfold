"""af3-any-model's diffusion head for Chai-1, one denoise step on the REAL trunk, with chai's own pair input.

    cd /tmp/claude-1000/ref && JAX_PLATFORMS=cuda ESM_EMB=<esm2.npy> MODEL_DIR=<chai1 blob dir> \
      PYTHONPATH=.:dev/oracles ~/.venv-lfjax/bin/python <repo>/tools/oracle/dump_chai1_denoise.py

🔴 af3-any-model's chai diffusion conditioned on `[z_trunk | pair_init]` with pair_init = the TRUNK's z_init
(diffusion_head.py, evoformer.py), where chai-lab's is its STRUCTURE token-pair features
(`token_pair_structure_input_feats`, chai1.py): relRMS 9.2 apart on 6MRR, measured on chai-lab's own tensors.
Fixed there since (its c38fec3, model.Chai1StructurePair, 2.4e-7 from this file's structure_pair). This
still hands the reference head chai's structure pair in that slot, so it reads the same on either side of it - built here from the bundle's
`diffuser/chai1_structure_token_pair` weights (tools/export_chai1_structure_pair.py) - and the oracle is chai at
that seam and af3-any-model everywhere else. `Z_STRUCT_OUT=<file.npy>` writes the structure pair too, for the check
against chai-lab's captured diffusion input.

The trunk's outputs (single, pair, the diffusion's target_feat) come from oracle-dumps/af3-oracle-trunk-chai1.json,
so the step sees the conditioning a real fold would. Writes oracle-dumps/af3-oracle-denoise-chai1.json (inputs and
output, at NOISE) and af3-oracle-stages-chai1.json (the head's seams), the formats native/af3/export-model.mjs reads.
"""
import os, sys, json, pathlib
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
import numpy as np
import fold_check
from alphafold3.model import feat_batch
from alphafold3.model.network import atom_cross_attention as ACA
from alphafold3.model.network import diffusion_transformer as DT
from alphafold3.model.network import diffusion_head as DH
from atom_parity import flat_atom_features

REPO = pathlib.Path(__file__).resolve().parents[2]
NOISE = float(os.environ.get("NOISE", "16.0"))
TRACE = {}
def _put(name, a):
    a = np.asarray(a); TRACE[name] = {"shape": list(a.shape), "data": a.astype(np.float32).ravel().tolist()}
_enc, _dec, _tx, _cond = (ACA.atom_cross_att_encoder, ACA.atom_cross_att_decoder,
                          DT.Transformer.__call__, DH.DiffusionHead._conditioning)
def enc(*a, **k):
    out = _enc(*a, **k)
    if k.get("conditioning_only"): return out
    _put("encoder.tokenAct", out.token_act); _put("encoder.skipConnection", out.skip_connection); return out
def dec(*a, **k):
    out = _dec(*a, **k); _put("decoder.update", out); return out
def tx(self, *a, **k):
    out = _tx(self, *a, **k); _put("transformer.out", out)
    _put("transformer.act", k.get("act", a[0] if a else None))
    _put("transformer.singleCond", k["single_cond"]); _put("transformer.pairCond", k["pair_cond"]); return out
def cond(self, *a, **k):
    out = _cond(self, *a, **k); single, pair = out
    if single is not None: _put("conditioning.single", single)
    if pair is not None: _put("conditioning.pair", pair)
    return out
ACA.atom_cross_att_encoder = enc; ACA.atom_cross_att_decoder = dec
DT.Transformer.__call__ = tx; DH.DiffusionHead._conditioning = cond
DH.atom_cross_attention.atom_cross_att_encoder = enc
DH.atom_cross_attention.atom_cross_att_decoder = dec


def bundle_tensor(name):
    root = REPO / "model-chai1-f32"
    m = json.loads((root / "manifest.json").read_text())["tensors"][name]
    n = int(np.prod(m["shape"]))
    return np.fromfile(root / m["file"], "<f4", count=n, offset=m["byteOffset"]).reshape(m["shape"])


def structure_pair(tf):
    """chai's 163 token-pair columns (alphabetical generators, chai-lab's own definitions), through the structure
    half: docking 0:6 (class 5, no constraint), relative chain 6:12, relative entity 12:15, residue separation
    15:82, token separation 82:149, the two restraints' masked columns 155 and 162."""
    ri = np.asarray(tf.residue_index).astype(np.int64); ti = np.asarray(tf.token_index).astype(np.int64)
    asym = np.asarray(tf.asym_id).astype(np.int64)
    ent = np.unique(np.asarray(tf.entity_id), return_inverse=True)[1].astype(np.int64)
    sym = np.unique(np.asarray(tf.sym_id), return_inverse=True)[1].astype(np.int64)
    n = ri.shape[0]
    same_chain = asym[:, None] == asym[None]
    rss = np.where(same_chain, np.clip(ri[:, None] - ri[None] + 33, 0, 65), 66)
    rts = np.where(same_chain & (ri[:, None] == ri[None]), np.clip(ti[:, None] - ti[None] + 32, 0, 65), 66)
    rel_e = ent[:, None] - ent[None]
    rchain = np.where(rel_e != 0, 5, np.clip(sym[:, None] - sym[None] + 2, 0, 4))
    rent = np.clip(rel_e + 1, 0, 2)
    W = bundle_tensor("diffuser/chai1_structure_token_pair/weights").astype(np.float64)   # [163, 256]
    b = bundle_tensor("diffuser/chai1_structure_token_pair/bias").astype(np.float64)
    z = (b + W[5] + W[155] + W[162])[None, None] + W[6 + rchain] + W[12 + rent] + W[15 + rss] + W[82 + rts]
    return z.astype(np.float32)


def main():
    import haiku as hk, jax, jax.numpy as jnp
    from alphafold3.model import params as afp
    # SEQ= / SEP=<position> / USER_CCD=<cif>: another job (a phosphoserine, to reach the atomised-residue
    # paths 6MRR cannot), its trunk conditioning RANDOM (SYNTH=1) since no trunk dump exists for it - both
    # sides read the same inputs, so exactness is what this measures, not geometry; SUFFIX= names the outputs
    seq = os.environ.get("SEQ") or fold_check.parse_ca(os.path.expanduser("~/6MRR.pdb"))[0]
    chains = None
    if os.environ.get("SEP"):
        from alphafold3.common import folding_input
        chains = [folding_input.ProteinChain(id="A", sequence=seq, ptms=[("SEP", int(os.environ["SEP"]))],
                                             unpaired_msa="", paired_msa="", templates=[])]
    if os.environ.get("USER_CCD"):
        import functools
        from alphafold3.constants import decoded_ccd
        decoded_ccd.get_ccd = functools.partial(decoded_ccd.get_ccd, user_ccd=open(os.environ["USER_CCD"]).read())
    batch, cfg, model_dir = fold_check._fold_setup("chai1", seq, os.environ.get("MODEL_DIR") or None, chains=chains)
    cfg.global_config.bfloat16 = "none"
    fb = feat_batch.Batch.from_data_dict(batch)
    feats = flat_atom_features(fb)
    n_tok = int(np.asarray(fb.token_features.mask).shape[0]); max_atoms = feats["mask"].shape[1]
    if os.environ.get("SYNTH"):
        r = np.random.default_rng(1)
        s = (r.normal(size=(n_tok, 384)) * 0.5).astype(np.float32)
        z = (r.normal(size=(n_tok, n_tok, 256)) * 0.5).astype(np.float32)
        s_struct = (r.normal(size=(n_tok, 384)) * 0.5).astype(np.float32)
    else:
        trunk = json.loads((REPO / "oracle-dumps" / "af3-oracle-trunk-chai1.json").read_text())["stages"]
        get = lambda k: np.asarray(trunk[k]["data"], np.float32).reshape(trunk[k]["shape"])
        s, z, s_struct = get("single"), get("pair"), get("target_feat_structure")
    z_struct = structure_pair(fb.token_features)
    if os.environ.get("Z_STRUCT_OUT"):
        np.save(os.environ["Z_STRUCT_OUT"], z_struct)
    rng = np.random.default_rng(0)
    pos = (rng.normal(size=(n_tok, max_atoms, 3)) * NOISE).astype(np.float32) * feats["mask"][..., None]
    full = afp.get_model_haiku_params(model_dir=model_dir)
    emb = {"single": jnp.asarray(s), "pair": jnp.asarray(z), "target_feat": jnp.asarray(s_struct),
           "pair_init": jnp.asarray(z_struct)}
    def fwd():
        return DH.DiffusionHead(cfg.heads.diffusion, cfg.global_config)(
            positions_noisy=jnp.asarray(pos), noise_level=jnp.asarray(NOISE, jnp.float32), batch=fb, embeddings=emb,
            use_conditioning=True)
    f = hk.transform(fwd)
    init = f.init(jax.random.PRNGKey(0))
    params = {}
    for sc in init:
        tail = sc[len("diffusion_head/"):] if sc.startswith("diffusion_head/") else sc
        key = "diffuser/~/diffusion_head" if tail in ("~", "diffusion_head") else "diffuser/~/diffusion_head/" + tail
        params[sc] = {k: np.asarray(full[key][k], np.float32) for k in init[sc]}
    TRACE.clear()
    x = np.asarray(f.apply(params, jax.random.PRNGKey(0)))
    print("tokens", n_tok, "out", x.shape, "rms %.4f" % float(np.sqrt((x ** 2).mean())))
    out = {"model": "chai1", "tokens": n_tok, "maxAtoms": int(max_atoms), "noise": NOISE,
           "seqChannels": int(s.shape[-1]), "pairChannels": int(z.shape[-1]),
           "source": "af3-any-model DiffusionHead, chai1, real trunk, chai's structure pair as pair_init", "inputs": {}}
    def put(name, a):
        a = np.asarray(a); out["inputs"][name] = {"shape": list(a.shape), "dtype": str(a.dtype),
                                                  "data": a.astype(np.float32).ravel().tolist()}
    put("single", s); put("pair", z); put("sInputs", s_struct); put("posNoisy", pos)
    put("atomMask", feats["mask"].astype(np.float32)); put("pairInit", z_struct)
    cross = fb.atom_cross_att
    for nm in ("token_atoms_to_queries", "queries_to_keys", "queries_to_token_atoms", "tokens_to_queries", "tokens_to_keys"):
        g = getattr(cross, nm); put(f"{nm}:gather_idxs", np.asarray(g.gather_idxs)); put(f"{nm}:gather_mask", np.asarray(g.gather_mask))
    for k in ("ref_pos", "ref_space_uid", "ref_mask", "ref_element", "ref_charge", "ref_atom_name_chars"):
        put(k, batch[k])
    put("seq_mask", fb.token_features.mask)
    out["output"] = {"shape": list(x.shape), "data": x.astype(np.float32).ravel().tolist()}
    tag = "chai1" + os.environ.get("SUFFIX", "")
    (REPO / "oracle-dumps" / f"af3-oracle-denoise-{tag}.json").write_text(json.dumps(out))
    stages = {"model": "chai1", "tokens": n_tok, "maxAtoms": int(max_atoms), "noise": NOISE, "stages": TRACE,
              "source": out["source"] + ", seams traced"}
    (REPO / "oracle-dumps" / f"af3-oracle-stages-{tag}.json").write_text(json.dumps(stages))
    print(f"wrote af3-oracle-denoise-{tag}.json and af3-oracle-stages-{tag}.json;", ", ".join(sorted(TRACE)))


if __name__ == "__main__":
    sys.argv = sys.argv[:1]          # (absl, inside the reference, parses them)
    main()
