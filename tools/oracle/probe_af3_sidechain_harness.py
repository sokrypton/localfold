"""Why does fold_check's AF3 contract side chains where run_alphafold's does not?

Same weights (~/ported/alphafold3/af3.bin.zst, Google's own), same sequence,
same query-only MSA, no templates. run_alphafold gives side-chain bond rms
0.051 A; fold_check gives 0.345 with 100% of them SHORT. So it is the HARNESS,
and this prints the config it uses and folds under each candidate difference.
"""
import os, sys
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
for e in ("/home/ubuntu/alphafold3/src", "/home/ubuntu/alphafold3",
          "/home/ubuntu/alphafold3/dev/oracles"):
    sys.path.insert(0, e)
import numpy as np
import fold_check

SEQ = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"
ARM = os.environ.get("ARM", "plain")

batch, cfg, model_dir = fold_check._fold_setup("alphafold3", SEQ, None)
d = cfg.heads.diffusion
print("model_dir", model_dir)
for name in ("eval", "sample", "num_samples", "steps"):
    if hasattr(d, name):
        print("  heads.diffusion.%s = %s" % (name, getattr(d, name)))
print("  num_recycles:", getattr(cfg, "num_recycles", "?"))
print("  tokens in unpadded batch:", np.asarray(batch["seq_mask"]).shape)
sys.stdout.flush()

buckets = None if ARM == "plain" else [int(ARM)]
msa = ">q\n%s\n" % SEQ
from alphafold3.common import folding_input
chains = [folding_input.ProteinChain(id="A", sequence=SEQ, ptms=[],
                                     unpaired_msa=msa, paired_msa="", templates=[])]
out, batch = fold_check.fold("alphafold3", SEQ, seed=1, chains=chains, buckets=buckets)

pos = np.asarray(out["diffusion_samples"]["atom_positions"])
print("samples", pos.shape)
mask = np.asarray(batch["ref_mask"])
ref = np.asarray(batch["ref_pos"])
names = np.asarray(batch["ref_atom_name_chars"])


def score(x):
    short = tot = 0
    errs = []
    T, A = mask.shape
    for t in range(T):
        live = [s for s in range(A) if mask[t, s]]
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                a, b = live[i], live[j]
                ideal = np.linalg.norm(ref[t, a] - ref[t, b])
                if ideal > 1.95 or ideal == 0:
                    continue
                nm = lambda s: "".join(chr(c + 32) for c in names[t, s] if c > 0).strip()
                if nm(a) in ("N", "CA", "C", "O", "OXT") and nm(b) in ("N", "CA", "C", "O", "OXT"):
                    continue
                seen = np.linalg.norm(x[t, a] - x[t, b])
                errs.append(seen - ideal); tot += 1; short += seen < ideal
    e = np.asarray(errs)
    return float(np.sqrt((e ** 2).mean())), float(e.mean()), 100.0 * short / tot, tot


for s in range(pos.shape[0]):
    r, m, sh, n = score(pos[s] * mask[..., None])
    print("ARM=%s sample %d  sidechain rms %.4f  mean %+.4f  short %3.0f%%  (n=%d)"
          % (ARM, s, r, m, sh, n))
