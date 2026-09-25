"""Synthyra's ConfidenceHead on OUR REAL trunk output, not on seeded normals.

    node tools/gpu-chrome.mjs tools/gpu/fold-esmfold2.js \
      --bundle=/model-esmfold2-int5 --confidence-inputs=1 > /tmp/ours.log
    ~/venv_ef2/bin/python tools/oracle/check_esmfold2_confidence_real.py /tmp/ours.json

🔴 THE ARM `dump_esmfold2_confidence.py` SAYS IT IS NOT. That dump's `z` and
`s_inputs` are seeded normals, so it gates the head's ARITHMETIC and its own
header says "a run on a real trunk's pair is a second arm that wants our own
fold to dump one". This is that arm: `--confidence-inputs=1` makes the fold
return the head's own inputs, and their module is run on exactly those.

🔴 AND IT IS THE ONLY THING HERE THAT COULD HAVE PRICED THE QUANTISATION. The
synthetic arm reads 1e-7 on a float32 head and says nothing about the bundle
that SHIPS. Measured on a 40-token fold, their head against ours on one trunk:

    bundle   mean pLDDT      pTM        pLDDT relRMS   PAE relRMS
    int5     56.215/57.502   0.472/0.491     3.57e-2      7.19e-2
    f32      58.205/58.200   0.554/0.554     1.42e-4      4.45e-4

The port is exact; the residual is int5 on the head, and it costs 2.0 pLDDT and
0.082 pTM - which is what a reader sees. See docs/EF2FAST.md.
"""
import sys, json, pathlib, zipfile, base64, tempfile, numpy as np, torch
SNAP = "/home/ubuntu/.cache/huggingface/hub/models--Synthyra--ESMFold2-600/snapshots/87f4ab1ea6ef882a30315b83d2cd5b2997358272"
OURS = sys.argv[1]

# their sources, unpacked by the function the existing dumper already uses
sys.path.insert(0, "/home/ubuntu/localfold/tools/oracle")
import importlib.util as _u
_spec = _u.spec_from_file_location("_dump", "/home/ubuntu/localfold/tools/oracle/dump_esmfold2_confidence.py")
_mod = _u.module_from_spec(_spec)
import types
try:
    _spec.loader.exec_module(_mod)
except SystemExit:
    pass
tmp = _mod.unpack_runtime(pathlib.Path(SNAP) / "fastplms_bundle.py")
sys.path.insert(0, str(tmp))

from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.models.esmfold2.modeling_esmfold2_experimental import ConfidenceHead
from safetensors.torch import load_file

config = ESMFold2Config(**json.loads((pathlib.Path(SNAP) / "config.json").read_text()))
head = ConfidenceHead(config).eval()
sd = load_file(str(pathlib.Path(SNAP) / "model.safetensors"))
hw = {k[len("confidence_head."):]: v for k, v in sd.items() if k.startswith("confidence_head.")}
missing, unexpected = head.load_state_dict(hw, strict=False)
print(f"  head tensors {len(hw)}, missing {len(missing)}, unexpected {len(unexpected)}")

d = json.load(open(OURS))["confidence"]
i = d["inputs"]; n = i["tokens"]; a = i["atoms"]
t = lambda x, shape, dt=torch.float32: torch.tensor(np.asarray(x, dtype=np.float32).reshape(shape), dtype=dt)
inputs = {
    "s_inputs": t(i["sInputs"], (1, n, -1)),
    "z": t(i["pair"], (1, n, n, -1)),
    "x_pred": t(i["coordinates"], (1, a, 3)),
    "distogram_atom_idx": torch.tensor(np.asarray(i["repAtom"]), dtype=torch.long).unsqueeze(0),
    "token_attention_mask": torch.ones(1, n, dtype=torch.long),
    "atom_to_token": torch.tensor(np.asarray(i["atomToToken"]), dtype=torch.long).unsqueeze(0),
    "atom_attention_mask": torch.tensor(np.asarray(i["atomMask"]), dtype=torch.long).unsqueeze(0),
    "asym_id": torch.tensor(np.asarray(i["asymId"]), dtype=torch.long).unsqueeze(0),
    "mol_type": torch.zeros(1, n, dtype=torch.long),
    # 🔴 THE ARGUMENT THAT WAS MISSING ON BOTH SIDES. Their head takes this as
    # a keyword defaulting to None and adds it only when it is not None, so
    # while neither side passed it the two agreed perfectly and both were
    # wrong. An oracle that builds its own call cannot see an argument neither
    # side supplies; passing it is what makes this checker able to fail.
    "relative_position_encoding": t(i["pairBias"], (1, n, n, -1)),
}
with torch.no_grad():
    out = head(**inputs)

their_plddt = out["plddt"].squeeze().numpy() * 100.0
their_pae = out["pae"].squeeze().numpy()
our_plddt = np.asarray(d["plddt"], dtype=np.float64) * 100.0
our_pae = np.asarray(d["paeAll"], dtype=np.float64).reshape(n, n)

def rel(x, y):
    return float(np.sqrt(((x - y) ** 2).mean()) / (np.sqrt((y ** 2).mean()) + 1e-12))
print()
print(f"  tokens {n}")
print(f"  pLDDT  ours mean {our_plddt.mean():7.3f}   theirs {their_plddt.mean():7.3f}"
      f"   relRMS {rel(our_plddt, their_plddt):.3e}  max|d| {np.abs(our_plddt-their_plddt).max():.4f}")
print(f"  PAE    ours mean {our_pae.mean():7.3f}   theirs {their_pae.mean():7.3f}"
      f"   relRMS {rel(our_pae, their_pae):.3e}  max|d| {np.abs(our_pae-their_pae).max():.4f}")
print(f"  pTM    ours {d.get('ptm')}   theirs {float(out['ptm'].squeeze()):.6f}")
print(f"  iptm   ours {d.get('iptm')}   theirs {float(out['iptm'].squeeze()):.6f}")
