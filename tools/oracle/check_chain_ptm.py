"""Hold localfold-af3's pTM-family scores to AlphaFold 3's own reductions.

    AF3_PAE_OUT=/tmp/x.bin cuda/af3/localfold-af3 --job=<job.json> --out=/tmp/x.pdb ...
    ~/.venv-lfjax/bin/python tools/oracle/check_chain_ptm.py --dump=/tmp/x.bin --out=/tmp/x.pdb \
        [--log=<the fold's stdout>] [--af3=<an af3-any-model src/, else the installed one>] [--batch=<dump_af3_batch.py json, for frames_mask>]

    # the page's path (WebGPU): tools/gpu/fold.js --tm-check's result, the JSON it prints
    node tools/gpu-chrome.mjs tools/gpu/fold.js --tm-check ... > /tmp/w.log
    ~/.venv-lfjax/bin/python tools/oracle/check_chain_ptm.py --scores=/tmp/w.log

The fold's dump (AF3_PAE_OUT) carries, after its PAE / PDE / pLDDT, the two per-pair TM terms - the interface one
(d0 from the pair's chains) and the global one (d0 from every token) - and the token frames' mask its pTM anchors on.
This feeds them to af3-any-model's confidences.py (predicted_tm_score, chain_pairwise_predicted_tm_scores,
get_iptm_xchain) exactly as model.py does, and compares with what the fold wrote: pTM and ipTM (its stdout, 4
places) and chain_ptm, chain_iptm and chain_pair_iptm (its summary JSON, 2 places). It also checks the interface
term against the global one where their d0 must agree - every pair whose chains' tokens are the whole complex (all
of a single chain's; a two-chain complex's cross pairs) - which holds the kernel to itself without the logits; and
with --batch, the frames' mask against AF3's own featurisation's frames_mask.
"""
import argparse
import json
import re
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dump", default="")
ap.add_argument("--out", default="", help="the fold's --out (its _confidences.json and _summary_confidences.json beside)")
ap.add_argument("--scores", default="", help="tools/gpu/fold.js --tm-check's printed result: the page's path instead")
ap.add_argument("--log", default="", help="the fold's stdout, for its pTM and ipTM")
ap.add_argument("--af3", default="", help="an af3-any-model checkout's src/ (its compiled alphafold3.cpp beside it); default: the installed package")
ap.add_argument("--batch", default="", help="a dump_af3_batch.py JSON of the same job: its frames_mask")
a = ap.parse_args()
if a.af3: sys.path.insert(0, a.af3)
from alphafold3.model import confidences  # noqa: E402

if a.scores:
  # the page's: the JSON object tools/gpu-chrome.mjs prints, its tmCheck (chain scores keyed by asym id, "a|b" a pair)
  text = open(a.scores).read()
  res = json.loads(text[text.index("{\n"):text.rindex("}") + 1])
  tc = res["tmCheck"]
  aid = np.array(tc["asymId"], dtype=np.int32)
  live = sorted(set(aid.tolist()))
  asym = np.array([live.index(v) + 1 for v in aid], dtype=np.int32)
  n = len(asym)
  iface = np.array(tc["tmInterface"], np.float64).reshape(n, n)
  glob = np.array(tc["tmGlobal"], np.float64).reshape(n, n)
  frame = np.array(tc["frame"], bool)
  num = lambda v: np.nan if v is None else v
  summ = {"chain_ptm": [num(tc["chainPtm"].get(str(c))) for c in live],
          "chain_iptm": [num(tc["chainIptm"].get(str(c))) for c in live],
          "chain_pair_iptm": [[num(tc["chainPtm"].get(str(c))) if c == d else
                               num(tc["chainPairIptm"].get("%d|%d" % (min(c, d), max(c, d)))) for d in live] for c in live]}
  ids = live
  page = (tc["ptm"], tc["iptm"])
else:
  stem = re.sub(r"\.(pdb|cif)$", "", a.out)
  conf = json.load(open(stem + "_confidences.json"))
  summ = json.load(open(stem + "_summary_confidences.json"))
  chains = conf["token_chain_ids"]
  n = len(chains)
  ids = sorted(set(chains), key=lambda c: (len(c), c))
  asym = np.array([ids.index(c) + 1 for c in chains], dtype=np.int32)
  raw = np.fromfile(a.dump, dtype=np.float32)
  nn = n * n
  dense = (raw.size - 4 * nn - n) // n   # pae, pde, plddt [n dense], interface, global, frames
  if raw.size != 4 * nn + n * dense + n:
    sys.exit("the dump is %d floats: not this fold's (n %d) with the TM terms - an older binary?" % (raw.size, n))
  off = 2 * nn + n * dense
  iface = raw[off:off + nn].reshape(n, n).astype(np.float64)
  glob = raw[off + nn:off + 2 * nn].reshape(n, n).astype(np.float64)
  frame = raw[off + 2 * nn:].astype(bool)
  page = None
pair_mask = np.tile(frame[:, None], [1, n])   # (model.py's pae_single_mask: the rows alone)

fails = 0


def report(name, mine, ref, tol):
  global fails
  mine, ref = np.asarray(mine, float), np.asarray(ref, float)
  both = np.isfinite(ref) & np.isfinite(mine)
  same_nan = np.array_equal(np.isfinite(ref), np.isfinite(mine))
  err = float(np.max(np.abs(mine[both] - ref[both]))) if both.any() else 0.0
  ok = same_nan and err <= tol
  fails += not ok
  print("  %-16s max |ours - AF3's| %.4f (bar %g)%s  %s" % (name, err, tol, "" if same_nan else ", NaN where the other is not",
                                                              "ok" if ok else "FAIL"))


print("%d tokens, %d chains, %d tokens without a frame" % (n, len(ids), int((~frame).sum())))
# the interface term against the global one where their d0 is the same: a pair whose chains hold every token
counts = {c: int((asym == c).sum()) for c in set(asym)}
same = np.array([[(counts[asym[i]] if asym[i] == asym[j] else counts[asym[i]] + counts[asym[j]]) == n for j in range(n)] for i in range(n)])
if same.any():
  d = float(np.max(np.abs(iface[same] - glob[same])))
  ok = d <= 1e-6
  fails += not ok
  print("  interface = global where their d0 agrees (%d pairs): max |diff| %.2e  %s" % (int(same.sum()), d, "ok" if ok else "FAIL"))
if (~same).any():
  print("  (and where it does not, the interface term is the lower in %.1f%% of %d pairs, as a smaller d0 makes it)"
        % (100 * float(np.mean(iface[~same] <= glob[~same] + 1e-7)), int((~same).sum())))
# pTM and ipTM, global term (model.py's _compute_ptm)
ptm = confidences.predicted_tm_score(tm_adjusted_pae=glob, pair_mask=pair_mask, asym_id=asym, interface=False)
iptm = confidences.predicted_tm_score(tm_adjusted_pae=glob, pair_mask=pair_mask, asym_id=asym, interface=True)
if page is not None:   # (the page's are unrounded: the bars tighten)
  report("pTM", [page[0]], [ptm], 1e-5)
  if page[1] is not None: report("ipTM", [page[1]], [iptm], 1e-5)
elif a.log:
  m = re.findall(r"pTM ([0-9.]+)\s+ipTM ([0-9.na]+)", open(a.log).read())
  if m:
    lp, li = m[-1]
    report("pTM", [float(lp)], [ptm], 1e-4)
    if li != "nan": report("ipTM", [float(li)], [iptm], 1e-4)
# chain_pair_iptm on the interface term (model.py's _compute_chain_pair_iptm), chain_ptm its diagonal, chain_iptm
# its rows' off-diagonal means (get_iptm_xchain)
cpi = confidences.chain_pairwise_predicted_tm_scores(tm_adjusted_pae=iface, pair_mask=pair_mask, asym_id=asym)
f = lambda v: np.nan if v is None else v
bar = 1e-5 if page is not None else 0.0051   # (the page's unrounded; the JSON's 2 places)
report("chain_ptm", [f(v) for v in summ["chain_ptm"]], np.diagonal(cpi), bar)
report("chain_pair_iptm", [[f(v) for v in r] for r in summ["chain_pair_iptm"]], cpi, bar)
if len(ids) > 1:
  report("chain_iptm", [f(v) for v in summ["chain_iptm"]], confidences.get_iptm_xchain(cpi[None])[0], bar)
if a.batch:
  bt = json.load(open(a.batch))
  ref = np.asarray(bt["frames_mask"] if "frames_mask" in bt else bt["batch"]["frames_mask"], dtype=bool)[:n]
  dif = int((ref != frame).sum())
  fails += dif != 0
  print("  frames mask against AF3's featurisation: %d of %d tokens differ  %s" % (dif, n, "ok" if dif == 0 else "FAIL"))
print("FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
