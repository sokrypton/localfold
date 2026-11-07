#!/usr/bin/env python3
"""The chain-level pTM scores' TWO implementations, held to each other: cuda/featurise/ptm.h (every
native port - metal/af3, metal/ef2, cuda/af3, cuda/ef2) against shared/heads/tm-score.js (the page -
WebGPU and WebGL2 alike).

    python3 tools/check-chain-ptm-ports.py [--keep]

🔴 NOTHING HELD THEM TOGETHER. Both files say they are the other's twin and both were written from
AlphaFold 3's own definition (reported by Yeqing Lin: the chain-level scores were reduced from the
GLOBAL TM term where AF3 reduces them from the interface one, and pTM's anchors are the tokens with
frames) - but a fold on the page and a fold through a native port share no code, no fixture and no
gate, so a divergence between the two shows up as a reader's two backends disagreeing about a
published number. Every other check here compares a BACKEND against a backend of the same half.

It needs no weights, no device and no fixture: the cases are generated, the logits come from the same
xorshift32 both sides, and what is compared is the arithmetic and nothing else - the frame mask token
by token, the whole interface term, chain_pair_iptm, chain_ptm and chain_iptm. Three cases, each
reaching something the others cannot:

  * two-chains          the plain complex;
  * binder-in-three     the reported shape, and THREE chains - the only arrangement where
                        chain_iptm's mean-of-pairs differs from the pooled reduction it replaced;
  * ligands-and-padding an ion, a two-atom ligand, a bent four-atom one and a colinear three-atom
                        one, plus padding, which is where the frame mask decides anything at all.

🔴 ONE DIFFERENCE IS DELIBERATE AND IS NOT CHECKED: on a NaN term the native reduction PROPAGATES
(ptm.h: "a broken term must not score") where the page's `reduceTmScore` lets `>` drop that anchor and
scores the rest. The generated terms are finite, so this gate cannot see it; it is recorded here
because it is the one place the two definitions are meant to part.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The cases, stated once per language and next to each other so that a drift in the construction is
# visible - and the driver compares each case's token count, chain ids and chain sizes before it
# compares a score, so a drift fails rather than being scored.
CASES_H = r"""
#pragma once
#include <string>
#include <vector>

struct Case {
  std::string name;
  int bins = 24;
  std::vector<int> asym, isLigand, residueIndex;
  std::vector<float> live, pos, posLive;
  int n() const { return (int)asym.size(); }
};

inline void polymer(Case& c, int asym, int tokens, float mask = 1.f) {
  for (int i = 0; i < tokens; ++i) {
    c.asym.push_back(asym); c.isLigand.push_back(0); c.residueIndex.push_back(i); c.live.push_back(mask);
    c.pos.insert(c.pos.end(), {0.f, 0.f, 0.f}); c.posLive.push_back(1.f);
  }
}

inline void ligand(Case& c, int asym, const std::vector<std::vector<float>>& atoms) {
  for (const auto& a : atoms) {
    c.asym.push_back(asym); c.isLigand.push_back(1); c.residueIndex.push_back(0); c.live.push_back(1.f);
    c.pos.insert(c.pos.end(), {a[0], a[1], a[2]}); c.posLive.push_back(1.f);
  }
}

inline std::vector<Case> cases() {
  std::vector<Case> out;
  { Case c; c.name = "two-chains"; polymer(c, 0, 30); polymer(c, 1, 25); out.push_back(c); }
  { Case c; c.name = "binder-in-three";
    polymer(c, 0, 76); polymer(c, 1, 55); polymer(c, 2, 40); out.push_back(c); }
  { Case c; c.name = "ligands-and-padding";
    polymer(c, 0, 20); polymer(c, 1, 12);
    ligand(c, 2, {{0, 0, 0}});                                                 // an ion: no frame
    ligand(c, 3, {{0, 0, 0}, {1.5f, 0, 0}});                                   // two atoms: no frame
    ligand(c, 4, {{0, 0, 0}, {1.5f, 0, 0}, {0.7f, 1.3f, 0}, {2.1f, 1.4f, 0}}); // bent: frames
    ligand(c, 5, {{0, 0, 0}, {1.5f, 0, 0}, {3.0f, 0, 0}});                     // colinear: no frames
    polymer(c, 6, 5, 0.f);                                                     // padding
    out.push_back(c); }
  return out;
}

struct Rng { unsigned x; explicit Rng(unsigned s) : x(s ? s : 1u) {} unsigned next() {
  x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; } };
inline double uniform(Rng& r) { return (double)(r.next() >> 8) / 16777216.0; }

inline std::vector<float> logitsFor(const Case& c, unsigned seed) {
  Rng r(seed);
  std::vector<float> lg((size_t)c.n() * c.n() * c.bins);
  for (size_t i = 0; i < lg.size(); ++i) lg[i] = (float)(uniform(r) * 8.0 - 4.0);
  return lg;
}
inline std::vector<float> centresFor(int bins) {
  std::vector<float> c(bins);
  for (int b = 0; b < bins; ++b) c[b] = (float)(0.5 + b * 31.0 / bins);
  return c;
}
"""

NATIVE_CPP = r"""
#include <cstdio>
#include <string>
#include <vector>
#include "ptm.h"
#include "cases.h"

static void jnum(double v, bool first) {
  if (!first) printf(", ");
  if (v != v) printf("null"); else printf("%.17g", v);
}

int main() {
  printf("{\"cases\": [");
  const std::vector<Case> all = cases();
  for (size_t ci = 0; ci < all.size(); ++ci) {
    const Case& c = all[ci];
    const int n = c.n();
    const std::vector<float> lg = logitsFor(c, 0x5eed1234u + (unsigned)ci);
    const std::vector<float> centres = centresFor(c.bins);
    const ptm::Layout L = ptm::layout(c.asym.data(), c.live.data(), n);
    const std::vector<float> term = ptm::interfaceTerm(lg.data(), n, c.bins, L,
                                                       ptm::tables(L, centres.data(), c.bins));
    std::vector<char> plive(n);
    for (int i = 0; i < n; ++i) plive[i] = c.posLive[i] > 0;
    const std::vector<char> frame = ptm::frameMask(n, c.isLigand.data(), c.asym.data(),
                                                   c.residueIndex.data(), c.pos.data(), plive.data());
    const ptm::Chains cs = ptm::chainScores(term.data(), n, L, frame.data());
    const int nc = L.nc();
    printf("%s{\"name\": \"%s\", \"tokens\": %d, \"chains\": %d, \"frame\": \"",
           ci ? ", " : "", c.name.c_str(), n, nc);
    for (int i = 0; i < n; ++i) printf("%d", frame[i] ? 1 : 0);
    printf("\", \"ids\": [");
    for (int a = 0; a < nc; ++a) jnum(L.ids[a], a == 0);
    printf("], \"counts\": [");
    for (int a = 0; a < nc; ++a) jnum(L.count[a], a == 0);
    printf("], \"term\": [");
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) jnum(term[(size_t)i * n + j], i == 0 && j == 0);
    printf("], \"chainPtm\": [");
    for (int a = 0; a < nc; ++a) jnum(cs.ptm[a], a == 0);
    printf("], \"chainIptm\": [");
    for (int a = 0; a < nc; ++a) jnum(cs.iptm[a], a == 0);
    printf("], \"chainPair\": [");
    for (int a = 0; a < nc; ++a) for (int b = 0; b < nc; ++b) jnum(cs.pair[(size_t)a * nc + b], a == 0 && b == 0);
    printf("]}");
  }
  printf("]}\n");
  return 0;
}
"""

PAGE_MJS = r"""
import { chainPairTmScores, chainTmInfo, interfaceTmTerm, perChainTmScores, tokenFrameMask }
  from "%(root)s/shared/heads/tm-score.js";

const cases = () => {
  const out = [];
  const blank = (name) => ({ name, bins: 24, asym: [], isLigand: [], residueIndex: [], live: [],
    pos: [], posLive: [] });
  const polymer = (c, asym, tokens, mask = 1) => {
    for (let i = 0; i < tokens; i += 1) {
      c.asym.push(asym); c.isLigand.push(0); c.residueIndex.push(i); c.live.push(mask);
      c.pos.push(0, 0, 0); c.posLive.push(1);
    }
  };
  const ligand = (c, asym, atoms) => {
    for (const a of atoms) {
      c.asym.push(asym); c.isLigand.push(1); c.residueIndex.push(0); c.live.push(1);
      c.pos.push(a[0], a[1], a[2]); c.posLive.push(1);
    }
  };
  { const c = blank("two-chains"); polymer(c, 0, 30); polymer(c, 1, 25); out.push(c); }
  { const c = blank("binder-in-three");
    polymer(c, 0, 76); polymer(c, 1, 55); polymer(c, 2, 40); out.push(c); }
  { const c = blank("ligands-and-padding");
    polymer(c, 0, 20); polymer(c, 1, 12);
    ligand(c, 2, [[0, 0, 0]]);
    ligand(c, 3, [[0, 0, 0], [1.5, 0, 0]]);
    ligand(c, 4, [[0, 0, 0], [1.5, 0, 0], [0.7, 1.3, 0], [2.1, 1.4, 0]]);
    ligand(c, 5, [[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]]);
    polymer(c, 6, 5, 0);
    out.push(c); }
  return out;
};

const logitsFor = (c, seed) => {
  let x = (seed >>> 0) || 1;
  const next = () => {
    x = (x ^ (x << 13)) >>> 0;
    x = (x ^ (x >>> 17)) >>> 0;
    x = (x ^ (x << 5)) >>> 0;
    return x;
  };
  const n = c.asym.length;
  const lg = new Float32Array(n * n * c.bins);
  for (let i = 0; i < lg.length; i += 1) lg[i] = (next() >>> 8) / 16777216 * 8 - 4;
  return lg;
};
const centresFor = (bins) => Float32Array.from({ length: bins }, (unused, b) => 0.5 + b * 31 / bins);

const report = [];
const all = cases();
for (let ci = 0; ci < all.length; ci += 1) {
  const c = all[ci];
  const n = c.asym.length;
  const logits = logitsFor(c, 0x5eed1234 + ci);
  const centres = centresFor(c.bins);
  const term = interfaceTmTerm(logits, n, chainTmInfo(c.asym, c.live, centres));
  const frame = tokenFrameMask({ tokens: n, dense: 1, isLigand: c.isLigand, asymId: c.asym,
    residueIndex: c.residueIndex, refPos: Float32Array.from(c.pos),
    refMask: Float32Array.from(c.posLive) });
  const { chains, chainPtm, chainIptm } = perChainTmScores(term, n, c.asym, c.live, frame);
  const pairs = chainPairTmScores(term, n, c.asym, c.live, frame).scores;
  const nc = chains.length;
  const finite = (v) => (v === undefined || Number.isNaN(v) ? null : v);
  const pair = [];
  for (let a = 0; a < nc; a += 1) {
    for (let b = 0; b < nc; b += 1) {
      const key = chains[a] < chains[b] ? `${chains[a]}|${chains[b]}` : `${chains[b]}|${chains[a]}`;
      pair.push(finite(a === b ? chainPtm.get(chains[a]) : pairs.get(key)));
    }
  }
  report.push({ name: c.name, tokens: n, chains: nc, ids: chains,
    counts: chains.map((id) => c.asym.reduce((s, a, i) => s + (a === id && c.live[i] > 0 ? 1 : 0), 0)),
    frame: Array.from(frame).join(""),
    term: Array.from(term),
    chainPtm: chains.map((id) => finite(chainPtm.get(id))),
    chainIptm: chains.map((id) => finite(chainIptm.get(id))),
    chainPair: pair });
}
process.stdout.write(`${JSON.stringify({ cases: report })}\n`);
"""

# The interface term is a float32 softmax on both sides and the reduction a double sum, so the bars
# are the arithmetic's and not a tolerance anyone chose: the term to a float32 epsilon, the scores
# tighter because a mean of many terms averages that away.
TERM_BOUND = 1e-6
SCORE_BOUND = 1e-7


def worst(a, b, key):
    """The largest absolute difference in a case's `key`, or a description of a structural one."""
    u, v = a[key], b[key]
    if len(u) != len(v):
        return f"{key}: {len(u)} values against {len(v)}"
    out = 0.0
    for i, (p, q) in enumerate(zip(u, v)):
        if (p is None) != (q is None):
            return f"{key}[{i}]: {p} against {q}"
        if p is not None:
            out = max(out, abs(p - q))
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true", help="keep the generated sources and binary")
    args = ap.parse_args(argv)

    work = tempfile.mkdtemp(prefix="chain-ptm-ports-")
    try:
        with open(os.path.join(work, "cases.h"), "w") as f:
            f.write(CASES_H)
        with open(os.path.join(work, "native.cpp"), "w") as f:
            f.write(NATIVE_CPP)
        with open(os.path.join(work, "page.mjs"), "w") as f:
            f.write(PAGE_MJS % {"root": ROOT})
        build = subprocess.run(
            ["clang++" if sys.platform == "darwin" else "g++", "-std=c++17", "-O2",
             "-I" + os.path.join(ROOT, "cuda", "featurise"), "-I" + work,
             "-o", os.path.join(work, "native"), os.path.join(work, "native.cpp")],
            capture_output=True, text=True)
        if build.returncode != 0:
            print(build.stderr.strip())
            return f"the native harness would not build (cuda/featurise/ptm.h)"
        native = json.loads(subprocess.run([os.path.join(work, "native")],
                                           capture_output=True, text=True, check=True).stdout)["cases"]
        page = json.loads(subprocess.run(["node", os.path.join(work, "page.mjs")],
                                         capture_output=True, text=True, check=True).stdout)["cases"]
    finally:
        if args.keep:
            print(f"sources kept in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    if len(native) != len(page):
        return f"{len(native)} native cases against {len(page)} page ones"
    problems = []
    for a, b in zip(native, page):
        if a["name"] != b["name"]:
            problems.append(f"{a['name']} against {b['name']}")
            continue
        # The shape first: two sides that built different cases cannot be compared on a score.
        for key in ("tokens", "chains", "ids", "counts", "frame"):
            if a[key] != b[key]:
                problems.append(f"{a['name']}: {key} {a[key]} against {b[key]}")
        gaps = {key: worst(a, b, key) for key in ("term", "chainPtm", "chainIptm", "chainPair")}
        for key, gap in gaps.items():
            bound = TERM_BOUND if key == "term" else SCORE_BOUND
            if isinstance(gap, str) or gap > bound:
                problems.append(f"{a['name']}: {key} {gap} (bound {bound})")
        shown = {k: (v if isinstance(v, str) else f"{v:.2e}") for k, v in gaps.items()}
        frameless = a["frame"].count("0")
        print(f"{a['name']:22s} {a['tokens']:4d} tokens, {a['chains']} chains, "
              f"{frameless} without a frame  "
              + "  ".join(f"{k} {v}" for k, v in shown.items()))
    if problems:
        print("\n".join(f"FAIL {p}" for p in problems))
        return f"{len(problems)} disagreement(s) between ptm.h and shared/heads/tm-score.js"
    print("ptm.h and shared/heads/tm-score.js agree: the frame mask token for token, the interface "
          "term, chain_pair_iptm, chain_ptm and chain_iptm")
    return None


if __name__ == "__main__":
    failure = main(sys.argv[1:])
    if failure is not None:
        sys.exit(f"FAIL: {failure}")
