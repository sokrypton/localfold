/**
 * The chain-level pTM scores as AlphaFold 3 defines them.
 *
 * 🔴 REPORTED BY YEQING LIN: a short binder's chain pTM read ~0.10 high against
 * AF3 over 400 BinderBench designs. Three gaps, all checked here against
 * af3-any-model's confidence_head.py / model.py / confidences.py:
 *
 * - the chain-level scores reduce AF3's INTERFACE TM term, d0 from the pair's
 *   own chains (chainTmInfo), not the global one, d0 from every token;
 * - `chain_iptm` is the mean of a chain's pair ipTMs (iptm_xchain), not the best
 *   anchor's mean over every other chain's tokens pooled;
 * - only tokens with frames anchor an alignment (pae_single_mask) - an ion
 *   aligns nothing (tokenFrameMask, features.py's Frames).
 *
 * tools/oracle/check_chain_ptm.py holds whole folds, the CUDA port's and the
 * page's, to AF3's own functions; these hold the pieces.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { chainTmInfo, interfaceTmTerm, perChainTmScores, tmPerBinFor, tmScoreD0, tokenFrameMask }
  from "../shared/heads/tm-score.js";

const centres = Float64Array.from({ length: 64 }, (unused, bin) => (31 / 62) * bin + 31 / 124);

describe("chainTmInfo: AF3's num_interface_tokens", () => {
  it("weights a chain's own pairs at its own d0 and a cross pair at the two chains' summed", () => {
    // a 55-token binder against a 116-token target (the report's 04_pdl1, 171 tokens)
    const chainOf = [...new Array(55).fill(1), ...new Array(116).fill(2)];
    const info = chainTmInfo(chainOf, new Array(171).fill(1), centres);
    assert.equal(info[171], 2);
    const table = (a, b) => info.subarray(172 + (a * 2 + b) * 64, 172 + (a * 2 + b + 1) * 64);
    const close = (x, y) => x.every((v, k) => Math.abs(v - y[k]) < 1e-6);
    assert.ok(close(table(0, 0), tmPerBinFor(centres, tmScoreD0(55))), "the binder's diagonal at d0(55)");
    assert.ok(close(table(1, 1), tmPerBinFor(centres, tmScoreD0(116))), "the target's at d0(116)");
    assert.ok(close(table(0, 1), tmPerBinFor(centres, tmScoreD0(171))), "the cross pair at d0(55 + 116)");
    // and d0(55) is the 2.44 A the report computes, against the whole complex's 4.88
    assert.ok(Math.abs(tmScoreD0(55) - 2.44) < 0.01 && Math.abs(tmScoreD0(171) - 4.88) < 0.01);
  });

  it("counts d0 from the tokens it is told to, not the ones it indexes (OpenDDE's structural tokens)", () => {
    const residues = { chainOf: [1, 1, 1, 2, 2], live: [1, 1, 1, 1, 1] };
    const structural = [1, 1, 1, 1, 1, 1, 2, 2, 2];   // (subtokens: more of them than residues)
    const info = chainTmInfo(structural, new Array(9).fill(1), centres, residues);
    const diag = info.subarray(10, 10 + 64);
    assert.ok(diag.every((v, k) => Math.abs(v - tmPerBinFor(centres, tmScoreD0(3))[k]) < 1e-6));
  });

  it("interfaceTmTerm applies the tables to a pair's softmax, zero for a token that is not live", () => {
    const logits = Float32Array.from({ length: 4 * 64 }, (unused, k) => Math.sin(k));
    const info = chainTmInfo([1, -1], [1, 1], centres);
    const term = interfaceTmTerm(logits, 2, info);
    assert.equal(term[1], 0);
    assert.equal(term[2], 0);
    assert.ok(term[0] > 0 && term[0] < 1);
  });
});

describe("perChainTmScores: AF3's chain_ptm and chain_iptm", () => {
  /** three chains of four tokens; the term `t[a][b]` between chains a and b */
  const three = (t) => {
    const asymId = Int32Array.from({ length: 12 }, (unused, i) => Math.floor(i / 4) + 1);
    const term = new Float64Array(144);
    for (let i = 0; i < 12; i += 1) for (let j = 0; j < 12; j += 1) term[i * 12 + j] = t[asymId[i] - 1][asymId[j] - 1];
    return { asymId, term, seqMask: new Float32Array(12).fill(1) };
  };

  it("takes chain_iptm as the mean of the chain's pair ipTMs (iptm_xchain), not one pooled alignment", () => {
    const { asymId, term, seqMask } = three([[0.9, 0.8, 0.2], [0.8, 0.9, 0.5], [0.2, 0.5, 0.9]]);
    const { chainIptm, chainPtm } = perChainTmScores(term, 12, asymId, seqMask);
    // chain 1's pair ipTMs: 0.8 with chain 2, 0.2 with chain 3 - their mean, 0.5
    assert.ok(Math.abs(chainIptm.get(1) - 0.5) < 1e-12, `chain_iptm[1] ${chainIptm.get(1)}`);
    assert.ok(Math.abs(chainIptm.get(3) - 0.35) < 1e-12);
    assert.ok(Math.abs(chainPtm.get(2) - 0.9) < 1e-12);
  });

  it("anchors only on tokens with frames", () => {
    const asymId = Int32Array.from([1, 1, 2]);
    const term = Float64Array.from([0.5, 0.5, 0.4, 0.5, 0.5, 0.4, 0.9, 0.9, 0.9]);   // token 2's row high
    const seqMask = new Float32Array(3).fill(1);
    const withIon = perChainTmScores(term, 3, asymId, seqMask, Uint8Array.from([1, 1, 0]));
    const without = perChainTmScores(term, 3, asymId, seqMask);
    // chain 2 is one frameless token: no anchor for its own pTM, and its row cannot lift the pair's ipTM
    assert.equal(withIon.chainPtm.get(2), undefined);
    assert.ok(Math.abs(withIon.chainIptm.get(1) - 0.4) < 1e-12);
    assert.ok(Math.abs(without.chainIptm.get(1) - 0.9) < 1e-12);
  });
});

describe("tokenFrameMask: AF3's Frames.compute_features", () => {
  // one protein residue then ligand atoms, each its own token (dense 1)
  const batch = (ligands) => {
    const tokens = 1 + ligands.reduce((n, l) => n + l.atoms.length, 0);
    const isLigand = new Int32Array(tokens), asymId = new Int32Array(tokens), residueIndex = new Int32Array(tokens);
    const refPos = new Float32Array(tokens * 3), refMask = new Float32Array(tokens).fill(1);
    let t = 1;
    ligands.forEach((l, li) => {
      for (const xyz of l.atoms) {
        isLigand[t] = 1; asymId[t] = li + 2; residueIndex[t] = 1; refPos.set(xyz, t * 3); t += 1;
      }
    });
    asymId[0] = 1;
    return { tokens, dense: 1, isLigand, asymId, residueIndex, refPos, refMask };
  };

  it("gives every polymer token a frame, and none to an ion or a two-atom ligand", () => {
    const m = tokenFrameMask(batch([{ atoms: [[0, 0, 0]] }, { atoms: [[5, 0, 0], [6.2, 0, 0]] }]));
    assert.deepEqual(Array.from(m), [1, 0, 0, 0]);
  });

  it("refuses a colinear triple and keeps a bent one", () => {
    const straight = tokenFrameMask(batch([{ atoms: [[0, 0, 0], [1.5, 0, 0], [3, 0, 0]] }]));
    const bent = tokenFrameMask(batch([{ atoms: [[0, 0, 0], [1.5, 0, 0], [2.2, 1.3, 0]] }]));
    assert.deepEqual(Array.from(straight), [1, 0, 0, 0]);
    assert.deepEqual(Array.from(bent), [1, 1, 1, 1]);
  });
});
