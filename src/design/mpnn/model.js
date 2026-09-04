// ProteinMPNN / SolubleMPNN / LigandMPNN / MembraneMPNN inference.
//
// The structure encoder does not depend on the sequence, so `encode()` runs
// once per structure and every subsequent call -- sampling a design, scoring a
// sequence, reading off a per-position profile -- reuses its output. That split
// is the single largest speedup available here and is why the API exposes an
// `Encoded` handle rather than a one-shot `design()`.
//
// Conditioning is expressed as an autoregressive mask rather than a decoding
// order, following ColabDesign. `arMask[i][j] = 1` means "position i may see
// position j's amino acid". A decoding order induces the usual triangular mask;
// the all-zero mask gives backbone-only logits in a single pass.

import { Arena } from "./arena.js";
import { ALPHABET, MESSAGE_SCALE } from "./constants.js";
import {
  atomTypeEmbedding,
  computeBackbone,
  edgeFeatures,
  ligandNodeFeatures,
  ligandPairTable,
  nearestLigandAtoms,
  neighborGraph,
  pairRbf,
  sideChainContext,
} from "./features.js";
import { makeDecoderLayer, makeEncoderLayer } from "./layers.js";
import {
  labelNodeFeatures, naBackbone, naEdgeFeatures,
  NA_OMIT_LEGACY_RNA, NA_OMIT_UNK, N_POLYTYPES, POLYTYPE,
} from "./na.js";
import { gatherNodes, layerNorm, linear, logSoftmax, softmax } from "./ops.js";

const LIGAND_CHUNK = 32;
const DECODE_CHUNK = 96;

/**
 * Rows per atom-pair message batch, in both atom-context layers.
 *
 * The pair count is a property of the structure, not of the model, so anything
 * sized by it has no ceiling: see `_encodeLigandContext`. 8192 rows is 4 MB of
 * [rows, 128] scratch, small enough that the accelerator's three staging
 * copies stay under 13 MB, large enough that the per-call overhead is lost
 * against 8192 x 128 x 128 x 2 multiply-adds.
 */
const PAIR_CHUNK = 8192;

/**
 * Budget for the cached per-edge W1 projection, [3, L·K, 128] floats.
 *
 * Past this the decoder falls back to building the concatenation, which is
 * correct but roughly 2.5x slower -- and the fallback used to trigger silently
 * at L ≈ 1400, which is exactly where someone designing a large complex would
 * notice and have no idea why. 320 MB covers L ≈ 4500 at K = 48.
 */
export const DEFAULT_PREP_BUDGET = 320 << 20;

/** Conditioning modes accepted by `score()` and `profile()`. */
export const AR = {
  /** Backbone only: no position sees any amino acid. One pass, exact. */
  NONE: "none",
  /** Standard autoregressive mask induced by a decoding order. */
  ORDER: "order",
  /**
   * Pseudo-likelihood: every position sees every other position's amino acid
   * and none sees its own, in one pass. `ar_mask = 1 - I`, which is exactly
   * what the ProteinMPNN-in-JAX notebook calls `conditional`.
   *
   * This is not literally L separate conditionals. The mask is cyclic, so
   * across three decoder layers a residue's own identity can return to it by a
   * two-hop path -- i tells j at layer 0, j tells i at layer 1. But the L-pass
   * alternative is not a clean reference point either: putting position t last
   * still leaves the order of the other L-1 free, and the decoder is not
   * invariant to it. Measured on ubiquitin and streptavidin, two L-pass
   * profiles that differ only in that order disagree by 0.04-0.05 nats per
   * position (argmax agreeing 96%), and the single pass sits 0.10-0.11 nats
   * from either -- the same kind of number, about twice as large, for 1/L of
   * the work. It also errs high (mean NLL 1.33 vs 1.29 on ubiquitin), so it
   * does not flatter a sequence. See `test/pseudolikelihood.mjs`.
   */
  ALL_BUT_SELF: "all-but-self",
};

export class Encoded {
  constructor(fields) {
    Object.assign(this, fields);
  }
}

export class Model {
  /** @param {import("./weights.js").Weights} weights */
  constructor(weights) {
    this.w = weights;
    this.arena = new Arena();
    this.hidden = weights.hiddenDim;
    this.numLetters = weights.numLetters;
    this.modelType = weights.modelType;
    this.K = weights.kNeighbors;
    this.M = weights.atomContextNum;
    /** See DEFAULT_PREP_BUDGET. Lower it on memory-constrained devices. */
    this.prepBudget = DEFAULT_PREP_BUDGET;

    // `features.*` in the checkpoint; recorded on the weights object so the
    // featuriser can address it without knowing the model variant.
    weights.featurePrefix = "features";

    const a = this.arena;
    this.encoderLayers = [0, 1, 2].map(
      (i) => makeEncoderLayer(weights, a, `encoder_layers.${i}`, this.hidden),
    );
    this.decoderLayers = [0, 1, 2].map(
      (i) => makeDecoderLayer(weights, a, `decoder_layers.${i}`, this.hidden, this.hidden * 3),
    );

    if (this.modelType === "ligand_mpnn") {
      this.contextLayers = [0, 1].map(
        (i) => makeDecoderLayer(
          weights, a, `context_encoder_layers.${i}`, this.hidden, this.hidden * 2, "ctx",
        ),
      );
      this.atomLayers = [0, 1].map(
        (i) => makeDecoderLayer(
          weights, a, `y_context_encoder_layers.${i}`, this.hidden, this.hidden, "atom",
        ),
      );
    }
  }

  get isLigand() {
    return this.modelType === "ligand_mpnn";
  }

  get isMembrane() {
    return this.modelType === "per_residue_label_membrane_mpnn"
      || this.modelType === "global_label_membrane_mpnn";
  }

  /** NA-MPNN: protein, DNA and RNA positions in one graph. See na.js. */
  get isNA() {
    return this.modelType === "na_mpnn";
  }

  /**
   * Letters the sampler may draw when the caller supplies no bias.
   *
   * The reference expresses this as `bias = -1e8·omit_AA + bias_AA` and leaves
   * the sampler itself unrestricted, so this reproduces each model's default
   * `--omit_AA`: "X" everywhere, plus NA-MPNN's legacy RNA tokens, which its
   * `--na_shared_tokens` default folds onto the DNA ones.
   *
   * @returns {Float32Array} [numLetters] of 0 or -1e9
   */
  defaultOmit() {
    if (this._defaultOmit) return this._defaultOmit;
    const V = this.numLetters;
    const omit = new Float32Array(V);
    if (this.isNA) {
      for (const i of [NA_OMIT_UNK, ...NA_OMIT_LEGACY_RNA]) omit[i] = -1e9;
    } else {
      omit[V - 1] = -1e9; // "X"
    }
    this._defaultOmit = omit;
    return omit;
  }

  /**
   * Run the structure encoder.
   *
   * @param {object} inputs
   * @param {Float32Array} inputs.X            [L, 4, 3] backbone N, CA, C, O
   * @param {Float32Array} inputs.mask         [L] 1 where all four atoms exist
   * @param {Int32Array}   inputs.residueIdx   [L] renumbered residue indices
   * @param {Int32Array}   inputs.chainLabels  [L] integer chain id
   * @param {Float32Array} [inputs.ligandXyz]  [A, 3] heteroatom coordinates
   * @param {Int32Array}   [inputs.ligandType] [A] atom-type index
   * @param {Float32Array} [inputs.ligandMask] [A]
   * @param {Int32Array}   [inputs.membraneLabels] [L] 0/1/2, membrane models only
   * @param {boolean}      [inputs.useAtomContext=true]
   * @param {boolean}      [inputs.useSideChains=false] LigandMPNN only: add the
   *   side chains of nearby fixed residues to the atom context. Needs `xyz37`,
   *   `xyz37Mask` and `chainMask`, and makes the encoding depend on `chainMask`.
   * @param {Float32Array} [inputs.xyz37]     [L, 37, 3] all-atom coordinates
   * @param {Float32Array} [inputs.xyz37Mask] [L, 37]
   * @param {Float32Array} [inputs.chainMask] [L] 1 = designed, 0 = fixed
   * @returns {Encoded}
   */
  encode(inputs) {
    const { X, mask, residueIdx, chainLabels } = inputs;
    const L = mask.length;
    const w = this.w;
    const H = this.hidden;

    // NA-MPNN graphs 18 atom slots over protein and nucleic residues alike and
    // centres its neighbour search on CA + C1'; everything else graphs 5 atoms
    // over C-alpha. Both then run the identical encoder.
    const polytype = this.isNA
      ? (inputs.polytype ?? new Int32Array(L).fill(POLYTYPE.PP))
      : null;
    let bb;
    let EIdx;
    let K;
    let E;
    if (this.isNA) {
      bb = naBackbone(inputs.X16, inputs.X16Mask, polytype, L);
      ({ EIdx, K } = neighborGraph(bb.centre, mask, L, this.K));
      E = naEdgeFeatures(w, bb, EIdx, residueIdx, chainLabels, L, K);
    } else {
      bb = computeBackbone(X, L);
      let DNeighbors;
      ({ EIdx, DNeighbors, K } = neighborGraph(bb.CA, mask, L, this.K));
      E = edgeFeatures(w, bb, EIdx, DNeighbors, residueIdx, chainLabels, L, K);
    }

    const wE = w.linear("W_e");
    const hE = linear(E, wE.weight, wE.bias, L * K, wE.shape[1], H);
    let hV = new Float32Array(L * H);

    // Two families seed the node state from a per-residue integer label -- the
    // membrane models from buried/interface/soluble, NA-MPNN from the polymer
    // type. Same embedding, different width. Everything else starts from zero.
    if (this.isNA || this.isMembrane) {
      const labels = this.isNA ? polytype : (inputs.membraneLabels ?? new Int32Array(L));
      const classes = this.isNA ? N_POLYTYPES : 3;
      const V = labelNodeFeatures(w, labels, classes, L, H);
      const wV = w.linear("W_v");
      hV = linear(V, wV.weight, wV.bias, L, wV.shape[1], H);
    }

    const maskAttend = new Float32Array(L * K);
    for (let i = 0; i < L; i++) {
      for (let k = 0; k < K; k++) maskAttend[i * K + k] = mask[i] * mask[EIdx[i * K + k]];
    }

    let ligand = null;
    if (this.isLigand) {
      ligand = this._prepareLigand(inputs, bb, mask, L, EIdx, K);
    }

    for (const layer of this.encoderLayers) layer(hV, hE, EIdx, mask, maskAttend, L, K);

    if (this.isLigand) this._encodeLigandContext(hV, ligand, mask, L);

    return new Encoded({
      hV, hE, EIdx, K, L, mask, bb, residueIdx, chainLabels,
      ligand, model: this,
    });
  }

  /** Nearest context atoms per residue plus their node features. */
  _prepareLigand(inputs, bb, mask, L, EIdx, K) {
    const M = this.M;
    const xyz = inputs.ligandXyz ?? new Float32Array(0);
    const types = inputs.ligandType ?? new Int32Array(0);
    const ligMask = inputs.ligandMask ?? new Float32Array(types.length).fill(1);

    const sideChains = inputs.useSideChains === true && inputs.xyz37 !== undefined;
    let { Y, Yt, Ym, Yg, Dclosest } = nearestLigandAtoms(
      bb.CB, mask, xyz, types, ligMask, L, M);
    let poolSize = types.length;
    if (sideChains) {
      // Fixed residues only, so the pool depends on the design selection --
      // see `sideChainContext`. Positions the caller has not marked are treated
      // as designed, which is the conservative reading.
      const chainMask = inputs.chainMask ?? new Float32Array(L).fill(1);
      ({ Y, Yt, Ym, Yg, Dclosest, poolSize } = sideChainContext(
        bb.CB, mask, EIdx, K, inputs.xyz37, inputs.xyz37Mask, chainMask,
        { Y, Yt, Ym, Yg }, L, M));
    }
    if (inputs.useAtomContext === false) Ym.fill(0);
    const { V } = ligandNodeFeatures(this.w, bb, Y, Yt, L, M);

    const pairs = ligandPairTable(Yg, Y, L, M, poolSize);
    // The first atom of each distinct pair, so its type embedding can be added
    // to the cached message without another lookup table.
    const pairTypeA = new Int32Array(pairs.count);
    for (let i = 0; i < L; i++) {
      for (let m = 0; m < M; m++) {
        for (let n = 0; n < M; n++) {
          pairTypeA[pairs.pairId[(i * M + m) * M + n]] = Yt[i * M + m];
        }
      }
    }
    return {
      Y, Yt, Ym, Yg, Dclosest, V, M, pairs, pairTypeA,
      nAtoms: types.length, sideChains,
    };
  }

  /**
   * Fold the ligand context into the node states.
   *
   * This is where LigandMPNN spends nearly all of its time, because the two
   * atom-context layers message-pass among the M nearest ligand atoms *of every
   * residue*: L x M x M rows, 75625 of them for streptavidin at M = 25. Two
   * observations cut most of that away.
   *
   * First, an atom pair's contribution depends only on which two ligand atoms
   * it is, never on the residue looking at them. Biotin has 16 atoms, so there
   * are 289 distinct pairs including padding, not 75625. The atom-pair edge
   * embedding is pair-pure, and so is the *entire* message of the first layer,
   * because that layer's node input is the atom-type embedding -- also a
   * function of the atom alone. Both are computed once per distinct pair here.
   *
   * Second, the second layer's node input has absorbed a per-residue sum and is
   * no longer pair-pure, so its W2/W3 still run per (residue, atom, atom) --
   * but only over pairs whose mask is 1. When the ligand has fewer atoms than
   * M, the rest of every row is zero padding contributing nothing, which for
   * biotin is 59% of the rows.
   */
  _encodeLigandContext(hV, ligand, mask, L) {
    const w = this.w;
    const H = this.hidden;
    const M = ligand.M;
    const a = this.arena;
    const { pairId, count: nPairs, symOf, symCount } = ligand.pairs;

    const wV = w.linear("W_v");
    const wC = w.linear("W_c");
    const wNodesY = w.linear("W_nodes_y");
    const wEdgesY = w.linear("W_edges_y");
    const vC = w.linear("V_C");
    const vCNorm = w.norm("V_C_norm");

    const hEContext = linear(ligand.V, wV.weight, wV.bias, L * M, H, H);
    const hVC = linear(hV, wC.weight, wC.bias, L, H, H);

    // --- pair-pure quantities, once for the whole structure ----------------
    // LayerNorm(W_y_edges · RBF(d)) then W_edges_y. Everything here is a
    // function of the pair's distance alone, so it runs over the *unordered*
    // pairs -- half the rows, and half the [nPairs, 128] buffers, which is
    // what put a side-chain-context encode of a large complex over the
    // accelerator's memory ceiling.
    const rbf = pairRbf(ligand.pairs.dist, symCount);
    const yEdgesLin = w.linear("features.y_edges");
    const normYEdges = w.norm("features.norm_y_edges");
    let yEdges = linear(rbf, yEdgesLin.weight, yEdgesLin.bias, symCount, 16, H);
    layerNorm(yEdges, normYEdges.gamma, normYEdges.beta, symCount, H, yEdges);
    yEdges = linear(yEdges, wEdgesY.weight, wEdgesY.bias, symCount, H, H);

    // Atom-type node embedding, one row per element rather than per slot.
    const normYNodes = w.norm("features.norm_y_nodes");
    let nodeByType = atomTypeEmbedding(w, "features.y_nodes", H);
    layerNorm(nodeByType, normYNodes.gamma, normYNodes.beta, 120, H, nodeByType);
    nodeByType = linear(nodeByType, wNodesY.weight, wNodesY.bias, 120, H, H);

    // --- layer 0: the whole message is a function of the pair --------------
    const layer0 = this.atomLayers[0];
    const [W0v, W0e] = layer0.blocks;
    const pairMsg0 = a.f32("lig.pairMsg0", nPairs * H);
    {
      // W1 · [node(type of a) ‖ edge(pair)] = W1v · node + W1e · edge. The
      // edge half is distance-pure, so project it over unordered pairs and
      // gather; only the node half knows which of the two atoms is `a`, and
      // that is the only reason this row stays ordered at all.
      const nodePart = linear(nodeByType, W0v, layer0.bias, 120, H, H,
        a.f32("lig.nodePart", 120 * H));
      const edgePart = linear(yEdges, W0e, null, symCount, H, H,
        a.f32("lig.edgePart", symCount * H));
      // Chunked, because `nPairs` is not bounded by anything: side-chain
      // context puts every fixed residue's atoms in the pool, and 6VXX with
      // all 2916 residues fixed reaches 730805 ordered pairs. Materialising
      // W1's pre-activation for all of them is 374 MB here and asks the
      // accelerator to stage three times that -- which is what put the encode
      // over the WASM memory ceiling. The rows are independent, so a fixed
      // window costs nothing but the loop.
      const span = Math.min(nPairs, PAIR_CHUNK);
      const h1 = a.f32("lig.pairH1", span * H);
      for (let p0 = 0; p0 < nPairs; p0 += span) {
        const n = Math.min(span, nPairs - p0);
        for (let p = 0; p < n; p++) {
          const e = symOf[p0 + p] * H;
          const t = ligand.pairTypeA[p0 + p] * H;
          const o = p * H;
          for (let d = 0; d < H; d++) h1[o + d] = edgePart[e + d] + nodePart[t + d];
        }
        layer0.tail(h1.subarray(0, n * H), n,
          pairMsg0.subarray(p0 * H, (p0 + n) * H));
      }
    }

    const layer1 = this.atomLayers[1];
    const [W1v, W1e] = layer1.blocks;
    const pairEdge1 = linear(yEdges, W1e, null, symCount, H, H,
      a.f32("lig.pairEdge1", symCount * H));

    // --- per residue -------------------------------------------------------
    const yNodes = a.f32("lig.yNodes", L * M * H);
    for (let i = 0; i < L * M; i++) {
      yNodes.set(nodeByType.subarray(ligand.Yt[i] * H, (ligand.Yt[i] + 1) * H), i * H);
    }

    const dh = a.f32("lig.dh", L * M * H);
    const yNext = a.f32("lig.yNext", L * M * H);
    const scale = 1 / MESSAGE_SCALE;

    // Layer 0: gather the cached pair messages and sum them.
    dh.fill(0, 0, L * M * H);
    for (let i = 0; i < L; i++) {
      for (let m = 0; m < M; m++) {
        const row = i * M + m;
        if (ligand.Ym[row] === 0) continue;
        const to = row * H;
        for (let n = 0; n < M; n++) {
          if (ligand.Ym[i * M + n] === 0) continue;
          const from = pairId[row * M + n] * H;
          for (let d = 0; d < H; d++) dh[to + d] += pairMsg0[from + d];
        }
        for (let d = 0; d < H; d++) dh[to + d] *= scale;
      }
    }
    layer0.finish(yNodes, dh, ligand.Ym, L * M, yNext);
    yNodes.set(yNext.subarray(0, L * M * H));

    // Context layer 0.
    this._contextStep(0, hVC, hEContext, yNodes, mask, ligand, L, M);

    // Layer 1: node half is per (residue, atom) now, edge half is still cached.
    const pa = linear(yNodes, W1v, layer1.bias, L * M, H, H, a.f32("lig.pa", L * M * H));
    dh.fill(0, 0, L * M * H);
    {
      // Only unmasked pairs contribute, so build a compacted batch of rows.
      const CAP = PAIR_CHUNK;
      const h1 = a.f32("lig.h1", CAP * H);
      const msg = a.f32("lig.msg", CAP * H);
      const owner = new Int32Array(CAP);
      let n1 = 0;
      const flush = () => {
        if (!n1) return;
        layer1.tail(h1, n1, msg);
        for (let r = 0; r < n1; r++) {
          const to = owner[r] * H;
          const from = r * H;
          for (let d = 0; d < H; d++) dh[to + d] += msg[from + d];
        }
        n1 = 0;
      };
      for (let i = 0; i < L; i++) {
        for (let m = 0; m < M; m++) {
          const row = i * M + m;
          if (ligand.Ym[row] === 0) continue;
          const po = row * H;
          for (let n = 0; n < M; n++) {
            if (ligand.Ym[i * M + n] === 0) continue;
            const eo = symOf[pairId[row * M + n]] * H;
            const to = n1 * H;
            for (let d = 0; d < H; d++) h1[to + d] = pa[po + d] + pairEdge1[eo + d];
            owner[n1++] = row;
            if (n1 === CAP) flush();
          }
        }
      }
      flush();
    }
    for (let i = 0; i < L * M * H; i++) dh[i] *= scale;
    layer1.finish(yNodes, dh, ligand.Ym, L * M, yNext);
    yNodes.set(yNext.subarray(0, L * M * H));

    // Context layer 1.
    this._contextStep(1, hVC, hEContext, yNodes, mask, ligand, L, M);

    const projected = linear(hVC, vC.weight, vC.bias, L, H, H);
    layerNorm(projected, vCNorm.gamma, vCNorm.beta, L, H, projected);
    for (let i = 0; i < L * H; i++) hV[i] += projected[i];
  }

  /** One residue-level context layer: h_V_C attends over its ligand atoms. */
  _contextStep(it, hVC, hEContext, yNodes, mask, ligand, L, M) {
    const H = this.hidden;
    const a = this.arena;
    const chunk = 256;
    const cat = a.f32("lig.cat", chunk * M * H * 2);
    const out = a.f32("lig.hVCOut", chunk * H);

    for (let start = 0; start < L; start += chunk) {
      const rows = Math.min(chunk, L - start);
      for (let r = 0; r < rows * M; r++) {
        const src = (start * M + r) * H;
        cat.set(hEContext.subarray(src, src + H), r * 2 * H);
        cat.set(yNodes.subarray(src, src + H), r * 2 * H + H);
      }
      this.contextLayers[it](
        hVC.subarray(start * H, (start + rows) * H), cat,
        mask.subarray(start, start + rows), ligand.Ym.subarray(start * M, (start + rows) * M),
        rows, M, out.subarray(0, rows * H),
      );
      hVC.set(out.subarray(0, rows * H), start * H);
    }
  }

  // -------------------------------------------------------------------------
  // Decoding
  // -------------------------------------------------------------------------

  /**
   * Per-edge autoregressive masks.
   *
   * @returns {{bw: Float32Array, fw: Float32Array}} both [L, K]
   */
  _edgeMasks(enc, ar) {
    const { L, K, EIdx, mask } = enc;
    const bw = this.arena.f32("ar.bw", L * K);
    const fw = this.arena.f32("ar.fw", L * K);

    let rank = null;
    if (ar.type === AR.ORDER) {
      rank = new Int32Array(L);
      for (let s = 0; s < L; s++) rank[ar.order[s]] = s;
    }

    for (let i = 0; i < L; i++) {
      for (let k = 0; k < K; k++) {
        const j = EIdx[i * K + k];
        let visible;
        if (ar.type === AR.NONE) visible = 0;
        else if (ar.type === AR.ALL_BUT_SELF) visible = i === j ? 0 : 1;
        else visible = rank[i] > rank[j] ? 1 : 0;
        bw[i * K + k] = mask[i] * visible;
        fw[i * K + k] = mask[i] * (1 - visible);
      }
    }
    return { bw, fw };
  }

  /** Sequence embedding lookup: hS[i] = W_s.weight[S[i]]. */
  _embedSequence(S, L, out) {
    const table = this.w.get("W_s.weight");
    const H = this.hidden;
    out = out ?? new Float32Array(L * H);
    for (let i = 0; i < L; i++) {
      const src = S[i] * H;
      for (let d = 0; d < H; d++) out[i * H + d] = table[src + d];
    }
    return out;
  }

  /**
   * Per-layer pieces of W1 that do not depend on the sequence, cached on the
   * encoding.
   *
   * W1's input is [h_V_i ‖ h_E_ij ‖ h_S_j ‖ h_V_j]. The h_E block is the only
   * per-edge one, and it never changes -- not between decoder layers, not
   * between decode steps, not across a batch. Projecting it once turns the
   * decoder's dominant [rows·K, 512] × [512, 128] product into a gather.
   *
   * Returns null when the projection would be too large to hold, in which case
   * callers fall back to building the concatenation.
   */
  _decoderPrep(enc) {
    if (enc.prep !== undefined) return enc.prep;
    const { L, K, hE } = enc;
    const H = this.hidden;
    const bytes = this.decoderLayers.length * L * K * H * 4;
    if (bytes > this.prepBudget) {
      enc.prep = null;
      enc.prepSkippedBytes = bytes;
      return null;
    }
    const pe = [];
    const pnEnc = [];
    for (const layer of this.decoderLayers) {
      const [, We, , Wn] = layer.blocks;
      pe.push(linear(hE, We, null, L * K, H, H));
      pnEnc.push(linear(enc.hV, Wn, null, L, H, H));
    }
    enc.prep = { pe, pnEnc };
    return enc.prep;
  }

  /**
   * Teacher-forced decoder: one pass over all positions.
   *
   * @param {Encoded} enc
   * @param {Int32Array} S  [L] amino-acid indices in MPNN alphabet order
   * @param {{type: string, order?: Int32Array}} ar
   * @returns {Float32Array} logits [L, 21]
   */
  score(enc, S, ar = { type: AR.NONE }) {
    const { L, K, EIdx, hE, mask } = enc;
    const H = this.hidden;
    const a = this.arena;
    const edgeDim = H * 3;
    const prep = this._decoderPrep(enc);

    const { bw, fw } = this._edgeMasks(enc, ar);
    const hS = this._embedSequence(S, L, a.f32("dec.hS", L * H));
    const hVEnc = enc.hV;

    const hV = a.f32("dec.hV", L * H);
    hV.set(hVEnc);
    const hVNext = a.f32("dec.hVNext", L * H);

    for (let l = 0; l < this.decoderLayers.length; l++) {
      const layer = this.decoderLayers[l];

      if (prep) {
        // Fused path: W1 · [a ‖ b ‖ c ‖ d] = Σ W1x · x, and three of the four
        // blocks are per-residue, so they are projected once here and gathered
        // onto edges below instead of being multiplied per edge.
        const [Wv, , Ws, Wn] = layer.blocks;
        const pa = linear(hV, Wv, layer.bias, L, H, H, a.f32("dec.pa", L * H));
        const ps = linear(hS, Ws, null, L, H, H, a.f32("dec.ps", L * H));
        const pn = linear(hV, Wn, null, L, H, H, a.f32("dec.pn", L * H));
        const pe = prep.pe[l];
        const pnEnc = prep.pnEnc[l];

        for (let start = 0; start < L; start += DECODE_CHUNK) {
          const rows = Math.min(DECODE_CHUNK, L - start);
          const h1 = a.f32("dec.h1", rows * K * H);
          for (let r = 0; r < rows; r++) {
            const i = start + r;
            const m = mask[i];
            const ao = i * H;
            for (let k = 0; k < K; k++) {
              const j = EIdx[i * K + k];
              const dst = (r * K + k) * H;
              const eo = (i * K + k) * H;
              const b = bw[i * K + k];
              const f = fw[i * K + k];
              const jo = j * H;
              for (let d = 0; d < H; d++) {
                h1[dst + d] = m * pe[eo + d] + pa[ao + d]
                  + b * (ps[jo + d] + pn[jo + d]) + f * pnEnc[jo + d];
              }
            }
          }
          layer.applyPre(
            hV.subarray(start * H, (start + rows) * H), h1,
            mask.subarray(start, start + rows), null, rows, K,
            hVNext.subarray(start * H, (start + rows) * H),
          );
        }
      } else {
        for (let start = 0; start < L; start += DECODE_CHUNK) {
          const rows = Math.min(DECODE_CHUNK, L - start);
          const edges = a.f32("dec.edges", rows * K * edgeDim);
          for (let r = 0; r < rows; r++) {
            const i = start + r;
            for (let k = 0; k < K; k++) {
              const j = EIdx[i * K + k];
              const dst = (r * K + k) * edgeDim;
              const eo = (i * K + k) * H;
              const b = bw[i * K + k];
              const f = fw[i * K + k];
              for (let d = 0; d < H; d++) {
                const e = hE[eo + d];
                edges[dst + d] = (b + f) * e;
                edges[dst + H + d] = b * hS[j * H + d];
                edges[dst + 2 * H + d] = b * hV[j * H + d] + f * hVEnc[j * H + d];
              }
            }
          }
          layer(
            hV.subarray(start * H, (start + rows) * H), edges,
            mask.subarray(start, start + rows), null, rows, K,
            hVNext.subarray(start * H, (start + rows) * H),
          );
        }
      }
      hV.set(hVNext);
    }

    const wOut = this.w.linear("W_out");
    return linear(hV, wOut.weight, wOut.bias, L, H, this.numLetters);
  }

  /**
   * Per-position amino-acid distribution.
   *
   * `mode: AR.NONE` (default) is a single pass and asks "what does the backbone
   * alone want here"; `mode: AR.ALL_BUT_SELF` is the one-pass pseudo-likelihood.
   * `exact: true` instead runs L passes, each putting one position last.
   *
   * That last one is not more correct so much as more expensive. The other L-1
   * positions still have to be decoded in *some* order, and the answer depends
   * on which -- see the note on `AR.ALL_BUT_SELF`. `order` picks it; the
   * default is sequence order.
   *
   * @returns {{logits: Float32Array, probs: Float32Array}} [L, 21] each
   */
  profile(enc, {
    S = null, mode = AR.NONE, exact = false, order: others = null, onProgress = null,
  } = {}) {
    const { L } = enc;
    const V = this.numLetters;
    const seq = S ?? new Int32Array(L).fill(20);

    if (!exact) {
      const logits = this.score(enc, seq, { type: mode });
      return { logits, probs: rowSoftmax(logits, L, V) };
    }

    const logits = new Float32Array(L * V);
    const base = others ?? Int32Array.from({ length: L }, (_, i) => i);
    const order = new Int32Array(L);
    for (let target = 0; target < L; target++) {
      // Decode `target` last so it sees every other position's identity.
      let at = 0;
      for (let i = 0; i < L; i++) if (base[i] !== target) order[at++] = base[i];
      order[L - 1] = target;
      const step = this.score(enc, seq, { type: AR.ORDER, order });
      logits.set(step.subarray(target * V, (target + 1) * V), target * V);
      if (onProgress) onProgress(target + 1, L);
    }
    return { logits, probs: rowSoftmax(logits, L, V) };
  }

  /**
   * Autoregressive sampling.
   *
   * The encoder output is shared across the batch; only the L decode steps are
   * repeated. Each sample gets its own decoding order, so a batch explores
   * genuinely different orders rather than re-rolling one.
   *
   * @param {Encoded} enc
   * @param {object} opts
   * @param {number}  [opts.batch=1]
   * @param {number}  [opts.temperature=0.1]
   * @param {Int32Array} [opts.S]          starting sequence; used at fixed positions
   * @param {Float32Array} [opts.chainMask] [L] 1 = design, 0 = keep from S
   * @param {Float32Array} [opts.bias]      [L, 21] added to logits before sampling
   * @param {number[][]}  [opts.symmetry]   groups of positions forced to share an identity
   * @param {Int32Array}  [opts.order]      explicit decoding order, for replay and tests;
   *                                        symmetry groups are applied on top of it
   * @param {() => number} [opts.rng]       uniform [0,1); defaults to Math.random
   * @returns {{S: Int32Array[], logits: Float32Array[], score: number[], order: Int32Array[]}}
   */
  sample(enc, opts = {}) {
    const { L, K, EIdx, hE, mask } = enc;
    const H = this.hidden;
    const V = this.numLetters;
    const a = this.arena;
    const nLayers = this.decoderLayers.length;
    const edgeDim = H * 3;
    const B = Math.max(1, opts.batch ?? 1);
    const temperature = Math.max(opts.temperature ?? 0.1, 1e-6);
    const rng = opts.rng ?? Math.random;
    const bias = opts.bias ?? null;
    // A caller-supplied bias carries its own omissions, exactly as the
    // reference's `-1e8·omit_AA + bias_AA` does; only fall back to the model's
    // defaults when there is none.
    const omit = bias ? new Float32Array(this.numLetters) : this.defaultOmit();
    const table = this.w.get("W_s.weight");
    const wOut = this.w.linear("W_out");

    const prep = this._decoderPrep(enc);
    const startS = opts.S ?? new Int32Array(L).fill(20);
    const chainMask = new Float32Array(L);
    for (let i = 0; i < L; i++) {
      chainMask[i] = mask[i] * (opts.chainMask ? opts.chainMask[i] : 1);
    }

    const groups = normaliseSymmetry(opts.symmetry, L);
    // With tied positions the batch shares one order, so every sample reaches
    // the same group at the same step and the per-step work stays rectangular.
    const shared = groups.groups.length > 0 || opts.order !== undefined;
    const orders = [];
    for (let b = 0; b < B; b++) {
      if (shared && b > 0) {
        orders.push(orders[0]);
      } else if (opts.order !== undefined) {
        orders.push(groupSteps(opts.order, groups));
      } else {
        orders.push(decodingOrder(chainMask, groups, rng));
      }
    }
    const nSteps = orders[0].length;

    // stepOf[b][pos] -> the step at which sample b decodes pos. Turns the
    // "has j been decoded yet" test into one array read.
    const stepOf = [];
    for (let b = 0; b < B; b++) {
      const s = new Int32Array(L);
      for (let step = 0; step < orders[b].length; step++) {
        for (const t of orders[b][step]) s[t] = step;
      }
      stepOf.push(s);
    }

    const stacks = [];
    const hSs = [];
    const outS = [];
    const outLogits = [];
    for (let b = 0; b < B; b++) {
      const stack = [Float32Array.from(enc.hV)];
      for (let l = 0; l < nLayers; l++) stack.push(new Float32Array(L * H));
      stacks.push(stack);
      hSs.push(new Float32Array(L * H));
      outS.push(Int32Array.from(startS));
      outLogits.push(new Float32Array(L * V));
    }

    const h1 = a.f32("smp.h1", B * K * H);
    const edges = a.f32("smp.edges", B * K * edgeDim);
    const hVRow = a.f32("smp.hV", B * H);
    const hVOut = a.f32("smp.hVOut", B * H);
    const maskRow = a.f32("smp.mask", B);
    const paRow = a.f32("smp.pa", B * H);
    const projBatch = a.f32("smp.projBatch", B * H);
    const logitsBatch = a.f32("smp.logits", B * V);
    const totalLogits = new Float32Array(B * V);
    const vis = new Uint8Array(B * K);
    /** Positions decoded this step, whose W1s projections need refreshing. */
    const pending = [];
    const pendingRows = a.f32("smp.pendingRows", B * 8 * H);
    const pendingOut = a.f32("smp.pendingOut", B * 8 * H);

    // Running projections of the two things that change one row per step.
    // Maintaining them incrementally costs one 128x128 product per step; doing
    // it inside the neighbour loop instead would cost K of them.
    //   psCache[b][l][j] = W1s_l · h_S[j]          (set when j is decoded)
    //   pnCache[b][l][j] = W1n_l · h_V_stack[l][j]
    const psCache = [];
    const pnCache = [];
    if (prep) {
      for (let b = 0; b < B; b++) {
        psCache.push(Array.from({ length: nLayers }, () => new Float32Array(L * H)));
        const perLayer = [];
        // Layer 0's stack starts as the encoder output everywhere, so its
        // projection is the one already precomputed for the whole structure.
        perLayer.push(Float32Array.from(prep.pnEnc[0]));
        for (let l = 1; l < nLayers; l++) perLayer.push(new Float32Array(L * H));
        pnCache.push(perLayer);
      }
    }

    for (let step = 0; step < nSteps; step++) {
      const groupSize = orders[0][step].length;
      totalLogits.fill(0);

      for (let g = 0; g < groupSize; g++) {
        for (let b = 0; b < B; b++) {
          const t = orders[b][step][g];
          maskRow[b] = mask[t];
          const seen = stepOf[b];
          for (let k = 0; k < K; k++) {
            vis[b * K + k] = seen[EIdx[t * K + k]] < step ? 1 : 0;
          }
        }

        if (prep) {
          // Fused: no per-edge matmul at all, just gathers of the four
          // precomputed or cached projections.
          for (let l = 0; l < nLayers; l++) {
            const [Wv] = this.decoderLayers[l].blocks;
            const pe = prep.pe[l];
            const pnEnc = prep.pnEnc[l];
            // Gather the batch's current node states, then project all B rows
            // in one call. B separate n=1 products fall below the accelerator's
            // threshold and land back in JS, which is where they showed up in a
            // profile as ~9% of total runtime.
            for (let b = 0; b < B; b++) {
              const t = orders[b][step][g];
              hVRow.set(stacks[b][l].subarray(t * H, (t + 1) * H), b * H);
            }
            linear(hVRow, Wv, this.decoderLayers[l].bias, B, H, H, paRow);

            for (let b = 0; b < B; b++) {
              const t = orders[b][step][g];
              const m = mask[t];
              const ps = psCache[b][l];
              const pn = pnCache[b][l];
              for (let k = 0; k < K; k++) {
                const j = EIdx[t * K + k];
                const dst = (b * K + k) * H;
                const eo = (t * K + k) * H;
                const jo = j * H;
                const visible = vis[b * K + k];
                const bw = m * visible;
                const fw = m * (1 - visible);
                for (let d = 0; d < H; d++) {
                  h1[dst + d] = m * pe[eo + d] + paRow[b * H + d]
                    + bw * (ps[jo + d] + pn[jo + d]) + fw * pnEnc[jo + d];
                }
              }
            }

            this.decoderLayers[l].applyPre(hVRow, h1, maskRow, null, B, K, hVOut);

            // The next layer reads h_V_stack[l+1] at these positions, so its
            // projection is refreshed now -- for the whole batch at once, since
            // hVOut is already a contiguous [B, H].
            if (l + 1 < nLayers) {
              const [, , , Wn] = this.decoderLayers[l + 1].blocks;
              linear(hVOut, Wn, null, B, H, H, projBatch);
            }
            for (let b = 0; b < B; b++) {
              const t = orders[b][step][g];
              stacks[b][l + 1].set(hVOut.subarray(b * H, (b + 1) * H), t * H);
              if (l + 1 < nLayers) {
                pnCache[b][l + 1].set(projBatch.subarray(b * H, (b + 1) * H), t * H);
              }
            }
          }
        } else {
          for (let b = 0; b < B; b++) {
            const t = orders[b][step][g];
            const hS = hSs[b];
            for (let k = 0; k < K; k++) {
              const j = EIdx[t * K + k];
              const bw = mask[t] * vis[b * K + k];
              const dst = (b * K + k) * edgeDim;
              const eo = (t * K + k) * H;
              for (let d = 0; d < H; d++) {
                edges[dst + d] = mask[t] * hE[eo + d];
                edges[dst + H + d] = bw * hS[j * H + d];
              }
            }
          }
          for (let l = 0; l < nLayers; l++) {
            for (let b = 0; b < B; b++) {
              const t = orders[b][step][g];
              const stack = stacks[b][l];
              const m = mask[t];
              for (let k = 0; k < K; k++) {
                const j = EIdx[t * K + k];
                const src = (vis[b * K + k] ? stack : enc.hV).subarray(j * H, (j + 1) * H);
                const dst = (b * K + k) * edgeDim + 2 * H;
                for (let d = 0; d < H; d++) edges[dst + d] = m * src[d];
              }
              hVRow.set(stack.subarray(t * H, (t + 1) * H), b * H);
            }
            this.decoderLayers[l](hVRow, edges, maskRow, null, B, K, hVOut);
            for (let b = 0; b < B; b++) {
              const t = orders[b][step][g];
              stacks[b][l + 1].set(hVOut.subarray(b * H, (b + 1) * H), t * H);
            }
          }
        }

        for (let b = 0; b < B; b++) {
          const t = orders[b][step][g];
          hVRow.set(stacks[b][nLayers].subarray(t * H, (t + 1) * H), b * H);
        }
        linear(hVRow, wOut.weight, wOut.bias, B, H, V, logitsBatch);
        for (let b = 0; b < B; b++) {
          const t = orders[b][step][g];
          outLogits[b].set(logitsBatch.subarray(b * V, (b + 1) * V), t * V);
          const weight = groups.weightOf(t);
          for (let v = 0; v < V; v++) totalLogits[b * V + v] += weight * logitsBatch[b * V + v];
        }
      }

      // One draw per tied group, broadcast to its members.
      pending.length = 0;
      for (let b = 0; b < B; b++) {
        const members = orders[b][step];
        let best = 0;
        let bestVal = -Infinity;
        // Every letter is a candidate, as in the reference; what may not be
        // drawn is expressed as bias, either the caller's or the model's
        // default omissions. Restricting the loop instead was a latent bug --
        // it silently capped NA-MPNN's 33 letters at the 20 amino acids, so a
        // nucleotide could never be sampled.
        for (let v = 0; v < V; v++) {
          let x = totalLogits[b * V + v] + omit[v];
          // Bias is combined over the whole tied group with the same weights
          // the logits already use, not read from the group's first member.
          // Reading only the first silently ignored a bias set on any other
          // member -- invisible while bias was one row applied everywhere, and
          // a lie as soon as the interface shows bias per position. For a group
          // whose members share a bias row this is identical to before, since
          // the weights sum to 1 under the averaging mode.
          if (bias) {
            for (let k = 0; k < members.length; k++) {
              const t = members[k];
              x += groups.weightOf(t) * bias[t * V + v];
            }
          }
          // Gumbel-max draw: same distribution as softmax(x / T) sampling,
          // without building the distribution.
          const g = -Math.log(-Math.log(Math.max(rng(), 1e-20)));
          const val = x / temperature + g;
          if (val > bestVal) {
            bestVal = val;
            best = v;
          }
        }
        for (const t of orders[b][step]) {
          const aa = chainMask[t] > 0 ? best : startS[t];
          outS[b][t] = aa;
          hSs[b].set(table.subarray(aa * H, (aa + 1) * H), t * H);
          if (prep) pending.push([b, t, aa]);
        }
      }

      if (prep && pending.length) {
        // One product per layer over every position decoded this step, rather
        // than one per (position, layer).
        const n = pending.length;
        const rows = pendingRows.subarray(0, n * H);
        for (let r = 0; r < n; r++) {
          rows.set(table.subarray(pending[r][2] * H, (pending[r][2] + 1) * H), r * H);
        }
        for (let l = 0; l < nLayers; l++) {
          const [, , Ws] = this.decoderLayers[l].blocks;
          const out = pendingOut.subarray(0, n * H);
          linear(rows, Ws, null, n, H, H, out);
          for (let r = 0; r < n; r++) {
            psCache[pending[r][0]][l].set(out.subarray(r * H, (r + 1) * H), pending[r][1] * H);
          }
        }
      }
    }

    return {
      S: outS,
      logits: outLogits,
      score: outS.map((S, b) => sequenceScore(outLogits[b], S, chainMask, L, V)),
      order: orders.map(flattenOrder),
      seq: outS.map((S) => sequenceToString(S)),
    };
  }
}

function flattenOrder(order) {
  const out = [];
  for (const group of order) for (const t of group) out.push(t);
  return Int32Array.from(out);
}

/**
 * Symmetry groups, plus the per-position logit weights the reference applies
 * when several positions are tied.
 */
export function normaliseSymmetry(symmetry, L) {
  const groupOf = new Int32Array(L).fill(-1);
  const groups = [];
  const weights = new Float32Array(L).fill(1);
  if (symmetry) {
    for (const raw of symmetry) {
      const positions = [];
      for (const entry of raw) {
        const pos = typeof entry === "number" ? entry : entry.pos;
        const weight = typeof entry === "number" ? 1 : (entry.weight ?? 1);
        if (pos < 0 || pos >= L || groupOf[pos] !== -1) continue;
        groupOf[pos] = groups.length;
        weights[pos] = weight;
        positions.push(pos);
      }
      if (positions.length) groups.push(positions);
    }
  }
  return {
    groupOf,
    groups,
    weightOf: (i) => weights[i],
  };
}

/**
 * Collapse a flat position order into decoding steps, pulling each tied group
 * forward to where its first member appears.
 *
 * This is the reference's `new_decoding_order` construction: walk the order, and
 * the first time a position belonging to a symmetry group comes up, emit the
 * whole group and skip its other members later.
 */
export function groupSteps(flat, groups) {
  const steps = [];
  const placed = new Uint8Array(groups.groupOf.length);
  for (const t of flat) {
    if (placed[t]) continue;
    const g = groups.groupOf[t];
    const group = g === -1 ? [t] : groups.groups[g];
    for (const p of group) placed[p] = 1;
    steps.push(group);
  }
  return steps;
}

/**
 * A decoding order as a list of steps, each step a list of tied positions.
 * Fixed positions (chainMask 0) are decoded first so designed positions can see
 * them, matching `argsort((chain_mask + 1e-4) * |randn|)`.
 */
function decodingOrder(chainMask, groups, rng) {
  const L = chainMask.length;
  const keys = new Float64Array(L);
  for (let i = 0; i < L; i++) keys[i] = (chainMask[i] + 0.0001) * Math.abs(gaussian(rng));
  const flat = Array.from({ length: L }, (_, i) => i).sort((x, y) => keys[x] - keys[y]);
  return groupSteps(flat, groups);
}

/** Box-Muller, so decoding-order keys match the reference's |randn| shape. */
function gaussian(rng) {
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function rowSoftmax(logits, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    softmax(logits.subarray(i * cols, (i + 1) * cols), out.subarray(i * cols, (i + 1) * cols));
  }
  return out;
}

/**
 * Negative log-likelihood of each residue of `S` under `logits`.
 *
 * @param {Float32Array} logits [L, V]
 * @param {Int32Array} S [L]
 * @param {Float32Array} [out] [L]
 */
export function perPositionNLL(logits, S, L, V, out) {
  out = out ?? new Float32Array(L);
  const lp = new Float32Array(V);
  for (let i = 0; i < L; i++) {
    logSoftmax(logits.subarray(i * V, (i + 1) * V), lp);
    out[i] = -lp[S[i]];
  }
  return out;
}

/** Mean negative log-likelihood of `S` over designed positions. */
export function sequenceScore(logits, S, weight, L, V) {
  let total = 0;
  let n = 0;
  const lp = new Float32Array(V);
  for (let i = 0; i < L; i++) {
    if (weight[i] <= 0) continue;
    logSoftmax(logits.subarray(i * V, (i + 1) * V), lp);
    total += -lp[S[i]];
    n += 1;
  }
  return n === 0 ? 0 : total / n;
}

export function sequenceToString(S) {
  let out = "";
  for (let i = 0; i < S.length; i++) out += ALPHABET[S[i]] ?? "X";
  return out;
}

export { gatherNodes };
