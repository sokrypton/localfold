/**
 * OpenDDE's target-feat ATOM ENCODER, module by module, against af3-any-model.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-opendde-encoder-oracle.js \
 *       --model=/model-opendde-full-f32/manifest.json
 *
 * 🔴 THE FIRST OPENDDE TRUNK ORACLE DISAGREES AT `target_feat` - 9.65e-2 at
 * f32, on a bundle that agrees with the reference's params 481 of 481, so it is
 * the port's code and not the weights. The error lives in EIGHT of the 384
 * encoder channels and the other 439 agree to machine precision, which a
 * whole-tensor residual cannot say. This bisects the encoder: the five per-atom
 * embeddings summed, the transformer stack's output, the projection that feeds
 * the aggregation, and the finished target_feat.
 *
 * The dump is `CAPTURE=evoformer_conditioning BLOCKS=1 dump_af3_trunk_taps.py
 * opendde`, reduced to these scopes - `dump_af3_scopes.py` CANNOT produce them,
 * because its forward pass is the denoiser's and never runs this encoder.
 */
import { af3BatchFromA3m } from "../../src/af3/batch.js";
import { batchFromDump } from "./fold.js";
import { openAf3Store } from "../../src/af3/weights.js";
import { targetFeatureWeights } from "../../src/af3/diffusion-weights.js";
import { perAtomConditioning } from "../../src/af3/atom-conditioning-reference.js";
import { atomCrossAttentionEncoder } from "../../src/af3/atom-encoder-reference.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

// 🔴 MASKED, BECAUSE THE PADDED SLOTS ARE NOT THE SAME TENSOR AND ARE NOT
// SUPPOSED TO BE. The reference's per-atom embeddings are NONZERO on every one
// of the 1632 dense slots - `embed_ref_element` and `embed_ref_atom_name` embed
// index-0 one-hots, which are real vectors - and this port zeroes them. Both
// are then masked out at `mask_mean`, so the fold is identical and an unmasked
// relRMS over the whole tensor is dominated by rows neither model reads. It
// reported 0.466 on the conditioning and 0.137 on the stack while the two
// comparable seams were 0 and 1.79e-7, which cannot both be true and is the
// second time in this file that an impossible pair of residuals was the
// checker's fault rather than the port's.
const maskedRelRms = (ours, expected, rowMask, channels) => {
  let error = 0, scale = 0;
  for (let row = 0; row < rowMask.length; row += 1) {
    if (rowMask[row] === 0) continue;
    for (let c = 0; c < channels; c += 1) {
      const at = row * channels + c;
      const d = ours[at] - expected[at];
      error += d * d; scale += expected[at] * expected[at];
    }
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
};

const relRms = (ours, expected) => {
  let error = 0, scale = 0;
  for (let i = 0; i < expected.length; i += 1) {
    const d = ours[i] - expected[i];
    error += d * d; scale += expected[i] * expected[i];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
};

export async function main(device, args) {
  const sequence = option(args, "sequence",
    "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE");
  const dumpPath = option(args, "oracle", "/oracle-dumps/af3-oracle-encoder-opendde.json");
  const response = await fetch(dumpPath);
  if (!response.ok) throw new Error(`failed to load ${dumpPath}: ${response.status}`);
  const stages = (await response.json()).stages;
  const of = (name) => {
    const entry = stages[name];
    if (entry === undefined) throw new Error(`the dump has no ${name}`);
    return { data: Float32Array.from(entry.data), shape: entry.shape };
  };

  const store = await openAf3Store(option(args, "model",
    "/model-opendde-full-f32/manifest.json"));
  store.prefetch();
  // 🔴 `buildTargetFeat` IS CALLED WITH `weights.targetFeat`, so the per-atom
  // embedding weights it reads as `weights.reference` are
  // `targetFeatureWeights(store).reference` - NOT `atomReference(store)`, which
  // is the DIFFUSION head's reference table and a different tensor set. Passing
  // that one made this checker report the conditioning as 1.19 - near-orthogonal
  // to the reference - while `target_feat` was exact in 439 of its 447 columns,
  // which cannot both be true and is what caught it.
  const targetFeat = await targetFeatureWeights(store);
  const weights = { reference: targetFeat.reference, targetFeat };
  // 🔴 THE REFERENCE'S OWN BATCH, OR THIS CHECK CANNOT ANSWER. Two reasons, and
  // both make a sequence-featurised run meaningless here:
  //   - this port ships one idealised conformer set and the reference
  //     featurises CCD geometry, so `embed_ref_pos` disagrees by construction
  //     and drags the whole conditioning with it (0.26 on the real atoms);
  //   - the featuriser sizes its subsets from the REAL atom count (18 subsets,
  //     576 rows) where the reference keeps the dense grid (51 subsets, 1632),
  //     so `stackOut` and `projectForAggr` could only report a length mismatch.
  // `batchFromDump` pads to the dense grid deliberately, so `--dump=` fixes
  // both at once and what is left is the encoder's arithmetic.
  const dumpBatchPath = option(args, "dump", "");
  const batch = dumpBatchPath === ""
    ? af3BatchFromA3m(sequence, null, {}).batch
    : batchFromDump(await (async () => {
        const r = await fetch(dumpBatchPath);
        if (!r.ok) throw new Error(`failed to load ${dumpBatchPath}: ${r.status}`);
        return r.json();
      })());
  const encoderWeights = weights.targetFeat.encoder;
  const dialect = weights.targetFeat.dialect ?? encoderWeights?.dialect;

  if (args.includes("--raw-cond")) {
    const c = perAtomConditioning({
      positions: batch.refPos, mask: batch.refMask,
      element: batch.refElement, charge: batch.refCharge,
      atomNameChars: batch.refAtomNameChars,
    }, batch.tokens, batch.dense, weights.reference,
      weights.targetFeat.dialect ?? weights.targetFeat.encoder?.dialect);
    return { raw: Array.from(c, (v) => Number(v.toFixed(6))) };
  }
  const arms = [];
  // 1. The five per-atom embeddings, as one sum - which is what our reference
  //    produces and what the encoder consumes.
  const conditioning = perAtomConditioning({
    positions: batch.refPos, mask: batch.refMask,
    element: batch.refElement, charge: batch.refCharge,
    atomNameChars: batch.refAtomNameChars,
  }, batch.tokens, batch.dense, weights.reference, dialect);
  const embedNames = ["pos", "mask", "element", "charge", "atom_name"]
    .map((n) => `scope.evoformer_conditioning_embed_ref_${n}`);
  const summed = new Float32Array(of(embedNames[0]).data.length);
  for (const name of embedNames) {
    const term = of(name).data;
    for (let i = 0; i < summed.length; i += 1) summed[i] += term[i];
  }
  const rms = (a) => Math.sqrt(a.reduce((t, v) => t + v * v, 0) / a.length);
  arms.push({ stage: "perAtomConditioning (5 embeds summed), MASKED",
              relRms: maskedRelRms(conditioning, summed, batch.refMask, 128),
              unmaskedRelRms: Number(relRms(conditioning, summed).toExponential(2)),
              oursRms: Number(rms(conditioning).toFixed(5)),
              nativeRms: Number(rms(summed).toFixed(5)),
              // 🔴 HOW MANY ROWS ARE NONZERO ON EACH SIDE. A padded slot the
              // reference fills and we zero (or the reverse) is a layout
              // difference, not an arithmetic one, and relRMS cannot tell them
              // apart - it just reports "completely different".
              oursNonzeroRows: (() => { let n = 0;
                for (let r = 0; r < conditioning.length / 128; r += 1) {
                  let any = false;
                  for (let c = 0; c < 128; c += 1) if (conditioning[r * 128 + c] !== 0) { any = true; break; }
                  if (any) n += 1; } return n; })(),
              nativeNonzeroRows: (() => { let n = 0;
                for (let r = 0; r < summed.length / 128; r += 1) {
                  let any = false;
                  for (let c = 0; c < 128; c += 1) if (summed[r * 128 + c] !== 0) { any = true; break; }
                  if (any) n += 1; } return n; })(),
              // ...and the residual over the rows BOTH sides call real, which
              // is the arithmetic question with the layout one removed.
              relRmsOnSharedRows: (() => {
                let error = 0, scale = 0;
                for (let r = 0; r < summed.length / 128; r += 1) {
                  let a2 = 0, b2 = 0;
                  for (let c = 0; c < 128; c += 1) {
                    a2 += conditioning[r * 128 + c] ** 2; b2 += summed[r * 128 + c] ** 2;
                  }
                  if (a2 === 0 || b2 === 0) continue;
                  for (let c = 0; c < 128; c += 1) {
                    const d = conditioning[r * 128 + c] - summed[r * 128 + c];
                    error += d * d; scale += summed[r * 128 + c] ** 2;
                  }
                }
                return Number(Math.sqrt(error / Math.max(scale, 1e-30)).toExponential(2));
              })(),
              // 🔴 WHICH rows are real on each side. Packed (0..573) and dense
              // (t*24+slot) are both "574 nonzero rows" and are not the same
              // tensor; relRMS reports 1.2 for either and names neither.
              oursFirstNonzeroRows: (() => { const out = [];
                for (let r = 0; r < conditioning.length / 128 && out.length < 30; r += 1) {
                  for (let c = 0; c < 128; c += 1) {
                    if (conditioning[r * 128 + c] !== 0) { out.push(r); break; } } }
                return out; })(),
              // Row 0 (token 0, atom 0) on both sides, and each native term,
              // because a scale, an offset and a missing summand look the same
              // in a norm and different in eight numbers.
              row0: { ours: Array.from(conditioning.slice(0, 8), (v) => Number(v.toFixed(4))),
                      nativeSum: Array.from(summed.slice(0, 8), (v) => Number(v.toFixed(4))),
                      terms: Object.fromEntries(embedNames.map((n) => [
                        n.replace("scope.evoformer_conditioning_embed_ref_", ""),
                        Array.from(of(n).data.slice(0, 8), (v) => Number(v.toFixed(4)))])) },
              length: conditioning.length, expected: summed.length });
  // ...and each embedding on its own, so a disagreeing sum names its term.
  for (const name of embedNames) {
    const term = of(name);
    arms.push({ stage: name.replace("scope.evoformer_conditioning_", ""),
                nativeRms: Math.sqrt(term.data.reduce((t, v) => t + v * v, 0) / term.data.length),
                shape: term.shape, note: "reference only - our terms are summed in place" });
  }

  // 2. Through the encoder, with its own stage hooks.
  const seen = {};
  const shared = {
    shape: batch.shape, dialect,
    conditioning, atomMask: batch.refMask,
    refPos: batch.refPos, refSpaceUid: batch.refSpaceUid,
    tokenAtomsToQueries: batch.tokenAtomsToQueries,
    queriesToKeys: batch.queriesToKeys,
    queriesToTokenAtoms: batch.queriesToTokenAtoms,
  };
  atomCrossAttentionEncoder(shared, encoderWeights, (name, value) => { seen[name] = value; });
  for (const [ours, native] of [
    ["encoder.pairMlp3", "scope.evoformer_conditioning_pair_mlp_3"],
    ["encoder.stackOut", "scope.evoformer_conditioning_atom_transformer_encoder"],
    ["encoder.projectForAggr", "scope.evoformer_conditioning_project_atom_features_for_aggr"],
  ]) {
    const mine = seen[ours];
    const theirs = of(native);
    if (mine === undefined) { arms.push({ stage: ours, missing: true }); continue; }
    if (mine.length !== theirs.data.length) {
      arms.push({ stage: ours, lengthMismatch: [mine.length, theirs.data.length] });
      continue;
    }
    // The stack's rows are QUERIES, so its mask is the atom mask gathered
    // through `tokenAtomsToQueries` - the same convert the encoder does.
    const channels = theirs.data.length / (theirs.shape[0] * theirs.shape[1]);
    const rows = mine.length / channels;
    const queryMask = new Float32Array(rows);
    for (let q = 0; q < rows && q < batch.tokenAtomsToQueries.indices.length; q += 1) {
      const at = batch.tokenAtomsToQueries.indices[q];
      queryMask[q] = (batch.tokenAtomsToQueries.mask?.[q] ?? 1) === 0 ? 0
        : (batch.refMask[at] ?? 0);
    }
    arms.push({ stage: `${ours}, MASKED`,
                relRms: maskedRelRms(mine, theirs.data, queryMask, channels),
                unmaskedRelRms: Number(relRms(mine, theirs.data).toExponential(2)),
                rowsCompared: queryMask.reduce((t, v) => t + (v !== 0 ? 1 : 0), 0),
                shape: theirs.shape });
  }
  const worst = arms.filter((a) => a.relRms !== undefined)
    .reduce((w, a) => (w === null || a.relRms > w.relRms ? a : w), null);
  return {
    sequence: sequence.length, tokens: batch.tokens, dense: batch.dense,
    arms: arms.map((a) => (a.relRms === undefined ? a
      : { ...a, relRms: Number(a.relRms.toExponential(2)) })),
    // The FIRST stage past the bound is the one to read; everything after it
    // inherits.
    firstBad: arms.find((a) => a.relRms !== undefined && a.relRms > 1e-4)?.stage ?? null,
    worst: worst === null ? null : { stage: worst.stage, relRms: worst.relRms.toExponential(2) },
  };
}
