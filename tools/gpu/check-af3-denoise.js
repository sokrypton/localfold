/**
 * L3: one whole denoise step, against af3-any-model's own DiffusionModule.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-denoise.js \
 *       --model=/model-protenix2-f32/manifest.json
 *
 * 🔴 WHY THIS LEVEL AND NOT ANOTHER. Every constant found in the diffusion path
 * so far was caught by a SHAPE that threw - a 256-wide pair read through a
 * 128-wide stride, a trunk pair embedded at the wrong width. protenix2's last
 * defect cannot be caught that way: every tensor is the right shape and only
 * the ANSWER is wrong, and it shows up as a folded chain whose bonds are 0.73x
 * ideal while its CA-CA spacing is 0.93x. Nothing throws.
 *
 * This runs conditioning, atom encoder, token transformer, atom decoder and the
 * EDM scaling at once, against the reference's own `denoise_parity.ours` - the
 * level its README calls "the one that subsumes the others on the diffusion
 * side". A match here means the whole score model agrees for one step.
 *
 * 🔴 AND THE INPUTS ARE THE REFERENCE'S, NOT REBUILT. The atom windows and the
 * relative-position features cannot be synthesised, and rebuilding the trunk
 * activations on this side would compare two featurisers rather than two score
 * models. `tools/oracle/dump_af3_denoise.py` records what native was handed.
 */
import { Af3DiffusionHeadGpu } from "../../src/af3/diffusion/diffusion-head-webgpu.js";
import {
  diffusionConditioning, diffusionHead, diffusionTransformer,
  atomDecoder, scalings,
} from "../../src/af3/diffusion/diffusion-reference.js";
import { Af3DiffusionTransformerGpu } from "../../src/af3/diffusion/diffusion-transformer-webgpu.js";
import { Af3AtomDecoderGpu } from "../../src/af3/diffusion/atom-decoder-webgpu.js";
import { Af3DiffusionConditioningGpu } from "../../src/af3/diffusion/diffusion-conditioning-webgpu.js";
import { atomCrossAttentionEncoder as encodeCpu } from "../../src/af3/diffusion/atom-encoder-reference.js";
import { Af3AtomEncoderGpu } from "../../src/af3/diffusion/atom-encoder-webgpu.js";
import { layerNormSlow } from "../../src/af3/diffusion/atom-encoder-reference.js";
import { linear } from "../../src/af3/trunk/pairformer-reference.js";
import { af3Dialect, openAf3Store } from "../../src/af3/weights/weights.js";
import { perAtomConditioning } from "../../src/af3/diffusion/atom-conditioning-reference.js";
import { denseBondGeometry } from "./bond-geometry.js";
import { atomReference, diffusionWeights } from "../../src/af3/weights/diffusion-weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function relativeRms(actual, expected) {
  let error = 0, scale = 0;
  for (let i = 0; i < expected.length; i += 1) {
    const d = actual[i] - expected[i];
    error += d * d; scale += expected[i] * expected[i];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
}

export async function main(device, args) {
  const model = option(args, "model", "/model-protenix2-f32/manifest.json");
  const dumpPath = option(args, "dump",
    `/oracle-dumps/af3-oracle-denoise-${option(args, "name", "protenix2")}.json`);
  const response = await fetch(dumpPath);
  if (!response.ok) throw new Error(`failed to load ${dumpPath}: ${response.status}`);
  const dump = await response.json();
  const raw = (name) => Float32Array.from(dump.inputs[name].data);
  const ints = (name) => Int32Array.from(dump.inputs[name].data, (v) => Math.round(v));
  const gather = (name) => ({
    indices: ints(`${name}:gather_idxs`), mask: raw(`${name}:gather_mask`),
    count: dump.inputs[`${name}:gather_idxs`].shape.reduce((a, b) => a * b, 1),
  });

  const store = await openAf3Store(model);
  // 🔴 `--flags=key=false,...` OVERRIDES THE DIALECT ON BOTH SIDES, which is
  // what makes it a bisect rather than a change. If the GPU mishandles one
  // convention, turning that convention off - where both implementations agree
  // - collapses the GPU-vs-CPU gap and names it in one run.
  const overrides = Object.fromEntries(option(args, "flags", "").split(",")
    .filter(Boolean).map((pair) => {
      const at = pair.indexOf("=");
      if (at < 0) throw new Error(`--flags wants key=value, got ${pair}`);
      return [pair.slice(0, at), pair.slice(at + 1) === "true"];
    }));
  const dialect = { ...af3Dialect(store), ...overrides };
  // --supers= truncates the token transformer on BOTH sides, which is how the
  // error is attributed: if it grows with depth the transformer owns it, and if
  // it is flat the stages after it do.
  const weights = await diffusionWeights(store, Number(option(args, "supers", "6")));
  const tokens = dump.tokens;
  const dense = dump.maxAtoms;

  const residueIndex = new Int32Array(tokens);
  for (let t = 0; t < tokens; t += 1) residueIndex[t] = t;
  // 🔴 THE SHAPE IS A SUB-OBJECT AND THE ATOM WINDOW COUNTS COME OFF THE
  // GATHERS. `subsets`, `queries` and `keys` are the windowed attention's
  // layout - (subsets, queries) and (subsets, keys) - and the dump's gather
  // shapes state all three, so none of them is typed in.
  const qShape = dump.inputs["token_atoms_to_queries:gather_idxs"].shape;
  const kShape = dump.inputs["queries_to_keys:gather_idxs"].shape;
  // 🔴 THE PER-ATOM CONDITIONING IS AN INPUT HERE AND IS COMPUTED INSIDE THE
  // MODULE THERE, so it is built on this side from the SAME reference-conformer
  // numbers native was handed. Rebuilding it from a different conformer set
  // would compare two featurisers rather than two score models.
  const conditioning = perAtomConditioning({
    positions: raw("ref_pos"), mask: raw("ref_mask"),
    element: ints("ref_element"), charge: raw("ref_charge"),
    atomNameChars: ints("ref_atom_name_chars"),
    // ...and the diffusion head's OWN reference-embedding weights, which are a
    // separate scope from the encoder's blocks: `diffusion_embed_ref_*` under
    // the head rather than `evoformer_conditioning_embed_ref_*`.
  }, tokens, dense, await atomReference(store), dialect);

  const input = {
    tokens, dense, dialect, conditioning,
    shape: { tokens, dense, subsets: qShape[0], queries: qShape[1], keys: kShape[1] },
    noiseLevel: dump.noise,
    positionsNoisy: raw("posNoisy"),
    seqMask: raw("seq_mask"),
    trunkSingle: raw("single"), trunkPair: raw("pair"),
    targetFeat: raw("sInputs"),
    atomMask: raw("atomMask"),
    refPos: raw("ref_pos"), refSpaceUid: ints("ref_space_uid"),
    predDenseAtomMask: raw("atomMask"),
    tokenAtomsToQueries: gather("token_atoms_to_queries"),
    queriesToKeys: gather("queries_to_keys"),
    queriesToTokenAtoms: gather("queries_to_token_atoms"),
    tokensToQueries: gather("tokens_to_queries"),
    tokensToKeys: gather("tokens_to_keys"),
    features: { residueIndex, tokenIndex: residueIndex,
                asymId: new Int32Array(tokens).fill(1),
                entityId: new Int32Array(tokens).fill(1),
                symId: new Int32Array(tokens) },
  };

  const expected = Float32Array.from(dump.output.data);
  const rms = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
  // 🔴 THE REFERENCE'S OWN SEAMS, when a stage dump is beside the answer.
  // Every per-stage arm below this is a GPU-against-CPU differential - it says
  // the port agrees with ITSELF and cannot say where it stops agreeing with the
  // reference, which is useless exactly when the whole step is uncorrelated.
  // `dump_af3_denoise_stages.py` traces af3-any-model's four seams; with it,
  // the FIRST stage over bound is the defect and everything after it is that
  // stage's error carried forward.
  const stagePath = option(args, "stage-dump",
    `/oracle-dumps/af3-oracle-stages-${option(args, "name", "protenix2")}.json`);
  const stageResponse = await fetch(stagePath);
  const native = stageResponse.ok ? (await stageResponse.json()).stages : null;
  if (native === null) console.log(`  (no stage dump at ${stagePath})`);
  // 🔴 AND THE MODULES INSIDE A SEAM, when a scope dump is beside it.
  // `dump_af3_scopes.py` traces every hk.Module output, so a conditioning that
  // is 1.65x too large can be attributed to its initial projection, its noise
  // embedding or one of its two transitions rather than to "the conditioning".
  // The map is here rather than in the reference because the reference's stage
  // names are this port's and the scope names are haiku's.
  const scopePath = option(args, "scope-dump",
    `/oracle-dumps/af3-oracle-scopes-${option(args, "name", "protenix2")}.json`);
  const scopeResponse = await fetch(scopePath);
  const scopes = scopeResponse.ok ? (await scopeResponse.json()).scopes : null;
  const SCOPE_OF = {
    "conditioning.pairInitial": "diffusion_head/pair_cond_initial_projection",
    "conditioning.singleInitial": "diffusion_head/single_cond_initial_projection",
    "encoder.pairCondRow": "diffusion_head/diffusion_single_to_pair_cond_row",
    "encoder.pairCondCol": "diffusion_head/diffusion_single_to_pair_cond_col",
    "encoder.trunkPair": "diffusion_head/diffusion_embed_trunk_pair_cond",
    "encoder.pairMlp3": "diffusion_head/diffusion_pair_mlp_3",
    "encoder.projectForAggr": "diffusion_head/diffusion_project_atom_features_for_aggr",
    "encoder.embedPairOffsets": "diffusion_head/diffusion_embed_pair_offsets",
    "encoder.embedPairDistances": "diffusion_head/diffusion_embed_pair_distances",
    "encoder.embedPairOffsetsValid": "diffusion_head/diffusion_embed_pair_offsets_valid",
    "encoder.pairMlp1": "diffusion_head/diffusion_pair_mlp_1",
    "encoder.pairMlp2": "diffusion_head/diffusion_pair_mlp_2",
    "encoder.pairLogits":
      "diffusion_head/diffusion_atom_transformer_encoder/pair_logits_projection",
    "encoder.stackOut": "diffusion_head/diffusion_atom_transformer_encoder",
    "encoder.tokenAct": "encoder.tokenAct",
    "encoder.skipConnection": "encoder.skipConnection",
  };
  const against = (label, ours) => {
    const entry = native?.[label] ?? scopes?.[SCOPE_OF[label] ?? label];
    if (entry === undefined || ours === undefined) return;
    if (entry.data === undefined) {
      // 🔴 rms ALONE WHERE THE TENSOR WAS NOT CAPTURED. Weak - two different
      // tensors can share an rms - but it costs nothing and the scope dump
      // records it for all ~124 modules, so a term that is the wrong SIZE is
      // named without a second 40 MB round trip.
      console.log(`  native ${label}\t(rms only)\tours ${rms(ours).toFixed(4)}`
        + `\tnative ${entry.rms.toFixed(4)}`);
      return;
    }
    const want = Float32Array.from(entry.data);
    if (want.length !== ours.length) {
      console.log(`  native ${label}\tLENGTH ${ours.length} vs ${want.length}`);
      return;
    }
    const score = relativeRms(ours, want);
    let extra = "";
    // 🔴 WHERE THE ERROR SITS, NOT ONLY HOW BIG IT IS. An rms that matches to
    // four decimals beside a relRMS of 4.5e-3 is a MISPLACED tensor, not a
    // rescaled one - so a checker that prints only the ratio cannot tell a
    // gather index from a rounding mode. `spread` is the fraction of elements
    // carrying a tenth of the worst difference: near 1 is diffuse (arithmetic),
    // near 0 is a handful of entries (an index, a mask, a layout).
    if (score > 1e-5) {
      let worst = 0;
      for (let i = 0; i < want.length; i += 1) {
        const d = Math.abs(ours[i] - want[i]); if (d > worst) worst = d;
      }
      let hot = 0;
      for (let i = 0; i < want.length; i += 1) {
        if (Math.abs(ours[i] - want[i]) > worst * 0.1) hot += 1;
      }
      extra = `\tmax|d| ${worst.toExponential(2)} spread ${(hot / want.length).toExponential(1)}`;
      // 🔴 AND WHICH ROWS, when the error is concentrated. A `spread` of 1.2e-2
      // over a (tokens, channels) tensor is one token's worth of elements, and
      // "one token" and "every token a little" are completely different faults.
      if (hot / want.length < 0.2 && want.length % tokens === 0) {
        const width = want.length / tokens;
        const rows = [];
        for (let t = 0; t < tokens; t += 1) {
          let row = 0;
          for (let c = 0; c < width; c += 1) {
            const d = Math.abs(ours[t * width + c] - want[t * width + c]);
            if (d > row) row = d;
          }
          if (row > worst * 0.1) rows.push(t);
        }
        extra += `\trows ${rows.length}${rows.length <= 8 ? ` [${rows}]` : ""}`;
      }
    }
    console.log(`  native ${label}\t${score.toExponential(2)}`
      + `\tours rms ${rms(ours).toFixed(4)}\tnative rms ${rms(want).toFixed(4)}${extra}`);
  };


  const results = {};
  // Our own `act` at the transformer's input, kept so the arms below can
  // interpolate between it and the reference's.
  let cpuAct = null;
  if (option(args, "cpu", "on") !== "off") {
    // 🔴 THE STAGES COME OUT OF THE SHIPPED HEAD, NOT OUT OF A COPY OF IT. The
    // per-stage arms used to rebuild the pipeline here, and the rebuild fed the
    // atom encoder the RAW trunk pair where the head feeds it the CONDITIONING
    // pair - so `encoder.trunkPair` read 1.13e+0 on a model whose encoder is
    // right, and would have read the same on one whose encoder is wrong.
    const cpu = diffusionHead(input, weights, encodeCpu, (label, value) => {
      if (label === "transformer.act") cpuAct = Float32Array.from(value);
      against(label, value);
    });
    results.cpu = relativeRms(cpu, expected);
    console.log(`denoise CPU\ttokens=${tokens} noise=${dump.noise}`
      + `\trelRMS ${results.cpu.toExponential(2)}`
      + `\tours rms ${rms(cpu).toFixed(4)}\tnative rms ${rms(expected).toFixed(4)}`);
  }
  // 🔴 THE ATOM ENCODER ON ITS OWN, because "the whole score model differs by
  // 4.5e-1" names no stage. The CPU path agrees with native to 1.65e-6 on this
  // very input, so anything the GPU disagrees with the CPU about IS the defect
  // - and the encoder is the first stage that reads the conditioning pair at
  // this model's width.
  if (option(args, "stages", "on") !== "off") {
    const shared = {
      shape: input.shape, dialect, conditioning,
      atomMask: input.atomMask, refPos: input.refPos, refSpaceUid: input.refSpaceUid,
      tokenAtomsToQueries: input.tokenAtomsToQueries,
      queriesToKeys: input.queriesToKeys,
      queriesToTokenAtoms: input.queriesToTokenAtoms,
      tokensToQueries: input.tokensToQueries,
      tokensToKeys: input.tokensToKeys,
      tokenAtomsAct: new Float32Array(tokens * dense * 3),
      trunkSingleCond: input.trunkSingle,
      trunkPairCond: raw("pair"),
    };
    const cpuEnc = encodeCpu(shared, weights.encoder,
      (label, value) => against(label, value));
    against("encoder.tokenAct", cpuEnc.tokenAct);
    against("encoder.skipConnection", cpuEnc.skipConnection);
    const gpuEnc = await new Af3AtomEncoderGpu(device).run(shared, weights.encoder, {});
    for (const key of ["tokenAct", "skipConnection"]) {
      if (cpuEnc[key] === undefined || gpuEnc[key] === undefined) continue;
      console.log(`  atom-encoder ${key}\t`
        + `${relativeRms(gpuEnc[key], cpuEnc[key]).toExponential(2)}`);
    }

    // ...and the TOKEN TRANSFORMER, which is the first stage that reads the
    // conditioning pair at this model's own width (256 here, 128 under AF3).
    const cond = diffusionConditioning(input, weights.conditioning,
      (label, value) => against(label, value));
    // 🔴 THE REAL ACTIVATION, NOT A SYNTHETIC ONE. A made-up `act` made this arm
    // useless: AlphaFold 3 scored 3.83e-2 on it while its whole head agrees to
    // 9.87e-4, so the number said nothing about either. This is what the head
    // actually hands the transformer - the encoder's tokenAct plus the
    // normalised, projected conditioning single.
    const projected = linear(
      layerNormSlow(cond.single, tokens, weights.seqChannels,
                    weights.singleCondEmbeddingNormScale, null),
      tokens, weights.seqChannels, weights.perTokenChannels,
      weights.singleCondEmbeddingProjection);
    const act = Float32Array.from(cpuEnc.tokenAct);
    for (let i = 0; i < act.length; i += 1) act[i] += projected[i];
    against("conditioning.single", cond.single);
    against("conditioning.pair", cond.pair);
    against("transformer.act", act);
    const cpuTx = diffusionTransformer(act, cond.single, cond.pair, input.seqMask,
                                       tokens, weights.transformer);
    against("transformer.out", cpuTx);
    const gpuTx = await new Af3DiffusionTransformerGpu(device).run(
      act, cond.single, cond.pair, input.seqMask, tokens, weights.transformer, {});
    console.log(`  token-transformer\t`
      + `${relativeRms(gpuTx.output ?? gpuTx, cpuTx).toExponential(2)}`);

    // ...and the ATOM DECODER, the last stage and the one the fold's geometry
    // already pointed at: per-atom offsets compressed to 0.73x while the token
    // centres they hang off stayed at 0.93x.
    // --dec-blocks= truncates the decoder stack on BOTH sides: if the error
    // grows with depth its cross-attention blocks own it, and if it is flat the
    // pair logits, the broadcast projection or the position update do.
    const decBlocks = Number(option(args, "dec-blocks", String(weights.decoder.blocks.length)));
    const decWeights = { ...weights.decoder,
                         blocks: weights.decoder.blocks.slice(0, decBlocks) };
    const cpuDec = atomDecoder(cpuTx, cpuEnc, { ...shared, shape: input.shape },
                               decWeights);
    against("decoder.update", ArrayBuffer.isView(cpuDec) ? cpuDec
            : (cpuDec?.output ?? cpuDec?.positions ?? cpuDec?.update));
    const gpuDec = await new Af3AtomDecoderGpu(device).run(
      cpuTx, cpuEnc, { ...shared, shape: input.shape }, decWeights, {});
    const pick = (v) => (ArrayBuffer.isView(v) ? v : (v?.output ?? v?.positions ?? v?.update));
    const g = pick(gpuDec); const c = pick(cpuDec);
    if (!ArrayBuffer.isView(g) || !ArrayBuffer.isView(c)) {
      console.log(`  atom-decoder\t\tgpu keys ${Object.keys(gpuDec ?? {})}`
        + ` cpu keys ${Object.keys(cpuDec ?? {})}`);
    } else {
      let bad = 0;
      for (let i = 0; i < g.length; i += 1) if (!Number.isFinite(g[i])) bad += 1;
      console.log(`  atom-decoder\t\t${relativeRms(g, c).toExponential(2)}`
        + `\tgpu rms ${rms(g).toFixed(4)} (${bad} non-finite)\tcpu rms ${rms(c).toFixed(4)}`);
    }
  }

  // 🔴 THE CONDITIONING'S FULL OUTPUT, TRANSITIONS INCLUDED.
  // check-af3-diffusion-conditioning holds its INITIAL projection (1.89e-7 here)
  // and the encoder arm above was fed the CPU's pair, so a conditioning whose
  // two transitions diverge would pass both and still poison everything
  // downstream of it.
  if (option(args, "stages", "on") !== "off") {
    const cpuCond = diffusionConditioning(input, weights.conditioning);
    const gpuCond = await new Af3DiffusionConditioningGpu(device)
      .run(input, weights.conditioning, {});
    console.log(`  conditioning pair\t${relativeRms(gpuCond.pair, cpuCond.pair).toExponential(2)}`
      + `\tsingle ${relativeRms(gpuCond.single, cpuCond.single).toExponential(2)}`);
  }

  // 🔴 THE TOKEN TRANSFORMER FED THE REFERENCE'S OWN INPUT. A deep residual
  // stack AMPLIFIES whatever it is handed - boltz2 enters it at 1.87e-4 and
  // leaves at 2.15e-1, and both numbers are consistent with a transformer that
  // is exactly right and a 1.33x-per-block growth. Re-running it on native's
  // `transformer.act` separates the two readings in one arm: still 2e-1 means
  // the blocks are wrong, ~1e-5 means everything after the atom encoder is
  // right and the defect is upstream.
  if (native?.["transformer.act"] !== undefined
      && native?.["transformer.out"] !== undefined
      && option(args, "tx-from-native", "on") !== "off") {
    const cond = diffusionConditioning(input, weights.conditioning);
    const theirAct = Float32Array.from(native["transformer.act"].data);
    const ours = diffusionTransformer(theirAct, cond.single, cond.pair, input.seqMask,
                                      tokens, weights.transformer);
    const want = Float32Array.from(native["transformer.out"].data);
    console.log(`  transformer from native act\t${relativeRms(ours, want).toExponential(2)}`
      + `\tours rms ${rms(ours).toFixed(4)}\tnative rms ${rms(want).toFixed(4)}`);
    // 🔴 AND THE ENVELOPE THAT SAYS WHAT THAT NUMBER IS WORTH. A deep residual
    // stack with no normalisation between blocks amplifies its input, and how
    // much is a property of the WEIGHTS - so "the transformer disagrees by X"
    // is meaningless without knowing what a float32 rounding of its input is
    // worth at its output. The same act perturbed by 1e-6 relative, run against
    // itself: our reference accumulates in doubles and the oracle in float32,
    // so an agreement at or below this envelope is as close as the two
    // arithmetics can come.
    const probe = Float32Array.from(theirAct);
    let seed = 1;
    for (let i = 0; i < probe.length; i += 1) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      probe[i] += theirAct[i] * 1e-6 * ((seed / 0x7fffffff) * 2 - 1);
    }
    const shaken = diffusionTransformer(probe, cond.single, cond.pair, input.seqMask,
                                        tokens, weights.transformer);
    // ...and whether the gap is LINEAR in that perturbation, which separates an
    // ill-conditioned map from a discrete decision taken differently. Half the
    // actual input difference should give half the output difference if the
    // stack is merely amplifying; anything else is a branch.
    if (cpuAct !== null) {
      const half = Float32Array.from(theirAct);
      for (let i = 0; i < half.length; i += 1) half[i] += 0.5 * (cpuAct[i] - theirAct[i]);
      const halfway = diffusionTransformer(half, cond.single, cond.pair, input.seqMask,
                                           tokens, weights.transformer);
      console.log(`  transformer half-step\t${relativeRms(halfway, ours).toExponential(2)}`
        + `  (half the input gap; linear amplification halves the output gap)`);
    }
    console.log(`  transformer 1e-6 envelope\t`
      + `${relativeRms(shaken, ours).toExponential(2)}`
      + `  (what one float32 ulp on its input is worth at its output)`);
  }

  // 🔴 AND THE PURE INTERNAL DIFFERENTIAL, WHICH NEEDS NO ORACLE AT ALL. The
  // GPU head and the CPU head take the same inputs and the same weights, so any
  // gap between them is this port disagreeing with itself - no featurisation
  // convention, no dump, no alphabet. It is the number to read first when the
  // oracle arms disagree about whether the harness or the port is at fault.
  const cpuHead = results.cpu !== undefined
    ? diffusionHead(input, weights, encodeCpu) : undefined;
  const gpu = await new Af3DiffusionHeadGpu(device).run(input, weights, {});
  if (cpuHead !== undefined) {
    // 🔴 THE EDM SCALINGS, which multiply the whole output. skip and out are
    // functions of the noise level and sigma_data alone, so a difference here
    // is a different schedule and would scale every coordinate at once - which
    // is what a fold with uniformly compressed bonds looks like.
    const want = scalings(dump.noise);
    console.log(`  scalings gpu skip ${gpu.scalings.skip.toFixed(6)}`
      + ` out ${gpu.scalings.out.toFixed(6)} input ${gpu.scalings.input.toFixed(6)}`
      + `  |  cpu skip ${want.skip.toFixed(6)} out ${want.out.toFixed(6)}`
      + ` input ${want.input.toFixed(6)}`);
    results.gpuVsCpu = relativeRms(gpu.positions, cpuHead);
    console.log(`  GPU vs CPU head\t${results.gpuVsCpu.toExponential(2)}`
      + `  (same inputs, same weights)`);
  }
  results.gpu = relativeRms(gpu.positions, expected);
  console.log(`denoise GPU\ttokens=${tokens} noise=${dump.noise}`
    + `\trelRMS ${results.gpu.toExponential(2)}`
    + `\tours rms ${rms(gpu.positions).toFixed(4)}\tnative rms ${rms(expected).toFixed(4)}`);

  // 🔴 AND THE ANSWER AS CHEMISTRY, WHICH A relRMS CANNOT SEE. Two denoiser
  // outputs agreeing to 1e-5 is a statement about the arithmetic; whether
  // either of them is a molecule is a different question, and it is the one
  // that separated a port folding 6MRR to 0.65 A with side-chain bonds at
  // 0.339 A rms from a reference at 0.051 on the same sequence. The ideal
  // length is `ref_pos`'s own, so this scores native's output and this port's
  // by one rule. `clean` is the structure the noisy input was built from, when
  // the dump carries one - the floor both sides are aiming at.
  if (args.includes("--bonds")) {
    const layout = { refPos: Float32Array.from(dump.inputs.ref_pos.data),
                     refMask: Float32Array.from(dump.inputs.ref_mask.data),
                     nameChars: Float32Array.from(dump.inputs.ref_atom_name_chars.data),
                     tokens, maxAtoms: dump.maxAtoms };
    const show = (label, flat) => {
      const r = denseBondGeometry(flat, layout);
      console.log(`  bonds ${label.padEnd(16)} mainchain rms ${r.mainchain.rms.toFixed(4)}`
        + `  sidechain rms ${r.sidechain.rms.toFixed(4)}`
        + `  mean ${r.sidechain.mean.toFixed(4)}`
        + `  short ${(100 * r.sidechain.short / r.sidechain.bonds).toFixed(0)}%`
        + `  (n=${r.sidechain.bonds})`);
      return { mainchain: Number(r.mainchain.rms.toFixed(4)),
               sidechain: Number(r.sidechain.rms.toFixed(4)),
               sidechainMean: Number(r.sidechain.mean.toFixed(4)) };
    };
    results.bonds = {
      ...(dump.inputs.clean === undefined ? {}
        : { clean: show("clean input", Float32Array.from(dump.inputs.clean.data)) }),
      noisy: show("noisy input", Float32Array.from(dump.inputs.posNoisy.data)),
      native: show("native D", expected),
      ours: show("our D", gpu.positions),
    };
  }

  // 🔴 AND THE ARITHMETIC ENVELOPE, because for one of these models a constant
  // bound is the wrong question. boltz2's 24-block token transformer amplifies
  // its input by ~2.2e4 - measured, and LINEAR: half the input gap gives half
  // the output gap - so this reference accumulating in doubles and the oracle
  // accumulating in float32 cannot agree to better than about 1e-2 however
  // right the port is. At f32 the GPU reads 3.50e-3 against native where this
  // f64 CPU reads 2.20e-2, which is that effect and not a defect.
  //
  // The probe is one float32 ulp on every input, run against ourselves. 1e-7 is
  // the relative size of that ulp; the perturbation is deterministic so the
  // number is comparable between runs.
  if (results.cpu !== undefined && option(args, "envelope", "on") !== "off") {
    let seed = 7;
    const shake = (source) => {
      if (source === undefined) return source;
      const copy = Float32Array.from(source);
      for (let i = 0; i < copy.length; i += 1) {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        copy[i] += source[i] * 1e-7 * ((seed / 0x7fffffff) * 2 - 1);
      }
      return copy;
    };
    const probed = { ...input,
      trunkSingle: shake(input.trunkSingle), trunkPair: shake(input.trunkPair),
      targetFeat: shake(input.targetFeat), positionsNoisy: shake(input.positionsNoisy) };
    const shaken = diffusionHead(probed, weights, encodeCpu);
    const reference = diffusionHead(input, weights, encodeCpu);
    results.envelope = relativeRms(shaken, reference);
    console.log(`  arithmetic envelope\t${results.envelope.toExponential(2)}`
      + `  (one float32 ulp on every input, this reference against itself)`);
  }

  // 🔴 A SEPARATION CONTROL. These are coordinates in angstroms and a flat
  // output would score well against a flat reference.
  if (!(rms(expected) > 1e-2)) throw new Error(`the reference is flat: ${rms(expected)}`);
  // 🔴 THE GPU IS HELD TO THE ORACLE AND THE CPU IS HELD TO THE GPU, and for
  // one model that distinction is the whole difference between a pass and a
  // fail. boltz2's token transformer amplifies its input by ~2.2e4 - measured,
  // and linear - so THIS reference accumulating in doubles and the oracle
  // accumulating in float32 cannot agree to better than about 1e-2 however
  // right the port is. The GPU accumulates in float32 like the oracle and reads
  // 3.50e-3 there where this CPU reads 2.20e-2, which says the model is right
  // and the arithmetic differs. Holding both to one bound would fail the
  // implementation that SHIPS on the behaviour of the one that does not.
  //
  // The CPU arm is still a gate: it must track the GPU to within 100x, which a
  // convention implemented on one side and not the other never does.
  const flat = Number(option(args, "bound", "2e-2"));
  const bound = Math.max(flat, 4 * (results.envelope ?? 0));
  if (bound > flat) {
    console.log(`  bound raised to ${bound.toExponential(2)}`
      + ` by this model's own arithmetic envelope`);
  }
  if (!(results.gpu < bound)) {
    throw new Error(`one denoise step differs by ${results.gpu.toExponential(3)}, over `
      + `${bound.toExponential(0)} - conditioning, atom encoder, token transformer, `
      + "atom decoder and the EDM scaling all run here, so this is the whole score model");
  }
  if (results.cpu !== undefined
      && !(results.cpu < Math.max(bound, 100 * results.gpu))) {
    throw new Error(`the CPU reference differs by ${results.cpu.toExponential(3)} where `
      + `the GPU differs by ${results.gpu.toExponential(3)} - two implementations of `
      + "one model do not part by 100x over arithmetic, so one of them is missing a "
      + "convention the other has");
  }
  return { model, tokens, noise: dump.noise, ...results, bound };
}
