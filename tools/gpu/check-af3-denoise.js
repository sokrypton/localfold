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
import { Af3DiffusionHeadGpu } from "../../src/af3/diffusion-head-webgpu.js";
import {
  diffusionConditioning, diffusionHead, diffusionTransformer,
  atomDecoder, scalings,
} from "../../src/af3/diffusion-reference.js";
import { Af3DiffusionTransformerGpu } from "../../src/af3/diffusion-transformer-webgpu.js";
import { Af3AtomDecoderGpu } from "../../src/af3/atom-decoder-webgpu.js";
import { Af3DiffusionConditioningGpu } from "../../src/af3/diffusion-conditioning-webgpu.js";
import { atomCrossAttentionEncoder as encodeCpu } from "../../src/af3/atom-encoder-reference.js";
import { Af3AtomEncoderGpu } from "../../src/af3/atom-encoder-webgpu.js";
import { layerNormSlow } from "../../src/af3/atom-encoder-reference.js";
import { linear } from "../../src/af3/pairformer-reference.js";
import { af3Dialect, openAf3Store } from "../../src/af3/weights.js";
import { perAtomConditioning } from "../../src/af3/atom-conditioning-reference.js";
import { atomReference, diffusionWeights } from "../../src/af3/diffusion-weights.js";

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
  }, tokens, dense, await atomReference(store));

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
  const results = {};
  if (option(args, "cpu", "on") !== "off") {
    const cpu = diffusionHead(input, weights, encodeCpu);
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
    const cpuEnc = encodeCpu(shared, weights.encoder);
    const gpuEnc = await new Af3AtomEncoderGpu(device).run(shared, weights.encoder, {});
    for (const key of ["tokenAct", "skipConnection"]) {
      if (cpuEnc[key] === undefined || gpuEnc[key] === undefined) continue;
      console.log(`  atom-encoder ${key}\t`
        + `${relativeRms(gpuEnc[key], cpuEnc[key]).toExponential(2)}`);
    }

    // ...and the TOKEN TRANSFORMER, which is the first stage that reads the
    // conditioning pair at this model's own width (256 here, 128 under AF3).
    const cond = diffusionConditioning(input, weights.conditioning);
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
    const cpuTx = diffusionTransformer(act, cond.single, cond.pair, input.seqMask,
                                       tokens, weights.transformer);
    const gpuTx = await new Af3DiffusionTransformerGpu(device).run(
      act, cond.single, cond.pair, input.seqMask, tokens, weights.transformer, {});
    console.log(`  token-transformer\t`
      + `${relativeRms(gpuTx.output ?? gpuTx, cpuTx).toExponential(2)}`);

    // ...and the ATOM DECODER, the last stage and the one the fold's geometry
    // already pointed at: per-atom offsets compressed to 0.73x while the token
    // centres they hang off stayed at 0.93x.
    const cpuDec = atomDecoder(cpuTx, cpuEnc, { ...shared, shape: input.shape },
                               weights.decoder);
    const gpuDec = await new Af3AtomDecoderGpu(device).run(
      cpuTx, cpuEnc, { ...shared, shape: input.shape }, weights.decoder, {});
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

  // 🔴 A SEPARATION CONTROL. These are coordinates in angstroms and a flat
  // output would score well against a flat reference.
  if (!(rms(expected) > 1e-2)) throw new Error(`the reference is flat: ${rms(expected)}`);
  const bound = Number(option(args, "bound", "2e-2"));
  const worst = Math.max(results.gpu, results.cpu ?? 0);
  if (!(worst < bound)) {
    throw new Error(`one denoise step differs by ${worst.toExponential(3)}, over `
      + `${bound.toExponential(0)} - conditioning, atom encoder, token transformer, `
      + "atom decoder and the EDM scaling all run here, so this is the whole score model");
  }
  return { model, tokens, noise: dump.noise, ...results, bound };
}
