/**
 * AF3's template embedder in WebGL2, against TWO references.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-template.js [--falsify]
 *
 * `cpu/af3/trunk/template.js`'s `templateEmbedding` is the f64 reference and
 * `Af3TemplateEmbedderGpu` the WGSL one, both in this page, with the two also
 * held to each other.
 *
 * 🔴 THE EMPTY-SLOT ARM IS THE ONE THAT MATTERS, WHICH IS THE OPPOSITE OF WHAT
 * IT LOOKS LIKE. "On a de novo protein with FOUR EMPTY TEMPLATE SLOTS its output
 * measures std 13.1 against a pair whose own std is 55 - about a quarter of the
 * representation entering the MSA stack." Nine features are summed and only six
 * are geometry; the other three are the query's own aatype twice and the query
 * pair itself, so an empty slot is a learned transform of the query run through
 * two blocks, a LayerNorm, a relu and a projection. Anything treating a missing
 * template as a zero is about 25% wrong from the first block and still folds to
 * something plausible.
 *
 * 🔴 AND THE PER-SLOT TAP IS COMPARED, NOT ONLY THE SUM. AF3 captures its own
 * after the blocks and the LayerNorm, before the summation and the output
 * projection, "so a checker can hold ONE slot's 64 channels to it and see the
 * geometry on its own. Summed and projected, a wrong unit vector and a wrong
 * distogram bin are the same number."
 *
 * 🔴 AND TWO CHAINS, BECAUSE THE CHAIN MASK ONCE DEFAULTED TO EVERYTHING AND
 * THAT WAS WORTH relRMS 1.09. "The cross-chain geometry is most of the module's
 * answer, so a permissive default is not a small error. It went unnoticed because
 * every check had a ONE-CHAIN query, where all-ones and per-chain are the same
 * array."
 *
 * 🔴 AND THE EMPTY-SLOT RESTYPE DIALECTS ARE ARMS. "OpenDDE and protenix2...
 * fill their one empty template with the GAP restype and zero-pads the rest",
 * and only the FIRST EMPTY slot gets it - writing 0 everywhere "put row 0 (ALA)
 * where row 21 belongs and was worth 2.83e-1 on the module".
 */
import { templateEmbedding as cpuTemplate } from "../../cpu/af3/trunk/template.js";
import { Af3TemplateEmbedderGpu } from "../../webgpu/af3/trunk/template.js";
import { setDeviceTuning, setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  RESTYPES, releaseTemplateWeights, slotCodes, slotGeometry, templateEmbedding,
  uploadTemplateWeights,
} from "../../webgl/af3/trunk/template.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { DGRAM_BINS } from "../../shared/af3/featurise/template-features.js";
import { generator } from "./af2-synthetic-block.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const bar = Number(option(args, "bar", "2e-5"));
  const tokens = Number(option(args, "tokens", "6"));
  const templates = Number(option(args, "templates", "4"));
  const occupied = Number(option(args, "occupied", "0"));
  // 🔴 64 BECAUSE THE WGSL TRANSITION CHUNKS AGAINST A 128-WIDE WORKGROUP and
  // refuses "chunk 64 is not a multiple of the workgroup 128" - so with the
  // template stack's widening factor of 2, 64 channels is the narrowest shape
  // it will run, and it is also AF3's own.
  const channels = Number(option(args, "channels", "64"));
  const queryChannels = Number(option(args, "query-channels", "32"));
  const blocks = Number(option(args, "blocks", "2"));
  const emptyAatype = option(args, "empty-aatype", null);
  const heads = 4;
  const dimension = 8;
  const pairs = tokens * tokens;

  const random = generator(161051);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };
  const triangleGroup = () => ({
    leftNormInputScale: noise(channels, 0.2, 1),
    leftNormInputOffset: noise(channels, 0.1),
    projection: noise(channels * channels * 2, 0.4 / Math.sqrt(channels)),
    gate: noise(channels * channels * 2, 0.4 / Math.sqrt(channels)),
    centerNormScale: noise(channels, 0.2, 1),
    centerNormOffset: noise(channels, 0.1),
    outputProjection: noise(channels * channels, 0.4 / Math.sqrt(channels)),
    gatingLinear: noise(channels * channels, 0.4 / Math.sqrt(channels)),
  });
  const gridGroup = () => {
    const width = heads * dimension;
    return {
      heads, dimension,
      actNormScale: noise(channels, 0.2, 1),
      actNormOffset: noise(channels, 0.1),
      pairBiasProjection: noise(channels * heads, 1 / Math.sqrt(channels)),
      qProjection: noise(width * channels, 1 / Math.sqrt(channels)),
      kProjection: noise(width * channels, 1 / Math.sqrt(channels)),
      vProjection: noise(channels * width, 1 / Math.sqrt(channels)),
      gatingQuery: noise(width * channels, 1 / Math.sqrt(channels)),
      outputProjection: noise(width * channels, 0.4 / Math.sqrt(width)),
    };
  };
  // 🔴 THE TEMPLATE STACK'S TRANSITION WIDENS BY **2**, NOT 4, and only the
  // weight's shape says so - "a wrong factor here reads them at the wrong stride
  // rather than failing". Both sides derive it from the tensor.
  const factor = 2;
  const transitionGroup = () => ({
    inputLayerNormScale: noise(channels, 0.2, 1),
    inputLayerNormOffset: noise(channels, 0.1),
    transition1: noise(channels * channels * factor * 2, 1 / Math.sqrt(channels)),
    transition2: noise(channels * factor * channels, 0.4 / Math.sqrt(channels * factor)),
  });
  const weights = {
    channels, queryChannels,
    queryEmbeddingNormScale: noise(queryChannels, 0.2, 1),
    queryEmbeddingNormOffset: noise(queryChannels, 0.1),
    templatePairEmbedding8: noise(queryChannels * channels, 1 / Math.sqrt(queryChannels)),
    templatePairEmbedding2: noise(RESTYPES * channels, 0.3),
    templatePairEmbedding3: noise(RESTYPES * channels, 0.3),
    templatePairEmbedding0: noise(DGRAM_BINS * channels, 0.3),
    templatePairEmbedding1: noise(channels, 0.3),
    templatePairEmbedding4: noise(channels, 0.3),
    templatePairEmbedding5: noise(channels, 0.3),
    templatePairEmbedding6: noise(channels, 0.3),
    templatePairEmbedding7: noise(channels, 0.3),
    outputLayerNormScale: noise(channels, 0.2, 1),
    outputLayerNormOffset: noise(channels, 0.1),
    outputLinear: noise(channels * queryChannels, 0.4 / Math.sqrt(channels)),
    blocks: Array.from({ length: blocks }, () => ({
      triangleMultiplicationOutgoing: triangleGroup(),
      triangleMultiplicationIncoming: triangleGroup(),
      pairAttention1: gridGroup(),
      pairAttention2: gridGroup(),
      pairTransition: transitionGroup(),
    })),
  };

  // Two chains, so the per-slot chain mask is not an all-ones array.
  const asymId = new Int32Array(tokens);
  for (let t = 0; t < tokens; t += 1) asymId[t] = t < Math.ceil(tokens / 2) ? 0 : 1;
  const pairMask = new Float32Array(pairs).fill(1);
  if (pairs > 2) pairMask[1] = 0;
  const pair = noise(pairs * queryChannels, 1);

  // `--occupied=N` fills the first N slots with a real template. AF3's dense
  // atom layout is what `templateGeometry` reads by default.
  const DENSE = 24;
  const slots = [];
  for (let slot = 0; slot < templates; slot += 1) {
    if (slot >= occupied) { slots.push(null); continue; }
    const aatype = new Int32Array(tokens);
    for (let t = 0; t < tokens; t += 1) aatype[t] = Math.floor(random() * 20);
    const atomMask = new Float32Array(tokens * DENSE).fill(1);
    // One residue with no atoms at all, which the gap-where-no-atoms dialect
    // reads, and one with no pseudo-beta.
    for (let a = 0; a < DENSE; a += 1) atomMask[1 * DENSE + a] = 0;
    atomMask[2 * DENSE + 3] = 0;
    slots.push({ aatype, atomMask, atomPositions: noise(tokens * DENSE * 3, 11) });
  }
  const dialect = {
    swapTransposedBias: false,
    ...(emptyAatype === null ? {} : { emptyTemplateAatype: Number(emptyAatype) }),
  };

  setDeviceTuning(device, { allowHostWeightPacking: true });
  setHalfPrecision(device, false);
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };
  const uploaded = uploadTemplateWeights(execution, weights);
  // The control skews half of each pair row's channels: the pair's first stop
  // is the query LayerNorm, and a whole-row affine change is invisible to one.
  const pairTensor = upload(falsify
    ? Float32Array.from(pair,
      (v, i) => (i % queryChannels < queryChannels / 2 ? v * 1.01 + 0.01 : v))
    : pair, "af3.template.pair");
  const pairMaskTensor = upload(pairMask, "af3.template.pair-mask");
  const firstEmpty = slots.filter(Boolean).length;
  const codes = slots.map((template, slot) => upload(slotCodes({
    template, tokens, slot, firstEmpty,
    emptyTemplateAatype: dialect.emptyTemplateAatype ?? null,
  }), `af3.template.codes${slot}`));
  const geometry = slots.map((template, slot) => {
    const packed = slotGeometry({ template, tokens, asymId });
    return packed === undefined ? undefined : upload(packed, `af3.template.geom${slot}`);
  });

  const ourSlots = [];
  const update = templateEmbedding(execution, {
    pair: pairTensor, pairMask: pairMaskTensor, tokens, templates,
    weights: uploaded, codes, geometry,
    onSlot: (slot, values) => { ourSlots[slot] = values; },
  });
  const ours = Float32Array.from(execution.read(update));
  execution.release(update);
  for (const tensor of held) execution.release(tensor);
  releaseTemplateWeights(execution, uploaded);

  const input = {
    tokens, pair, pairMask, templates, slots, asymId,
  };
  const theirSlots = [];
  const expected = cpuTemplate(
    { ...input, onSlot: (slot, values) => { theirSlots[slot] = Float32Array.from(values); } },
    weights, dialect);
  const wgslSlots = [];
  const theirs = await new Af3TemplateEmbedderGpu(device).run(
    { ...input, onSlot: (slot, values) => { wgslSlots[slot] = Float32Array.from(values); } },
    weights, dialect);

  const arms = [];
  const failures = [];
  const arm = (name, got, want, falsifiable = true) => {
    const relative = relativeRms(got, want, name);
    const ok = relative <= bar;
    arms.push({
      arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok,
      ...(falsifiable ? {} : {
        falsifiable: false, reason: "both references read the unperturbed pair",
      }),
    });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  };
  arm("the embedding vs the f64 reference", ours, expected);
  arm("the embedding vs the WGSL", ours, theirs.output ?? theirs);
  arm("the two references agree", expected, theirs.output ?? theirs, false);
  // The per-slot taps, which is where a wrong geometry feature is attributable.
  for (let slot = 0; slot < templates; slot += 1) {
    if (theirSlots[slot] === undefined || ourSlots[slot] === undefined) continue;
    arm(`slot ${slot}${slots[slot] ? " (occupied)" : " (empty)"} vs the f64 reference`,
      ourSlots[slot], theirSlots[slot]);
    if (wgslSlots[slot] !== undefined) {
      arm(`slot ${slot} vs the WGSL`, ourSlots[slot], wgslSlots[slot]);
    }
  }

  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const falsifiable = arms.filter((row) => row.falsifiable !== false);
  const red = falsifiable.filter((row) => !row.ok).length;
  return {
    shape: { tokens, templates, occupied, channels, queryChannels, blocks, factor },
    chains: 2,
    emptyTemplateAatype: dialect.emptyTemplateAatype ?? "0 in every slot (AF3)",
    bar,
    falsify,
    arms,
    memory: execution.stats(),
    verdict: falsify
      ? (red === falsifiable.length
        ? `ok: all ${falsifiable.length} falsifiable arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${falsifiable.length} falsifiable arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} arms, ${occupied} of ${templates} slots occupied,`
          + " both references agreeing"
        : `FAILED: ${failures.join("; ")}`),
  };
}
