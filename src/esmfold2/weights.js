/**
 * One reading of the ESMFold2 bundle's 856 tensors, for every caller.
 *
 * 🔴 THE NAMES WERE SPELLED OUT IN FOUR PLACES BEFORE THIS FILE. The CPU fold,
 * the GPU fold and two checkers each built the same nested weight objects by
 * hand, so a converter that renamed a leaf broke them one at a time and each
 * failure named a different missing tensor. `read` is the only thing that
 * varies - `fetch` in a page, `readFileSync` in node - so it is the argument.
 *
 * Everything here is async because a shard may not be local.
 */

/** The six tensors one sliding-window atom block holds. */
const SWA_LEAVES = ["adaln", "qkv", "attnGate", "attnOut", "ffnUp", "ffnDown"];

/**
 * Where a weight object's tensors came from, for a caller that wants the CODES
 * rather than the values.
 *
 * 🔴 A SYMBOL, SO IT IS NOT A FIELD. Every loop in this port walks a weight
 * object's own enumerable properties - `Object.values(part)` in the denoiser's
 * byte accounting, for one - and a string key here would look like another
 * tensor to all of them.
 */
// One symbol for every loader; see src/runtime/weight-sources.js for what two
// of them cost. Imported AND re-exported, because a bare `export ... from`
// does not bind the name in this module and every use here is local.
import { SOURCES } from "../runtime/weight-sources.js";

export { SOURCES };

/**
 * 🔴 LAZY WHEN THE READER OFFERS A SOURCE, AND EAGER OTHERWISE. Decoding an
 * int5 bundle's denoiser on the main thread is 1.17 seconds before a fold's own
 * clock starts, and narrowing the same tensors to f16 inside `prepare` is
 * another 880 ms - and the GPU can do both. A reader that carries `.source`
 * (the bytes, undecoded) and `.decode` (the host fallback, synchronous) gets
 * getters and a `SOURCES` map; one that does not gets exactly what it always
 * got, which is every CPU checker in this port.
 */
const gather = async (read, entries) => {
  const out = {};
  if (typeof read.source !== "function" || typeof read.decode !== "function") {
    await Promise.all(entries.map(async ([name, path]) => { out[name] = await read(path); }));
    return out;
  }
  const sources = {};
  await Promise.all(entries.map(async ([name, path]) => {
    // 🔴 IN THE SHAPE THE DEVICE DECODER READS, not only in the shape `decode`
    // does. `planBlockUpload` wants a store, a tensor name and a range; it
    // never calls the entry, so hanging those four off the source object is
    // enough to let AF3's pair-track packer take an ESMFold2 block - which is
    // 828 ms of `trunk 0` against 32 for the same block warm. See
    // src/runtime/weight-sources.js.
    const source = await read.source(path);
    source.store = { tensorSource: () => source };
    source.tensorName = path;
    source.first = 0;
    source.count = (source.record?.shape ?? []).reduce((total, extent) => total * extent, 1);
    sources[name] = source;
  }));
  for (const [name] of entries) {
    let decoded;
    Object.defineProperty(out, name, {
      enumerable: true,
      get() { return (decoded ??= read.decode(sources[name])); },
    });
  }
  Object.defineProperty(out, SOURCES, { value: sources });
  return out;
};

/**
 * `gathered` with extra fields on it, WITHOUT spreading.
 *
 * 🔴 A SPREAD OVER A LAZY OBJECT IS AN EAGER DECODE. `{ ...gathered, blocks }`
 * copies enumerable own properties, which for a getter means calling it - so
 * every one of these lines decoded the tensor it was trying to defer, and threw
 * the `SOURCES` symbol away on the way past. It cost 932 ms at load and left
 * the token blocks taking the host path with no sign that anything was wrong,
 * because the answer was right either way.
 */
function extend(gathered, fields) {
  for (const [name, value] of Object.entries(fields)) {
    Object.defineProperty(gathered, name, { value, enumerable: true, configurable: true });
  }
  return gathered;
}

/** `blocks` sliding-window blocks under one prefix. */
export async function atomStackBlocks(read, prefix, blocks) {
  return Promise.all(Array.from({ length: blocks }, (_, layer) =>
    gather(read, SWA_LEAVES.map((leaf) => [leaf, `${prefix}/blocks/${layer}/${leaf}`]))));
}

/**
 * The inputs embedder, or the diffusion module's own copy of it.
 *
 * 🔴 THE DIFFUSION'S COPY HAS A `coordsLinear` AND THE INPUTS EMBEDDER DOES
 * NOT, which is the whole difference between them: the noisy coordinates enter
 * the running ACTIVATION and never the conditioning. Both are otherwise the
 * same module with different parameters, and `toToken` is 384 wide for the
 * inputs embedder and 768 for the diffusion's.
 */
export async function atomEncoderWeights(read, prefix, blocks, { withCoordinates = false } = {}) {
  const entries = [
    ["atomLinear", `${prefix}/linear`],
    ["atomNormScale", `${prefix}/norm/scale`],
    ["atomNormOffset", `${prefix}/norm/offset`],
    ["atomToToken", `${prefix}/toToken`],
  ];
  if (withCoordinates) entries.push(["coordsLinear", `${prefix}/coordsLinear`]);
  return extend(await gather(read, entries),
                { blocks: await atomStackBlocks(read, prefix, blocks) });
}

export async function atomDecoderWeights(read, prefix, blocks) {
  return extend(await gather(read, [
    ["tokenToAtom", `${prefix}/tokenToAtom`],
    ["normScale", `${prefix}/norm/scale`],
    ["normOffset", `${prefix}/norm/offset`],
    ["outputLinear", `${prefix}/outputLinear`],
  ]), { blocks: await atomStackBlocks(read, prefix, blocks) });
}

const TRIANGLE = ["leftNormInputScale", "leftNormInputOffset", "centerNormScale",
  "centerNormOffset", "outputProjection", "gatingLinear", "projection", "gate"];
const TRANSITION = ["inputLayerNormScale", "inputLayerNormOffset",
  "transition1", "transition2"];

/** One trunk block, in the shapes src/af3/pair-track-gpu.js wants. */
export async function trunkBlockWeights(read, layer) {
  const group = (name, leaves) =>
    gather(read, leaves.map((leaf) => [leaf, `blocks/${layer}/${name}/${leaf}`]));
  const [outgoing, incoming, transition] = await Promise.all([
    group("triangleMultiplicationOutgoing", TRIANGLE),
    group("triangleMultiplicationIncoming", TRIANGLE),
    group("pairTransition", TRANSITION),
  ]);
  return {
    triangleMultiplicationOutgoing: outgoing,
    triangleMultiplicationIncoming: incoming,
    pairTransition: transition,
  };
}

export async function featuriserWeights(read) {
  return gather(read, [
    ["relPos", "featuriser/relPos"],
    ["tokenBonds", "featuriser/tokenBonds"],
    ["zInit1", "featuriser/zInit1"],
    ["zInit2", "featuriser/zInit2"],
    ["recycleScale", "recycle/norm/scale"],
    ["recycleOffset", "recycle/norm/offset"],
    ["recycleProjection", "recycle/projection"],
    ["distogramWeights", "distogram/weights"],
    ["distogramBias", "distogram/bias"],
  ]);
}

/**
 * The diffusion conditioning.
 *
 * 🔴 ITS TRANSITIONS DO NOT FUSE THEIR GATE, WHERE THE TOKEN TRANSFORMER'S
 * DOES, IN THE SAME MODULE. `TransitionLayer` has `a_proj` and `b_proj` as two
 * Linears and computes `out_proj(silu(a) * b)`; the token transition's
 * `lin_swish` is ONE Linear of `2 * hidden` split in half, gate first. Reading
 * either as the other indexes one matrix at half its stride.
 */
export async function conditioningWeights(read, { pairTransitions = 2, singleTransitions = 2 } = {}) {
  const transition = (kind, layer) => gather(read, [
    ["normScale", `diffusion/${kind}/${layer}/norm/scale`],
    ["normOffset", `diffusion/${kind}/${layer}/norm/offset`],
    ["aProjection", `diffusion/${kind}/${layer}/aProjection`],
    ["bProjection", `diffusion/${kind}/${layer}/bProjection`],
    ["outProjection", `diffusion/${kind}/${layer}/outProjection`],
  ]);
  const [base, zTransitions, sTransitions] = await Promise.all([
    gather(read, [
      ["zInputNormScale", "diffusion/zInputNorm/scale"],
      ["zInputNormOffset", "diffusion/zInputNorm/offset"],
      ["zProjection", "diffusion/zProjection"],
      ["sInputNormScale", "diffusion/sInputNorm/scale"],
      ["sInputNormOffset", "diffusion/sInputNorm/offset"],
      ["sProjection", "diffusion/sProjection"],
      ["fourierWeights", "diffusion/fourier/weights"],
      ["fourierOffsets", "diffusion/fourier/offsets"],
      ["noiseNormScale", "diffusion/noiseNorm/scale"],
      ["noiseNormOffset", "diffusion/noiseNorm/offset"],
      ["noiseProjection", "diffusion/noiseProjection"],
    ]),
    Promise.all(Array.from({ length: pairTransitions }, (_, l) => transition("zTransitions", l))),
    Promise.all(Array.from({ length: singleTransitions }, (_, l) => transition("sTransitions", l))),
  ]);
  return extend(base, { zTransitions, sTransitions });
}

/** One token-transformer block: an attention half and a transition half. */
export async function tokenBlockWeights(read, layer) {
  const at = (kind, leaves) => gather(read,
    leaves.map((leaf) => [leaf.split("/").pop(),
                          `diffusion/tokenBlocks/${layer}/${kind}/${leaf}`]));
  const adaln = (kind) => gather(read, [
    ["singleScale", `diffusion/tokenBlocks/${layer}/${kind}/adaln/singleScale`],
    ["gateWeights", `diffusion/tokenBlocks/${layer}/${kind}/adaln/gateWeights`],
    ["gateBias", `diffusion/tokenBlocks/${layer}/${kind}/adaln/gateBias`],
    ["shiftWeights", `diffusion/tokenBlocks/${layer}/${kind}/adaln/shiftWeights`],
  ]);
  const [attention, attentionAdaln, transition, transitionAdaln] = await Promise.all([
    at("attention", ["queryWeights", "queryBias", "kvWeights", "gateWeights", "outWeights",
                     "outGateWeights", "outGateBias", "pairNormScale", "pairNormOffset",
                     "pairBiasWeights"]),
    adaln("attention"),
    at("transition", ["swishWeights", "outWeights", "outGateWeights", "outGateBias"]),
    adaln("transition"),
  ]);
  return {
    attention: extend(attention, { adaln: attentionAdaln }),
    transition: extend(transition, { adaln: transitionAdaln }),
  };
}

/** Everything the denoiser needs that is not one of its stacks. */
export async function denoiserWeights(read, { tokenBlocks }) {
  const [conditioning, blocks, rest] = await Promise.all([
    conditioningWeights(read),
    Promise.all(Array.from({ length: tokenBlocks }, (_, l) => tokenBlockWeights(read, l))),
    gather(read, [
      ["stepNormScale", "diffusion/stepNorm/scale"],
      ["stepNormOffset", "diffusion/stepNorm/offset"],
      ["singleToToken", "diffusion/singleToToken"],
      ["tokenNormScale", "diffusion/tokenNorm/scale"],
      ["tokenNormOffset", "diffusion/tokenNorm/offset"],
    ]),
  ]);
  return extend(rest, { conditioning, tokenBlocks: blocks });
}
