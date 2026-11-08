/**
 * ESMFold2's atom transformer in WebGL2 against `cpu/esmfold2/atom-transformer.js`.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-esmfold2-atom.js [--falsify]
 *
 * 🔴 THE WINDOW IS OVER RANK AMONG VALID ATOMS, SO THE LAYOUT HAS PADDING IN
 * THE MIDDLE. "Two atoms 64 apart in the array can be adjacent in rank if the
 * atoms between them are padding", and a layout whose padding is all at the end
 * cannot tell rank from index - so this one masks atoms THROUGHOUT and runs a
 * `--window` arm narrow enough to exclude somebody.
 *
 * 🔴 AND THE DEFAULT IS DENSE, which is the finding `docs/EF2FAST.md` records:
 * `swa_window_size: 128` is in the config and unused, and reading it as a window
 * "cost 2.5 pLDDT". Both are arms here and the gate asserts they differ, because
 * a window wider than the molecule IS dense and would make the arm vacuous.
 *
 * 🔴 AND THE bfloat16 ARM IS A REAL ARM. The native module downcasts q, k and v
 * whatever the checkpoint's dtype, so an f32 port "CANNOT agree with it below
 * about 1e-4" - this port can do either, the device rounding is
 * `floatBitsToUint` arithmetic, and the gate holds the two to EACH OTHER's
 * reference rather than mixing them.
 *
 * 🔴 AND THE ROPE TABLE IS bfloat16 IN A float32 MODEL, which is worth 2.4e-3
 * through the attention. It is built on the host on both sides, so what this
 * gate compares is the arithmetic around it - and `--rope-precision=f32` is the
 * arm that says the table's rounding reaches the answer.
 *
 * 🔴 AND ref_element AND ref_atom_name_chars ARE INDICES, NOT ONE-HOTS. The
 * model is handed them one-hot and the FEATURISER produces indices; "passing the
 * one-hot makes this read its first `atoms` entries - all 0 or 1 - as element
 * indices... the encoder's output comes back at relRMS 0.89 with corr 0.72,
 * which reads exactly like a wrong convention somewhere in three SWA blocks".
 * The f64 reference REFUSES the wrong one by length, and this gate asserts that
 * refusal rather than trusting it.
 */
import {
  atomDecoder as cpuDecoder, atomFeatures, ATOM_FEATURES, buildRope,
  inputsEmbedder as cpuEmbedder,
} from "../../cpu/esmfold2/atom-transformer.js";
import {
  atomDecoder, inputsEmbedder,
} from "../../webgl/esmfold2/atom-transformer.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { generator } from "./af2-synthetic-block.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const bar = Number(option(args, "bar", "5e-6"));
  const tokens = Number(option(args, "tokens", "6"));
  const perToken = Number(option(args, "per-token", "4"));
  const channels = Number(option(args, "channels", "32"));
  const heads = Number(option(args, "heads", "4"));
  const hidden = Number(option(args, "hidden", "64"));
  const tokenChannels = Number(option(args, "token-channels", "48"));
  const blocks = Number(option(args, "blocks", "2"));
  const windowSize = Number(option(args, "window", "0"));
  const precision = option(args, "precision", "f32");
  const ropePrecision = option(args, "rope-precision", "bf16");
  const atoms = tokens * perToken;
  const headDim = channels / heads;

  const random = generator(2718281);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };

  // 🔴 PADDING THROUGHOUT, not at the end: the window is over RANK.
  const mask = new Float32Array(atoms).fill(1);
  for (let atom = 0; atom < atoms; atom += 1) if (atom % 5 === 3) mask[atom] = 0;
  const atomToToken = new Int32Array(atoms);
  for (let atom = 0; atom < atoms; atom += 1) atomToToken[atom] = Math.floor(atom / perToken);
  const refPos = noise(atoms * 3, 4);
  const refCharge = noise(atoms, 1);
  const refElement = Int32Array.from({ length: atoms }, (_, i) => (i * 7 + 3) % 128);
  const refAtomNameChars = Int32Array.from(
    { length: atoms * 4 }, (_, i) => (i * 11 + 5) % 64);
  const refSpaceUid = Float32Array.from({ length: atoms }, (_, i) => Math.floor(i / perToken));
  // The rope table must fill headDim / 2 exactly: 3 * spatial + uid.
  const spatialPairs = 1;
  const uidPairs = (headDim >> 1) - 3 * spatialPairs;
  if (uidPairs < 1) {
    throw new RangeError(
      `head_dim ${headDim}: 3 spatial pairs plus the uid pairs must fill ${headDim >> 1},`
      + " and the upstream code zero-pads a short table rather than failing");
  }
  const ropeOptions = { spatialPairs, uidPairs, precision: ropePrecision };
  const shape = {
    atoms, tokens, channels, heads, blocks, hidden, tokenChannels,
    rope: ropeOptions, attentionPrecision: precision,
    ...(windowSize > 0 ? { windowSize } : {}),
  };

  const blockWeights = () => ({
    adaln: noise(channels * channels * 6, 1 / Math.sqrt(channels)),
    qkv: noise(channels * channels * 3, 1 / Math.sqrt(channels)),
    attnGate: noise(channels * channels, 1 / Math.sqrt(channels)),
    attnOut: noise(channels * channels, 1 / Math.sqrt(channels)),
    ffnUp: noise(channels * hidden * 2, 1 / Math.sqrt(channels)),
    ffnDown: noise(hidden * channels, 1 / Math.sqrt(hidden)),
  });
  const encoderWeights = {
    atomLinear: noise(ATOM_FEATURES * channels, 1 / Math.sqrt(ATOM_FEATURES)),
    atomNormScale: noise(channels, 0.2, 1),
    atomNormOffset: noise(channels, 0.1),
    coordsLinear: noise(6 * channels, 0.3),
    atomToToken: noise(channels * tokenChannels, 1 / Math.sqrt(channels)),
    blocks: Array.from({ length: blocks }, blockWeights),
  };
  const decoderWeights = {
    tokenToAtom: noise(tokenChannels * channels, 1 / Math.sqrt(tokenChannels)),
    normScale: noise(channels, 0.2, 1),
    normOffset: noise(channels, 0.1),
    outputLinear: noise(channels * 3, 0.3 / Math.sqrt(channels)),
    blocks: Array.from({ length: blocks }, blockWeights),
  };

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };
  const uploadTree = (value, label) => {
    if (value instanceof Float32Array) return upload(value, label);
    if (Array.isArray(value)) return value.map((v, i) => uploadTree(v, `${label}.${i}`));
    if (typeof value === "number" || typeof value === "boolean") return value;
    return Object.fromEntries(Object.entries(value).map(
      ([key, v]) => [key, uploadTree(v, `${label}.${key}`)]));
  };
  // 🔴 THE SKEW GOES TO THE DEVICE ALONE, and the reference position is where
  // it has to land: it reaches the 389 features AND the rope table, which is
  // every arm. The charge reaches the features alone.
  const skew = (values, period) => Float32Array.from(values,
    (v, i) => (i % period < period / 2 ? v * 1.01 + 0.01 : v));
  // 🔴 THE DIFFUSION'S COPY OF THIS ENCODER TAKES THE NOISY COORDINATES INTO THE
  // ACTIVATION AND NEVER INTO THE CONDITIONING, with `pred_r1` ZEROS rather than
  // absent when there is no previous prediction - "the projection is 6 channels
  // wide either way; a port that fed it three would read the second half of the
  // matrix at the wrong offset". `--noisy` is that path and `--previous` adds
  // the second half.
  const noisy = flag(args, "noisy") ? noise(atoms * 3, 6) : undefined;
  const previous = flag(args, "previous") ? noise(atoms * 3, 6) : undefined;
  const features = {
    refPos: upload(falsify ? skew(refPos, 3) : refPos, "ef2.ref-pos"),
    ...(noisy === undefined ? {} : {
      noisyCoordinates: upload(falsify ? skew(noisy, 3) : noisy, "ef2.noisy"),
    }),
    ...(previous === undefined ? {} : {
      previousCoordinates: upload(previous, "ef2.previous"),
    }),
    refCharge: upload(falsify ? skew(refCharge, 2) : refCharge, "ef2.ref-charge"),
    mask: upload(mask, "ef2.mask"),
    refElement: upload(Float32Array.from(refElement), "ef2.ref-element"),
    refAtomNameChars: upload(Float32Array.from(refAtomNameChars), "ef2.ref-names"),
    atomToToken: upload(Float32Array.from(atomToToken), "ef2.atom-to-token"),
  };
  const host = { refPos, refSpaceUid, mask };
  const uploadedEncoder = uploadTree(encoderWeights, "w.ef2.encoder");
  const uploadedDecoder = uploadTree(decoderWeights, "w.ef2.decoder");

  const arms = [];
  const failures = [];
  const separations = {};
  const ourStages = {};
  const encoded = inputsEmbedder(execution, {
    features, host, shape, weights: uploadedEncoder,
    onStage: (stage, tensor) => {
      ourStages[stage] = Float32Array.from(execution.read(tensor));
    },
  });
  const ourTokenAct = Float32Array.from(execution.read(encoded.tokenAct));
  const update = atomDecoder(execution, {
    tokenAct: encoded.tokenAct, skip: encoded.atomState,
    conditioning: encoded.conditioning, rope: encoded.rope,
    valid: encoded.valid, rank: encoded.rank, index: encoded.index,
    shape, weights: uploadedDecoder,
  });
  const ourUpdate = Float32Array.from(execution.read(update));
  execution.release(update);
  for (const tensor of [encoded.atomState, encoded.conditioning, encoded.tokenAct,
    encoded.rope.cos, encoded.rope.sin, encoded.rank]) {
    execution.release(tensor);
  }

  // The f64 reference, on the UNPERTURBED inputs.
  const hostFeatures = {
    refPos, refCharge, refElement, refAtomNameChars, mask, atomToToken, refSpaceUid,
    ...(noisy === undefined ? {} : { noisyCoordinates: noisy }),
    ...(previous === undefined ? {} : { previousCoordinates: previous }),
  };
  const theirs = cpuEmbedder(hostFeatures, shape, encoderWeights);
  const theirUpdate = cpuDecoder(theirs.tokenAct, theirs.atomState, theirs.conditioning,
    theirs.rope, hostFeatures, shape, decoderWeights);

  const report = (name, mine, them) => {
    const relative = relativeRms(mine, them, name);
    const ok = relative <= bar;
    arms.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  };
  report("the 389 atom features", ourStages["atom-features"],
    atomFeatures(hostFeatures, atoms));
  // c0 is the reference's `conditioning`, which it returns.
  report("c0, the conditioning", ourStages.c0, theirs.conditioning);
  report("the atom stack's output", ourStages[`block${blocks - 1}`], theirs.atomState);
  report("the per-token pooling", ourTokenAct, theirs.tokenAct);
  report("the decoder's coordinate update", ourUpdate, theirUpdate);

  // 🔴 THE ROPE TABLE'S OWN ROUNDING, measured rather than asserted: the table
  // is the same on both sides, so this compares two of this port's own runs and
  // says what the bfloat16 storage is worth here.
  {
    const exact = buildRope(refPos, refSpaceUid, atoms, headDim,
      { ...ropeOptions, precision: "f32" });
    const rounded = buildRope(refPos, refSpaceUid, atoms, headDim,
      { ...ropeOptions, precision: "bf16" });
    separations["the rope table in bfloat16"] = Number(
      relativeRms(rounded.cos, exact.cos, "rope").toPrecision(3));
    if (!(separations["the rope table in bfloat16"] > 1e-4)) {
      failures.push("the bfloat16 rope table is indistinguishable from the exact one");
    }
  }

  // 🔴 AND THE WINDOW MUST MOVE THE ANSWER, or a windowed arm is vacuous.
  {
    const narrow = { ...shape, windowSize: 4 };
    const windowed = inputsEmbedder(execution, {
      features, host, shape: narrow, weights: uploadedEncoder,
    });
    const values = Float32Array.from(execution.read(windowed.tokenAct));
    for (const tensor of [windowed.atomState, windowed.conditioning, windowed.tokenAct,
      windowed.rope.cos, windowed.rope.sin, windowed.rank]) {
      execution.release(tensor);
    }
    const theirWindowed = cpuEmbedder(hostFeatures, narrow, encoderWeights);
    report("a window of 4, against the reference", values, theirWindowed.tokenAct);
    separations["a window of 4 against dense"] = Number(
      relativeRms(values, ourTokenAct, "window").toPrecision(3));
    if (!(separations["a window of 4 against dense"] > 1e-3)) {
      failures.push("a window of 4 computes the dense answer, so the arm is vacuous");
    }
  }

  // The refusal the reference makes, which this port inherits by using indices.
  const refusals = [];
  {
    let message = null;
    try {
      atomFeatures({ ...hostFeatures, refElement: new Float32Array(atoms * 128) }, atoms);
    } catch (error) {
      message = String(error.message);
    }
    const ok = message !== null && message.includes("INDICES");
    refusals.push({
      refusal: "ref_element handed over as a one-hot rather than as indices",
      ok, said: message ?? "did not throw",
    });
    if (!ok) failures.push("a one-hot ref_element was NOT refused");
  }

  for (const tensor of held) execution.release(tensor);
  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const red = arms.filter((row) => !row.ok).length;
  return {
    shape: {
      atoms, tokens, channels, heads, headDim, hidden, tokenChannels, blocks,
      windowSize: windowSize > 0 ? windowSize : "dense",
      noisyCoordinates: noisy !== undefined,
      previousCoordinates: previous !== undefined,
      attentionPrecision: precision, ropePrecision,
      paddedAtoms: atoms - mask.reduce((a, b) => a + b, 0),
      atomFeatures: ATOM_FEATURES,
    },
    bar,
    falsify,
    arms,
    separations,
    refusals,
    memory: execution.stats(),
    verdict: falsify
      ? (red === arms.length
        ? `ok: all ${arms.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${arms.length} arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} arms against the f64 reference, ${refusals.length} refusal`
        : `FAILED: ${failures.join("; ")}`),
  };
}
