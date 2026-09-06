// ESMFold2's diffusion conditioning, against the module's own first call.
//
//     .venv-esm/bin/python tools/esmc/dump-esmfold2-trunk.py \
//         --sequence-length 40 --esmc esmc-600m --out oracle-dumps/esmfold2-trunk-40-lm.json
//     python3 tools/export_esmfold2_trunk.py
//     node tools/check-esmfold2-conditioning.js
//
// 🔴 IT TAKES THE MODULE'S OWN ARGUMENTS, NOT THE TRUNK'S OUTPUT. `z_trunk`,
// `rel_pos`, `s_inputs` and `t_hat` are all recorded, so this is `f(x) == y` and
// says nothing about, and cannot be broken by, anything upstream. The
// end-to-end checker is what joins them.
//
// 🔴 AND THE SAMPLER CALLS THIS FIFTEEN TIMES, SO THE DUMP KEEPS THE FIRST. `z`
// is cached across the steps and `s` is not, and a hook keeping the LAST call
// would record a `t_hat` from one step beside an `s` from another.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";

import { readTensor } from "../src/reference/dtype.js";
import { diffusionConditioning } from "../src/esmfold2/diffusion-reference.js";

const ROOT = new URL("..", import.meta.url).pathname;
const bundleDirectory = process.argv[2] ?? join(ROOT, "model-esmfold2-trunk-f32");
const dumpPath = process.argv[3] ?? join(ROOT, "oracle-dumps", "esmfold2-trunk-40-lm.json");
const BOUND = 2e-6;

function loadBundle(directory) {
  const manifest = JSON.parse(readFileSync(join(directory, "manifest.json"), "utf8"));
  const shards = new Map();
  const tensors = {};
  for (const [name, record] of Object.entries(manifest.tensors)) {
    if (!shards.has(record.file)) {
      const raw = readFileSync(join(directory, record.file));
      shards.set(record.file,
        raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
    }
    tensors[name] = readTensor(record, shards.get(record.file), record.byteOffset ?? 0, true);
  }
  return { manifest, tensors };
}

const relative = (got, want) => {
  let error = 0, total = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i] - want[i];
    error += d * d;
    total += want[i] * want[i];
  }
  return Math.sqrt(error / total);
};

const dump = JSON.parse(readFileSync(dumpPath, "utf8"));
const { manifest, tensors } = loadBundle(bundleDirectory);
if (dump.conditioning == null) {
  throw new Error(`${dumpPath} has no conditioning; re-dump with a build that hooks it`);
}
const conditioning = dump.conditioning;
const argument = (name) => Float32Array.from(conditioning.arguments[name].values);
const tokens = conditioning.singleShape[1];
const shape = {
  tokens,
  pairChannels: manifest.trunk.pairChannels,
  singleInputs: manifest.trunk.singleInputs,
  tokenChannels: manifest.trunk.tokenChannels2,
  multiplier: manifest.trunk.transitionMultiplier,
};
const transitions = (kind) => [0, 1].map((layer) => ({
  normScale: tensors[`diffusion/${kind}Transitions/${layer}/norm/scale`],
  normOffset: tensors[`diffusion/${kind}Transitions/${layer}/norm/offset`],
  aProjection: tensors[`diffusion/${kind}Transitions/${layer}/aProjection`],
  bProjection: tensors[`diffusion/${kind}Transitions/${layer}/bProjection`],
  outProjection: tensors[`diffusion/${kind}Transitions/${layer}/outProjection`],
}));
const weights = {
  zInputNormScale: tensors["diffusion/zInputNorm/scale"],
  zInputNormOffset: tensors["diffusion/zInputNorm/offset"],
  zProjection: tensors["diffusion/zProjection"],
  sInputNormScale: tensors["diffusion/sInputNorm/scale"],
  sInputNormOffset: tensors["diffusion/sInputNorm/offset"],
  sProjection: tensors["diffusion/sProjection"],
  fourierWeights: tensors["diffusion/fourier/weights"],
  fourierOffsets: tensors["diffusion/fourier/offsets"],
  noiseNormScale: tensors["diffusion/noiseNorm/scale"],
  noiseNormOffset: tensors["diffusion/noiseNorm/offset"],
  noiseProjection: tensors["diffusion/noiseProjection"],
  zTransitions: transitions("z"),
  sTransitions: transitions("s"),
};

const tHat = conditioning.arguments.t_hat.values[0];
console.log(`${dump.esmfold2}: ${tokens} tokens, t_hat ${tHat.toPrecision(6)}, `
  + `${shape.tokenChannels} token channels\n`);

let failures = 0;
const report = (label, score, limit = BOUND) => {
  const ok = score <= limit;
  if (!ok) failures += 1;
  console.log(`  ${label.padEnd(32)} relRMS ${score.toExponential(3)}   `
    + `${ok ? "ok" : "FAILED"}`);
};

const run = (options = {}) => diffusionConditioning(
  argument("z_trunk"), argument("relative_position_encoding"), argument("s_inputs"),
  tHat, shape, options.weights ?? weights, manifest.trunk.sigmaData);

const got = run();
report("conditioning, pair", relative(got.pair, Float32Array.from(conditioning.pair)));
report("conditioning, single", relative(got.single, Float32Array.from(conditioning.single)));

// 🔴 THE CONTROLS. The transition's gate and value are SEPARATE tensors here
// where every other transition in this tree fuses them, and the pair
// conditioning CONCATENATES its two inputs where adding them conforms. Both are
// 50/50 and both return a plausible tensor when wrong, so each is run the other
// way round to show the check can see it.
const swapped = {
  ...weights,
  zTransitions: weights.zTransitions.map((block) => ({
    ...block, aProjection: block.bProjection, bProjection: block.aProjection })),
  sTransitions: weights.sTransitions.map((block) => ({
    ...block, aProjection: block.bProjection, bProjection: block.aProjection })),
};
const other = run({ weights: swapped });
for (const [label, mine, theirs] of [
  ["...gate and value swapped, pair", other.pair, conditioning.pair],
  ["...gate and value swapped, single", other.single, conditioning.single],
]) {
  const score = relative(mine, Float32Array.from(theirs));
  const discriminates = score > 1e-2;
  if (!discriminates) failures += 1;
  console.log(`  ${label.padEnd(32)} relRMS ${score.toExponential(3)}   `
    + `${discriminates ? "discriminates" : "DOES NOT"}`);
}

console.log(failures === 0
  ? "\nthe diffusion conditioning agrees with ESMFold2"
  : `\n${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
