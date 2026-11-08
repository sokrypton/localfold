/**
 * AF3's per-atom conditioning in WebGL2 against
 * `cpu/af3/diffusion/atom-conditioning.js`.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-atom-conditioning.js [--falsify]
 *
 * "This is the bottom of the model... Its output is where 384 of
 * `target_feat`'s 447 columns come from."
 *
 * 🔴 THE ARM THAT MATTERS IS THE CHARGED ONE, AND NO PROTEIN-ONLY INPUT IS IT.
 * AF3 feeds `arcsinh(charge)` and four models of the lineage feed the raw
 * charge: "the two are IDENTICAL AT ZERO. Every standard amino acid conformer in
 * the CCD is neutral, so a test on protein alone can never tell them apart; it
 * first matters on a ligand." So the arms carry charges of -2, -1, 0, +1 and +2
 * and assert the two conventions DISAGREE - a gate whose input is all-neutral
 * would pass with either.
 *
 * 🔴 AND THE FOUR NAME CHARACTERS ARE SUMMED BEFORE THEY REACH THE RUNNING
 * VALUE, which is a ROUNDING fact: "the matmul committed `act + (a+b+c+d)` and
 * `(((act+a)+b)+c)+d` differed in 169,390 of 696,512 floats". The bar here is
 * tight enough that the wrong order would show.
 *
 * 🔴 AND FIVE EMBEDDINGS ARE SUMMED AND ALL FIVE ARE BIAS-FREE, "so a missing
 * term is not a constant offset that a later LayerNorm absorbs - it is a
 * direction removed from every atom". Every one is non-zero in these arms,
 * including on masked slots, because the mask is applied LAST.
 *
 * The pair conditioning's own two extra terms are arms as well: the projected
 * offset between two atoms of one token, and the INVERSE squared distance - "so
 * nearby atoms give a large feature and distant ones tend to zero rather than
 * growing without bound", where the plain squared distance is the obvious
 * reading and grows instead.
 */
import {
  perAtomConditioning as cpuPerAtom, perAtomPairConditioning as cpuPairCond,
} from "../../cpu/af3/diffusion/atom-conditioning.js";
import {
  perAtomConditioning, perAtomPairConditioning, releaseConditioningWeights,
  uploadConditioningWeights,
} from "../../webgl/af3/diffusion/atom-conditioning.js";
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
  const bar = Number(option(args, "bar", "2e-6"));
  const tokens = Number(option(args, "tokens", "5"));
  const dense = Number(option(args, "dense", "4"));
  const channels = Number(option(args, "channels", "16"));
  const pairChannels = Number(option(args, "pair-channels", "8"));
  const rows = tokens * dense;

  const random = generator(3141592);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };

  const weights = {
    channels, pairChannels,
    embedRefPos: noise(3 * channels, 0.4),
    embedRefMask: noise(channels, 0.3),
    embedRefCharge: noise(channels, 0.3),
    embedRefElement: noise(128 * channels, 0.3),
    embedRefAtomName: noise(4 * 64 * channels, 0.3),
    singleToPairCondRow: noise(channels * pairChannels, 1 / Math.sqrt(channels)),
    singleToPairCondCol: noise(channels * pairChannels, 1 / Math.sqrt(channels)),
    embedPairOffsets: noise(3 * pairChannels, 0.3),
    embedPairDistances: noise(pairChannels, 0.3),
  };

  // 🔴 CHARGES OF -2 TO +2, which is what separates the two conventions - and a
  // masked slot whose five embeddings are all non-zero, which is what the
  // mask-last rule is about.
  const mask = new Float32Array(rows).fill(1);
  const charge = new Float32Array(rows);
  const element = new Float32Array(rows);
  const atomNameChars = new Float32Array(rows * 4);
  for (let index = 0; index < rows; index += 1) {
    charge[index] = (index % 5) - 2;
    element[index] = [1, 6, 7, 8, 16, 200][index % 6];   // ...including one out of range
    for (let character = 0; character < 4; character += 1) {
      // ...including one out-of-range code, which must simply not fire.
      atomNameChars[index * 4 + character] = (index * 7 + character * 13) % 70;
    }
    if (index % 7 === 3) mask[index] = 0;
  }
  const positions = noise(rows * 3, 3);
  const reference = { positions, mask, element, charge, atomNameChars };

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };
  const uploaded = uploadConditioningWeights(execution, weights);
  // The control skews the POSITIONS, which reach the single conditioning, the
  // pair offsets and the inverse distance - every arm.
  const positionsTensor = upload(falsify
    ? Float32Array.from(positions, (v) => v * 1.01 + 0.01)
    : positions, "af3.cond.positions");
  const inputs = {
    positions: positionsTensor,
    mask: upload(mask, "af3.cond.mask"),
    element: upload(element, "af3.cond.element"),
    charge: upload(charge, "af3.cond.charge"),
    atomNameChars: upload(atomNameChars, "af3.cond.names"),
  };

  const arms = [];
  const failures = [];
  const singles = {};
  for (const rawRefCharge of [false, true]) {
    const name = rawRefCharge ? "the raw charge (boltz2, chai1, rf3, ef2)" : "arcsinh (AF3)";
    const act = perAtomConditioning(execution, {
      ...inputs, weights: uploaded, tokens, dense, rawRefCharge,
    });
    const mine = Float32Array.from(execution.read(act));
    singles[name] = mine;
    const expected = cpuPerAtom(reference, tokens, dense, weights, { rawRefCharge });
    const relative = relativeRms(mine, expected, name);
    const ok = relative <= bar;
    arms.push({ arm: `per-atom, ${name}`,
      relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);

    if (!rawRefCharge) {
      // The pair conditioning, off AF3's own single conditioning.
      const pairAct = perAtomPairConditioning(execution, {
        act, positions: positionsTensor, weights: uploaded,
        tokens, dense, channels, pairChannels,
      });
      const minePair = Float32Array.from(execution.read(pairAct));
      execution.release(pairAct);
      const expectedPair = cpuPairCond(reference, expected, tokens, dense, weights);
      const pairRelative = relativeRms(minePair, expectedPair, "pair");
      const pairOk = pairRelative <= bar;
      arms.push({ arm: "per-atom pair conditioning",
        relativeRms: Number(pairRelative.toPrecision(4)), bar, ok: pairOk });
      if (!pairOk) {
        failures.push(`pair: relRMS ${pairRelative.toExponential(3)} over ${bar}`);
      }
    }
    execution.release(act);
  }

  // 🔴 AND THE TWO CHARGE CONVENTIONS MUST DISAGREE, or the arms above are one
  // arm twice and a protein-only input would have passed either.
  const chargeMoves = Number(relativeRms(
    singles["the raw charge (boltz2, chai1, rf3, ef2)"], singles["arcsinh (AF3)"],
    "charge.moves").toPrecision(3));
  if (!(chargeMoves > 1e-3)) {
    failures.push(
      `the two charge conventions agree to ${chargeMoves} - this input cannot separate them`);
  }

  for (const tensor of held) execution.release(tensor);
  releaseConditioningWeights(execution, uploaded);
  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const red = arms.filter((row) => !row.ok).length;
  return {
    shape: { tokens, dense, channels, pairChannels },
    charges: [-2, -1, 0, 1, 2],
    maskedSlots: rows - mask.reduce((a, b) => a + b, 0),
    bar,
    falsify,
    arms,
    chargeMoves,
    memory: execution.stats(),
    verdict: falsify
      ? (red === arms.length
        ? `ok: all ${arms.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${arms.length} arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} arms against the f64 reference, the two charge`
          + ` conventions differing by ${chargeMoves}`
        : `FAILED: ${failures.join("; ")}`),
  };
}
