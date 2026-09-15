/**
 * OpenDDE's CONFIDENCE HEAD against af3-any-model's own.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-opendde-confidence-oracle.js \
 *       --model=/model-opendde-full-f32/manifest.json
 *
 * 🔴 THIS HEAD HAD NO ORACLE AT ALL. `check-opendde-confidence.js` is
 * differential - it says the GPU agrees with `hostPairReadouts`, which is our
 * own code - so a misreading shared by both halves is invisible to it, which is
 * exactly how boltz2's head was written against a reading of the reference
 * rather than against its numbers. And `dump_af3_confidence.py` cannot reach
 * this one: OpenDDE's head is its own parametrisation with no
 * `left_target_feat_project`, so that script dies with a KeyError. The dump
 * here is `tools/oracle/dump_af3_opendde_confidence.py`, which drives the
 * reference's `OpenDDEConfidenceHead` with 0 unmapped scopes.
 *
 * 🔴 WHAT IT MEASURES, on both machines, with the precision pins in place
 * (8dbb8cd):
 *
 *                                        A100       M2
 *     predicted_lddt                     1.14e-7    1.18e-7
 *     predicted_experimentally_resolved  9.53e-8    -
 *     full_pae                           8.46e-7    9.11e-7
 *     full_pde                           9.73e-7    1.20e-6
 *
 * 🔴 AND THE BOUND WAS 1e-2 BECAUSE THIS FIRST READ 4.7e-3 AND I CALLED IT "NOT
 * PRECISION". It was precision. The control was `--f16=off`, which CANNOT REACH
 * THE MATRIX KERNELS - so on a device with matrix units the flag moves nothing
 * and an unchanged number reads as proof that precision is not the cause. It is
 * proof of nothing at all. The M2, which has no matrix units at these widths,
 * saw the same flag remove the whole error; pinning the four settings AF3's
 * confidence head already carried took the A100 from 4.68e-3 to 8.46e-7.
 * **A control arm that cannot vary the thing under test is not a control.**
 *
 * 🔴 AND IT TAKES THE REFERENCE'S OWN INPUTS. The head's s/z/s_inputs are
 * seeded and its atom layout synthesised, both recorded in the dump - so this
 * compares the FORWARD and nothing else. Deriving the inputs on both sides
 * would let a shared featurisation error pass.
 */
import { openddeConfidence } from "../../src/af3/confidence/opendde-confidence.js";
import { openAf3Store, openddeConfidenceWeights, af3Dialect }
  from "../../src/af3/weights/weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
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
  const dumpPath = option(args, "dump", "/oracle-dumps/af3-oracle-confidence-opendde.json");
  const response = await fetch(dumpPath);
  if (!response.ok) throw new Error(`failed to load ${dumpPath}: ${response.status}`);
  const dump = await response.json();
  const of = (name) => {
    const entry = dump.stages[name];
    if (entry === undefined) throw new Error(`the dump has no ${name}`);
    return Float32Array.from(entry.data);
  };

  const store = await openAf3Store(option(args, "model",
    "/model-opendde-full-f32/manifest.json"));
  store.prefetch();
  const dialect = af3Dialect(store);
  const weights = await openddeConfidenceWeights(store);

  const tokens = dump.tokens;
  const slots = dump.slots;
  const atomCount = tokens * slots;
  const got = await openddeConfidence(device, {
    tokens,
    singleInputs: of("in.targetFeat"),
    single: of("in.single"),
    pair: of("in.pair"),
    coordinates: of("in.coordinates"),
    seqMask: of("in.seqMask"),
    atomToToken: Int32Array.from(of("in.atomToToken")),
    atomToSlot: Int32Array.from(of("in.atomToSlot")),
    atomCount,
  }, weights, dialect);

  // 🔴 THE VALUES, NOT THE LOGITS. The reference exposes both and this head's
  // own history is that a logits-vs-logits comparison passed while the reduced
  // score was unusable - see the note in opendde_confidence.py. The logits are
  // checked too, because a reduction can hide a small logit error and a wrong
  // reduction shows only in the value.
  const arms = [];
  for (const [ours, native] of [
    [got.plddt, "out.predicted_lddt"],
    [got.pae, "out.full_pae"],
    [got.pde, "out.full_pde"],
    [got.resolved, "out.predicted_experimentally_resolved"],
  ]) {
    const expected = of(native);
    if (ours === undefined) { arms.push({ stage: native, missing: true }); continue; }
    if (ours.length !== expected.length) {
      arms.push({ stage: native, lengthMismatch: [ours.length, expected.length] });
      continue;
    }
    arms.push({ stage: native, relRms: relRms(ours, expected), length: expected.length });
  }
  // 1e-5, which is two orders above what both machines measure and two below
  // what the unpinned head read. A bound left at the defect's own magnitude
  // would have let the defect back in silently.
  const bound = Number(option(args, "bound", "1e-5"));
  const worst = arms.filter((a) => a.relRms !== undefined)
    .reduce((w, a) => (w === null || a.relRms > w.relRms ? a : w), null);
  const result = {
    model: dump.model, tokens, slots, bound,
    arms: arms.map((a) => (a.relRms === undefined ? a
      : { ...a, relRms: Number(a.relRms.toExponential(2)) })),
    worst: worst === null ? null
      : { stage: worst.stage, relRms: worst.relRms.toExponential(2) },
    agrees: arms.length > 0 && arms.every((a) => a.relRms !== undefined && a.relRms <= bound),
  };
  // A gate that reports and does not fail is not a gate.
  if (!result.agrees) {
    throw new Error(`OpenDDE's confidence head disagrees with af3-any-model: `
      + `${JSON.stringify(result.arms)}`);
  }
  return result;
}
