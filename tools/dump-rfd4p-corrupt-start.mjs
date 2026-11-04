/**
 * The reference's CORRUPTED START for a partial or inverse-folding run, which no oracle carried.
 *
 *     node tools/dump-rfd4p-corrupt-start.mjs <out dir> --preset=<id>
 *       [--seed=0] [--checkpoint=model-rfd4p-int8/model.safetensors]
 *
 * 🔴 WHY THIS EXISTS. `cuda/rfd4p/export_input.mjs` writes a generation's initial state as
 * `gen/x0Bb` / `gen/x0Ll` from `sampleInitialNoise({ ..., rng: seededRng(seed) })` - an
 * UNCONDITIONAL start, on that block's own stated assumption that "nothing consumes the stream
 * before `sample_noise`". For a partial or inverse run neither half holds: the run starts from
 * `corruptBatch`'s xt rather than from pure noise, and the encoder's reparameterisation consumes
 * `n * latent` normals before the backbone's are drawn. So those two tensors describe a first step
 * such a run never takes, and `metal/rfd4p` had to decline comparing against them - which left the
 * corrupted-mode START, the one place a wrong clean target shows up first, compared against nothing.
 * It was a wrong clean target: see metal/rfd4p/sampler.mm's note on `e/coorsNm` against `in/caNm`.
 *
 * This writes the start the reference actually takes, under its own `gen/cStart*` keys, into
 * `<gen>-start`, which `metal/rfd4p` loads beside the oracle and compares the corrupted modes against.
 * The alternative was to teach the exporter the corrupted start, and `cuda/` belongs to another agent.
 *
 * 🔴 AND THE DRAW ORDER IS THE REFERENCE'S, NOT A RE-DERIVATION. `cpu/rfd4p/reference.js` creates one
 * rng, hands it to `encodeLatentsCpu` (which draws `n * latent` eps unless deterministic) and then to
 * `corruptBatch` (bb x0, ll x0, per-group offsets, in that order). This file calls the same two
 * functions with one rng in the same order, so the stream position is theirs by construction rather
 * than by arithmetic that could drift from it.
 */
import { mkdirSync, openSync, readFileSync, writeSync, closeSync } from "node:fs";
import { buildBatch } from "../shared/rfd4p/pipeline/index.js";
import { parseSafetensors } from "../shared/rfd4p/safetensors.js";
import { seededRng } from "../shared/rfd4p/flow.js";
import { corruptBatch, resolvePartialT, resolveInverseFoldingMethod, buildPartialSchedule,
  sampleNoisedTargetBackbone, RESAMPLED_FORWARD_MARGINAL } from "../shared/rfd4p/partial.js";
import { flowEmitMasks } from "../shared/rfd4p/flow.js";
import { encodeLatentsCpu } from "../cpu/rfd4p/encoder.js";
import { getRfd4pPresetById } from "../shared/rfd4p/presets.js";
import { DEFAULT_SAMPLING } from "../shared/rfd4p/flow.js";

const args = process.argv.slice(2);
const out = args.find((a) => !a.startsWith("--"));
const option = (name, fallback) =>
  args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
if (!out) {
  console.error("usage: dump-rfd4p-corrupt-start.mjs <out dir> --preset=<id> [--seed=0] [--checkpoint=<f>]");
  process.exit(1);
}
const presetId = option("preset", "partial_diffusion_1prw");
const seed = Number(option("seed", "0"));
const checkpoint = option("checkpoint", "model-rfd4p-int8/model.safetensors");
const preset = getRfd4pPresetById(presetId);
if (!preset) throw new Error(`no preset ${presetId}`);

// 🔴 THE SAME TWO BRANCHES `export_input.mjs` HAS, and the condition is copied from it rather than
// guessed: a preset that names a structure and is not `specMode` is built FROM ITS ANNOTATIONS
// (`buildBatch(bytes)`), and only a spec-mode preset goes through the contig pipeline. Taking the
// spec branch for an annotated preset dies in `fresh.js` with "Unresolved link valence ... load an
// annotated CIF with input_as_annotated_cif: true", which is that file telling you the same thing.
const cifPath = preset.cifUrl?.replace(/^\.\//, "");
const fromAnnotations = cifPath != null
  && (preset.fromJobFile ? preset.specMode === false : preset.specMode !== true);
const built = fromAnnotations
  ? buildBatch(new Uint8Array(readFileSync(cifPath)), { seed })
  : buildBatch({
    contig: preset.contig,
    ...(preset.conditions === undefined ? {} : { conditions: preset.conditions }),
    ...(preset.inputAsAnnotatedCif ? { input_as_annotated_cif: true } : {}),
    ...(cifPath ? { input: [cifPath] } : {}),
  }, { seed, loadInput: (f) => readFileSync(f) });
const { batch, n } = built;

const raw = readFileSync(checkpoint);
const parsed = parseSafetensors(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
const weights = parsed.tensors ?? parsed;
// `.config`, as export_input.mjs reads it: the bundle wraps it beside its format version and digest
const cfg = JSON.parse(readFileSync(checkpoint.replace(/model\.safetensors$/, "config.json"), "utf8")).config;
const encCfg = cfg.autoencoder_config.nn_ae.encoder;

const field = (key) => batch[key]?.data ?? batch[key];
// 🔴 THE REFERENCE'S OWN CALL SHAPE, not the batch. `flowEmitMasks` indexes `m.padding_mask[i]`
// directly, so handing it the batch - whose fields are `{data}` wrappers - makes every mask ZERO,
// `corruptBatch` skips every token and the dump is all zeros. It does not fail loudly: the port then
// reported relRMS 5.5e+15 against it, which is a near-zero DENOMINATOR and not a near-infinite error.
const emit = flowEmitMasks({
  padding_mask: field("padding_mask"),
  is_in_context_conditioning: field("is_in_context_conditioning"),
  supports_local_latent: field("supports_local_latent"),
}, n);
// 🔴 `--nsteps=` MUST MATCH THE EXPORT THIS IS COMPARED AGAINST, because the corruption TIME is a
// function of it: `resolvePartialT` turns the preset's step FRACTION into a t on the schedule for
// `nsteps` steps, so the preset's own 60 gives t_bb 0.9465 where the four-step export's meta says
// 0.9633. Read `meta/baseNsteps` out of the `-gen` dir and pass it; a mismatch here compares two
// different corruptions and looks exactly like a wrong port.
const nsteps = Number(option("nsteps", String(preset.nsteps)));
const partialTDict = resolvePartialT({ ...preset, nsteps }, DEFAULT_SAMPLING);
if (!Object.keys(partialTDict).length) throw new Error(`${presetId} is not a corrupted-start preset`);
resolveInverseFoldingMethod(preset);          // validates the method the preset names, as the reference does

// ONE rng, in the reference's order: the encoder's eps, then corruptBatch's three draws.
const rng = seededRng(seed);
const enc = encodeLatentsCpu(batch, { n, weights, cfg: encCfg, rng });
const coorsNm = batch.coords_nm
  ? field("coords_nm")
  : Float32Array.from(field("coords"), (v) => Math.fround(v / 10));
// the clean backbone is the atom37 CA slot, which is what the reference designs against
const caNm = new Float32Array(n * 3);
for (let i = 0; i < n; i += 1) {
  for (let k = 0; k < 3; k += 1) {
    const v = coorsNm[(i * 37 + 1) * 3 + k];
    caNm[i * 3 + k] = Number.isFinite(v) ? v : 0;
  }
}
const corrupted = corruptBatch({
  n,
  x1: { bb_ca: caNm, local_latents: enc.zLatent },
  partialTDict,
  emit,
  coordCondMask: field("coord_cond_mask"),
  moleculeIid: field("molecule_iid"),
  rng,
  centeringScale: DEFAULT_SAMPLING.centering?.scale ?? 0.2,
});

// 🔴 AND INVERSE FOLDING RE-NOISES THE TARGET INSTEAD, which is a different start and not a corruption:
// its partial t is 0 for both modalities, so `corruptBatch`'s xt would be the pure draw. The reference
// replaces the backbone with `sampleNoisedTargetBackbone` at the SCHEDULE's first time, drawing from the
// same rng AFTER corruptBatch - so the stream position matters as much as the formula and both come from
// calling the reference's own helpers in its own order.
const startBb = Float32Array.from(corrupted.xt.bb_ca);
const startLl = Float32Array.from(corrupted.xt.local_latents);
if (resolveInverseFoldingMethod(preset) === RESAMPLED_FORWARD_MARGINAL
    && (preset.inverseFolding ?? preset.inverse_folding)) {
  const sched = buildPartialSchedule(partialTDict, nsteps, DEFAULT_SAMPLING,
    { fullNsteps: (preset.partialTrajectoryFractionSteps ?? null) != null ? nsteps : null });
  const noised = sampleNoisedTargetBackbone({
    n,
    targetBackbone: Float32Array.from(corrupted.x1.bb_ca),
    tBb: sched.bb_ca.ts[0],
    emitBbCa: emit.bb_ca,
    coordCondMask: field("coord_cond_mask"),
    moleculeIid: field("molecule_iid"),
    rng,
    centeringScale: DEFAULT_SAMPLING.centering?.scale ?? 0.2,
  });
  startBb.set(noised);
}

mkdirSync(out, { recursive: true });
const fd = openSync(`${out}/model.bin`, "w");
const lines = [];
let offset = 0;
// the SAME keys the port already compares, so `--gen=<this dir>` needs no change on the other side
for (const [name, values] of Object.entries({
  // 🔴 OWN KEYS, NOT `gen/x0Bb`. `Model::loadInput` refuses a tensor it has already seen ("gen/x0Bb
  // is in two places") - deliberately, since silently shadowing an oracle tensor is how a comparison
  // starts measuring the wrong thing - so this cannot override the exporter's start and does not try.
  "gen/cStartBb": startBb,
  "gen/cStartLl": startLl,
  "gen/cCleanBb": corrupted.x1.bb_ca,
  "gen/cCleanLl": corrupted.x1.local_latents,
})) {
  const data = values instanceof Float32Array ? values : Float32Array.from(values);
  lines.push(`t ${name} ${offset} ${data.length}`);
  writeSync(fd, Buffer.from(data.buffer, data.byteOffset, data.byteLength));
  offset += data.length;
}
closeSync(fd);
writeSync(openSync(`${out}/model.idx`, "w"), lines.join("\n") + "\n");
console.log(`${out}: the reference's corrupted start for ${presetId} `
  + `(nsteps ${nsteps}, t bb ${partialTDict.bb_ca ?? 0}, ll ${partialTDict.local_latents ?? 0}, ${n} tokens)`);
