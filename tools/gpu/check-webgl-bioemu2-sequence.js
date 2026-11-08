/**
 * BioEmu2's sequence embedder in WebGL2 against the WGSL one, in the same page.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-bioemu2-sequence.js [--falsify]
 *
 * It compares all three of the stage's outputs - `z_init`, `s_init` and the MSA
 * input `m0` - against `Bioemu2Gpu`'s own `trunkInit` and `msaInputs` on the
 * same bundle and the same features, which is what `check-webgl-bioemu2-trunk.js`
 * has been taking from the WGSL side all along.
 *
 * 🔴 THE FEATURES ARE BUILT TO REACH EVERY BRANCH, because every one of them is
 * a different line of the same two kernels and a fold with a plain protein
 * exercises about half:
 *   - TWO CHAINS AND TWO ENTITIES, so the relative encoding's four blocks each
 *     have a pair that falls out of range and one that does not - and the four
 *     fall back on four DIFFERENT conditions (different chain, not the same
 *     chain and residue, different entity, same entity).
 *   - A NONZERO METHOD PER TOKEN, which indexes two tables (the single's
 *     embedding and the pair's), so a port that read one for both agrees
 *     wherever the method is zero.
 *   - CONTACTS OF ALL THREE KINDS: class 0 (`encoding_unspecified`, what a fold
 *     with no restraint gets on every pair), class 1 (`encoding_unselected`) and
 *     real contacts at DIFFERENT thresholds, which run the fourier embedding and
 *     the encoder per pair and so are the only arm that can see them.
 *
 * 🔴 AND THE s_init ARM CANNOT SEE ITS PROJECTION, which is worth knowing before
 * reading its residual as coverage: `linearRows` lives in `cpu/bioemu2/host.js`
 * and BOTH backends import it, so what the arm checks is the method embedding
 * and the plumbing around it. The projection's own gate is the CPU suite.
 *
 * 🔴 AND THE SEPARATIONS ARE HERE BECAUSE AGREEING WITH THE REFERENCE DOES NOT
 * SAY A TERM IS LIVE: two ports that both drop the contact table agree
 * perfectly. Each separation re-runs THIS port with one input group flattened
 * and asserts the answer moved.
 */
import { Bioemu2Gpu } from "../../webgpu/bioemu2/network.js";
import { BIOEMU2, bundleReader } from "../../shared/bioemu2/weights.js";
import { sequenceEmbedder } from "../../webgl/bioemu2/sequence.js";
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
  // 🔴 THE MEASURED FLOOR IS EXACTLY ZERO ON ALL THREE ARMS, at 8 tokens and at
  // 12, and that is the one residual in this tree a reader should NOT suspect
  // the instrument over. There is no long accumulation here and no regrouping:
  // both backends add the same five terms of z_init in the same order, from the
  // same host projections, so the two land on the same f32 bits - and the
  // separations below read 0.027 to 0.83 through the same `relativeRms`, which
  // is what says the helper is measuring something. A 1% skew of `s_inputs`
  // reads 1.6e-2 / 6.6e-2 / 3.8e-2.
  //
  // 🔴 THE BAR IS NOT ZERO ALL THE SAME: another device may contract a multiply
  // and an add into one fma and reorder nothing else, which moves the last place
  // without being a defect. 1e-6 is four orders below the control.
  const bar = Number(option(args, "bar", "1e-6"));
  const bundle = option(args, "model", "/model-bioemu2-int5").replace(/\/$/, "");
  const tokens = Number(option(args, "tokens", "8"));
  const sequences = Number(option(args, "sequences", "4"));
  const windows = Number(option(args, "windows", "2"));
  const atoms = windows * BIOEMU2.windowQueries;
  const pairs = tokens * tokens;
  if (tokens < 4) {
    throw new RangeError(
      "bioemu2 sequence gate: fewer than 4 tokens leaves the two chains with"
      + " nothing in them, and the relative encoding's chain blocks go inert");
  }

  const random = generator(2718281);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };

  const reader = bundleReader(bundle);
  const read = (name) => reader(name);
  const P = "sequence_embedder";
  const hostWeights = {};
  for (const name of ["s_init/weight", "z_init_1/weight", "z_init_2/weight",
    "method_conditioning/embedding/weight", "method_conditioning/pair_embedding/weight",
    "token_bonds/weight", "rel_pos/embed_rel_pos/weight", "rel_pos/embed_rel_token/weight",
    "rel_pos/embed_same_entity/weight", "rel_pos/embed_rel_chain/weight",
    "msa_module/msa_proj/weight", "msa_module/s_proj/weight"]) {
    hostWeights[name] = Float32Array.from(await read(`${P}/${name}`));
  }
  const contact = {};
  for (const name of ["encoder/weight", "encoder/bias", "encoding_unspecified",
    "encoding_unselected", "fourier_embedding/proj/weight",
    "fourier_embedding/proj/bias"]) {
    contact[name] = Float32Array.from(await read(`${P}/contact_conditioning/${name}`));
  }
  // The method table's own length says how many methods there are, so the
  // features index it without a constant written here.
  const methods = hostWeights["method_conditioning/embedding/weight"].length
    / BIOEMU2.single;

  // 🔴 CLASS 0 ON EVERY PAIR, THEN CLASS 1 ON THE NEIGHBOURS AND REAL CONTACTS
  // ON A FEW - and the real ones at three different thresholds, because the
  // threshold is what the fourier embedding reads.
  const contactConditioning = new Float32Array(pairs * 5);
  const thresholds = new Float32Array(pairs).fill(8);
  for (let pair = 0; pair < pairs; pair += 1) contactConditioning[pair * 5] = 1;
  let contacts = 0;
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const pair = i * tokens + j;
      const apart = Math.abs(i - j);
      if (apart === 1) {
        contactConditioning[pair * 5] = 0;
        contactConditioning[pair * 5 + 1] = 1;
      } else if (apart === 3 || apart === 5) {
        contactConditioning[pair * 5] = 0;
        contactConditioning[pair * 5 + 2 + (apart === 3 ? 0 : 1)] = 1;
        thresholds[pair] = apart === 3 ? 6 : 12;
        contacts += 1;
      }
    }
  }
  if (contacts === 0) {
    throw new RangeError(
      "bioemu2 sequence gate: no pair reached class 2, so the contact encoder"
      + " - the only per-pair arithmetic in the stage - is untested");
  }

  const features = {
    atom_pad_mask: new Int32Array(atoms).fill(1),
    atom_mask: new Float32Array(atoms).fill(1),
    template_coords: noise(atoms * 3, 6),
    charge: noise(atoms, 1),
    element: Int32Array.from({ length: atoms }, (_, i) => (i * 7 + 3) % 128),
    name_encoding: Int32Array.from({ length: atoms * 4 }, (_, i) => (i * 11 + 5) % 64),
    ref_space_uid: new Int32Array(atoms),
    token_idx_at_atom: Int32Array.from({ length: atoms },
      (_, i) => Math.min(tokens - 1, Math.floor(i / Math.ceil(atoms / tokens)))),
    res_type: Int32Array.from({ length: tokens }, (_, i) => (i * 5 + 1) % 32),
    profile: noise(tokens * 33, 0.3, 0.3),
    deletion_mean: noise(tokens, 0.2, 0.2),
    pocket_feature: new Int32Array(tokens),
    // two chains and two entities, so the relative encoding's four blocks differ
    asym_id: Int32Array.from({ length: tokens }, (_, t) => (t < tokens / 2 ? 0 : 1)),
    entity_id: Int32Array.from({ length: tokens }, (_, t) => (t < tokens / 2 ? 0 : 1)),
    residue_index: Int32Array.from({ length: tokens }, (_, t) => t % Math.ceil(tokens / 2)),
    sym_id: Int32Array.from({ length: tokens }, (_, t) => (t < tokens / 2 ? 0 : 1)),
    method_feature: Int32Array.from({ length: tokens }, (_, t) => t % methods),
    contact_conditioning: contactConditioning,
    contact_threshold: thresholds,
    token_bonds: (() => {
      const out = new Float32Array(pairs);
      for (let t = 0; t + 1 < tokens; t += 1) {
        out[t * tokens + t + 1] = 1;
        out[(t + 1) * tokens + t] = 1;
      }
      return out;
    })(),
    msa: Int32Array.from({ length: sequences * tokens }, (_, i) => (i * 3 + 1) % 32),
    msa_mask: new Float32Array(sequences * tokens).fill(1),
    has_deletion: noise(sequences * tokens, 0.5, 0.5),
    deletion_value: noise(sequences * tokens, 0.3, 0.3),
    msa_paired: Float32Array.from({ length: sequences * tokens },
      (_, i) => (i % 3 === 0 ? 1 : 0)),
  };

  const sInputs = noise(tokens * BIOEMU2.sInputs, 1);
  // f16 weights off: two implementations, not two precisions.
  const net = new Bioemu2Gpu(device, reader, { half: false, trunkPrecision: "f32" });
  const g = net.upload(features);
  const program = await net.trunkProgram(g, sInputs);
  const theirs = {
    "z_init": Float32Array.from(await net.ops.read(program.zInit)),
    "s_init": Float32Array.from(program.sInit),
  };
  const msa = await net.msaInputs(g, sInputs);
  theirs.m0 = Float32Array.from(await net.ops.read(msa.m0));

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  // 🔴 THE SKEW GOES TO THIS PORT'S s_inputs ALONE, which is the one input all
  // three outputs are built from: s_init reads it, both pairwise projections
  // read it, and so does the MSA module's single projection. A skew anywhere
  // else leaves an arm green.
  const skewed = Float32Array.from(sInputs,
    (v, i) => (i % 2 === 0 ? v * 1.01 + 0.01 : v));
  const run = (inputs, overrides = {}) => sequenceEmbedder(execution, {
    features: { ...features, ...overrides }, sInputs: inputs, tokens, sequences,
    hostWeights, contact,
  });

  const out = run(falsify ? skewed : sInputs);
  const mine = {
    "z_init": Float32Array.from(execution.read(out.zInit)),
    "s_init": out.sInit,
    m0: Float32Array.from(execution.read(out.m0)),
  };

  const arms = [];
  const failures = [];
  for (const name of ["z_init", "s_init", "m0"]) {
    const relative = relativeRms(mine[name], theirs[name], name);
    const ok = relative <= bar;
    arms.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  }
  if (msa.valid !== out.valid) {
    failures.push(`valid rows ${out.valid} against the reference's ${msa.valid}`);
  }

  // The separations: each flattens one input group and asserts z_init moved.
  const separations = [];
  if (!falsify) {
    const flattened = [
      ["the contact conditioning", {
        contact_conditioning: (() => {
          const out2 = new Float32Array(pairs * 5);
          for (let pair = 0; pair < pairs; pair += 1) out2[pair * 5] = 1;
          return out2;
        })(),
      }],
      // 🔴 ONE CHAIN, NOT "every sym_id plus one": adding a constant to every
      // identity leaves every DIFFERENCE unchanged, which is how a chain-block
      // separation reads 0 while the block works.
      ["the chain and entity blocks", {
        asym_id: new Int32Array(tokens),
        entity_id: new Int32Array(tokens),
        sym_id: new Int32Array(tokens),
      }],
      ["the token bonds", { token_bonds: new Float32Array(pairs) }],
      ["the method", { method_feature: new Int32Array(tokens) }],
    ];
    for (const [name, overrides] of flattened) {
      const other = run(sInputs, overrides);
      const moved = relativeRms(
        Float32Array.from(execution.read(other.zInit)), mine.z_init, name);
      execution.release(other.zInit);
      execution.release(other.m0);
      const ok = moved > 1e-3;
      separations.push({ arm: name, relativeRms: Number(moved.toPrecision(4)), ok });
      if (!ok) failures.push(`${name} is inert in z_init: relRMS ${moved.toExponential(3)}`);
    }
  }

  execution.release(out.zInit);
  execution.release(out.m0);
  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }

  const red = arms.filter((row) => !row.ok).length;
  return {
    bundle,
    shape: {
      tokens, sequences, atoms, pair: BIOEMU2.pair, single: BIOEMU2.single,
      msa: BIOEMU2.msa, sInputs: BIOEMU2.sInputs, methods,
      contactPairs: contacts, validRows: msa.valid,
    },
    reference: "the WGSL trunkInit and msaInputs, f16 weights off, same bundle",
    bar,
    falsify,
    arms,
    separations,
    memory: execution.stats(),
    verdict: falsify
      ? (red === arms.length
        ? `ok: all ${arms.length} arms go red on a 1% perturbation of s_inputs`
        : `VACUOUS: only ${red} of ${arms.length} arms noticed it`)
      : (failures.length === 0
        ? `ok: z_init, s_init and m0 against the WGSL sequence embedder,`
          + ` ${separations.length} separations live`
        : `FAILED: ${failures.join("; ")}`),
  };
}
