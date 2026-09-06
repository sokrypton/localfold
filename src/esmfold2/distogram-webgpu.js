/**
 * ESMFold2's distogram head, and the contact map a page draws from it.
 *
 *     logits = distogram_head(z + z.transpose(-2, -3))
 *
 * 🔴 THE HEAD SYMMETRISES, AND THAT IS PART OF THE HEAD. A distance is
 * symmetric and the trunk's pair is not, so `z` alone conforms in shape and
 * returns a plausible distogram - which is why tools/check-esmfold2-fold.js
 * runs the unsymmetrised form as a control and measures it at 5.29e-1 against
 * the head's own 4.82e-5.
 *
 * 🔴 AND THE BIN EDGES ARE NOT IN THIS CHECKPOINT'S CONFIG FOR THIS HEAD.
 * `distogram_bins: 128` is stated and no range is; the CONFIDENCE head - which
 * is disabled in this checkpoint and has its own 128 bins - carries
 * `min_dist: 2.0, max_dist: 52.0`. So `CONTACT_EDGES` is that range, borrowed,
 * and it is a borrowing rather than a fact: `distogramLogits` returns LOGITS
 * and every distance in this file is downstream of an assumption a caller can
 * replace. A contact map is a picture and is the right place to take it; a
 * reported distance would not be.
 */
import { GRID_WIDTH, LANES, createLinearShader, linearGrid } from "../esmc/block-webgpu.js";

/** Borrowed from the disabled confidence head's own 128 bins. See above. */
export const CONTACT_EDGES = { minimum: 2, maximum: 52 };
/** What counts as a contact, which is the usual 8 A between pseudo-betas. */
export const CONTACT_ANGSTROMS = 8;

/**
 * ...and what it should be when the two ends are not both residues.
 *
 * 🔴 8 ANGSTROMS IS A PSEUDO-BETA CONVENTION AND A LIGAND TOKEN HAS NO SIDE
 * CHAIN. The distogram predicts a distance between one representative atom per
 * token, and for a residue that atom stands in for a whole side chain's reach
 * while for a ligand it IS the atom. So the threshold that means "these touch"
 * differs by what the pair is, and it is measured rather than reasoned:
 * `tools/calibrate-contact-cutoff.py` sweeps it against real depositions, with
 * real atomic contact - any heavy atom pair under 5 A - as the ground truth.
 * Best F1 over 14 entries and 41,000 real contacts:
 *
 * | pair | cutoff | F1 | at 8 A |
 * |---|---|---|---|
 * | protein-protein | **8 A** | 0.767 | the convention, confirmed |
 * | ligand-protein | **7 A** | 0.707 | 0.629 |
 * | ligand-nucleic | **7 A** | 0.764 | |
 * | nucleic-protein | **10 A** | 0.607 | 0.444 |
 * | nucleic-nucleic | **9 A** | 0.777 | |
 * | ligand-ligand | **5 A** | **1.000** | 0.696 |
 *
 * 🔴 AND THE LIGAND-LIGAND ROW IS EXACT, WHICH IS THE POINT RATHER THAN A
 * FLUKE. Both representatives ARE the heavy atoms, so the representative
 * distance is not an approximation of the ground truth - it is the ground
 * truth, and the only thing 8 A was doing there was being the wrong
 * definition. That holds whether the two atoms are in one molecule or two;
 * what differs between those is what the number MEANS - inside a molecule the
 * geometry came from the CCD conformer the model was handed, so a "prediction"
 * there is a copy - and not where the line sits.
 */
export const CONTACT_ANGSTROMS_BY_KIND = {
  "protein-protein": 8, "nucleic-protein": 10, "ligand-protein": 7,
  "nucleic-nucleic": 9, "ligand-nucleic": 7, "ligand-ligand": 5,
};

/** protein 0 and 3 for a ligand, as `molType` numbers them. */
const KIND_NAME = ["protein", "nucleic", "nucleic", "ligand"];

/**
 * The number of bins under the threshold THIS PAIR's kinds ask for, per pair.
 *
 * 🔴 IT IS A PAIRS-SIZED ARRAY BECAUSE THE PASS IS CHUNKED. The contact shader
 * reads a slice of the logits, so its cell index is chunk-relative and it
 * cannot recover i and j to look a kind up. An array sliced the same way needs
 * no shader arithmetic and no second dispatch; it costs one int per pair, which
 * is the size of the contact map it is computing.
 */
export function contactBinCountsByPair(molType, tokens, bins,
                                       edges = CONTACT_EDGES) {
  const cache = new Map();
  const out = new Int32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const kind = [KIND_NAME[molType[i]] ?? "protein",
                    KIND_NAME[molType[j]] ?? "protein"].sort().join("-");
      if (!cache.has(kind)) {
        cache.set(kind, contactBinCount(bins, edges,
          CONTACT_ANGSTROMS_BY_KIND[kind] ?? CONTACT_ANGSTROMS));
      }
      out[i * tokens + j] = cache.get(kind);
    }
  }
  return out;
}

/**
 * How sharply the distogram knows a distance, per residue - swept, not chosen.
 *
 * 🔴 THIS IS NOT A pLDDT AND MUST NEVER BE SHOWN AS ONE. This checkpoint has no
 * confidence head: 820 tensors and not one named confidence, plddt, pae or pde,
 * and `model.confidence_head` is None on the loaded model. What this is, is an
 * ORDERING - "the model knows where this residue goes more precisely than that
 * one" - with nothing to calibrate a NUMBER against.
 *
 * 🔴 AND THE THREE CONSTANTS ARE THE OUTCOME OF AN 11,400-ARM SWEEP OVER 60
 * FOLDS, not a guess. `tools/gpu/probe-esmfold2-confidence.js` scores per-pair
 * measures against per-residue lDDT-Ca, with targets corrupted at 0, 15 and 40%
 * so the label has range at all - 43 of 46 real targets fold above 0.9. The
 * winners, ranked on realistic rates by their WORST fold:
 *
 * | arm | median Spearman | worst fold |
 * |---|---|---|
 * | mode 3, sep 4, cut 12 | 0.504 | 0.310 |
 * | **mode 2, sep 3, cut 12** | **0.538** | **0.307** |
 * | mode 1.5, sep 3, cut 12 | 0.541 | 0.300 |
 *
 * against a buriedness baseline of about 0.19-0.26.
 *
 * 🔴 THE RADIUS IS IN ANGSTROMS BECAUSE THE BIN GRID IS BORROWED. `CONTACT_EDGES`
 * comes from the DISABLED confidence head, so "the height of the mode" is partly
 * an artefact of a 0.39 A grid nobody chose; the mass within 2 A of it is not.
 * Radius 0 - the exact bin - measured 0.415 against radius 2's 0.408 on clean
 * targets and is far less robust on corrupted ones.
 *
 * 🔴 AND THE CUTOFF IS ON THE PAIR, NOT ON THE BINS. Excluding pairs the model
 * places beyond 12 A is worth about 0.10 of Spearman; excluding the non-contact
 * BINS - ColabDesign's `con` loss, which is a design objective rather than a
 * confidence - scores 0.175, BELOW the buriedness baseline. The two sound alike
 * and are opposite.
 */
export const CERTAINTY = { radius: 2, separation: 3, cutoff: 12 };

/**
 * The two numbers the partner rule needs, one pair per token: `asymId` and
 * `residueIndex`.
 *
 * 🔴 A LIGAND IS ONE TOKEN PER HEAVY ATOM, SO A SEPARATION ON THE TOKEN INDEX
 * MEANS NOTHING THERE. Every atom of a component shares one asym id and one
 * residue number (`featurise.js` writes 1), so this rule drops a ligand's whole
 * self-block - which is right, because its internal geometry comes from the
 * CCD conformer that was handed to the model and is not a prediction at all.
 * Measured on ubiquitin plus ATP, 76 residues and 31 atoms:
 *
 * | | no ligand | with ATP, token gap | with ATP, this rule |
 * |---|---|---|---|
 * | mean certainty, PROTEIN tokens | 0.9496 | 0.8989 | (see check-esmfold2-certainty) |
 *
 * ...a 0.05 shift on the protein's own numbers, with one residue moving 0.47,
 * caused by a molecule the protein's confidence should not depend on.
 *
 * 🔴 AND IT IS EXACTLY THE OLD RULE FOR ONE UNMODIFIED PROTEIN CHAIN, which is
 * what every measurement behind `CERTAINTY` was made on: there the residue
 * number and the token index differ by a constant, so their differences agree.
 * A complex changes: two tokens in different chains are no longer excluded for
 * being near each other in the array, which they never should have been.
 */
export function partnerKeys({ asymId, residueIndex }, tokens) {
  const keys = new Int32Array(tokens * 2);
  for (let token = 0; token < tokens; token += 1) {
    keys[token * 2] = asymId[token];
    keys[token * 2 + 1] = residueIndex[token];
  }
  return keys;
}

/** `out[i, j] = pair[i, j] + pair[j, i]`, which is what the head is handed. */
export function createSymmetriseShader({ tokens, channels }) {
  const pairs = tokens * tokens;
  return `
@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${pairs * channels}u) { return; }
  let cell = i / ${channels}u;
  let c = i % ${channels}u;
  let row = cell / ${tokens}u;
  let column = cell % ${tokens}u;
  output[i] = pair[i] + pair[(column * ${tokens}u + row) * ${channels}u + c];
}`;
}

/**
 * softmax over the bins, then the mass below `CONTACT_ANGSTROMS`.
 *
 * 🔴 THE BIAS IS ADDED HERE RATHER THAN BY THE PROJECTION, because
 * `createLinearShader` has none and a whole extra pass to add 128 numbers to
 * every row would cost more than the branch. It is the same arithmetic either
 * way; what it must not be is forgotten, and a distogram missing its bias is
 * still a distribution because the softmax renormalises - it just puts its mass
 * in the wrong bins.
 */
export function createContactShader({ pairs, bins }) {
  return `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
// ...how many bins are under THIS pair's threshold; see CONTACT_ANGSTROMS_BY_KIND.
@group(0) @binding(2) var<storage, read> contactBins: array<i32>;
@group(0) @binding(3) var<storage, read_write> contacts: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let cell = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (cell >= ${pairs}u) { return; }
  let base = cell * ${bins}u;
  let near_bins = u32(contactBins[cell]);
  var largest = -3.0e38;
  for (var b = 0u; b < ${bins}u; b += 1u) {
    largest = max(largest, logits[base + b] + bias[b]);
  }
  var total = 0.0;
  var near = 0.0;
  for (var b = 0u; b < ${bins}u; b += 1u) {
    let weight = exp(logits[base + b] + bias[b] - largest);
    total += weight;
    if (b < near_bins) { near += weight; }
  }
  contacts[cell] = near / max(total, 1.0e-30);
}`;
}

/**
 * Per pair: the mass within `radius` of the mode, and where the mode is.
 *
 * Both are needed by the aggregation below - the first is the quantity, the
 * second decides which pairs are kept.
 */
export function createCertaintyPairShader({ pairs, bins }, span) {
  return `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> mass: array<f32>;
@group(0) @binding(3) var<storage, read_write> mode: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let cell = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (cell >= ${pairs}u) { return; }
  let base = cell * ${bins}u;
  var largest = -3.0e38;
  var argmax = 0u;
  for (var b = 0u; b < ${bins}u; b += 1u) {
    let value = logits[base + b] + bias[b];
    if (value > largest) { largest = value; argmax = b; }
  }
  var total = 0.0;
  var near = 0.0;
  for (var b = 0u; b < ${bins}u; b += 1u) {
    let weight = exp(logits[base + b] + bias[b] - largest);
    total += weight;
    // ...unsigned, so the distance between bins is taken the long way round.
    let away = select(argmax - b, b - argmax, b > argmax);
    if (away <= ${span}u) { near += weight; }
  }
  mass[cell] = near / max(total, 1.0e-30);
  mode[cell] = f32(argmax);
}`;
}

/**
 * Per residue: the mean of that mass over the pairs a residue keeps.
 *
 * 🔴 EVERY KEPT PAIR, NOT A TOP-N. Truncating to a residue's best partners cost
 * 0.08 of Spearman in the sweep and got worse the harder it truncated.
 */
export function createCertaintyShader({ tokens, separation, modeCutoffBin }) {
  return `
@group(0) @binding(0) var<storage, read> mass: array<f32>;
@group(0) @binding(1) var<storage, read> mode: array<f32>;
// 🔴 THE PARTNER RULE IS DATA, NOT ARITHMETIC ON THE TOKEN INDEX. Component x
// is the asym id and y the residue number; see the comment on partnerKeys.
@group(0) @binding(2) var<storage, read> partner: array<vec2<i32>>;
@group(0) @binding(3) var<storage, read_write> certainty: array<f32>;

var<workgroup> partial_sum: array<f32, ${LANES}>;
var<workgroup> partial_count: array<f32, ${LANES}>;
var<workgroup> partial_loose: array<f32, ${LANES}>;
var<workgroup> partial_loose_count: array<f32, ${LANES}>;

@compute @workgroup_size(${LANES})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let token = group.x + group.y * ${GRID_WIDTH}u;
  if (token >= ${tokens}u) { return; }
  var total = 0.0;
  var count = 0.0;
  var loose = 0.0;
  var looseCount = 0.0;
  for (var other = local.x; other < ${tokens}u; other += ${LANES}u) {
    // A partner is excluded only when it is a SEQUENCE neighbour: the same
    // chain, and within 'separation' residues. Across chains there is no
    // neighbourhood to exclude, and inside one ligand every atom shares the
    // residue number, so a gap of zero drops all of them.
    let here = partner[token];
    let there = partner[other];
    if (here.x == there.x && abs(here.y - there.y) <= ${separation}) { continue; }
    let cell = token * ${tokens}u + other;
    // 🔴 THE UNFILTERED MEAN IS KEPT AS A FALLBACK, because "no partner inside
    // the cutoff" is NO DATA and zero is a colour. A terminal residue the model
    // places away from everything has an empty filtered mean, and writing 0
    // there paints it as the least confident residue in the structure - which
    // is a claim, and the wrong one.
    loose += mass[cell];
    looseCount += 1.0;
    if (mode[cell] > ${modeCutoffBin}.0) { continue; }
    total += mass[cell];
    count += 1.0;
  }
  partial_sum[local.x] = total;
  partial_count[local.x] = count;
  partial_loose[local.x] = loose;
  partial_loose_count[local.x] = looseCount;
  workgroupBarrier();
  for (var stride = ${LANES / 2}u; stride > 0u; stride >>= 1u) {
    if (local.x < stride) {
      partial_sum[local.x] += partial_sum[local.x + stride];
      partial_count[local.x] += partial_count[local.x + stride];
      partial_loose[local.x] += partial_loose[local.x + stride];
      partial_loose_count[local.x] += partial_loose_count[local.x + stride];
    }
    workgroupBarrier();
  }
  if (local.x == 0u) {
    // ...the cutoff where there is anything inside it, every pair where there
    // is not, and only a chain shorter than the separation gets nothing.
    if (partial_count[0] > 0.0) {
      certainty[token] = partial_sum[0] / partial_count[0];
    } else if (partial_loose_count[0] > 0.0) {
      certainty[token] = partial_loose[0] / partial_loose_count[0];
    } else {
      certainty[token] = 0.0;
    }
  }
}`;
}

/**
 * Per pair: the mass within `radius` of the distance THIS STRUCTURE has.
 *
 * 🔴 THE OTHER ONE ASKS WHAT THE MODEL EXPECTS; THIS ONE ASKS WHETHER IT GOT
 * IT. `createCertaintyPairShader` centres on the distribution's mode and needs
 * no coordinates, so it is fixed for a fold. This centres on the distance the
 * SAMPLER produced, so it changes every step - which is what lets a trajectory
 * be coloured by its own agreement rather than by the final answer's, and a
 * frame that has not converged says so instead of wearing the last frame's
 * colour.
 *
 * They scored a tie on the sweep - median 0.537 against 0.535, worst fold 0.361
 * against 0.363 - so this costs nothing in accuracy and buys a per-frame
 * reading.
 */
export function createObservedMassShader({ tokens, bins }, span, edges = CONTACT_EDGES) {
  const width = (edges.maximum - edges.minimum) / bins;
  return `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read> positions: array<f32>;
@group(0) @binding(3) var<storage, read_write> mass: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let cell = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (cell >= ${tokens * tokens}u) { return; }
  let i = cell / ${tokens}u;
  let j = cell % ${tokens}u;
  let dx = positions[i * 3u] - positions[j * 3u];
  let dy = positions[i * 3u + 1u] - positions[j * 3u + 1u];
  let dz = positions[i * 3u + 2u] - positions[j * 3u + 2u];
  let separation = sqrt(dx * dx + dy * dy + dz * dz);
  let raw = (separation - ${edges.minimum}.0) / ${width};
  let observed = u32(clamp(raw, 0.0, ${bins - 1}.0));

  let base = cell * ${bins}u;
  var largest = -3.0e38;
  for (var b = 0u; b < ${bins}u; b += 1u) {
    largest = max(largest, logits[base + b] + bias[b]);
  }
  var total = 0.0;
  var near = 0.0;
  for (var b = 0u; b < ${bins}u; b += 1u) {
    let weight = exp(logits[base + b] + bias[b] - largest);
    total += weight;
    let away = select(observed - b, b - observed, b > observed);
    if (away <= ${span}u) { near += weight; }
  }
  mass[cell] = near / max(total, 1.0e-30);
}`;
}

/** How many of `bins` have their centre inside `CONTACT_ANGSTROMS`. */
export function contactBinCount(bins, edges = CONTACT_EDGES,
                                threshold = CONTACT_ANGSTROMS) {
  const width = (edges.maximum - edges.minimum) / bins;
  let count = 0;
  for (let bin = 0; bin < bins; bin += 1) {
    if (edges.minimum + (bin + 0.5) * width < threshold) count += 1;
  }
  return count;
}

/**
 * Build the contact map for a trunk's final pair, in row chunks.
 *
 * @param context { device, allocator, cache, submit }
 * @returns {Float32Array} one probability per token pair
 */
export async function encodeContactMap(context, { tokens, channels, bins, pair,
                                                  weights, bias, partners, molType,
                                                  chunk = 8192,
                                                  wantLogits = false,
                                                  retainForFrames = false }) {
  const { allocator, cache, submit, device } = context;
  const pairs = tokens * tokens;
  const storage = GPUBufferUsage.STORAGE;
  const height = Math.min(chunk, pairs);
  const heights = [...new Set([height, pairs % height].filter((h) => h > 0))];
  const key = `esmfold2-disto:${tokens}:${channels}:${bins}`;
  const symmetrise = await cache.get(`${key}:sym`,
    createSymmetriseShader({ tokens, channels }));
  // 🔴 THE CERTAINTY RIDES ON THE SAME PROJECTION. Its per-pair pass reads the
  // logits chunk the contact pass has just been handed, so the distogram is
  // projected ONCE and both readings come off it. Computing it separately would
  // be a second pass over the largest tensor in the fold.
  const width = (CONTACT_EDGES.maximum - CONTACT_EDGES.minimum) / bins;
  const span = Math.round(CERTAINTY.radius / width);
  const modeCutoffBin = Math.floor((CERTAINTY.cutoff - CONTACT_EDGES.minimum) / width);
  const certaintyPair = {};
  for (const rows of heights) {
    certaintyPair[rows] = await cache.get(`${key}:certain:${rows}`,
      createCertaintyPairShader({ pairs: rows, bins }, span));
  }
  const certaintyPass = await cache.get(`${key}:certain-token:${tokens}`,
    createCertaintyShader({ tokens, separation: CERTAINTY.separation, modeCutoffBin }));
  const project = {};
  const contact = {};

  for (const rows of heights) {
    project[rows] = await cache.get(`${key}:project:${rows}`,
      createLinearShader({ rows, inner: channels, outer: bins }, false));
    contact[rows] = await cache.get(`${key}:contact:${rows}`,
      createContactShader({ pairs: rows, bins }));
  }

  const held = [];
  const keep = (allocation) => { held.push(allocation); return allocation; };
  const elementwise = (elements) => {
    const groups = Math.ceil(elements / LANES);
    return [Math.min(GRID_WIDTH, groups), Math.ceil(groups / GRID_WIDTH)];
  };
  try {
    const symmetric = keep(allocator.allocate("esmfold2.disto.sym",
      pairs * channels * 4, storage));
    const projection = keep(allocator.upload("w.esmfold2.disto", weights, storage));
    const biasBuffer = keep(allocator.upload("w.esmfold2.disto-bias", bias, storage));
    // 🔴 THE LOGITS ARE A CHUNK UNLESS A CALLER WANTS THEM ALL. The contact map
    // needs one chunk's worth at a time - it collapses 128 bins to one number
    // per pair as it goes - and keeping every logit is `bins` times the pair
    // representation, 46 MiB at 300 tokens. A caller scoring a STRUCTURE
    // against the distribution needs them, because the distances it scores do
    // not exist until the sampler has run.
    // 🔴 THE WHOLE THING IS KEPT WHEN THE FRAMES WILL BE SCORED, which is
    // `bins` times the pair representation - 46 MiB at 300 tokens - and is the
    // price of colouring a trajectory by its own agreement rather than by the
    // final answer's. It is released the moment the sampler finishes.
    const wholeLogits = wantLogits || retainForFrames;
    const logits = keep(allocator.allocate("esmfold2.disto.logits",
      (wholeLogits ? pairs : height) * bins * 4,
      storage | (wantLogits ? GPUBufferUsage.COPY_SRC : 0)));
    const contacts = keep(allocator.allocate("esmfold2.disto.contacts",
      pairs * 4, storage | GPUBufferUsage.COPY_SRC));
    const mass = keep(allocator.allocate("esmfold2.disto.mass", pairs * 4, storage));
    const modes = keep(allocator.allocate("esmfold2.disto.mode", pairs * 4, storage));
    const certainty = keep(allocator.allocate("esmfold2.disto.certainty",
      Math.max(16, tokens * 4), storage | GPUBufferUsage.COPY_SRC));
    // 🔴 THE RULE IS REQUIRED, NOT DEFAULTED. A caller with no chain ids would
    // silently get the token-index rule back, which is the bug this replaced.
    if (partners === undefined || partners.length !== tokens * 2) {
      throw new Error("encodeContactMap needs partners: partnerKeys(features, tokens)");
    }
    const partnerBuffer = keep(allocator.upload("esmfold2.disto.partners",
      partners, storage));
    if (molType === undefined || molType.length !== tokens) {
      throw new Error("encodeContactMap needs molType, one per token");
    }
    const binCounts = keep(allocator.upload("esmfold2.disto.contact-bins",
      contactBinCountsByPair(molType, tokens, bins), storage));
    await submit("esmfold2.distogram", [
      ["symmetrise", symmetrise, [pair, symmetric], ...elementwise(pairs * channels)],
    ]);
    for (let start = 0; start < pairs; start += height) {
      const rows = Math.min(height, pairs - start);
      await submit("esmfold2.distogram", [
        ["project", project[rows],
         [{ buffer: symmetric.buffer, byteOffset: start * channels * 4,
            byteSize: rows * channels * 4 }, projection,
          wholeLogits
            ? { buffer: logits.buffer, byteOffset: start * bins * 4,
                byteSize: rows * bins * 4 }
            : logits],
         ...linearGrid(rows, bins)],
        ["certainty-pair", certaintyPair[rows],
         [wholeLogits
            ? { buffer: logits.buffer, byteOffset: start * bins * 4,
                byteSize: rows * bins * 4 }
            : logits, biasBuffer,
          { buffer: mass.buffer, byteOffset: start * 4, byteSize: rows * 4 },
          { buffer: modes.buffer, byteOffset: start * 4, byteSize: rows * 4 }],
         ...elementwise(rows)],
        ["contacts", contact[rows],
         [wholeLogits
            ? { buffer: logits.buffer, byteOffset: start * bins * 4,
                byteSize: rows * bins * 4 }
            : logits, biasBuffer,
          { buffer: binCounts.buffer, byteOffset: start * 4, byteSize: rows * 4 },
          { buffer: contacts.buffer, byteOffset: start * 4, byteSize: rows * 4 }],
         ...elementwise(rows)],
      ]);
    }
    await submit("esmfold2.certainty", [
      ["certainty", certaintyPass, [mass, modes, partnerBuffer, certainty],
       Math.min(GRID_WIDTH, tokens), Math.ceil(tokens / GRID_WIDTH)],
    ]);
    const readback = keep(allocator.allocate("esmfold2.disto.readback", pairs * 4,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
    const readCertainty = keep(allocator.allocate("esmfold2.disto.certainty-readback",
      Math.max(16, tokens * 4), GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
    const encoder = device.createCommandEncoder({ label: "esmfold2.disto.readback" });
    encoder.copyBufferToBuffer(contacts.buffer, 0, readback.buffer, 0, pairs * 4);
    encoder.copyBufferToBuffer(certainty.buffer, 0, readCertainty.buffer, 0, tokens * 4);
    device.queue.submit([encoder.finish()]);
    await readback.buffer.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readback.buffer.getMappedRange().slice(0));
    readback.buffer.unmap();
    await readCertainty.buffer.mapAsync(GPUMapMode.READ);
    const perToken = new Float32Array(readCertainty.buffer.getMappedRange().slice(0));
    readCertainty.buffer.unmap();
    // 🔴 THE RETAINED BUFFERS LEAVE `held` NOW, NOT WHEN THEY ARE RELEASED. The
    // first version handed the scorer a closure that spliced them out of the
    // release list - and that closure runs when the CALLER is finished, long
    // after this function's `finally` has already freed them. The failure was
    // "[Buffer esmfold2.disto.certainty-readback] is destroyed" on the first
    // frame, which names the buffer and not the lifetime.
    const retained = retainForFrames
      ? [logits, biasBuffer, modes, mass, certainty, readCertainty, partnerBuffer] : [];
    for (const allocation of retained) {
      const at = held.indexOf(allocation);
      if (at >= 0) held.splice(at, 1);
    }
    const frames = retainForFrames ? await framesScorer({
      device, allocator, cache, submit, key, tokens, bins, span, modeCutoffBin,
      logits, biasBuffer, modes, mass, certainty, readCertainty, partnerBuffer,
      release: () => { for (const allocation of retained) allocation.release(); },
    }) : undefined;
    if (!wantLogits) return { contacts: out, certainty: perToken, frames };
    // 🔴 THE BIAS IS NOT IN THE BUFFER, because the projection has none and the
    // contact pass adds it as it reads. A caller taking the logits away has to
    // be handed the bias too, or it will softmax a distribution missing 128
    // numbers - which is still a distribution, just the wrong one.
    const readLogits = keep(allocator.allocate("esmfold2.disto.logits-readback",
      pairs * bins * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
    const second = device.createCommandEncoder({ label: "esmfold2.disto.logits" });
    second.copyBufferToBuffer(logits.buffer, 0, readLogits.buffer, 0, pairs * bins * 4);
    device.queue.submit([second.finish()]);
    await readLogits.buffer.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(readLogits.buffer.getMappedRange().slice(0));
    readLogits.buffer.unmap();
    return { contacts: out, certainty: perToken, frames, logits: values, bias };
  } finally {
    for (let at = held.length - 1; at >= 0; at -= 1) held[at].release();
  }
}

/**
 * A closure that scores one structure at a time against the retained distogram.
 *
 * 🔴 IT REUSES THE AGGREGATION AND ONLY THE PER-PAIR PASS CHANGES. The filter
 * is still on the MODE - what the model predicts, which does not move between
 * frames - and only the quantity being averaged is recomputed against where the
 * sampler currently has the atoms. Recomputing the filter too would let a frame
 * change which pairs it is judged on, and a score whose denominator moves is
 * not comparable down a trajectory.
 */
async function framesScorer(context) {
  const { device, allocator, cache, submit, key, tokens, bins, span } = context;
  const { modeCutoffBin, logits, biasBuffer, modes, mass, certainty, readCertainty,
          partnerBuffer } = context;
  const storage = GPUBufferUsage.STORAGE;
  const observed = await cache.get(`${key}:observed:${tokens}`,
    createObservedMassShader({ tokens, bins }, span));
  const aggregate = await cache.get(`${key}:certain-token:${tokens}`,
    createCertaintyShader({ tokens, separation: CERTAINTY.separation, modeCutoffBin }));
  const positions = allocator.allocate("esmfold2.disto.frame-positions",
    Math.max(16, tokens * 3 * 4), storage | GPUBufferUsage.COPY_DST);
  const elementwise = (elements) => {
    const groups = Math.ceil(elements / LANES);
    return [Math.min(GRID_WIDTH, groups), Math.ceil(groups / GRID_WIDTH)];
  };
  return {
    /** @param {Float32Array} representative one xyz per token */
    async score(representative) {
      device.queue.writeBuffer(positions.buffer, 0, representative);
      await submit("esmfold2.certainty.frame", [
        ["observed", observed, [logits, biasBuffer, positions, mass],
         ...elementwise(tokens * tokens)],
        ["aggregate", aggregate, [mass, modes, partnerBuffer, certainty],
         Math.min(GRID_WIDTH, tokens), Math.ceil(tokens / GRID_WIDTH)],
      ]);
      const encoder = device.createCommandEncoder({ label: "esmfold2.certainty.frame" });
      encoder.copyBufferToBuffer(certainty.buffer, 0, readCertainty.buffer, 0, tokens * 4);
      device.queue.submit([encoder.finish()]);
      await readCertainty.buffer.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(readCertainty.buffer.getMappedRange().slice(0));
      readCertainty.buffer.unmap();
      return out;
    },
    release() { positions.release(); context.release(); },
  };
}
