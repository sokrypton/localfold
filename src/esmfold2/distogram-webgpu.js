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
import { MOL_NONPOLYMER, MOL_PROTEIN } from "./featurise.js";
import {
  CLASS_AMINO, CLASS_LIGAND, CLASS_NUCLEIC, CLASS_PROTEIN, PSEUDO_BETA_RESIDUES,
  contactAngstromsForClasses, contactBinsByPair,
} from "../heads/contact-threshold.js";

/** Borrowed from the disabled confidence head's own 128 bins. See above. */
export const CONTACT_EDGES = { minimum: 2, maximum: 52 };
/** What counts as a contact, which is the usual 8 A between pseudo-betas. */
// ...re-exported so callers of this head keep one import; the tables and the
// calibration behind them are in ../heads/contact-threshold.js, because AF3's
// distogram asks the identical question of the identical geometry.
export {
  CONTACT_ANGSTROMS, CONTACT_ANGSTROMS_BY_KIND, LIGAND_PROTEIN_ANGSTROMS,
} from "../heads/contact-threshold.js";

/**
 * The threshold one pair asks for, given both ends' kinds and residue types.
 *
 * 🔴 ONE FUNCTION, BECAUSE A METRIC'S TWO HALVES MUST ASK THE SAME QUESTION. A
 * checker computing "actual" at 8 A against a map computing "predicted" at 7
 * reports a precision about nothing.
 */
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
export function partnerKeys({ asymId, residueIndex, molType }, tokens) {
  const keys = new Int32Array(tokens * 4);
  for (let token = 0; token < tokens; token += 1) {
    keys[token * 4] = asymId[token];
    keys[token * 4 + 1] = residueIndex[token];
    // ...0 protein, 1 nucleic, 2 ligand. See PARTNER_ANGSTROMS.
    const mol = molType?.[token] ?? MOL_PROTEIN;
    keys[token * 4 + 2] = mol === MOL_NONPOLYMER ? 2 : (mol === MOL_PROTEIN ? 0 : 1);
  }
  return keys;
}

/**
 * How far away a partner can be and still say something, by what the PARTNER
 * is - and which tokens may be partners at all.
 *
 * 🔴 AF3's OWN lDDT IS SHAPED THIS WAY, AND IT IS NOT SYMMETRIC.
 * `all_atom_plddt_loss` in OpenFold3 builds its pair mask as
 *
 *     (dx_gt < 15) * protein_atom_mask[..., None, :]
 *   + (dx_gt < 30) * nucleotide_atom_mask[..., None, :]
 *
 * - the radius is chosen by the kind of the atom in the SECOND index, the one
 * doing the scoring, and a ligand atom appears in neither term. Its `rep_index`
 * says the same thing from the other side: CA for a standard protein residue,
 * C1' for a standard nucleotide, and a padding sentinel for a ligand or an
 * atomized residue, so those contribute no representative atom. **Every atom is
 * SCORED; only polymer representatives do the SCORING.**
 *
 * That is what "treat a ligand like a protein" actually means, and it removes
 * the fallback by construction rather than by adding a tier: a ligand token has
 * partners - the polymer around it - under the same rule as everyone else.
 *
 * 🔴 AND THE PROTEIN RADIUS STAYS AT THE ONE THAT WAS SWEPT HERE. AF3's 15 A is
 * for a different quantity (a distance-difference test against a true
 * structure, not a distogram's peakedness), and this repository's own sweep
 * peaked at 12-14 A over 11,400 arms - close enough to be reassuring and not a
 * reason to move. What is taken from AF3 is the SHAPE: a nucleotide reaches
 * twice as far, because a base pair's partners are further off than a side
 * chain's.
 */
export const PARTNER_ANGSTROMS = { protein: 12, nucleic: 24 };


export function contactAngstromsFor(molTypeI, molTypeJ, residueTypeI, residueTypeJ) {
  return contactAngstromsForClasses(esmfold2Class(molTypeI, residueTypeI),
                                    esmfold2Class(molTypeJ, residueTypeJ));
}

/**
 * ESMFold2's `molType` and `residueType` as a contact class.
 *
 * 🔴 THE OFFSET IS TWO AND IT IS WRITTEN DOWN HERE, not assumed at a call site.
 * ESMFold2 numbers residues by three-letter code from 2, which is
 * `PSEUDO_BETA_RESIDUES`'s own order - see the note there, and
 * test/esmfold2-certainty-partners.test.js, which asserts it rather than
 * trusting it.
 */
function esmfold2Class(molType, residueType) {
  if (molType === MOL_NONPOLYMER) return CLASS_LIGAND;
  if (molType !== MOL_PROTEIN) return CLASS_NUCLEIC;
  const at = residueType - 2;
  return at >= 0 && at < PSEUDO_BETA_RESIDUES.length ? CLASS_AMINO + at : CLASS_PROTEIN;
}

export function contactBinCountsByPair(molType, residueType, tokens, bins,
                                       edges = CONTACT_EDGES) {
  const classes = new Int32Array(tokens);
  for (let token = 0; token < tokens; token += 1) {
    classes[token] = esmfold2Class(molType[token], residueType[token]);
  }
  // ...ESMFold2's grid counts a bin whose CENTRE is under the threshold; AF3's
  // counts one whose top edge is. Both are prefixes and neither is the other.
  return contactBinsByPair(classes, tokens,
    (angstroms) => contactBinCount(bins, edges, angstroms));
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
export function createCertaintyShader({ tokens, separation, cutoffBins }) {
  return `
@group(0) @binding(0) var<storage, read> mass: array<f32>;
@group(0) @binding(1) var<storage, read> mode: array<f32>;
// 🔴 THE PARTNER RULE IS DATA, NOT ARITHMETIC ON THE TOKEN INDEX. x is the asym
// id, y the residue number and z the chemistry - 0 protein, 1 nucleic, 2
// ligand. See partnerKeys and PARTNER_ANGSTROMS.
@group(0) @binding(2) var<storage, read> partner: array<vec4<i32>>;
@group(0) @binding(3) var<storage, read_write> certainty: array<f32>;
// ...and the interface reading beside it, which is a different question and is
// -1 where the token has no cross-chain partner at all.
@group(0) @binding(4) var<storage, read_write> interface_certainty: array<f32>;

var<workgroup> partial_sum: array<f32, ${LANES}>;
var<workgroup> partial_count: array<f32, ${LANES}>;
var<workgroup> partial_cross: array<f32, ${LANES}>;
var<workgroup> partial_cross_count: array<f32, ${LANES}>;

@compute @workgroup_size(${LANES})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let token = group.x + group.y * ${GRID_WIDTH}u;
  if (token >= ${tokens}u) { return; }
  var total = 0.0;
  var count = 0.0;
  var cross = 0.0;
  var crossCount = 0.0;
  for (var other = local.x; other < ${tokens}u; other += ${LANES}u) {
    // A partner is excluded only when it is a SEQUENCE neighbour: the same
    // chain, and within 'separation' residues. Across chains there is no
    // neighbourhood to exclude, and inside one ligand every atom shares the
    // residue number, so a gap of zero drops all of them.
    let here = partner[token];
    let there = partner[other];
    if (here.x == there.x && abs(here.y - there.y) <= ${separation}) { continue; }
    // 🔴 A LIGAND IS SCORED, NEVER SCORING. AF3's lDDT gives a ligand atom no
    // representative and admits only protein and nucleotide atoms as the
    // partner index; this is that rule. It is also what lets a ligand token be
    // scored at all without a fallback - the polymer around it is its partner
    // set, under everyone else's cutoff.
    if (there.z == 2) { continue; }
    let same_chain = here.x == there.x;
    let cell = token * ${tokens}u + other;
    // ...and how far that partner may be is the PARTNER's question, not the
    // pair's: a nucleotide reaches twice as far as a residue does.
    let reach = select(${cutoffBins.protein}.0, ${cutoffBins.nucleic}.0, there.z == 1);
    if (mode[cell] > reach) { continue; }
    // 🔴 THE TWO ARE KEPT APART, WHICH IS AF3's OWN DISTINCTION. A chain can be
    // folded well and docked badly, and one mean over both says neither: on a
    // two-chain fold, chain B read 0.712 within itself, 0.370 across the
    // interface, and 0.630 mixed - so the number a reader saw was pulled down
    // by an interface question they had not asked.
    if (same_chain) {
      total += mass[cell];
      count += 1.0;
    } else {
      cross += mass[cell];
      crossCount += 1.0;
    }
  }
  partial_sum[local.x] = total;
  partial_count[local.x] = count;
  partial_cross[local.x] = cross;
  partial_cross_count[local.x] = crossCount;
  workgroupBarrier();
  for (var stride = ${LANES / 2}u; stride > 0u; stride >>= 1u) {
    if (local.x < stride) {
      partial_sum[local.x] += partial_sum[local.x + stride];
      partial_count[local.x] += partial_count[local.x + stride];
      partial_cross[local.x] += partial_cross[local.x + stride];
      partial_cross_count[local.x] += partial_cross_count[local.x + stride];
    }
    workgroupBarrier();
  }
  if (local.x == 0u) {
    // ...the cutoff where there is anything inside it, every pair where there
    // is not, and only a chain shorter than the separation gets nothing.
    interface_certainty[token] = select(-1.0,
      partial_cross[0] / max(partial_cross_count[0], 1.0), partial_cross_count[0] > 0.0);
    // 🔴 ONE RULE FOR EVERY TOKEN, AND NO FALLBACK UNDER IT - WHICH IS AF3's
    // OWN lDDT. A token is scored on every partner it is not TRIVIALLY close to
    // and that is allowed to score, whatever chain that partner is in. The
    // trivial exclusion says the same thing about all three chemistries, which
    // is why they need no branch: a residue's i+1 neighbour is 3.8 A apart in
    // every structure, and a ligand's atoms sit at the spacing the CCD
    // conformer HANDED the model, so both are an input rather than a
    // prediction. What differs is only which partners survive, and that is a
    // fact about the molecule rather than a special case in the code.
    //
    // 🔴 AND IT DOES NOT SPLIT WITHIN FROM ACROSS, BECAUSE pLDDT DOES NOT. A
    // local score is about a token's neighbourhood, and a residue at an
    // interface really does have neighbours in the other chain - AF3's lDDT
    // admits them and so does this. The pTM/ipTM question is a PER-CHAIN one
    // and it is answered per chain, by interface_certainty beside this.
    let scored = partial_sum[0] + partial_cross[0];
    let scoredCount = partial_count[0] + partial_cross_count[0];
    certainty[token] = select(-1.0, scored / max(scoredCount, 1.0), scoredCount > 0.0);
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
                                                  weights, bias, partners,
                                                  molType, residueType,
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
  // ...one bin cutoff per partner chemistry; see PARTNER_ANGSTROMS.
  const binOf = (angstroms) =>
    Math.floor((angstroms - CONTACT_EDGES.minimum) / width);
  const cutoffBins = { protein: binOf(PARTNER_ANGSTROMS.protein),
                       nucleic: binOf(PARTNER_ANGSTROMS.nucleic) };
  const certaintyPair = {};
  for (const rows of heights) {
    certaintyPair[rows] = await cache.get(`${key}:certain:${rows}`,
      createCertaintyPairShader({ pairs: rows, bins }, span));
  }
  const certaintyPass = await cache.get(`${key}:certain-token:${tokens}`,
    createCertaintyShader({ tokens, separation: CERTAINTY.separation, cutoffBins }));
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
    const interfaceCertainty = keep(allocator.allocate("esmfold2.disto.interface",
      Math.max(16, tokens * 4), storage | GPUBufferUsage.COPY_SRC));
    // 🔴 THE RULE IS REQUIRED, NOT DEFAULTED. A caller with no chain ids would
    // silently get the token-index rule back, which is the bug this replaced.
    if (partners === undefined || partners.length !== tokens * 4) {
      throw new Error("encodeContactMap needs partners: partnerKeys(features, tokens)");
    }
    const partnerBuffer = keep(allocator.upload("esmfold2.disto.partners",
      partners, storage));
    if (molType === undefined || molType.length !== tokens
        || residueType === undefined || residueType.length !== tokens) {
      throw new Error("encodeContactMap needs molType and residueType per token");
    }
    const binCounts = keep(allocator.upload("esmfold2.disto.contact-bins",
      contactBinCountsByPair(molType, residueType, tokens, bins), storage));
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
      ["certainty", certaintyPass,
       [mass, modes, partnerBuffer, certainty, interfaceCertainty],
       Math.min(GRID_WIDTH, tokens), Math.ceil(tokens / GRID_WIDTH)],
    ]);
    const readback = keep(allocator.allocate("esmfold2.disto.readback", pairs * 4,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
    const readCertainty = keep(allocator.allocate("esmfold2.disto.certainty-readback",
      Math.max(16, tokens * 4), GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
    const readInterface = keep(allocator.allocate("esmfold2.disto.interface-readback",
      Math.max(16, tokens * 4), GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
    const encoder = device.createCommandEncoder({ label: "esmfold2.disto.readback" });
    encoder.copyBufferToBuffer(contacts.buffer, 0, readback.buffer, 0, pairs * 4);
    encoder.copyBufferToBuffer(certainty.buffer, 0, readCertainty.buffer, 0, tokens * 4);
    encoder.copyBufferToBuffer(interfaceCertainty.buffer, 0, readInterface.buffer, 0,
                               tokens * 4);
    device.queue.submit([encoder.finish()]);
    await readback.buffer.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readback.buffer.getMappedRange().slice(0));
    readback.buffer.unmap();
    await readCertainty.buffer.mapAsync(GPUMapMode.READ);
    const perToken = new Float32Array(readCertainty.buffer.getMappedRange().slice(0));
    readCertainty.buffer.unmap();
    await readInterface.buffer.mapAsync(GPUMapMode.READ);
    // ...-1 where a token has no cross-chain partner, which a monomer's every
    // token does. A caller reads that as "not applicable", not as a low score.
    const perTokenInterface = new Float32Array(
      readInterface.buffer.getMappedRange().slice(0));
    readInterface.buffer.unmap();
    // 🔴 THE RETAINED BUFFERS LEAVE `held` NOW, NOT WHEN THEY ARE RELEASED. The
    // first version handed the scorer a closure that spliced them out of the
    // release list - and that closure runs when the CALLER is finished, long
    // after this function's `finally` has already freed them. The failure was
    // "[Buffer esmfold2.disto.certainty-readback] is destroyed" on the first
    // frame, which names the buffer and not the lifetime.
    const retained = retainForFrames
      ? [logits, biasBuffer, modes, mass, certainty, readCertainty, partnerBuffer,
         interfaceCertainty] : [];
    for (const allocation of retained) {
      const at = held.indexOf(allocation);
      if (at >= 0) held.splice(at, 1);
    }
    const frames = retainForFrames ? await framesScorer({
      device, allocator, cache, submit, key, tokens, bins, span, cutoffBins,
      logits, biasBuffer, modes, mass, certainty, readCertainty, partnerBuffer,
      interfaceCertainty,
      release: () => { for (const allocation of retained) allocation.release(); },
    }) : undefined;
    if (!wantLogits) {
      return { contacts: out, certainty: perToken, interface: perTokenInterface, frames };
    }
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
    return { contacts: out, certainty: perToken, interface: perTokenInterface,
             frames, logits: values, bias };
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
  const { cutoffBins, logits, biasBuffer, modes, mass, certainty, readCertainty,
          partnerBuffer, interfaceCertainty } = context;
  const storage = GPUBufferUsage.STORAGE;
  const observed = await cache.get(`${key}:observed:${tokens}`,
    createObservedMassShader({ tokens, bins }, span));
  const aggregate = await cache.get(`${key}:certain-token:${tokens}`,
    createCertaintyShader({ tokens, separation: CERTAINTY.separation, cutoffBins }));
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
        ["aggregate", aggregate,
         [mass, modes, partnerBuffer, certainty, interfaceCertainty],
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
