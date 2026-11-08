/**
 * BioEmu2's sequence embedder in WebGL2: `s_init`, `z_init` and the MSA input.
 *
 *     s_init = s_init(s_inputs) + method_embedding[method]
 *     z_init = rel_pos(residue, token, entity, chain)
 *            + z_init_1(s_inputs)[i] + z_init_2(s_inputs)[j]
 *            + method_pair_embedding, in both of those
 *            + token_bonds * w
 *            + the contact conditioning's class for the pair
 *     m0     = msa_proj([one_hot | has_deletion | deletion_value | paired])
 *            + s_proj(s_inputs)[token]
 *
 * 🔴 THE PER-TOKEN PROJECTIONS ARE HOST WORK AND THE PAIRWISE SUM IS NOT, which
 * is the same split every model in this tree makes: `z_init_1` and `z_init_2`
 * are 2T rows of 455 channels, and broadcasting them first would make them 2T^2.
 *
 * 🔴 AND THE CONTACT CONDITIONING'S CLASSES 0 AND 1 ARE LEARNED CONSTANTS while
 * a real contact (class >= 2) runs an encoder over its own threshold - so a fold
 * with no restraints still gets a term, `encoding_unspecified`, on every pair.
 * Leaving it out because "there are no contacts" drops a constant the model was
 * trained with.
 *
 * 🔴 AND THE RELATIVE ENCODING'S FOUR BLOCKS CLAMP AND FALL BACK DIFFERENTLY:
 * the residue block sends a different-CHAIN pair out of range, the token block a
 * pair that is not in the same chain AND the same residue, and the chain block a
 * pair whose ENTITIES differ - while `embed_same_entity` is added only when they
 * are the same. Four nearly-identical lines, four different conditions.
 */
import { ONE_TARGET, PRELUDE, memoSource } from "../runtime/execution.js";
import { BIOEMU2 } from "../../shared/bioemu2/weights.js";
import { linearRows } from "../../cpu/bioemu2/host.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

/** The four relative-position tables, summed into the pair. */
const relativeSourceUncached = (tokens) => `${PRELUDE}${ONE_TARGET}
const int L = ${tokens};
const int C4 = 64;
const int ELEMENTS = ${tokens * tokens * 256};
uniform sampler2D identityTex;
uniform sampler2D resTex;
uniform sampler2D tokTex;
uniform sampler2D entityTex;
uniform sampler2D chainTex;

int clipBin(int value, int high) { return min(max(value, 0), high); }

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;

  // The four lanes a token carries: asym, entity, residue, sym.
  int asymI = int(fetch1(identityTex, i * 4));
  int asymJ = int(fetch1(identityTex, j * 4));
  int entityI = int(fetch1(identityTex, i * 4 + 1));
  int entityJ = int(fetch1(identityTex, j * 4 + 1));
  int resI = int(fetch1(identityTex, i * 4 + 2));
  int resJ = int(fetch1(identityTex, j * 4 + 2));
  int symI = int(fetch1(identityTex, i * 4 + 3));
  int symJ = int(fetch1(identityTex, j * 4 + 3));
  bool sameChain = asymI == asymJ;
  bool sameResidue = resI == resJ;
  bool sameEntity = entityI == entityJ;

  int dres = sameChain ? clipBin(resI - resJ + 32, 64) : 65;
  // ...the token block's offset is the TOKEN POSITION, i - j, not a feature.
  int dtok = (sameChain && sameResidue) ? clipBin(i - j + 32, 64) : 65;
  int dchain = sameEntity ? clipBin(symI - symJ + 2, 4) : 5;

  vec4 total = fetch4(resTex, dres * C4 + group)
    + fetch4(tokTex, dtok * C4 + group)
    + fetch4(chainTex, dchain * C4 + group);
  if (sameEntity) total += fetch4(entityTex, group);
  result = total;
}`;

/**
 * The rest of `z_init` in one draw: the two per-token projections broadcast, the
 * bond term and the contact class.
 */
const assembleSourceUncached = (tokens) => `${PRELUDE}${ONE_TARGET}
const int L = ${tokens};
const int C4 = 64;
const int ELEMENTS = ${tokens * tokens * 256};
uniform sampler2D relativeTex;
uniform sampler2D abTex;
uniform sampler2D bondTex;
uniform sampler2D bondWeightTex;
uniform sampler2D classTex;
uniform sampler2D slotTex;
uniform sampler2D tableTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;

  vec4 total = fetch4(relativeTex, texel)
    + fetch4(abTex, i * C4 + group)
    + fetch4(abTex, (L + j) * C4 + group)
    + fetch1(bondTex, pair) * fetch4(bondWeightTex, group);
  int cls = int(fetch1(classTex, pair));
  if (cls == 0) {
    total += fetch4(tableTex, group);
  } else if (cls == 1) {
    total += fetch4(tableTex, C4 + group);
  } else {
    total += fetch4(tableTex, 2 * C4 + int(fetch1(slotTex, pair)) * C4 + group);
  }
  result = total;
}`;

/** `m0[s][t][c] = msa_proj(one_hot, deletions, paired) + s_proj(s_inputs)[t]`. */
const msaInputSourceUncached = (sequences, tokens) => `${PRELUDE}${ONE_TARGET}
const int L = ${tokens};
const int C4 = 32;
const int ELEMENTS = ${sequences * tokens * 128};
uniform sampler2D codeTex;
uniform sampler2D featureTex;
uniform sampler2D weightTex;
uniform sampler2D singleTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int row = texel / C4;
  int token = row % L;
  int code = int(fetch1(codeTex, row));
  // The table is (36, 128) here - transposed from the bundle's (128, 36) by the
  // caller, so that a row of it is one texel fetch.
  vec4 total = fetch4(weightTex, code * C4 + group)
    + fetch1(featureTex, row * 3) * fetch4(weightTex, 33 * C4 + group)
    + fetch1(featureTex, row * 3 + 1) * fetch4(weightTex, 34 * C4 + group)
    + fetch1(featureTex, row * 3 + 2) * fetch4(weightTex, 35 * C4 + group)
    + fetch4(singleTex, token * C4 + group);
  result = total;
}`;

/**
 * The relative encoding alone, as its own tensor.
 *
 * 🔴 IT IS HERE AND NOT IN THE DENOISER BECAUSE THE DENOISER RUNS IT TOO, from
 * its own `rel_pos` group: the pairwise conditioner concatenates `z_trunk` with
 * an encoding of exactly these four tables over exactly these four conditions.
 * Two copies of this kernel is two places for a clamp to be wrong in.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{features: any, tokens: number, tables: {relPos: Tensor,
 *   relToken: Tensor, sameEntity: Tensor, relChain: Tensor}}} call
 * @returns {Tensor} `tokens^2 * 256`, the caller's to release
 */
export function relativeEncoding(execution, call) {
  const { features: f, tokens, tables } = call;
  const identity = execution.upload(tokenIdentity(f, tokens), "bioemu2.identity");
  const out = execution.allocate(tokens * tokens * 256, "bioemu2.relative");
  execution.dispatch({
    source: relativeSource(tokens),
    output: out,
    inputs: {
      identityTex: identity,
      resTex: tables.relPos,
      tokTex: tables.relToken,
      entityTex: tables.sameEntity,
      chainTex: tables.relChain,
    },
    label: `bioemu2.relative:${tokens}`,
  });
  execution.release(identity);
  return out;
}

/**
 * The contact conditioning's table: two learned constants and one encoder output
 * per contact pair.
 *
 * 🔴 THE FOURIER TERM IS OF THE THRESHOLD, NOT OF THE DISTANCE, and the
 * threshold is `(t - 4) / 16` before it is embedded - the same rescaling the
 * encoder's own four leading columns use.
 *
 * @returns {{table: Float32Array, classes: Float32Array, slots: Float32Array}}
 */
export function contactTable(weights, features, tokens) {
  const pairs = tokens * tokens;
  const classes = new Float32Array(pairs);
  const slots = new Float32Array(pairs).fill(-1);
  const special = [];
  for (let pair = 0; pair < pairs; pair += 1) {
    let cls = 0;
    for (let k = 0; k < 5; k += 1) {
      if (features.contact_conditioning[pair * 5 + k]) cls = k;
    }
    classes[pair] = cls;
    if (cls < 2) continue;
    const threshold = (features.contact_threshold[pair] - 4) / 16;
    const row = new Float64Array(260);
    for (let k = 0; k < 3; k += 1) {
      row[k] = features.contact_conditioning[pair * 5 + 2 + k];
    }
    row[3] = threshold;
    for (let k = 0; k < 256; k += 1) {
      row[4 + k] = Math.cos(2 * Math.PI * (threshold
        * weights["fourier_embedding/proj/weight"][k]
        + weights["fourier_embedding/proj/bias"][k]));
    }
    const encoded = new Float32Array(256);
    for (let c = 0; c < 256; c += 1) {
      let sum = weights["encoder/bias"][c];
      for (let k = 0; k < 260; k += 1) {
        sum += row[k] * weights["encoder/weight"][c * 260 + k];
      }
      encoded[c] = sum;
    }
    slots[pair] = special.length;
    special.push(encoded);
  }
  const table = new Float32Array(Math.max(1, special.length) * 256 + 512);
  table.set(weights.encoding_unspecified, 0);
  table.set(weights.encoding_unselected, 256);
  special.forEach((values, index) => table.set(values, 512 + index * 256));
  return { table, classes, slots };
}

/** The four per-token lanes the relative encoding reads, interleaved. */
export function tokenIdentity(features, tokens) {
  const out = new Float32Array(tokens * 4);
  for (let token = 0; token < tokens; token += 1) {
    out[token * 4] = features.asym_id[token];
    out[token * 4 + 1] = features.entity_id[token];
    out[token * 4 + 2] = features.residue_index[token];
    out[token * 4 + 3] = features.sym_id[token];
  }
  return out;
}

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{features: any, sInputs: Float32Array, tokens: number,
 *   sequences: number, hostWeights: Record<string, Float32Array>,
 *   contact: Record<string, Float32Array>}} call
 *   `hostWeights` is the `sequence_embedder` group under its own names and
 *   `contact` its `contact_conditioning` subgroup, the same split the WGSL side
 *   reads them in.
 * @returns {{zInit: Tensor, sInit: Float32Array, m0: Tensor, valid: number}}
 *   `zInit` and `m0` are the caller's to release; everything else this builds
 *   goes back before it returns.
 */
export function sequenceEmbedder(execution, call) {
  const {
    features: f, sInputs, tokens, sequences, hostWeights: w, contact,
  } = call;
  const pairs = tokens * tokens;
  const single = BIOEMU2.single;
  const width = BIOEMU2.sInputs;
  const channels = BIOEMU2.msa;
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(
      values instanceof Float32Array ? values : Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };

  // s_init, on the host: T rows of 384 channels.
  const sInit = linearRows(sInputs, tokens, width, w["s_init/weight"], single);
  for (let token = 0; token < tokens; token += 1) {
    const from = f.method_feature[token] * single;
    for (let c = 0; c < single; c += 1) {
      sInit[token * single + c] += w["method_conditioning/embedding/weight"][from + c];
    }
  }

  // The two pairwise projections, also per token, with the method's pair term.
  const rows = linearRows(sInputs, tokens, width, w["z_init_1/weight"], 256);
  const columns = linearRows(sInputs, tokens, width, w["z_init_2/weight"], 256);
  for (let token = 0; token < tokens; token += 1) {
    const from = f.method_feature[token] * 256;
    for (let c = 0; c < 256; c += 1) {
      rows[token * 256 + c] += w["method_conditioning/pair_embedding/weight"][from + c];
      columns[token * 256 + c] += w["method_conditioning/pair_embedding/weight"][from + c];
    }
  }
  const ab = new Float32Array(2 * tokens * 256);
  ab.set(rows);
  ab.set(columns, tokens * 256);

  const relative = relativeEncoding(execution, {
    features: f,
    tokens,
    tables: {
      relPos: upload(w["rel_pos/embed_rel_pos/weight"], "w.rel-pos"),
      relToken: upload(w["rel_pos/embed_rel_token/weight"], "w.rel-token"),
      sameEntity: upload(w["rel_pos/embed_same_entity/weight"], "w.same-entity"),
      relChain: upload(w["rel_pos/embed_rel_chain/weight"], "w.rel-chain"),
    },
  });
  const { table, classes, slots } = contactTable(contact, f, tokens);
  const zInit = execution.allocate(pairs * 256, "bioemu2.z-init");
  execution.dispatch({
    source: assembleSource(tokens),
    output: zInit,
    inputs: {
      relativeTex: relative,
      abTex: upload(ab, "bioemu2.z-init.ab"),
      bondTex: upload(Float32Array.from(f.token_bonds), "bioemu2.bonds"),
      bondWeightTex: upload(w["token_bonds/weight"], "w.token-bonds"),
      classTex: upload(classes, "bioemu2.contact-class"),
      slotTex: upload(slots, "bioemu2.contact-slot"),
      tableTex: upload(table, "bioemu2.contact-table"),
    },
    label: `bioemu2.z-init:${tokens}`,
  });
  execution.release(relative);

  // m0, and the number of VALID MSA rows the outer product mean divides by.
  const singleProjected = linearRows(sInputs, tokens, width,
    w["msa_module/s_proj/weight"], channels);
  // The bundle's (128, 36) transposed, so a row of the table is one texel fetch.
  const msaProj = w["msa_module/msa_proj/weight"];
  const transposed = new Float32Array(36 * channels);
  for (let c = 0; c < channels; c += 1) {
    for (let k = 0; k < 36; k += 1) {
      transposed[k * channels + c] = msaProj[c * 36 + k];
    }
  }
  const msaFeatures = new Float32Array(sequences * tokens * 3);
  for (let row = 0; row < sequences * tokens; row += 1) {
    msaFeatures[row * 3] = f.has_deletion[row];
    msaFeatures[row * 3 + 1] = f.deletion_value[row];
    msaFeatures[row * 3 + 2] = f.msa_paired[row];
  }
  const m0 = execution.allocate(sequences * tokens * channels, "bioemu2.m0");
  execution.dispatch({
    source: msaInputSource(sequences, tokens),
    output: m0,
    inputs: {
      codeTex: upload(Float32Array.from(f.msa), "bioemu2.msa-codes"),
      featureTex: upload(msaFeatures, "bioemu2.msa-features"),
      weightTex: upload(transposed, "w.msa-proj"),
      singleTex: upload(singleProjected, "bioemu2.msa-single"),
    },
    label: `bioemu2.m0:${sequences}x${tokens}`,
  });
  for (const tensor of held) execution.release(tensor);

  // 🔴 A ROW IS VALID IF ANY OF ITS COLUMNS IS LIVE, and the divisor is at least
  // one: an all-masked alignment would otherwise divide by zero.
  let valid = 0;
  for (let sequence = 0; sequence < sequences; sequence += 1) {
    let any = false;
    for (let token = 0; token < tokens; token += 1) {
      any = any || f.msa_mask[sequence * tokens + token] !== 0;
    }
    if (any) valid += 1;
  }
  return { zInit, sInit, m0, valid: Math.max(valid, 1) };
}

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const relativeSource = memoSource(relativeSourceUncached);
export const assembleSource = memoSource(assembleSourceUncached);
export const msaInputSource = memoSource(msaInputSourceUncached);
