/**
 * The fused template embedder's feature columns (protenix2's 108, boltz2's 109, rosettafold3's 66), dense and
 * sparse - out of webgpu/af3/trunk/template-webgpu.js (which re-exports them): featurisation, which the CUDA port's
 * exporter and the page both build from, so it does not live beside the WebGPU embedder.
 */
import { DGRAM_BINS, boltz2TemplateFeatures, rosettafold3TemplateFeatures, templateGeometry } from "./template-features.js";

/**
 * The fused embedder's 108 feature columns for ONE slot.
 *
 * 🔴 EMPTY SLOTS ONLY, AND IT SAYS SO RATHER THAN GUESSING. A de novo fold has
 * four empty slots and their columns are constant: every geometry feature is
 * exactly zero and both restype blocks are one-hot at GAP. That is measured -
 * `EMPTY=1 tools/oracle/dump_af3_template.py protenix2` - not inferred from the
 * featuriser, whose empty-slot convention differs per vendor (protenix fills the
 * first slot with GAP, opendde all four, intellifold2 deliberately uses 0).
 *
 * A slot WITH a template needs the real featuriser: Boltz's frame convention,
 * the 39 bin edges, the 32-class remap and the multichain masking, all specified
 * in docs/AF3.md and gated by nothing yet. Building it from that specification
 * and checking it against a reference built the same way would prove nothing.
 */
export function fusedTemplateFeatures(template, tokens, width, dialect,
                                      multichainMask2d = undefined, useGap = true) {
  if (template !== undefined && template !== null) {
    // 🔴 THE 108 COLUMNS ARE THE NINE-PROJECTION EMBEDDER'S OWN FEATURES,
    // CONCATENATED. 39 distogram + 1 pseudo-beta mask + 32 restype_i + 32
    // restype_j + 3 unit vector + 1 backbone frame mask = 108, and
    // `templateGeometry` already computes four of the six for AF3's path -
    // measured against af3-any-model's `our_features`, the distogram and both
    // masks are EXACT and the unit vector is 3.64e-7. So this is a
    // concatenation of things this port has had all along, which is why the
    // refusal that stood here was costing more than it protected: boltz2 and
    // protenix2 could not take a template at all while their forward scored
    // 1.52e-7 against the oracle.
    //
    // 🔴 AND `restype_i` VARIES ALONG j, `restype_j` ALONG i. The name says
    // which index the tensor varies along in the reference's own naming, not
    // which one selects its value; built the other way both columns score 1.36
    // and the fold is plausible. The aatype needs NO remap - the table
    // recovered from the dump is the identity - which is worth stating because
    // docs/AF3.md's "32-class remap" reads as though it does.
    // See tools/gpu/check-fused-template-features.js.
    // 🔴 BOLTZ-2's 109 ARE A DIFFERENT CONSTRUCTION, NOT WIDER ONES. 38 bins on
    // different edges, a unit vector that is a SIGN, a restype vocabulary
    // shifted by two over 33 classes, and `restype_i` varying along i where
    // protenix2's varies along j. See boltz2TemplateFeatures.
    if (dialect?.boltz2TemplateFeatures === true) {
      return boltz2TemplateFeatures(template, multichainMask2d, tokens);
    }
    // 🔴 AND RoseTTAFold3's 66 ARE NOT A TEMPLATE IN THE OTHER TWO'S SENSE.
    // They are a CA-CA distance histogram, a coverage flag and a noise level -
    // distance-distribution conditioning rather than a geometry embedding -
    // riding the identical weight scopes, which is why only `a_proj`'s first
    // dimension tells the three apart: 66 against 108 and 109. See
    // rosettafold3TemplateFeatures.
    if (dialect?.rosettafold3TemplateFeatures === true) {
      return rosettafold3TemplateFeatures(template, multichainMask2d, tokens);
    }
    const columnsFor = dialect?.fusedTemplateLayout;
    if (columnsFor === undefined || columnsFor === null) {
      throw new Error("dialect.fusedTemplateLayout has no default: protenix2's"
        + " 108 columns are 39/1/32/32/3/1 and boltz2's 109 are not the same"
        + " widths, and guessing the bin count is guessing the model");
    }
    const { distogramBins, restypes } = columnsFor;
    const geometry = templateGeometry(template, multichainMask2d, tokens);
    if (distogramBins !== DGRAM_BINS) {
      throw new Error(`this dialect wants ${distogramBins} distogram bins and`
        + ` templateGeometry computes ${DGRAM_BINS}: the bin edges are not the`
        + " same feature and nothing here has measured the other set");
    }
    const features = new Float32Array(tokens * tokens * width);
    const restypeI = distogramBins + 1;
    const restypeJ = restypeI + restypes;
    const vectorAt = restypeJ + restypes;
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        const pair = i * tokens + j;
        const base = pair * width;
        for (let bin = 0; bin < distogramBins; bin += 1) {
          features[base + bin] = geometry.distogram[pair * distogramBins + bin];
        }
        features[base + distogramBins] = geometry.pseudoBetaMask2d[pair];
        const ci = template.aatype[j], cj = template.aatype[i];
        if (ci >= 0 && ci < restypes) features[base + restypeI + ci] = 1;
        if (cj >= 0 && cj < restypes) features[base + restypeJ + cj] = 1;
        for (let axis = 0; axis < 3; axis += 1) {
          features[base + vectorAt + axis] = geometry.unitVector[pair * 3 + axis];
        }
        features[base + vectorAt + 3] = geometry.backboneMask2d[pair];
      }
    }
    return features;
  }
  // 🔴 WHICH COLUMNS AN EMPTY SLOT SETS IS THE DIALECT'S, AND THE TWO FUSED
  // MODELS DISAGREE. protenix2's 108 columns carry GAP in both restype blocks;
  // boltz2's 109 are all zero. Its whole feature ORDER is different too -
  // distogram 38 against 39, restypes 33 against 32 - which is why `a_proj`
  // refused a 108-wide build with "wants 109" rather than folding something
  // plausible.
  // 🔴 AND THE GAP GOES IN EVERY EMPTY SLOT HERE, NOT ONLY THE FIRST - WHICH IS
  // THE OPPOSITE OF THE NINE-PROJECTION PATH. protenix2's `template_aatype` is
  // 21 in slot 0 and 0 in slots 1..3, exactly like OpenDDE's, so gating the gap
  // to the first empty slot is the obvious symmetry - and MEASURED it makes the
  // trunk's `z_after_template` seam WORSE, 3.89e-3 to 4.55e-3. The fused
  // embedder does not consume `template_aatype`; it consumes 108 columns that
  // protenix's own featuriser builds, and those are not the same array. Left as
  // it is, on the measurement rather than on the symmetry. `useGap` is kept as
  // the arm for re-running that comparison.
  // 🔴 AND A PADDED SLOT IS NOT AN EMPTY ONE. protenix2's featuriser fills its
  // ONE empty template with the GAP restype and zero-pads the rest - but
  // `template_aatype = 0` is zero-padding of the AATYPE, and `one_hot(0, 32)`
  // is NOT a zero row: it sets restype column 0. So the four slots the trunk
  // runs are [gap, restype-0, restype-0, restype-0], not [gap, 0, 0, 0] and
  // not four gaps.
  //
  // All four measured against af3-any-model's own `evoformer/template_embedding`
  // on 6MRR, which has no template: four gaps read 7.39e-3, gap-then-ZERO read
  // worse still, and this reads what is below. Written the wrong way twice
  // before the batch dump was read carefully enough to notice that a one-hot of
  // zero is a one.
  const columns = emptyTemplateColumns(dialect, useGap);
  const pairs = tokens * tokens;
  const features = new Float32Array(pairs * width);
  for (let index = 0; index < pairs; index += 1) {
    for (const column of columns) features[index * width + column] = 1;
  }
  return features;
}

// The columns an empty slot sets to 1, every pair alike (see fusedTemplateFeatures' empty branch)
export function emptyTemplateColumns(dialect, useGap = true) {
  const layout = dialect?.fusedTemplateLayout;
  const gapColumns = dialect?.emptyTemplateRestypeColumns;
  const columns = useGap ? gapColumns
    : (layout === undefined || layout === null || gapColumns === null ? []
      : [layout.distogramBins + 1, layout.distogramBins + 1 + layout.restypes]);
  if (columns === undefined || columns === null) {
    throw new Error("dialect.emptyTemplateRestypeColumns has no default: an "
      + "empty template slot carries GAP under protenix2 and zeros under "
      + "boltz2, and guessing either is a different model");
  }
  return columns;
}

// 🔴 THE FEATURES SPARSE, AS THE SHADER CONSUMES THEM. Dense, a slot was 108-109 floats a pair - an empty slot's
// included, whose rows are all one row - built on the host and uploaded: 864 MB at 1,020 tokens for protenix2 with
// no template at all (measured through the native port's exporter, which built the same arrays). Each row instead
// carries its nonzero (column, value) pairs IN COLUMN ORDER, K the most any row has and the rest padding: the
// shader adds the same terms in the same order the dense loop did when it skipped the zeros, so the embed is
// byte-identical. Layout: [K, then rows x K x (column, f32 bits)], padding column SPARSE_PAD.
export const SPARSE_PAD = 0xFFFFFFFF;
const ONE_BITS = 0x3F800000;
export function sparseTemplateFeatures(dense, width) {
  const rows = dense.length / width;
  let K = 1;
  for (let r = 0; r < rows; r += 1) {
    let nz = 0;
    for (let c = 0; c < width; c += 1) if (dense[r * width + c] !== 0) nz += 1;
    if (nz > K) K = nz;
  }
  const packed = new Uint32Array(1 + rows * K * 2);
  const bits = new Float32Array(1), word = new Uint32Array(bits.buffer);
  packed[0] = K;
  for (let r = 0; r < rows; r += 1) {
    let at = 1 + r * K * 2, k = 0;
    for (let c = 0; c < width; c += 1) {
      const v = dense[r * width + c];
      if (v === 0) continue;
      bits[0] = v; packed[at] = c; packed[at + 1] = word[0]; at += 2; k += 1;
    }
    for (; k < K; k += 1, at += 2) packed[at] = SPARSE_PAD;
  }
  return packed;
}
// fusedTemplateFeatures, sparse - an empty slot's rows written from its columns, never built dense
export function fusedTemplateFeaturesSparse(template, tokens, width, dialect, multichainMask2d = undefined,
                                            useGap = true) {
  if (template !== undefined && template !== null) {
    return sparseTemplateFeatures(fusedTemplateFeatures(template, tokens, width, dialect, multichainMask2d, useGap), width);
  }
  const columns = [...emptyTemplateColumns(dialect, useGap)].sort((a, b) => a - b);
  const K = Math.max(1, columns.length), rows = tokens * tokens;
  const row = new Uint32Array(K * 2);
  for (let k = 0; k < K; k += 1) {
    row[2 * k] = k < columns.length ? columns[k] : SPARSE_PAD;
    row[2 * k + 1] = k < columns.length ? ONE_BITS : 0;
  }
  const packed = new Uint32Array(1 + rows * K * 2);
  packed[0] = K;
  for (let r = 0; r < rows; r += 1) packed.set(row, 1 + r * K * 2);
  return packed;
}
