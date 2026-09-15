/**
 * AF3's template embedder, which contributes whether or not there are templates.
 *
 * 🔴 THIS IS THE MODULE EVERYONE SKIPS AND NOBODY SHOULD. On a de novo protein
 * with FOUR EMPTY TEMPLATE SLOTS its output measures std 13.1 against a pair
 * whose own std is 55 - about a quarter of the representation entering the MSA
 * stack. It is not a residual correction that a missing template makes zero.
 *
 * The reason is visible in the feature list below. Nine features are summed
 * into the embedding and only six of them are template geometry; the other
 * three are the QUERY's own aatype (twice, once per axis) and the query pair
 * representation itself. With no template the geometry vanishes and those three
 * do not, so the module becomes a learned transform of the query - and then
 * runs it through two pairformer blocks, a LayerNorm, a relu and a projection.
 *
 * AF2-multimer had the identical trap. It cost this project a week there.
 *
 * 🔴 REAL TEMPLATES USED TO RAISE, AND THE REASON WAS GOOD. The six geometry
 * features - a 39-bin pseudo-beta distogram, its mask, three components of a
 * unit vector in each residue's backbone frame, and the backbone mask - are
 * all identically zero when no template is present, so nothing here could tell
 * a correct implementation of them from a wrong one, and writing them anyway
 * would have been a hundred lines no measurement covers.
 *
 * `tools/oracle/dump_af3_trunk.py --template <pdb>` produces the measurement.
 * The features themselves are in src/af3/template-features.js, kept separate
 * because they are arithmetic over coordinates and can be checked before any
 * embedding is involved; `tools/oracle/check_af3_template.js` does both.
 */
import {
  gridSelfAttention, layerNorm, linear, transition, triangleMultiplication,
} from "./pairformer-reference.js";
import {
  DGRAM_BINS, coverageOf, multichainMaskFor, templateGeometry,
} from "./template-features.js";

const RESTYPES = 31;

/**
 * The template stack's own channel width, off the weights.
 *
 * 🔴 IT IS 64 IN FIVE CHECKPOINTS AND 256 IN IntelliFold-2, and it was a
 * `const CHANNELS = 64` here and in template-webgpu.js. `templateWeights`
 * reads it from the norm after the stack - `output_layer_norm/scale` for the
 * nine-projection embedder and `v_norm/scale` for the fused one - which is the
 * one tensor both forms carry that states it. Required rather than defaulted:
 * a caller handing weights from an older loader gets an error, not AF3's 64
 * silently applied to a wider stack.
 */
function stackChannels(weights) {
  const channels = weights.channels;
  if (!Number.isInteger(channels) || channels < 1) {
    throw new Error("template weights carry no `channels`: it is read from "
      + "output_layer_norm/scale (or v_norm/scale), never assumed to be 64");
  }
  return channels;
}

/** One block of the template stack: the pair half of a pairformer block. */
/** `transition1` is [CHANNELS, CHANNELS * factor * 2]; the gated form doubles. */
export function templateTransitionFactor(pairTransition, CHANNELS) {
  if (!Number.isInteger(CHANNELS)) {
    throw new Error("templateTransitionFactor needs the stack's channel width");
  }
  const factor = pairTransition.transition1.length / (CHANNELS * CHANNELS * 2);
  if (!Number.isInteger(factor) || factor < 1) {
    throw new Error(`a template transition1 of ${pairTransition.transition1.length} `
      + `is not CHANNELS * CHANNELS * 2 * factor for CHANNELS ${CHANNELS}`);
  }
  return factor;
}

function templateBlock(pair, pairMask, tokens, weights, dialect, CHANNELS) {
  let act = Float32Array.from(pair);
  const add = (delta) => {
    for (let index = 0; index < act.length; index += 1) act[index] += delta[index];
  };
  add(triangleMultiplication(act, pairMask, tokens, CHANNELS, "outgoing",
                             weights.triangleMultiplicationOutgoing, dialect));
  add(triangleMultiplication(act, pairMask, tokens, CHANNELS, "incoming",
                             weights.triangleMultiplicationIncoming, dialect));
  add(gridSelfAttention(act, pairMask, tokens, CHANNELS, false,
                        weights.pairAttention1, dialect));
  add(gridSelfAttention(act, pairMask, tokens, CHANNELS, true,
                        weights.pairAttention2, dialect));
  // ...factor 2 here, against the trunk's 4. See transition() in
  // pairformer-reference.js.
  // 🔴 THE FACTOR IS THE TENSOR'S, NOT A CONSTANT. AlphaFold 3 and protenix2
  // run a factor of 2 here against the trunk's 4; boltz2 runs 4, and its
  // transition1 is [64, 512] where theirs is [64, 256]. Typed in, both this and
  // the GPU path read a 512-wide gate as though it were 256 - each in its own
  // way, which is why they disagreed by 9e-1 rather than agreeing on a wrong
  // answer. `transition1` is [channels, channels * factor * 2]: the gated form
  // doubles it.
  add(transition(act, tokens * tokens, CHANNELS, weights.pairTransition,
                 templateTransitionFactor(weights.pairTransition, CHANNELS)));
  return act;
}

/**
 * The template embedding added to the pair representation.
 *
 * @param {{pair: Float32Array, tokens: number, pairMask: Float32Array,
 *          templates: number, slots?: (object|undefined)[],
 *          multichainMask2d?: ArrayLike<number>,
 *          onSlot?: (slot: number, embedded: Float32Array) => void}} input
 *   `slots` holds one entry per OCCUPIED slot - `{aatype, atomPositions,
 *   atomMask}` in AF3's dense-24 layout - with `undefined` for an empty one.
 * @param {object} weights
 * @param {{swapTransposedBias: boolean}} dialect
 * @returns {Float32Array} tokens * tokens * pairChannels
 */
/**
 * The FUSED template embedder - boltz2's module, which protenix2 also runs.
 *
 *     v = z_proj(z_norm(z)) + a_proj(a)
 *     v = v + pairformer(v)   x2
 *     v = v_norm(v);  mean over slots;  u_proj(relu(u))
 *
 * 🔴 IT TAKES THE 108 FEATURE COLUMNS, IT DOES NOT BUILD THEM. That is the seam
 * the reference's own template_parity.py uses, and its docstring is explicit
 * about why: the featuriser and the forward are separate jobs, so a gate that
 * derives features on both sides cannot tell a wrong projection from a wrong
 * frame convention. `oracle-dumps/af3-oracle-template-protenix2.json` carries
 * both halves separately for the same reason. The featuriser is NOT written
 * here; docs/AF3.md has its specification, including the two traps the
 * reference paid for.
 *
 * 🔴 AND A SUM OF PROJECTIONS IS ONE PROJECTION OF THE CONCATENATION, which is
 * why this is the same model as AF3's nine `template_pair_embedding_*` and not
 * a second one. The packing differs; the arithmetic does not.
 */
export function fusedTemplateEmbedding(input, weights, dialect) {
  const CHANNELS = stackChannels(weights);
  const { tokens, pair, pairMask, templates } = input;
  const pairs = tokens * tokens;
  const features = input.templateFeatures;
  if (features === undefined) {
    throw new Error("the fused template embedder needs `templateFeatures`: the "
      + "108 concatenated columns per pair, which this module projects rather "
      + "than derives. See docs/AF3.md for their order.");
  }
  const width = weights.featureWidth;
  if (features.length !== pairs * width) {
    throw new Error(`templateFeatures has ${features.length} elements; `
      + `expected ${pairs * width} (${pairs} pairs x ${width})`);
  }

  // v = z_proj(z_norm(z)) + a_proj(a). The query half does not depend on the
  // slot, so it is computed once - as in AF3's, and for the same reason.
  const normalised = layerNorm(pair, pairs, weights.queryChannels,
                               weights.queryEmbeddingNormScale,
                               weights.queryEmbeddingNormOffset);
  const queryTerm = linear(normalised, pairs, weights.queryChannels, CHANNELS,
                           weights.zProjection);
  const featureTerm = linear(features, pairs, width, CHANNELS, weights.aProjection);

  // 🔴 boltz2 MASKS BY WHAT THE TEMPLATE COVERS, SO AN EMPTY SLOT CONTRIBUTES
  // EXACTLY NOTHING. Measured, not inferred: af3-any-model's own module returns
  // rms 0.0000 for boltz2 with no template supplied, where protenix2's returns
  // 12.46 and AlphaFold 3's is also live. Both of ours produced ~1.0 and ~1.6 -
  // the z-dependent half that the other dialects keep - and disagreed with each
  // other because they build it differently, which is why this looked like a
  // GPU bug for a while and is a missing convention.
  const visibility = dialect?.templateVisibilityByCoverage === true;
  const summed = new Float32Array(pairs * CHANNELS);
  for (let slot = 0; slot < templates; slot += 1) {
    const here = input.slots?.[slot];
    if (visibility && (here === undefined || here === null)) continue;
    let act = new Float32Array(pairs * CHANNELS);
    for (let index = 0; index < act.length; index += 1) {
      act[index] = queryTerm[index] + featureTerm[index];
    }
    // 🔴 boltz2 WRAPS THE WHOLE STACK IN A RESIDUAL AND protenix2 DOES NOT,
    // though protenix2 inherited the rest of this module from it. The
    // reference's note is the one to keep: "protenix inherited the shared
    // forward and got the wrong convention; rf3 escaped by not inheriting it.
    // Either a per-vendor convention is named -- as it now is here -- or the
    // next subclass gets whichever behaviour its parent happened to have."
    // Worth 1.62e-2 on boltz2's whole trunk, where the pairformer, the MSA
    // stack and the embedder all pass on their own.
    if (dialect?.templateStackOuterResidual === undefined) {
      throw new Error("dialect.templateStackOuterResidual has no default: boltz2 "
        + "adds the stack's input back to its output and protenix2 does not");
    }
    const before = dialect.templateStackOuterResidual
      ? Float32Array.from(act) : undefined;
    for (let index = 0; index < weights.blocks.length; index += 1) {
      act = templateBlock(act, pairMask, tokens, weights.blocks[index], dialect, CHANNELS);
    }
    if (before !== undefined) {
      for (let index = 0; index < act.length; index += 1) act[index] += before[index];
    }
    act = layerNorm(act, pairs, CHANNELS, weights.outputLayerNormScale,
                    weights.outputLayerNormOffset);
    input.onSlot?.(slot, act);
    for (let index = 0; index < summed.length; index += 1) summed[index] += act[index];
  }

  // 🔴 DIVIDED BY THE SLOT COUNT, which is `templateMeanOverAllSlots` and is
  // what AF3's own path already does - see the note at the bottom of
  // templateEmbedding. So that dialect flag describes LocalFold's existing
  // arithmetic rather than asking for new arithmetic.
  const scale = 1 / (1e-7 + templates);
  const output = new Float32Array(pairs * weights.queryChannels);
  const relu = new Float32Array(pairs * CHANNELS);
  for (let index = 0; index < summed.length; index += 1) {
    relu[index] = Math.max(0, summed[index] * scale);
  }
  const projected = linear(relu, pairs, CHANNELS, weights.queryChannels,
                           weights.outputLinear);
  output.set(projected);
  return output;
}

export function templateEmbedding(input, weights, dialect) {
  const CHANNELS = stackChannels(weights);
  const { tokens, pair, pairMask, templates } = input;
  const pairs = tokens * tokens;
  const slots = input.slots ?? [];
  if (slots.length > templates) {
    throw new RangeError(`${slots.length} templates for ${templates} slots`);
  }
  // 🔴 THE OLD FLAG STILL REFUSES, RATHER THAN BEING IGNORED. Callers wrote
  // `templateOccupied: <does the dump have a template>` to fail loudly when
  // one appeared, back when this path could not handle it. Now that it can,
  // dropping the flag would turn that deliberate noise into silence: a dump
  // WITH a template would be folded WITHOUT one and simply score worse.
  if (input.templateOccupied === true && slots.filter(Boolean).length === 0) {
    throw new Error("templateOccupied is true but no slots were given:"
      + " pass `slots` with {aatype, atomPositions, atomMask} per template");
  }
  const EMPTY_MASK = new Float32Array(pairs);
  // 🔴 THE MASK IS PER SLOT AND IS NOT ALLOWED TO DEFAULT TO "EVERYTHING". It
  // did, and a two-chain query with a template on each chain then scored
  // relRMS 1.09 against AF3 - the cross-chain geometry is most of the module's
  // answer, so a permissive default is not a small error. It went unnoticed
  // because every check had a ONE-CHAIN query, where all-ones and per-chain
  // are the same array.
  const chainMaskFor = (template) => {
    if (input.multichainMask2d !== undefined) return input.multichainMask2d;
    if (input.asymId === undefined) {
      if (template === undefined || template === null) {
        // An empty slot has no geometry to mask, so the mask is unread.
        return EMPTY_MASK;
      }
      throw new Error("a template needs `asymId` (or `multichainMask2d`):"
        + " AF3 masks the geometry features across chains, and assuming one"
        + " chain silently lets a template speak about pairs it has never"
        + " seen in one coordinate frame");
    }
    return multichainMaskFor(input.asymId, tokens, {
      coverage: coverageOf(template, tokens),
      // ...opt in, and only where one structure covered both chains. See
      // multichainMaskFor.
      spanChains: template.spanChains === true,
    });
  };

  // Feature 8: the query pair representation, normalised. It does not depend on
  // the slot, so it and its projection are computed ONCE - which is most of the
  // module's arithmetic when the slots are empty and all of it when there are
  // none.
  const normalised = layerNorm(pair, pairs, weights.queryChannels,
                               weights.queryEmbeddingNormScale,
                               weights.queryEmbeddingNormOffset);
  const queryTerm = linear(normalised, pairs, weights.queryChannels, CHANNELS,
                           weights.templatePairEmbedding8);

  const summed = new Float32Array(pairs * CHANNELS);
  for (let slot = 0; slot < templates; slot += 1) {
    const template = slots[slot];
    const act = Float32Array.from(queryTerm);

    // Features 2 and 3: the TEMPLATE's aatype, once along each axis. An empty
    // slot still contributes a row of each weight rather than nothing, which is
    // half of why an empty slot is not a no-op.
    //
    // 🔴 AND WHICH ROW IS THE DIALECT'S, AND ONLY THE FIRST EMPTY SLOT GETS IT.
    // OpenDDE and protenix2 take protenix's featuriser, which "fills its one
    // empty template with the GAP restype and zero-pads the rest": on a query
    // with NO template their `template_aatype` is 21 across slot 0 and 0 across
    // slots 1..3, which the reference's own batch dumps show exactly. Writing 0
    // everywhere put row 0 (ALA) where row 21 belongs and was worth 2.83e-1 on
    // the module and 2.07e-2 on the trunk's `z_after_template` seam - a defect
    // that needed no template to appear, and the last open one in OpenDDE's
    // trunk. AlphaFold 3, openbind0 and boltz2 write 0 in every slot.
    const gap = dialect?.emptyTemplateAatype ?? null;
    // The first EMPTY slot, not slot zero: with a real template in slot 0 the
    // featuriser's "one empty template" is the first unoccupied one.
    const firstEmpty = slots.filter(Boolean).length;
    const emptyCode = gap !== null && slot === firstEmpty ? gap : 0;
    const oneHot = new Float32Array(tokens * RESTYPES);
    for (let token = 0; token < tokens; token += 1) {
      const code = template ? template.aatype[token] : emptyCode;
      if (code >= 0 && code < RESTYPES) oneHot[token * RESTYPES + code] = 1;
    }
    const row = linear(oneHot, tokens, RESTYPES, CHANNELS,
                       weights.templatePairEmbedding2);
    const column = linear(oneHot, tokens, RESTYPES, CHANNELS,
                          weights.templatePairEmbedding3);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        const base = (i * tokens + j) * CHANNELS;
        for (let c = 0; c < CHANNELS; c += 1) {
          // ...feature 2 is aatype[None, :, :], so it varies along j; feature 3
          // is aatype[:, None, :] and varies along i.
          act[base + c] += row[j * CHANNELS + c] + column[i * CHANNELS + c];
        }
      }
    }

    // Features 0, 1, 4, 5, 6 and 7: the geometry. All exactly zero for an empty
    // slot - the distogram is multiplied by a pseudo-beta mask that is zero,
    // the unit vectors by a backbone mask that is zero, and the two masks are
    // themselves two of the features - so an empty slot skips the work rather
    // than computing zeros.
    if (template !== undefined && template !== null) {
      const geometry = templateGeometry(template, chainMaskFor(template), tokens);
      for (let index = 0; index < pairs; index += 1) {
        const base = index * CHANNELS;
        for (let bin = 0; bin < DGRAM_BINS; bin += 1) {
          const value = geometry.distogram[index * DGRAM_BINS + bin];
          if (value === 0) continue;
          for (let c = 0; c < CHANNELS; c += 1) {
            act[base + c] += value * weights.templatePairEmbedding0[bin * CHANNELS + c];
          }
        }
        // 🔴 FEATURES 1, 4, 5, 6 AND 7 ARE SCALARS TIMES A [64] VECTOR, not
        // matrix products. AF3 builds them with `num_input_dims=0`, which makes
        // the weight a per-channel scale rather than a projection - so reading
        // any of these four as a [1, 64] matmul is right by accident and
        // reading them as [39, 64] or [31, 64] is a shape error that only
        // shows up as a wrong answer.
        const scalars = [
          [geometry.pseudoBetaMask2d[index], weights.templatePairEmbedding1],
          [geometry.unitVector[index * 3], weights.templatePairEmbedding4],
          [geometry.unitVector[index * 3 + 1], weights.templatePairEmbedding5],
          [geometry.unitVector[index * 3 + 2], weights.templatePairEmbedding6],
          [geometry.backboneMask2d[index], weights.templatePairEmbedding7],
        ];
        for (const [value, weight] of scalars) {
          if (value === 0) continue;
          for (let c = 0; c < CHANNELS; c += 1) act[base + c] += value * weight[c];
        }
      }
    }

    let embedded = act;
    for (let index = 0; index < weights.blocks.length; index += 1) {
      embedded = templateBlock(embedded, pairMask, tokens, weights.blocks[index], dialect,
                               CHANNELS);
    }
    embedded = layerNorm(embedded, pairs, CHANNELS, weights.outputLayerNormScale,
                         weights.outputLayerNormOffset);
    // 🔴 REPORTED PER SLOT, BECAUSE THE SUM HIDES WHICH SLOT WAS WRONG. AF3
    // captures `single_template_embedding/__call__#k` at exactly this point -
    // after the two blocks and the LayerNorm, before the summation and the
    // output projection - so a checker can hold ONE slot's 64 channels to it
    // and see the geometry on its own. Summed and projected, a wrong unit
    // vector and a wrong distogram bin are the same number.
    input.onSlot?.(slot, embedded);
    for (let index = 0; index < summed.length; index += 1) summed[index] += embedded[index];
  }

  // 🔴 DIVIDED BY THE SLOT COUNT, NOT BY HOW MANY SLOTS ARE REAL. Four empty
  // slots each produce the SAME embedding, so the sum is four times one of them
  // and the division puts it back - the module behaves as though there were
  // exactly one template, whatever the slot count. With one real template among
  // four slots the real one is therefore worth a QUARTER of what it would be
  // alone, which is AF3's arithmetic and not an oversight to correct.
  const scale = 1 / (1e-7 + templates);
  for (let index = 0; index < summed.length; index += 1) {
    // ...relu before the projection, so the module can only add along the
    // directions its output_linear selects from a non-negative combination.
    summed[index] = Math.max(0, summed[index] * scale);
  }
  return linear(summed, pairs, CHANNELS, weights.queryChannels, weights.outputLinear);
}
