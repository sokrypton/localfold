/**
 * The subgroup-matrix attention's GEOMETRY, and the device questions it asks.
 *
 * 🔴 IT IS HERE TO BREAK A CYCLE, AND THE CYCLE WAS REAL REUSE RATHER THAN A
 * MISFILING. AlphaFold 3's grid attention imported these seven symbols from
 * `src/kernels/attention-matrix.js` while AlphaFold 2's template module
 * imported AlphaFold 3's template featuriser, so `af3` and `evoformer` each
 * depended on the other and neither could be read, moved or reasoned about
 * alone. The other two cycles in `src/` were one leaf file filed under the
 * wrong heading; this one needed the shared surface taking out.
 *
 * What is shared is a GEOMETRY and four capability questions - what tile the
 * units want, how wide a padded head is, how the strides dodge the banks, how
 * many bytes that costs, and whether this device can run any of it. None of
 * that is AlphaFold 2's, and none of it emits a kernel: the kernel stays in
 * `evoformer/attention-matrix.js`, which imports these back and re-exports them
 * so its own callers are unchanged.
 *
 * 🔴 AND `allowsAttentionSubgroupSize` CAME WITH IT, because
 * `supportsAttentionMatrix` needs it and leaving it in `attention.js` would
 * have made `runtime` depend on `evoformer` - a worse layering than the cycle
 * it replaced. It is eight lines of `device.adapterInfo`, which is a runtime
 * question wherever it is written. `attention.js` imports and re-exports it,
 * so `probe-kernel.js` and `attention-subgroup-size.test.js` are unchanged.
 */
import { deviceProfile } from "../runtime/device-profile.js";

/** The subgroup size every attention kernel here indexes for. */
export const ATTENTION_SUBGROUP_SIZE = 32;

/**
 * Whether this device's subgroup may be the size a kernel indexes for.
 *
 * 🔴 THE RANGE IS ON `adapterInfo`, NOT ON `limits`.
 * `device.limits.maxSubgroupSize` is undefined, so a check defaulting it to
 * zero refuses every device including this one - silently, as an unsupported
 * kernel rather than an error.
 *
 * 🔴 UNKNOWN IS TREATED AS ALLOWED, DELIBERATELY. A browser that exposes the
 * subgroups feature without the size range would otherwise lose a fast path it
 * has been running correctly.
 */
export function allowsAttentionSubgroupSize(device, size = ATTENTION_SUBGROUP_SIZE) {
  const info = device.adapterInfo ?? device.info ?? {};
  const min = info.subgroupMinSize;
  const max = info.subgroupMaxSize;
  if (typeof min === "number" && size < min) return false;
  if (typeof max === "number" && size > max) return false;
  return true;
}

/** Lanes a subgroup, which every index here assumes. */
export const ATTENTION_MATRIX_SUBGROUP_SIZE = 32;
/** The unit's M, N and K on every device that offers f16 with an f32 result. */
export const ATTENTION_MATRIX_UNIT = 16;
const UNIT = ATTENTION_MATRIX_UNIT;

/**
 * 🔴 THE GEOMETRY IS A KNOB, BECAUSE THE FIRST GUESS AT IT WAS 1.26x SLOWER.
 * Two subgroups and a key tile of 32 - the shape this started as - measured
 * 147.0 ms against the register kernels' 116.9 across an 825-residue block's
 * four attentions. Nothing about that number says the units cannot pay; it says
 * a key tile of 32 pays six workgroup barriers and a full rescale of the
 * running output for every 32 keys, and the rescale alone moves ROWS x HEAD
 * floats through workgroup memory three times a tile. Both of those are FIXED
 * per tile, so both amortise in `keyTile` - and `subgroups` is how many queries
 * share one staged key tile. They are swept, not chosen: see
 * `attentionMatrixTile` in src/runtime/device-profile.js.
 */
export const ATTENTION_MATRIX_DEFAULT_TILE = { subgroups: 2, keyTile: 32 };

/**
 * The resolved geometry: what the shader, the storage and the dispatch share.
 *
 * Takes `{subgroups, keyTile}` or the string `"2x32"`, because `--tune=` splits
 * its argument on commas and so cannot carry an object at all.
 */
export function attentionMatrixGeometry(requested) {
  let asked = requested ?? {};
  if (typeof asked === "string") {
    const parts = asked.split("x").map(Number);
    if (parts.length !== 2 || !parts.every(Number.isSafeInteger)) {
      throw new RangeError(`a matrix attention tile reads "subgroupsXkeys"; got ${requested}`);
    }
    asked = { subgroups: parts[0], keyTile: parts[1] };
  }
  const subgroups = asked.subgroups ?? ATTENTION_MATRIX_DEFAULT_TILE.subgroups;
  const keyTile = asked.keyTile ?? ATTENTION_MATRIX_DEFAULT_TILE.keyTile;
  if (!Number.isSafeInteger(subgroups) || subgroups < 1 || subgroups > 16) {
    throw new RangeError(`subgroups wants 1..16, got ${subgroups}`);
  }
  if (!Number.isSafeInteger(keyTile) || keyTile < UNIT || keyTile % UNIT !== 0) {
    throw new RangeError(`keyTile wants a multiple of ${UNIT}, got ${keyTile}`);
  }
  return {
    subgroups,
    keyTile,
    // One subgroup owns one UNIT-row block of queries.
    rows: subgroups * UNIT,
    lanes: subgroups * ATTENTION_MATRIX_SUBGROUP_SIZE,
  };
}

/**
 * A head narrower than the unit is padded up to it.
 *
 * The channels past the head are never written and workgroup memory starts at
 * zero, so contracting over the padded width is contracting over the head. It
 * buys multiplies this kernel is not short of: the extra-MSA stack's heads are
 * eight channels wide and wait on memory, not on the units.
 */
export const paddedHead = (headDim) => Math.max(headDim, UNIT);

/**
 * 🔴 EVERY STRIDE IS ODD OR TWO PAST A POWER OF TWO, AGAINST THE BANKS.
 * Workgroup memory is thirty-two banks of four bytes, so lane `i` reading
 * element `i * stride` lands in bank `(i * stride) % 32` for f32 and
 * `(i * stride / 2) % 32` for f16. A head width of 32 is the worst case there:
 * the transposed key tile is read one row apart per lane, and at a stride of 32
 * halves that is `(16i) % 32` - two banks for thirty-two lanes. Padding to 34
 * gives `(17i) % 32`, and 17 is coprime with 32, so no two lanes collide.
 */
export const strides = (headDim, g) => ({
  head: paddedHead(headDim) + 2,
  key: g.keyTile + 2,
  // 🔴 THE SCORE ARRAY HOLDS BOTH S AND THE P V RESULT, so its row must fit the
  // wider of the two. A key tile of 16 against a head of 32 would otherwise
  // stride the second store by 17 and write each row over the next one.
  score: Math.max(g.keyTile, paddedHead(headDim)) + 1,
});

/** Workgroup bytes the kernel declares, which a device must permit. */
export function attentionMatrixStorageBytes(headDim, geometry) {
  const g = attentionMatrixGeometry(geometry);
  const s = strides(headDim, g);
  const rows = g.rows;
  const keys = g.keyTile;
  const head = paddedHead(headDim);
  // A matrix load or store reaches `offset + stride * rows`, not the last
  // element it touches, so each array is sized by that reach.
  return rows * s.head * 2                    // staged queries, f16
    + head * s.key * 2                        // staged keys, transposed, f16
    + keys * s.head * 2                       // staged values, f16
    // scores, then the P V result of each key tile, then - once, in the
    // epilogue - the running output on its way out of the registers.
    + rows * s.score * 4
    + rows * s.key * 2                        // probabilities, f16
    + keys * 4;                               // the mask, as an additive term
}

/** Whether this device can run it: the units, the shape, and the room. */
export function supportsAttentionMatrix(device, headDim, geometry) {
  if (!Number.isSafeInteger(headDim) || headDim % 4 !== 0 || headDim > 32 || headDim < 4) {
    return false;
  }
  let g;
  try { g = attentionMatrixGeometry(geometry); } catch { return false; }
  if (device?.features?.has("chromium-experimental-subgroup-matrix") !== true) return false;
  if (device.features.has("shader-f16") !== true) return false;
  if (device.features.has("subgroups") !== true) return false;
  const limits = device.limits ?? {};
  if ((limits.maxComputeInvocationsPerWorkgroup ?? 256) < g.lanes) return false;
  if ((limits.maxComputeWorkgroupStorageSize ?? 16384) < attentionMatrixStorageBytes(headDim, g)) {
    return false;
  }
  // 🔴 THE SUBGROUP MUST BE ABLE TO BE THIRTY-TWO LANES. Every index here
  // divides the workgroup into two subgroups of 32; a device whose subgroup is
  // 16 or 64 would silently pair the wrong lanes.
  //
  // 🔴 AND THE RANGE IS ON adapterInfo, NOT ON limits, WHICH COST A DEBUGGING
  // ROUND. `device.limits.maxSubgroupSize` is undefined here, so a check
  // defaulting it to zero refuses every device including this one - silently,
  // as an unsupported kernel rather than an error. src/kernels/attention.js
  // already had `allowsAttentionSubgroupSize` reading the right place.
  if (!allowsAttentionSubgroupSize(device, ATTENTION_MATRIX_SUBGROUP_SIZE)) return false;
  // 🔴 AND THE UNITS MUST BE THE SHAPE THIS SHADER DECLARES, WHICH HAVING THEM
  // AT ALL DOES NOT SAY. Every matrix in here is `<f16, 16, 16>` - UNIT is not
  // a tunable - and Metal supports 8x8 ONLY. So an M2 passes every check above,
  // announcing `chromium-experimental-subgroup-matrix`, `shader-f16`,
  // `subgroups` and a 32-lane subgroup, and then fails at pipeline creation
  // with "the MSL backend only supports 8x8 subgroup matrices" - a
  // GPUPipelineError out of a function whose whole contract is that a device
  // which will never compile this kernel is refused here instead.
  //
  // It became reachable when `matrixCapabilityTuning` started answering
  // `attentionMatrix` from the feature list, because the feature list says the
  // units exist and never says how big they are: AF2 and the multimer stopped
  // folding on this M2 entirely. The configs are the only place the shape is
  // written down.
  return deviceProfile(device).matrixConfigs.some(
    (c) => c.componentType === "f16" && c.M === UNIT && c.N === UNIT && c.K === UNIT);
}
