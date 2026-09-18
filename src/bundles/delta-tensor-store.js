/**
 * A model stored as a DELTA on another, read as though it were whole.
 *
 * AlphaFold 2 is five models and a bundle is 97 MiB, so offering all five is
 * half a gigabyte a visitor and 97 MiB every time they switch. The five are one
 * training run continued five ways: the difference between two of them stores
 * at three bits where a model costs eight, which is 43 MiB, and folds the same
 * - 5CAJ chain A with model_3_ptm is 1.95 A from a delta against 1.94 from its
 * own bundle. tools/pack_delta_model.py writes one; this reads one.
 *
 * A delta bundle is an ordinary quantised bundle - int3, group 128,
 * asymmetric, the codec ESM-C already ships - plus a `delta` header saying what
 * each tensor is:
 *
 *   addTo   decode and ADD to the base's value
 *   whole   take from the delta and ignore the base (the structure module,
 *           which is never quantised, and the norms and biases)
 *   absent  the base has it and this model does not - model_3, model_4 and
 *           model_5 have no template embedder at all, and a name left in the
 *           manifest would have `templateWeights` build a stage the checkpoint
 *           cannot fill
 *
 * Anything the header does not mention comes from the base unchanged: the
 * residue-geometry tables are residue_constants and identical in every model.
 *
 * 🔴 THE BASE IS ROUNDED TO f16 BEFORE THE DELTA IS ADDED, because that is what
 * it was subtracted from. The packer reads the base BUNDLE and takes the f16 a
 * fold will hold, so reconstructing from the float32 decode instead would add
 * the delta to a slightly different number than it was made against. The
 * difference is small - f16 is 11 bits of mantissa against int8's 7 - and
 * getting it right costs one pass.
 */
export class DeltaTensorStore {
  #base;
  #delta;
  #header;
  #cache = new Map();

  /**
   * @param {object} base   the store the delta is added to
   * @param {object} delta  the store holding the delta bundle
   */
  constructor(base, delta) {
    const header = delta.manifest.delta;
    if (header === undefined) {
      throw new Error("this bundle carries no `delta` header - it is not a delta");
    }
    this.#base = base;
    this.#delta = delta;
    this.#header = { addTo: new Set(header.addTo), whole: new Set(header.whole),
                     absent: new Set(header.absent) };
    // 🔴 THE MANIFEST IS THE BASE'S, MINUS WHAT THIS MODEL DOES NOT HAVE. Every
    // consumer reads shapes and parameter tables from it, and both are the
    // base's - the two models are the same graph. What differs is which
    // stages exist, so an absent tensor is removed and the section that named
    // it goes with it, which is what makes `templateWeights` return null rather
    // than gather a tensor nothing can supply.
    const tensors = {};
    for (const [name, record] of Object.entries(base.manifest.tensors)) {
      if (!this.#header.absent.has(name)) tensors[name] = record;
    }
    this.manifest = { ...base.manifest, tensors,
                      bundle: { ...base.manifest.bundle, model: header.model ?? "delta" },
                      delta: header };
    if (this.#header.absent.size > 0) {
      for (const section of ["templateEmbedding"]) {
        const parameters = base.manifest[section]?.parameters;
        if (parameters === undefined) continue;
        const names = Object.values(parameters).flatMap((leaves) => Object.values(leaves));
        if (names.some((name) => this.#header.absent.has(name))) {
          if (!names.every((name) => this.#header.absent.has(name))) {
            throw new Error(`${section} is only partly absent from this delta, which is a`
              + " packing fault rather than a model without the stage");
          }
          delete this.manifest[section];
        }
      }
    }
  }

  #from(name) {
    if (this.#header.absent.has(name)) throw new Error(`missing tensor ${name}`);
    if (this.#header.whole.has(name) || this.#header.addTo.has(name)) return this.#delta;
    return this.#base;
  }

  async open(name) {
    if (this.#header.absent.has(name)) throw new Error(`missing tensor ${name}`);
    if (this.#header.addTo.has(name)) {
      await Promise.all([this.#base.open(name), this.#delta.open(name)]);
      return;
    }
    await this.#from(name).open(name);
  }

  shape(name) {
    if (this.#header.absent.has(name)) throw new Error(`missing tensor ${name}`);
    return this.#base.shape(name);
  }

  tensor(name) {
    let value = this.#cache.get(name);
    if (value === undefined) { value = this.#read(name); this.#cache.set(name, value); }
    return value;
  }

  async #read(name) {
    if (!this.#header.addTo.has(name)) return this.#from(name).tensor(name);
    const [held, added] = await Promise.all([
      this.#base.tensorAsFloat16(name), this.#delta.tensor(name),
    ]);
    if (held.length !== added.length) {
      throw new Error(`${name} is ${added.length} in the delta and ${held.length} in the base`);
    }
    const out = new Float32Array(held.length);
    for (let index = 0; index < out.length; index += 1) out[index] = held[index] + added[index];
    return out;
  }

  async tensorAsFloat16(name) {
    if (!this.#header.addTo.has(name)) return this.#from(name).tensorAsFloat16(name);
    return Float16Array.from(await this.tensor(name));
  }

  // 🔴 THERE IS DELIBERATELY NO `tensorSource` HERE, AND ITS ABSENCE IS THE
  // INTERFACE. A source is a shard plus a record - "the codes are these bytes"
  // - and no shard holds this model's codes: the base's are the base's and the
  // delta's are a difference. Returning either would decode the wrong numbers
  // into a buffer nothing downstream could tell was wrong. AlphaFoldFixture
  // asks whether the store HAS the method and reads through `tensor` when it
  // does not, so leaving it off routes every tensor through the reconstruction
  // above, which is always right.
  //
  // What it costs is the device weight decode: the codes never reach the GPU,
  // so the host does the work. Applying the delta ON the device is
  // `planBlockUpload(..., { accumulate: true })` - gated by
  // tools/gpu/check-delta-upload.js, which holds it to zero differing f16
  // results on real parameters - and wiring it needs the resident packer to run
  // a second pass over the entries it just filled, not a source invented here.

  prefetch() {
    this.#base.prefetch?.();
    this.#delta.prefetch?.();
  }
}
