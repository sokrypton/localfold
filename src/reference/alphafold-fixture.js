import { readTensorRange } from "./dtype.js";

function transpose(input, rows, columns) {
  const output = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      output[column * rows + row] = input[row * columns + column];
    }
  }
  return output;
}

/**
 * Where a descriptor's tensors came from, for a caller that wants the CODES.
 *
 * 🔴 A SYMBOL, SO IT IS NOT A FIELD, for the reason src/esmfold2/weights.js
 * gives: every packer here walks a descriptor's own properties and a string key
 * would look like another tensor to all of them.
 */
// One symbol for every loader; see src/runtime/weight-sources.js for what two
// of them cost. Imported AND re-exported, because a bare `export ... from`
// does not bind the name in this module and every use here is local.
import { SOURCES } from "../runtime/weight-sources.js";

export { SOURCES };

export class AlphaFoldFixture {
  store;
  manifest;

  constructor(store) {
    this.store = store;
    this.manifest = store.manifest;
  }

  static fromStore(store) { return new AlphaFoldFixture(store); }

  tensor(name) { return this.store.tensor(name); }
  shape(name) { return this.store.shape(name); }

  async #parameter(
    parameters,
    module,
    name,
    block,
    blocks,
  ) {
    const tensorName = parameters[module]?.[name];
    if (tensorName === undefined) throw new Error(`missing ${module}/${name}`);
    const value = await this.store.tensor(tensorName);
    if (block === undefined) return value;
    if (blocks === undefined) throw new Error("stacked parameter requires a block count");
    const size = value.length / blocks;
    return value.subarray(block * size, (block + 1) * size);
  }

  /**
   * A descriptor's tensors, LAZILY when the store can hand over their bytes.
   *
   * 🔴 AF2's BUNDLE IS int8 AND DECODING IT IS 445 ms BEFORE A FOLD BEGINS.
   * 283 of its 337 tensors are int8 at a group of 64 and 93.1 M elements, and
   * the host loop that widens them is already at JavaScript's floor - benched,
   * `out[i] = codes[i]` with no arithmetic at all is 263 Melem/s against the
   * shipped 254. So the way out is not to run it: the GPU decoder takes int8
   * now, and every packer that can bind codes instead of values leaves this
   * getter uncalled.
   *
   * The eager path is what a store with no `tensorSource` gets, which is every
   * fixture built over a plain object rather than an HTTP store.
   *
   * @param {[string, string, string][]} entries [property, module, key]
   * @param {(name: string) => object} [reshape] per property, a mapping the
   *   host applies and the device path must reproduce - see `transposed`.
   */
  async #gather(parameters, block, blocks, entries, reshape) {
    const out = {};
    const sources = {};
    const lazy = typeof this.store.tensorSource === "function"
      && typeof this.store.open === "function";
    for (const [property, module, key] of entries) {
      const tensorName = parameters[module]?.[key];
      if (tensorName === undefined) throw new Error(`missing ${module}/${key}`);
      const shaped = reshape?.(property);
      if (!lazy) {
        const value = await this.#parameter(parameters, module, key, block, blocks);
        out[property] = shaped === undefined
          ? value : transpose(value, shaped.rows, shaped.columns);
        continue;
      }
      // eslint-disable-next-line no-await-in-loop
      await this.store.open(tensorName);
      const source = this.store.tensorSource(tensorName);
      const elements = (source.record.shape ?? []).reduce((a, b) => a * b, 1);
      const count = block === undefined ? elements : elements / blocks;
      const first = block === undefined ? 0 : block * count;
      sources[property] = { ...source, tensorName, first, count,
                            ...(shaped === undefined ? {} : { transpose: shaped }) };
      let decoded;
      Object.defineProperty(out, property, {
        enumerable: true,
        get() {
          if (decoded !== undefined) return decoded;
    const range = readTensorRange(source.record, source.buffer, source.byteOffset,
                                        first, count, true);
          decoded = shaped === undefined
            ? range : transpose(range, shaped.rows, shaped.columns);
          return decoded;
        },
      });
    }
    if (lazy) Object.defineProperty(out, SOURCES, { value: sources });
    return out;
  }

  #parameterShape(parameters, module, name, stacked) {
    const tensorName = parameters[module]?.[name];
    if (tensorName === undefined) throw new Error(`missing ${module}/${name}`);
    const shape = this.store.shape(tensorName);
    return stacked ? shape.slice(1) : shape;
  }

  async #attention(
    parameters, root, block, blocks,
  ) {
    const attentionRoot = `${root}/attention`;
    const weights = await this.#gather(parameters, block, blocks, [
      ["queryNormScale", `${root}/query_norm`, "scale"],
      ["queryNormOffset", `${root}/query_norm`, "offset"],
      ["queryWeight", attentionRoot, "query_w"],
      ["keyWeight", attentionRoot, "key_w"],
      ["valueWeight", attentionRoot, "value_w"],
      ["gatingWeight", attentionRoot, "gating_w"],
      ["gatingBias", attentionRoot, "gating_b"],
      ["outputWeight", attentionRoot, "output_w"],
      ["outputBias", attentionRoot, "output_b"],
    ]);
    return {
      heads: this.#parameterShape(parameters, attentionRoot, "gating_b", true)[0],
      attention: weights,
    };
  }

  async #triangleAttention(
    parameters, root, block, blocks,
  ) {
    const result = await this.#attention(parameters, root, block, blocks);
    // 🔴 SPREAD, WHICH IS SAFE HERE AND WOULD NOT BE ONE LEVEL DOWN. `result`
    // holds `heads` and a REFERENCE to the gathered attention object, so
    // copying it copies the reference; spreading the gathered object itself
    // would call every getter and decode the block. See src/esmfold2/weights.js,
    // where exactly that cost a second.
    // 🔴 NAMED THE WAY THE PAIR-BIAS DESCRIPTOR NAMES IT, not the way this
    // object does. `packAttentionWeights` reads `pairBias.projectionWeight`, so
    // the gathered object IS that descriptor's base - see `pairBiasFrom` in
    // src/evoformer/block.js - and `pairProjectionWeight` is a getter onto it
    // for the readers that predate this.
    const pairBias = await this.#gather(parameters, block, blocks, [
      ["projectionWeight", root, "feat_2d_weights"],
    ]);
    return { ...result, pairBias,
             get pairProjectionWeight() { return pairBias.projectionWeight; } };
  }

  async #transition(
    parameters, root, block, blocks,
  ) {
    return this.#gather(parameters, block, blocks, [
      ["layerNormScale", `${root}/input_layer_norm`, "scale"],
      ["layerNormOffset", `${root}/input_layer_norm`, "offset"],
      ["firstWeight", `${root}/transition1`, "weights"],
      ["firstBias", `${root}/transition1`, "bias"],
      ["secondWeight", `${root}/transition2`, "weights"],
      ["secondBias", `${root}/transition2`, "bias"],
    ]);
  }

  async #triangle(
    parameters, root, channels, block, blocks,
  ) {
    const hidden = this.#parameterShape(parameters, `${root}/left_projection`, "bias", true)[0];
    // Every projection is stored `[in][out]` and every kernel indexes
    // `[out][in]`, so all six are transposes - which `#gather` records on the
    // source so the device packer can reproduce them, or cancel them.
    const shapes = {
      linearAPWeight: { rows: channels, columns: hidden },
      linearAGWeight: { rows: channels, columns: hidden },
      linearBPWeight: { rows: channels, columns: hidden },
      linearBGWeight: { rows: channels, columns: hidden },
      linearZWeight: { rows: hidden, columns: channels },
      linearGWeight: { rows: channels, columns: channels },
    };
    return this.#gather(parameters, block, blocks, [
      ["layerNormInWeight", `${root}/layer_norm_input`, "scale"],
      ["layerNormInBias", `${root}/layer_norm_input`, "offset"],
      ["linearAPWeight", `${root}/left_projection`, "weights"],
      ["linearAPBias", `${root}/left_projection`, "bias"],
      ["linearAGWeight", `${root}/left_gate`, "weights"],
      ["linearAGBias", `${root}/left_gate`, "bias"],
      ["linearBPWeight", `${root}/right_projection`, "weights"],
      ["linearBPBias", `${root}/right_projection`, "bias"],
      ["linearBGWeight", `${root}/right_gate`, "weights"],
      ["linearBGBias", `${root}/right_gate`, "bias"],
      ["layerNormOutWeight", `${root}/center_layer_norm`, "scale"],
      ["layerNormOutBias", `${root}/center_layer_norm`, "offset"],
      ["linearZWeight", `${root}/output_projection`, "weights"],
      ["linearZBias", `${root}/output_projection`, "bias"],
      ["linearGWeight", `${root}/gating_linear`, "weights"],
      ["linearGBias", `${root}/gating_linear`, "bias"],
    ], (name) => shapes[name]);
  }

  async #outerProductMean(
    parameters, block, blocks,
  ) {
    return this.#gather(parameters, block, blocks, [
      ["layerNormScale", "outer_product_mean/layer_norm_input", "scale"],
      ["layerNormOffset", "outer_product_mean/layer_norm_input", "offset"],
      ["leftWeight", "outer_product_mean/left_projection", "weights"],
      ["leftBias", "outer_product_mean/left_projection", "bias"],
      ["rightWeight", "outer_product_mean/right_projection", "weights"],
      ["rightBias", "outer_product_mean/right_projection", "bias"],
      ["outputWeight", "outer_product_mean", "output_w"],
      ["outputBias", "outer_product_mean", "output_b"],
    ]);
  }

  async mainStackWeights(pairChannels = 128) {
    const { parameters, blocks } = this.manifest.evoformerStack;
    const result = [];
    for (let block = 0; block < blocks; block += 1) {
      const rowBase = await this.#attention(parameters, "msa_row_attention_with_pair_bias", block, blocks);
      // ...its three pair-bias tensors gathered too, so the whole attention
      // pack can bind codes; see attentionPackOrder.
      const rowPairBias = await this.#gather(parameters, block, blocks, [
        ["layerNormScale", "msa_row_attention_with_pair_bias/feat_2d_norm", "scale"],
        ["layerNormOffset", "msa_row_attention_with_pair_bias/feat_2d_norm", "offset"],
        ["projectionWeight", "msa_row_attention_with_pair_bias", "feat_2d_weights"],
      ]);
      const row = {
        ...rowBase,
        pairBias: rowPairBias,
        get pairLayerNormScale() { return rowPairBias.layerNormScale; },
        get pairLayerNormOffset() { return rowPairBias.layerNormOffset; },
        get pairProjectionWeight() { return rowPairBias.projectionWeight; },
      };
      result.push({
        msaRowAttention: row,
        msaColumnAttention: await this.#attention(parameters, "msa_column_attention", block, blocks),
        msaTransition: await this.#transition(parameters, "msa_transition", block, blocks),
        outerProductMean: await this.#outerProductMean(parameters, block, blocks),
        triangleMultiplicationOutgoing: await this.#triangle(
          parameters, "triangle_multiplication_outgoing", pairChannels, block, blocks,
        ),
        triangleMultiplicationIncoming: await this.#triangle(
          parameters, "triangle_multiplication_incoming", pairChannels, block, blocks,
        ),
        triangleAttentionStarting: await this.#triangleAttention(
          parameters, "triangle_attention_starting_node", block, blocks,
        ),
        triangleAttentionEnding: await this.#triangleAttention(
          parameters, "triangle_attention_ending_node", block, blocks,
        ),
        pairTransition: await this.#transition(parameters, "pair_transition", block, blocks),
      });
    }
    return result;
  }

  /**
   * AF2-multimer's template embedder.
   *
   * 🔴 IT RUNS EVEN WITH NO TEMPLATES. Multimer's config has template.enabled
   * true and the embedding wrapper adds this to the pair unconditionally;
   * masking the templates off leaves the biases and layer norms still
   * contributing. Skipping it put the pair track 30% out from the first block.
   *
   * The pair stack is an ordinary evoformer pair block at 64 channels, so it
   * reads with the same helpers as the extra-MSA stack.
   */
  async templateEmbeddingWeights(pairChannels = 64) {
    const section = this.manifest.templateEmbedding;
    if (section?.parameters === undefined) return undefined;
    const parameters = section.parameters;
    const blocks = section.pairStackBlocks ?? 2;
    const S = "template_embedding/single_template_embedding/";
    const IT = `${S}template_embedding_iteration/`;
    const scoped = (prefix) => Object.fromEntries(
      Object.entries(parameters)
        .filter(([name]) => name.startsWith(prefix))
        .map(([name, value]) => [name.slice(prefix.length), value]),
    );
    const stackParameters = scoped(IT);
    const stack = [];
    for (let block = 0; block < blocks; block += 1) {
      stack.push({
        triangleMultiplicationOutgoing: await this.#triangle(
          stackParameters, "triangle_multiplication_outgoing", pairChannels, block, blocks,
        ),
        triangleMultiplicationIncoming: await this.#triangle(
          stackParameters, "triangle_multiplication_incoming", pairChannels, block, blocks,
        ),
        triangleAttentionStarting: await this.#triangleAttention(
          stackParameters, "triangle_attention_starting_node", block, blocks,
        ),
        triangleAttentionEnding: await this.#triangleAttention(
          stackParameters, "triangle_attention_ending_node", block, blocks,
        ),
        pairTransition: await this.#transition(stackParameters, "pair_transition", block, blocks),
      });
    }
    const single = scoped(S);
    const embedding = async(index) => ({
      weight: await this.#parameter(single, `template_pair_embedding_${index}`, "weights"),
      bias: await this.#parameter(single, `template_pair_embedding_${index}`, "bias"),
    });
    return {
      stack,
      pairEmbeddings: await Promise.all([0, 1, 2, 3, 4, 5, 6, 7, 8].map(embedding)),
      queryNormScale: await this.#parameter(single, "query_embedding_norm", "scale"),
      queryNormOffset: await this.#parameter(single, "query_embedding_norm", "offset"),
      outputNormScale: await this.#parameter(single, "output_layer_norm", "scale"),
      outputNormOffset: await this.#parameter(single, "output_layer_norm", "offset"),
      outputWeight: await this.#parameter(parameters, "template_embedding/output_linear", "weights"),
      outputBias: await this.#parameter(parameters, "template_embedding/output_linear", "bias"),
    };
  }

  async extraPairStackWeights(pairChannels = 128) {
    const { parameters, blocks } = this.manifest.extraMsaStack;
    const result = [];
    for (let block = 0; block < blocks; block += 1) {
      result.push({
        outerProductMean: await this.#outerProductMean(parameters, block, blocks),
        triangleMultiplicationOutgoing: await this.#triangle(
          parameters, "triangle_multiplication_outgoing", pairChannels, block, blocks,
        ),
        triangleMultiplicationIncoming: await this.#triangle(
          parameters, "triangle_multiplication_incoming", pairChannels, block, blocks,
        ),
        triangleAttentionStarting: await this.#triangleAttention(
          parameters, "triangle_attention_starting_node", block, blocks,
        ),
        triangleAttentionEnding: await this.#triangleAttention(
          parameters, "triangle_attention_ending_node", block, blocks,
        ),
        pairTransition: await this.#transition(parameters, "pair_transition", block, blocks),
      });
    }
    return result;
  }

  async extraStackWeights(pairChannels = 128) {
    const { parameters, blocks } = this.manifest.extraMsaStack;
    const pairWeights = await this.extraPairStackWeights(pairChannels);
    const result = [];
    for (let block = 0; block < blocks; block += 1) {
      const rowBase = await this.#attention(parameters, "msa_row_attention_with_pair_bias", block, blocks);
      const rowPairBias = await this.#gather(parameters, block, blocks, [
        ["layerNormScale", "msa_row_attention_with_pair_bias/feat_2d_norm", "scale"],
        ["layerNormOffset", "msa_row_attention_with_pair_bias/feat_2d_norm", "offset"],
        ["projectionWeight", "msa_row_attention_with_pair_bias", "feat_2d_weights"],
      ]);
      const root = "msa_column_global_attention";
      const attention = `${root}/attention`;
      const parameter = (module, name) => this.#parameter(parameters, module, name, block, blocks);
      result.push({
        ...pairWeights[block],
        msaRowAttention: {
          ...rowBase,
          pairBias: rowPairBias,
          get pairLayerNormScale() { return rowPairBias.layerNormScale; },
          get pairLayerNormOffset() { return rowPairBias.layerNormOffset; },
          get pairProjectionWeight() { return rowPairBias.projectionWeight; },
        },
        msaColumnGlobalAttention: {
          queryNormScale: await parameter(`${root}/query_norm`, "scale"),
          queryNormOffset: await parameter(`${root}/query_norm`, "offset"),
          queryWeight: await parameter(attention, "query_w"), keyWeight: await parameter(attention, "key_w"),
          valueWeight: await parameter(attention, "value_w"), gatingWeight: await parameter(attention, "gating_w"),
          gatingBias: await parameter(attention, "gating_b"), outputWeight: await parameter(attention, "output_w"),
          outputBias: await parameter(attention, "output_b"),
          heads: this.#parameterShape(parameters, attention, "gating_b", true)[0],
        },
        msaTransition: await this.#transition(parameters, "msa_transition", block, blocks),
      });
    }
    return result;
  }

  async embeddingWeights() {
    const p = this.manifest.embedding.parameters;
    const parameter = (module, name) => this.#parameter(p, module, name);
    return {
      preprocess1dWeight: await parameter("preprocess_1d", "weights"),
      preprocess1dBias: await parameter("preprocess_1d", "bias"),
      preprocessMsaWeight: await parameter("preprocess_msa", "weights"),
      preprocessMsaBias: await parameter("preprocess_msa", "bias"),
      leftSingleWeight: await parameter("left_single", "weights"),
      leftSingleBias: await parameter("left_single", "bias"),
      rightSingleWeight: await parameter("right_single", "weights"),
      rightSingleBias: await parameter("right_single", "bias"),
      previousPositionWeight: await parameter("prev_pos_linear", "weights"),
      previousPositionBias: await parameter("prev_pos_linear", "bias"),
      previousMsaNormScale: await parameter("prev_msa_first_row_norm", "scale"),
      previousMsaNormOffset: await parameter("prev_msa_first_row_norm", "offset"),
      previousPairNormScale: await parameter("prev_pair_norm", "scale"),
      previousPairNormOffset: await parameter("prev_pair_norm", "offset"),
      relativePositionWeight: await parameter("pair_activiations", "weights"),
      relativePositionBias: await parameter("pair_activiations", "bias"),
      extraMsaWeight: await parameter("extra_msa_activations", "weights"),
      extraMsaBias: await parameter("extra_msa_activations", "bias"),
    };
  }

  async templateWeights() {
    const p = this.manifest.templateEmbedding.parameters;
    const blocks = 2;
    const root = "single_template_embedding/template_pair_stack/__layer_stack_no_state";
    const blockWeights = [];
    for (let block = 0; block < blocks; block += 1) {
      blockWeights.push({
        triangleAttentionStarting: await this.#triangleAttention(
          p, `${root}/triangle_attention_starting_node`, block, blocks,
        ),
        triangleAttentionEnding: await this.#triangleAttention(
          p, `${root}/triangle_attention_ending_node`, block, blocks,
        ),
        triangleMultiplicationOutgoing: await this.#triangle(
          p, `${root}/triangle_multiplication_outgoing`, 64, block, blocks,
        ),
        triangleMultiplicationIncoming: await this.#triangle(
          p, `${root}/triangle_multiplication_incoming`, 64, block, blocks,
        ),
        pairTransition: await this.#transition(p, `${root}/pair_transition`, block, blocks),
      });
    }
    return {
      embeddingBias: await this.#parameter(p, "single_template_embedding/embedding2d", "bias"),
      // 🔴 THE WEIGHT WAS NEVER LOADED, ONLY THE BIAS. With every template
      // masked the whole 88-channel concatenation is zero and `embedding2d`
      // contributes its bias alone, so the weight was unreachable - the same
      // shape of gap as AF3's six unread template projections.
      embeddingWeight: await this.#parameter(
        p, "single_template_embedding/embedding2d", "weights"),
      blockWeights,
      outputNormScale: await this.#parameter(p, "single_template_embedding/output_layer_norm", "scale"),
      outputNormOffset: await this.#parameter(p, "single_template_embedding/output_layer_norm", "offset"),
      valueWeight: await this.#parameter(p, "attention", "value_w"),
      outputWeight: await this.#parameter(p, "attention", "output_w"),
      outputBias: await this.#parameter(p, "attention", "output_b"),
      heads: this.#parameterShape(p, "attention", "value_w", false)[1],
    };
  }

  async structureWeights() {
    const p = this.manifest.structureModule.parameters;
    const parameter = (module, name) => this.#parameter(p, module, name);
    const root = "fold_iteration";
    const ipa = `${root}/invariant_point_attention`;
    const sidechain = `${root}/rigid_sidechain`;
    return {
      initialize: {
        singleProjectionWeight: await this.#parameter(this.manifest.embedding.parameters, "single_activations", "weights"),
        singleProjectionBias: await this.#parameter(this.manifest.embedding.parameters, "single_activations", "bias"),
        singleNormScale: await parameter("single_layer_norm", "scale"),
        singleNormOffset: await parameter("single_layer_norm", "offset"),
        initialProjectionWeight: await parameter("initial_projection", "weights"),
        initialProjectionBias: await parameter("initial_projection", "bias"),
      },
      ipa: {
        pairNormScale: await parameter("pair_layer_norm", "scale"),
        pairNormOffset: await parameter("pair_layer_norm", "offset"),
        queryScalarWeight: await parameter(`${ipa}/q_scalar`, "weights"),
        queryScalarBias: await parameter(`${ipa}/q_scalar`, "bias"),
        keyValueScalarWeight: await parameter(`${ipa}/kv_scalar`, "weights"),
        keyValueScalarBias: await parameter(`${ipa}/kv_scalar`, "bias"),
        queryPointWeight: await parameter(`${ipa}/q_point_local`, "weights"),
        queryPointBias: await parameter(`${ipa}/q_point_local`, "bias"),
        keyValuePointWeight: await parameter(`${ipa}/kv_point_local`, "weights"),
        keyValuePointBias: await parameter(`${ipa}/kv_point_local`, "bias"),
        trainablePointWeights: await parameter(ipa, "trainable_point_weights"),
        attention2dWeight: await parameter(`${ipa}/attention_2d`, "weights"),
        attention2dBias: await parameter(`${ipa}/attention_2d`, "bias"),
        outputWeight: await parameter(`${ipa}/output_projection`, "weights"),
        outputBias: await parameter(`${ipa}/output_projection`, "bias"),
      },
      postAttention: {
        attentionNormScale: await parameter(`${root}/attention_layer_norm`, "scale"),
        attentionNormOffset: await parameter(`${root}/attention_layer_norm`, "offset"),
        transitionWeights: [await parameter(`${root}/transition`, "weights"),
          await parameter(`${root}/transition_1`, "weights"), await parameter(`${root}/transition_2`, "weights")],
        transitionBiases: [await parameter(`${root}/transition`, "bias"),
          await parameter(`${root}/transition_1`, "bias"), await parameter(`${root}/transition_2`, "bias")],
        transitionNormScale: await parameter(`${root}/transition_layer_norm`, "scale"),
        transitionNormOffset: await parameter(`${root}/transition_layer_norm`, "offset"),
        affineWeight: await parameter(`${root}/affine_update`, "weights"),
        affineBias: await parameter(`${root}/affine_update`, "bias"),
      },
      sidechain: {
        inputWeight: await parameter(`${sidechain}/input_projection`, "weights"),
        inputBias: await parameter(`${sidechain}/input_projection`, "bias"),
        initialInputWeight: await parameter(`${sidechain}/input_projection_1`, "weights"),
        initialInputBias: await parameter(`${sidechain}/input_projection_1`, "bias"),
        residual1Weights: [await parameter(`${sidechain}/resblock1`, "weights"),
          await parameter(`${sidechain}/resblock2`, "weights")],
        residual1Biases: [await parameter(`${sidechain}/resblock1`, "bias"),
          await parameter(`${sidechain}/resblock2`, "bias")],
        residual2Weights: [await parameter(`${sidechain}/resblock1_1`, "weights"),
          await parameter(`${sidechain}/resblock2_1`, "weights")],
        residual2Biases: [await parameter(`${sidechain}/resblock1_1`, "bias"),
          await parameter(`${sidechain}/resblock2_1`, "bias")],
        angleWeight: await parameter(`${sidechain}/unnormalized_angles`, "weights"),
        angleBias: await parameter(`${sidechain}/unnormalized_angles`, "bias"),
      },
    };
  }

  /**
   * AlphaFold 2's distogram head, or undefined on a bundle without one.
   *
   * 🔴 OPTIONAL ON PURPOSE. The head was appended to the published bundles
   * long after they shipped (tools/add_distogram_head.py), so a store pointed
   * at an older copy - a cached one, a fork, a remote that has not been
   * re-uploaded - simply has no `distogramHead` section. Returning undefined
   * costs the contact map and nothing else; throwing would cost the fold.
   */
  async distogramHeadWeights() {
    const section = this.manifest.distogramHead;
    if (section === undefined) return undefined;
    // 🔴 A TENSOR IN THE SHARDS, LIKE EVERY OTHER WEIGHT. It was base64 in the
    // manifest for a while - 44 KB of text carried beside the table that
    // declared it - which existed only to avoid rewriting published bytes. The
    // cost was a bundle that was not the whole model: a reader of `model/` got
    // weights whose distogram head lived somewhere else, in another encoding,
    // reachable only through a special case here. It is appended to the last
    // shard now, so the 227 MB before it are untouched and this is an ordinary
    // read.
    //
    // 🔴 WHICH MAKES THE PUBLISH ORDER LOAD-BEARING. The manifests are compiled
    // into the page while the shards come from a pinned remote, so a manifest
    // naming bytes the pinned commit does not have breaks every AF2 fold. See
    // tools/add_distogram_head.py: upload, re-pin, then regenerate.
    const [halfLogitsWeights, halfLogitsBias] = await Promise.all([
      this.tensor(section.weights), this.tensor(section.bias),
    ]);
    return {
      halfLogitsWeights,
      halfLogitsBias,
      bins: section.bins,
      firstBreak: section.firstBreak,
      lastBreak: section.lastBreak,
    };
  }

  async confidenceWeights()

  {
    const lp = this.manifest.confidenceHeads.parameters.predictedLddt;
    const pp = this.manifest.confidenceHeads.parameters.predictedAlignedError;
    const parameter = (map, module, name) => this.#parameter(map, module, name);
    return {
      lddt: {
        normScale: await parameter(lp, "input_layer_norm", "scale"),
        normOffset: await parameter(lp, "input_layer_norm", "offset"),
        act0Weight: await parameter(lp, "act_0", "weights"), act0Bias: await parameter(lp, "act_0", "bias"),
        act1Weight: await parameter(lp, "act_1", "weights"), act1Bias: await parameter(lp, "act_1", "bias"),
        logitsWeight: await parameter(lp, "logits", "weights"), logitsBias: await parameter(lp, "logits", "bias"),
      },
      pae: {
        logitsWeight: await parameter(pp, "logits", "weights"), logitsBias: await parameter(pp, "logits", "bias"),
      },
    };
  }

  async geometryTables() {
    return {
      defaultFrames: await this.tensor("geometryDefaultFrames"),
      atom14ToGroup: await this.tensor("geometryAtom14ToGroup"),
      atom14Positions: await this.tensor("geometryAtom14Positions"),
      atom14Mask: await this.tensor("geometryAtom14Mask"),
    };
  }

  async queryOnlyFeatureTables() {
    return {
      atom37ToAtom14: await this.tensor("geometryAtom37ToAtom14"),
      atom37Mask: await this.tensor("geometryAtom37Mask"),
    };
  }
}
