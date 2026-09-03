function transpose(input, rows, columns) {
  const output = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      output[column * rows + row] = input[row * columns + column];
    }
  }
  return output;
}

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

  #parameterShape(parameters, module, name, stacked) {
    const tensorName = parameters[module]?.[name];
    if (tensorName === undefined) throw new Error(`missing ${module}/${name}`);
    const shape = this.store.shape(tensorName);
    return stacked ? shape.slice(1) : shape;
  }

  async #attention(
    parameters, root, block, blocks,
  ) {
    const parameter = (module, name) =>
      this.#parameter(parameters, module, name, block, blocks);
    const attentionRoot = `${root}/attention`;
    const weights = {
      queryNormScale: await parameter(`${root}/query_norm`, "scale"),
      queryNormOffset: await parameter(`${root}/query_norm`, "offset"),
      queryWeight: await parameter(attentionRoot, "query_w"),
      keyWeight: await parameter(attentionRoot, "key_w"),
      valueWeight: await parameter(attentionRoot, "value_w"),
      gatingWeight: await parameter(attentionRoot, "gating_w"),
      gatingBias: await parameter(attentionRoot, "gating_b"),
      outputWeight: await parameter(attentionRoot, "output_w"),
      outputBias: await parameter(attentionRoot, "output_b"),
    };
    return {
      heads: this.#parameterShape(parameters, attentionRoot, "gating_b", true)[0],
      attention: weights,
    };
  }

  async #triangleAttention(
    parameters, root, block, blocks,
  ) {
    const result = await this.#attention(parameters, root, block, blocks);
    return {
      ...result,
      pairProjectionWeight: await this.#parameter(parameters, root, "feat_2d_weights", block, blocks),
    };
  }

  async #transition(
    parameters, root, block, blocks,
  ) {
    const parameter = (module, name) =>
      this.#parameter(parameters, module, name, block, blocks);
    return {
      layerNormScale: await parameter(`${root}/input_layer_norm`, "scale"),
      layerNormOffset: await parameter(`${root}/input_layer_norm`, "offset"),
      firstWeight: await parameter(`${root}/transition1`, "weights"),
      firstBias: await parameter(`${root}/transition1`, "bias"),
      secondWeight: await parameter(`${root}/transition2`, "weights"),
      secondBias: await parameter(`${root}/transition2`, "bias"),
    };
  }

  async #triangle(
    parameters, root, channels, block, blocks,
  ) {
    const parameter = (module, name) =>
      this.#parameter(parameters, module, name, block, blocks);
    const hidden = this.#parameterShape(parameters, `${root}/left_projection`, "bias", true)[0];
    const projection = async(module, inputChannels, outputChannels) =>
      transpose(await parameter(`${root}/${module}`, "weights"), inputChannels, outputChannels);
    return {
      layerNormInWeight: await parameter(`${root}/layer_norm_input`, "scale"),
      layerNormInBias: await parameter(`${root}/layer_norm_input`, "offset"),
      linearAPWeight: await projection("left_projection", channels, hidden),
      linearAPBias: await parameter(`${root}/left_projection`, "bias"),
      linearAGWeight: await projection("left_gate", channels, hidden),
      linearAGBias: await parameter(`${root}/left_gate`, "bias"),
      linearBPWeight: await projection("right_projection", channels, hidden),
      linearBPBias: await parameter(`${root}/right_projection`, "bias"),
      linearBGWeight: await projection("right_gate", channels, hidden),
      linearBGBias: await parameter(`${root}/right_gate`, "bias"),
      layerNormOutWeight: await parameter(`${root}/center_layer_norm`, "scale"),
      layerNormOutBias: await parameter(`${root}/center_layer_norm`, "offset"),
      linearZWeight: await projection("output_projection", hidden, channels),
      linearZBias: await parameter(`${root}/output_projection`, "bias"),
      linearGWeight: await projection("gating_linear", channels, channels),
      linearGBias: await parameter(`${root}/gating_linear`, "bias"),
    };
  }

  async #outerProductMean(
    parameters, block, blocks,
  ) {
    const parameter = (module, name) =>
      this.#parameter(parameters, module, name, block, blocks);
    return {
      layerNormScale: await parameter("outer_product_mean/layer_norm_input", "scale"),
      layerNormOffset: await parameter("outer_product_mean/layer_norm_input", "offset"),
      leftWeight: await parameter("outer_product_mean/left_projection", "weights"),
      leftBias: await parameter("outer_product_mean/left_projection", "bias"),
      rightWeight: await parameter("outer_product_mean/right_projection", "weights"),
      rightBias: await parameter("outer_product_mean/right_projection", "bias"),
      outputWeight: await parameter("outer_product_mean", "output_w"),
      outputBias: await parameter("outer_product_mean", "output_b"),
    };
  }

  async mainStackWeights(pairChannels = 128) {
    const { parameters, blocks } = this.manifest.evoformerStack;
    const result = [];
    for (let block = 0; block < blocks; block += 1) {
      const rowBase = await this.#attention(parameters, "msa_row_attention_with_pair_bias", block, blocks);
      const row = {
        ...rowBase,
        pairLayerNormScale: await this.#parameter(
          parameters, "msa_row_attention_with_pair_bias/feat_2d_norm", "scale", block, blocks,
        ),
        pairLayerNormOffset: await this.#parameter(
          parameters, "msa_row_attention_with_pair_bias/feat_2d_norm", "offset", block, blocks,
        ),
        pairProjectionWeight: await this.#parameter(
          parameters, "msa_row_attention_with_pair_bias", "feat_2d_weights", block, blocks,
        ),
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
      const root = "msa_column_global_attention";
      const attention = `${root}/attention`;
      const parameter = (module, name) => this.#parameter(parameters, module, name, block, blocks);
      result.push({
        ...pairWeights[block],
        msaRowAttention: {
          ...rowBase,
          pairLayerNormScale: await parameter("msa_row_attention_with_pair_bias/feat_2d_norm", "scale"),
          pairLayerNormOffset: await parameter("msa_row_attention_with_pair_bias/feat_2d_norm", "offset"),
          pairProjectionWeight: await parameter("msa_row_attention_with_pair_bias", "feat_2d_weights"),
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
    return {
      halfLogitsWeights: await this.tensor(section.parameters.halfLogitsWeights),
      halfLogitsBias: await this.tensor(section.parameters.halfLogitsBias),
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
