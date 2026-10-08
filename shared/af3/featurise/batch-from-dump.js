// AlphaFold 3's own featurised batch (an oracle dump) in this port's batch shape - out of tools/gpu/fold.js (which
// re-exports it), so the CUDA port's exporter reads a dump without the WebGPU fold tool behind it.
const floats = (source) => Float32Array.from(source, (v) => Number(v));
const ints = (source) => Int32Array.from(source, (v) => Number(v));

/**
 * AF3's own batch, in the shape shared/af3/featurise/featurise.js produces, so the fold reads
 * one object either way and the two paths cannot silently diverge in what they
 * supply.
 */
export function batchFromDump(dump) {
  const tokens = dump.tokens;
  const dense = 24;
  // 🔴 PADDED, DELIBERATELY, AND ONLY HERE. The featuriser sizes its subsets
  // from the real atom count; this path does not, because its gathers ARE
  // AF3's - read straight out of the dump, at the dense grid's width - and a
  // subset count that disagreed with the arrays beside it is the failure this
  // repository keeps meeting. The two paths differ in shape and agree to every
  // digit in what they fold, which is what the padding was worth.
  const subsets = Math.ceil((tokens * dense) / 32);
  const raw = (name) => dump.inputs[name].data;
  // 🔴 count IS NOT DECORATION. convert() in atom-encoder.js sizes its
  // output from it, so a gather without one silently produces a zero-length
  // tensor - which reads downstream as a model that runs and folds a 17 A
  // spaghetti rather than as an error.
  const gather = (name) => {
    const indices = ints(raw(`${name}:gather_idxs`));
    return {
    indices, mask: floats(raw(`${name}:gather_mask`)), count: indices.length };
  };
  const refMask = floats(raw("ref_mask"));
  let atomCount = 0;
  for (const value of refMask) atomCount += value;
  return {
    sequence: dump.sequence, tokens, dense, subsets, atomCount,
    shape: { tokens, dense, subsets, queries: 32, keys: 128 },
    aatype: ints(raw("aatype")), profile: floats(raw("profile")),
    deletionMean: floats(raw("deletion_mean")),
    msa: ints(raw("msa")), msaMask: floats(raw("msa_mask")),
    // 🔴 THE ROW COUNT IS A SHAPE, AND THE DUMP IS THE ONLY PLACE IT IS WRITTEN
    // DOWN. AF3 pads its msa array to the crop size and records how many rows
    // are real in `numMsa`, so deriving it from the array's length reads the
    // padding as alignment. Absent, the trunk's shader was built with
    // `const SEQUENCES: u32 = undefinedu;` and the fold died in WGSL parsing -
    // which is a shape bug wearing a compiler error.
    sequences: dump.numMsa ?? 1,
    // ...and the chain identity, which the confidence head's ipTM reduction
    // indexes directly. Absent, it threw inside reduceTmScore AFTER the whole
    // fold had run, which is the most expensive place to discover a missing
    // field.
    asymId: ints(raw("asym_id")),
    deletionMatrix: floats(raw("deletion_matrix")),
    seqMask: floats(raw("seq_mask")),
    // 🔴 AND WHICH RESIDUE EACH TOKEN BELONGS TO, which the featuriser records
    // and this path did not. OpenDDE's structural layout reads it to decide a
    // token's molecule kind, so `fold-opendde.js --dump=` died in
    // `kindOfToken` reading undefined - a batch that is complete for AF3 and
    // incomplete for the family that re-tokenises. Derived from the dump's own
    // (asym_id, residue_index) pairs: a new pair starts a new residue, which is
    // exactly the grouping the featuriser builds.
    ...(() => {
      const asym = ints(raw("asym_id"));
      const residueIndex = ints(raw("residue_index"));
      const residueOfToken = new Int32Array(tokens).fill(-1);
      const chainOfResidue = [];
      const chains = new Map();
      let residue = -1, lastKey = null;
      for (let token = 0; token < tokens; token += 1) {
        const key = `${asym[token]}:${residueIndex[token]}`;
        if (key !== lastKey) {
          residue += 1; lastKey = key;
          if (!chains.has(asym[token])) chains.set(asym[token], chains.size);
          chainOfResidue.push(chains.get(asym[token]));
        }
        residueOfToken[token] = residue;
      }
      return { residueOfToken, chainOfResidue };
    })(),
    // chai1's tokens read ESM2 3B's last hidden state ([tokens, 2560], zeros on non-protein tokens): a batch
    // field only for that family, which a batch dumped for it carries
    ...(dump.inputs.esm_embeddings === undefined ? {} : { esmEmbeddings: floats(raw("esm_embeddings")) }),
    refPos: floats(raw("ref_pos")), refMask,
    refElement: ints(raw("ref_element")), refCharge: floats(raw("ref_charge")),
    refAtomNameChars: ints(raw("ref_atom_name_chars")),
    refSpaceUid: ints(raw("ref_space_uid")),
    predDenseAtomMask: floats(raw("pred_dense_atom_mask")),
    // 🔴 boltz2's `target_feat` READS ALL FOUR OF THESE and every other model
    // reads none of them, so they were in the dump and not in the batch. Its
    // InputEmbedder projects a mol_type one-hot and a modified flag onto the
    // single track; see `targetFeatures`.
    // Each independently, because a dump written for one model carries a
    // different subset - `af3-6mrr.json` has is_dna and no is_modified, and
    // reading them as a group threw inside `raw` on a fold that never needed
    // any of them.
    ...Object.fromEntries([["isDna", "is_dna"], ["isRna", "is_rna"],
                           ["isLigand", "is_ligand"], ["isModified", "is_modified"]]
      .filter(([, name]) => dump.inputs[name] !== undefined)
      .map(([field, name]) => [field, ints(raw(name))])),
    tokenAtomsToQueries: gather("token_atoms_to_queries"),
    queriesToKeys: gather("queries_to_keys"),
    queriesToTokenAtoms: gather("queries_to_token_atoms"),
    tokensToQueries: gather("tokens_to_queries"),
    tokensToKeys: gather("tokens_to_keys"),
    tokenAtomsToPseudoBeta: gather("token_atoms_to_pseudo_beta"),
    // 🔴 AND AT THE TOP LEVEL TOO, WHICH IS WHERE THE READERS LOOK. The
    // featuriser returns `residueIndex, tokenIndex, asymId, entityId, symId` as
    // fields of the batch; this path had them ONLY under `features`, so
    // `structural-tokens.js` read `batch.entityId` as undefined and
    // `fold-opendde.js --dump=` crashed AFTER the trunk comparison had already
    // printed - which reads as a broken tool rather than a missing field. Same
    // class as `residueOfToken` above: a batch that is complete for AF3 and
    // incomplete for the family that re-tokenises.
    residueIndex: ints(raw("residue_index")), tokenIndex: ints(raw("token_index")),
    entityId: ints(raw("entity_id")), symId: ints(raw("sym_id")),
    features: {
      residueIndex: ints(raw("residue_index")), tokenIndex: ints(raw("token_index")),
      asymId: ints(raw("asym_id")), entityId: ints(raw("entity_id")),
      symId: ints(raw("sym_id")),
    },
  };
}
