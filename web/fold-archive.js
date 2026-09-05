/**
 * A finished fold as the AlphaFold 3 server's archive, and back again.
 *
 * WHY THIS SHAPE AND NOT ONE OF OUR OWN. The server's layout is what every
 * script written against AlphaFold 3 already reads, and the reference archive
 * `fold_2026_09_01_10_17.zip` in the repository root is what this was written
 * against - file for file, key for key. A format nobody else writes would need
 * a reader written for it before anyone could use a fold from this page.
 *
 * 🔴 THE STRUCTURE IS A .pdb WHERE THE SERVER WRITES .cif. That is the one
 * deliberate difference. LocalFold has a PDB writer that every checker in the
 * repository already reads, and an mmCIF writer would be a second description
 * of the same atoms to keep in step for no gain here.
 *
 * 🔴 AND terms_of_use.md IS NOT COPIED. The server's is DeepMind's, about
 * DeepMind's service. Shipping it verbatim out of a different program would
 * misstate who is promising what to whom. README.md says what actually ran.
 *
 * 🔴 A FIELD WE DO NOT COMPUTE IS LEFT OUT, NOT FILLED IN. `has_clash` and
 * `chain_pair_pae_min` are both cheap to invent and would be read as the
 * model's opinion of the structure. An absent key is a question that was not
 * asked; a zero is an answer.
 */
import { CHAIN_IDS, paeMatrix, safeJobName } from "./prediction-results.js";
import { coordinateAtoms } from "../src/design/superpose-pdb.js";

/**
 * 🔴 TWO DECIMALS, WHICH IS WHAT THE SERVER WRITES. Not cosmetic: `full_data`
 * is three token-by-token matrices, and at full float64 spelling a 220-token
 * complex's file is several times the 410 KB the server's is. It is also what
 * stops float32 values arriving as 0.20000000298023224 - the matrices come off
 * the GPU as f32 and widening them to double prints the error.
 */
const round2 = (value) => Math.round(value * 100) / 100;
const matrix2 = (rows) => rows.map((row) => row.map(round2));

/** The chain letter a chain index is written as, which is the PDB writer's. */
export const chainLetter = (index) => CHAIN_IDS[index] ?? "?";

/**
 * 🔴 THE MSA FILENAMES RUN OUT AT 26 CHAINS, AND SAY SO RATHER THAN COLLIDE.
 * The server spells a chain lower case - `..._msa_chains_a.a3m` - and
 * CHAIN_IDS runs A-Z then a-z, so chain 0 and chain 26 are both "a" once
 * lower-cased and the second alignment would overwrite the first in the
 * archive. Every way of extending the scheme invents a convention the server
 * does not have, and an archive that quietly holds 26 of 30 alignments is worse
 * than one that was not written. Nothing on this page folds 27 chains.
 */
const MAX_NAMED_CHAINS = 26;

/**
 * Per-token chain letters and residue numbers.
 *
 * 🔴 DERIVED ONLY WHEN THE TOKENS ARE THE RESIDUES, AND CHECKED EITHER WAY.
 * AlphaFold 3 scores TOKENS - a ligand is one per heavy atom and a modified
 * residue one per atom - so a fold with either in it has more tokens than the
 * chain lengths account for, and numbering them as though it did not would put
 * every ligand token on the last polymer chain. The caller that featurised
 * knows the real layout and passes it; this is the fallback for the case where
 * one residue is one token, and it refuses rather than guesses when the count
 * says otherwise.
 */
export function tokenIdentifiers(chainLengths, tokens, given) {
  if (given?.chainIds !== undefined && given?.resIds !== undefined) {
    if (given.chainIds.length !== tokens) {
      throw new RangeError(`token chain ids are ${given.chainIds.length} for ${tokens} tokens`);
    }
    return { chainIds: given.chainIds, resIds: given.resIds };
  }
  const residues = chainLengths.reduce((total, length) => total + length, 0);
  if (residues !== tokens) {
    throw new RangeError(`${tokens} tokens against ${residues} residues:`
      + " this fold's token layout must be passed in, not inferred");
  }
  const chainIds = [];
  const resIds = [];
  chainLengths.forEach((length, chain) => {
    for (let within = 0; within < length; within += 1) {
      chainIds.push(chainLetter(chain));
      resIds.push(within + 1);
    }
  });
  return { chainIds, resIds };
}

/**
 * The request that produced this fold, in the server's own dialect.
 *
 * 🔴 COPIES STAY A COUNT. `expandEntities` turns two copies into two chains
 * because that is what the model is given, but the server's request says
 * `count: 2` on one entry - and a request that listed the same sequence twice
 * would come back from the server as a different job than the one that ran.
 */
export function jobRequestJson({ name, seed, entities }) {
  const sequences = [];
  for (const entity of entities ?? []) {
    const value = (entity.value ?? "").trim();
    if (value === "") continue;
    const count = Math.max(1, Number(entity.copies) || 1);
    if (entity.type === "protein") {
      sequences.push({ proteinChain: { sequence: value, count,
        useStructureTemplate: (entity.template?.kind ?? "none") !== "none" } });
    } else if (entity.type === "dna" || entity.type === "rna") {
      sequences.push({ [`${entity.type}Sequence`]: { sequence: value, count } });
    } else {
      sequences.push({ ligand: { ligand: value.toUpperCase(), count } });
    }
  }
  return `${JSON.stringify([{
    name,
    // A string, as the server writes it, and an array because a job may carry
    // several seeds. This page folds one at a time.
    modelSeeds: [String(seed ?? 0)],
    sequences,
    dialect: "alphafoldserver",
    version: 3,
  }], null, 2)}\n`;
}

/** The per-token and per-atom arrays, as `full_data_0.json`. */
export function fullDataJson({ confidence, pdb, tokenChainIds, tokenResIds }) {
  const tokens = tokenChainIds.length;
  // 🔴 THE ATOM ARRAYS ARE READ BACK OFF THE STRUCTURE IN THIS ARCHIVE, not
  // recomputed beside it. `atom_plddts` has to line up with the atoms of
  // `_model_0.pdb` record for record, and the only thing that guarantees that
  // is taking both from the same text - the writer drops an atom whose mask is
  // clear, so an independent walk over the sequence counts different atoms.
  const atoms = coordinateAtoms(pdb);
  const data = {
    atom_chain_ids: atoms.chains,
    atom_plddts: Array.from(atoms.bFactors, round2),
  };
  // ...before the PAE, where the server puts it, and only when the fold
  // actually produced one.
  if (confidence.contactProbs !== undefined) {
    data.contact_probs = matrix2(paeMatrix(confidence.contactProbs, tokens));
  }
  data.pae = matrix2(paeMatrix(confidence.predictedAlignedError, tokens));
  data.token_chain_ids = tokenChainIds;
  data.token_res_ids = tokenResIds;
  return `${JSON.stringify(data)}\n`;
}

/**
 * Which asym id is which chain, in chain order.
 *
 * 🔴 THE SCORE KEYS ARE ASYM IDS AND THEY ARE NOT THE CHAIN INDEX. AlphaFold 3
 * numbers its chains from ONE - `featurise.js` writes `identity.asymId + 1` -
 * while AlphaFold 2 uses contiguous blocks numbered from zero. Reading the keys
 * as indices produced a summary that looked complete and was not: a real
 * two-chain fold came out with `chain_pair_iptm` all null and
 * `chain_ptm: [null, 0.69]`, because "1|2" matched nothing and "1" matched the
 * second chain by accident. Every unit test passed - they were written with
 * 0-based keys, which is the AF2 convention and half the truth.
 *
 * So the ids are taken from the scores themselves and sorted: the nth distinct
 * asym id is the nth chain. That holds for both models without either being
 * named here.
 */
function asymOrder(confidence, chainCount) {
  const ids = new Set();
  const add = (value) => {
    const id = Number(value);
    if (Number.isFinite(id)) ids.add(id);
  };
  for (const key of Object.keys(confidence.chainPtm ?? {})) add(key);
  for (const key of Object.keys(confidence.chainIptm ?? {})) add(key);
  for (const key of Object.keys(confidence.chainPairIptm ?? {})) {
    for (const part of key.split("|")) add(part);
  }
  const sorted = [...ids].sort((a, b) => a - b);
  // ...and when the scores do not name every chain - a monomer, or a pair that
  // could not be scored - there is nothing to align against, so the plain
  // index is used and a missing entry stays null rather than being shifted onto
  // the wrong chain.
  return sorted.length === chainCount ? sorted
    : Array.from({ length: chainCount }, (_, index) => index);
}

/** The scalar scores, as `summary_confidences_0.json`. */
export function summaryConfidencesJson({ confidence, chainLengths, tokenChainIds }) {
  const chains = chainLengths.map((_, index) => chainLetter(index));
  const asym = asymOrder(confidence, chains.length);
  const summary = { chain_ids: tokenChainIds };

  // 🔴 THE PAIR MATRIX IS BUILT FROM THE MAP, NOT ASSUMED SQUARE-COMPLETE.
  // chainPairTmScores omits a pair it could not score - two chains that share
  // no admitted token - so a missing entry is "not scored" and is written as
  // null rather than as zero, which would read as an interface the model was
  // sure about and sure was bad.
  const pairs = confidence.chainPairIptm;
  if (pairs !== undefined && Object.keys(pairs).length > 0) {
    summary.chain_pair_iptm = chains.map((_, a) => chains.map((__, b) => {
      if (a === b) return null;
      const [first, second] = asym[a] < asym[b] ? [asym[a], asym[b]] : [asym[b], asym[a]];
      const value = pairs[`${first}|${second}`];
      return value === undefined ? null : round2(value);
    }));
  }
  // Per chain, in chain order, as the server writes them.
  const perChain = (values) => (values === undefined ? undefined
    : chains.map((_, index) => (values[asym[index]] === undefined
      ? null : round2(values[asym[index]]))));
  const chainPtm = perChain(confidence.chainPtm);
  const chainIptm = perChain(confidence.chainIptm);
  if (chainPtm !== undefined) summary.chain_ptm = chainPtm;
  if (chainIptm !== undefined && chains.length > 1) summary.chain_iptm = chainIptm;

  // 🔴 THE CLOSEST THE TWO CHAINS COME, IN THE MODEL'S OWN UNCERTAINTY. The
  // server writes it and it is a minimum over the PAE, so it needs nothing the
  // model did not already produce. Not symmetric, and its diagonal is real: the
  // server's own file has [[0.76, 0.83], [0.82, 0.76]], which is the minimum
  // over ORDERED pairs - row i in one chain, column j in the other.
  if (confidence.predictedAlignedError !== undefined && chains.length > 0) {
    const pae = confidence.predictedAlignedError;
    const tokens = tokenChainIds.length;
    const indexOf = new Map(chains.map((letter, index) => [letter, index]));
    const minima = chains.map(() => chains.map(() => Number.POSITIVE_INFINITY));
    for (let i = 0; i < tokens; i += 1) {
      const a = indexOf.get(tokenChainIds[i]);
      if (a === undefined) continue;
      for (let j = 0; j < tokens; j += 1) {
        const b = indexOf.get(tokenChainIds[j]);
        if (b === undefined) continue;
        const value = pae[i * tokens + j];
        if (value < minima[a][b]) minima[a][b] = value;
      }
    }
    summary.chain_pair_pae_min = minima.map((row) =>
      row.map((value) => (Number.isFinite(value) ? round2(value) : null)));
  }

  if (confidence.iptm !== undefined && !Number.isNaN(Number(confidence.iptm))) {
    summary.iptm = round2(confidence.iptm);
  }
  if (confidence.ptm !== undefined) summary.ptm = round2(confidence.ptm);
  if (confidence.multimerScore !== undefined) {
    summary.ranking_score = round2(confidence.multimerScore);
  } else if (summary.iptm !== undefined && summary.ptm !== undefined) {
    summary.ranking_score = round2(0.8 * confidence.iptm + 0.2 * confidence.ptm);
  } else if (summary.ptm !== undefined) {
    summary.ranking_score = summary.ptm;
  }
  // AF3's own definition: the fraction of residues the model is not confident
  // are ordered at all.
  if (confidence.plddt !== undefined && confidence.plddt.length > 0) {
    let disordered = 0;
    for (const value of confidence.plddt) if (value < 50) disordered += 1;
    summary.fraction_disordered = round2(disordered / confidence.plddt.length);
  }
  if (confidence.meanPlddt !== undefined) {
    summary.mean_plddt = round2(confidence.meanPlddt);
  }
  return `${JSON.stringify(summary, null, 2)}\n`;
}

function readme({ stem, model, settings, msaOrigin, templateCount }) {
  const lines = [
    `# ${stem}`,
    "",
    "Folded in a browser by LocalFold (https://localfold.org), which runs",
    `${model} on WebGPU. Nothing in this archive passed through a fold server.`,
    "",
    "## What ran",
    "",
    `- model: ${model}`,
  ];
  for (const [key, value] of Object.entries(settings ?? {})) {
    if (value !== undefined && value !== null && value !== "") lines.push(`- ${key}: ${value}`);
  }
  lines.push(
    `- alignment: ${msaOrigin}`,
    `- templates: ${templateCount === 0 ? "none" : `${templateCount} used`}`,
    "",
    "## Layout",
    "",
    "The AlphaFold 3 server's, with one difference: the structure is written as",
    "PDB rather than mmCIF.",
    "",
    "`msas/` holds one alignment per chain, split into the paired and unpaired",
    "blocks the model reads separately. Drop this whole .zip onto LocalFold's",
    "alignment upload box to fold again with exactly these alignments - the two",
    "blocks are not interchangeable, so re-uploading a single merged a3m would",
    "not reproduce this fold.",
    "",
  );
  return lines.join("\n");
}

/**
 * Every member of the archive, ready for writeZip.
 *
 * @returns {Map<string, string>}
 */
export function buildFoldArchive({
  stem, model, settings, entities, prediction, msas = {}, templates = [],
  msaOrigin = "none (single sequence)",
}) {
  const name = safeJobName(stem);
  const { confidence, pdb, chainLengths } = prediction;
  const tokens = Math.round(Math.sqrt(confidence.predictedAlignedError.length));
  const { chainIds, resIds } = tokenIdentifiers(chainLengths, tokens, prediction.tokens);

  if (chainLengths.length > MAX_NAMED_CHAINS) {
    throw new RangeError(`this archive names alignments by chain letter and`
      + ` cannot hold ${chainLengths.length} chains`);
  }

  const files = new Map();
  files.set(`${name}_job_request.json`, jobRequestJson({
    name: stem, seed: settings?.seed, entities,
  }));
  files.set(`${name}_model_0.pdb`, pdb);
  files.set(`${name}_summary_confidences_0.json`, summaryConfidencesJson({
    confidence, chainLengths, tokenChainIds: chainIds,
  }));
  files.set(`${name}_full_data_0.json`, fullDataJson({
    confidence, pdb, tokenChainIds: chainIds, tokenResIds: resIds,
  }));

  // 🔴 ONE FILE PER CHAIN PER BLOCK, WHICH IS THE WHOLE POINT. A merged a3m
  // cannot say which rows were paired, and AlphaFold 3 reads the paired block
  // first and takes its profile over the unpaired one alone - so a fold
  // restored from a single merged alignment is a different fold, silently.
  (msas.unpaired ?? []).forEach((text, chain) => {
    if (text) files.set(`msas/${name}_unpaired_msa_chains_${chainLetter(chain).toLowerCase()}.a3m`, text);
  });
  (msas.paired ?? []).forEach((text, chain) => {
    if (text) files.set(`msas/${name}_paired_msa_chains_${chainLetter(chain).toLowerCase()}.a3m`, text);
  });
  // 🔴 A MERGED ALIGNMENT IS NOT WRITTEN AS CHAIN A'S. A pasted a3m, or an
  // uploaded one that was not an archive, is one text covering every chain and
  // carrying no record of which rows were paired - so naming it
  // `_unpaired_msa_chains_a.a3m` would claim a split it does not have, and on a
  // complex would claim the whole alignment belongs to the first chain. It goes
  // under a name of its own, which is also what tells the reader on the way
  // back that there is nothing to reconstruct.
  if (msas.merged) files.set(`msas/${name}_merged_msa.a3m`, msas.merged);

  templates.forEach((template, index) => {
    if (!template?.text) return;
    // ...named by what it IS. The server's are always mmCIF; ours come from
    // the RCSB as PDB, from AlphaFold DB as PDB and from the MMseqs2 template
    // endpoint as mmCIF, so the extension follows the bytes.
    const cif = /^\s*(data_|#|loop_|_)/m.test(template.text.slice(0, 4096));
    const letter = chainLetter(template.chain ?? 0).toLowerCase();
    files.set(`templates/${name}_template_hit_${index}_chains_${letter}`
      + `.${cif ? "cif" : "pdb"}`, template.text);
  });

  files.set("README.md", readme({
    stem, model, settings, msaOrigin, templateCount: templates.length,
  }));
  return files;
}

/**
 * The per-chain alignments out of an archive, ready for `mergeSearchedChains`.
 *
 * 🔴 THIS IS THE HALF THAT MAKES THE ROUND TRIP MEAN ANYTHING. The upload box
 * used to take one merged a3m and, having no way to tell the blocks apart,
 * recorded the whole thing as UNPAIRED - which AlphaFold 3 reads differently
 * from what the search produced: it takes the paired block first and computes
 * its profile over the unpaired block alone. A fold restored from its own
 * downloaded alignment was quietly a different fold. Per-chain files carry the
 * distinction in their names, so it survives.
 *
 * The chain letter is the writer's own (`chainLetter`), lower-cased as the
 * server writes it, and it is what orders the arrays - not the order the
 * members happen to appear in the archive.
 *
 * @param {Map<string, string>} files from readZip
 * @returns {{chainA3ms: string[], pairedA3ms: Map<number, string>, chains: number}}
 */
export function msasFromArchive(files) {
  const unpaired = new Map();
  const paired = new Map();
  for (const [path, text] of files) {
    const match = /^msas\/.*_(paired|unpaired)_msa_chains_([a-z])\.a3m$/i.exec(path);
    if (match === null) continue;
    // ...case-insensitive going in, but CHAIN_IDS runs upper case THEN lower,
    // so a bare indexOf on the lower-case letter the server writes would read
    // "a" as chain 26. See MAX_NAMED_CHAINS.
    const chain = CHAIN_IDS.indexOf(match[2].toUpperCase());
    if (chain < 0 || chain >= MAX_NAMED_CHAINS) continue;
    (match[1].toLowerCase() === "paired" ? paired : unpaired).set(chain, text);
  }
  const highest = Math.max(-1, ...unpaired.keys(), ...paired.keys());
  const chainA3ms = [];
  for (let chain = 0; chain <= highest; chain += 1) {
    chainA3ms.push(unpaired.get(chain) ?? "");
  }
  let merged;
  for (const [path, text] of files) {
    if (/^msas\/.*_merged_msa\.a3m$/i.test(path)) merged = text;
  }
  return { chainA3ms, pairedA3ms: paired, chains: highest + 1, merged };
}
