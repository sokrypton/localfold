/**
 * A finished fold as the AlphaFold 3 server's archive, and back again.
 *
 * WHY THIS SHAPE AND NOT ONE OF OUR OWN. The server's layout is what every
 * script written against AlphaFold 3 already reads, and the reference archive
 * `tools/fixtures/fold_2026_09_01_10_17.zip` in the repository root is what this was written
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
export function tokenLayoutFrom(asymId, residueIndex) {
  // 🔴 THE ASYM IDS ARE SORTED AND TAKEN IN ORDER, NOT READ AS INDICES. AF3
  // numbers chains from ONE and this model's features are zero-based, so an id
  // used as an index is off by one in one of them - the same mistake
  // `asymOrder` records making on the score keys. Residue numbers are rebuilt
  // per chain from the order they appear, which makes one ligand ONE residue
  // however its atoms were numbered, as the server writes it.
  const order = [...new Set(asymId)].sort((a, b) => a - b);
  const chainIds = [];
  const resIds = [];
  const numbering = new Map();
  for (let token = 0; token < asymId.length; token += 1) {
    const chain = order.indexOf(asymId[token]);
    chainIds.push(chainLetter(chain));
    const key = `${chain}:${residueIndex[token]}`;
    if (!numbering.has(key)) {
      const seen = [...numbering.keys()].filter((k) => k.startsWith(`${chain}:`)).length;
      numbering.set(key, seen + 1);
    }
    resIds.push(numbering.get(key));
  }
  return { chainIds, resIds };
}

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
      // 🔴 A MODIFIED RESIDUE IS PART OF THE JOB, NOT A RENDERING OF IT. The
      // request is what a reader hands back to reproduce the fold, and one
      // that lists the parent sequence alone describes a DIFFERENT job - the
      // same mistake as dropping the templates line. The server's dialect
      // names them `ptmType`/`ptmPosition`, with the CCD code prefixed, so
      // they are written the way the server would read them back.
      const modifications = (entity.modifications ?? [])
        .filter((modification) => (modification.code ?? "").trim() !== "")
        .map((modification) => ({
          ptmType: `CCD_${modification.code.trim().toUpperCase()}`,
          ptmPosition: modification.position,
        }));
      sequences.push({ proteinChain: { sequence: value, count,
        // ...absent rather than empty, because `modifications: []` is a claim
        // that the chain was checked and carries none, and every unmodified
        // fold this page has ever written says nothing at all.
        ...(modifications.length === 0 ? {} : { modifications }),
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
export function fullDataJson({ confidence, alignedError, pdb, tokenChainIds, tokenResIds }) {
  const tokens = tokenChainIds.length;
  // 🔴 THE ATOM ARRAYS ARE READ BACK OFF THE STRUCTURE IN THIS ARCHIVE, not
  // recomputed beside it. `atom_plddts` has to line up with the atoms of
  // `_model_0.pdb` record for record, and the only thing that guarantees that
  // is taking both from the same text - the writer drops an atom whose mask is
  // clear, so an independent walk over the sequence counts different atoms.
  const atoms = coordinateAtoms(pdb);
  // 🔴 THE B-FACTOR IS NOT ALWAYS A pLDDT, AND THE KEY MUST NOT SAY IT IS. A
  // model with no confidence head writes something else in that column -
  // EF2-fast writes the distogram certainty, under a REMARK naming it - and
  // `atom_plddts` would hand a reader the model's opinion where it has none.
  // Same rule as `has_clash`: a field we do not compute is omitted, not
  // guessed. The presence of a pLDDT vector is what says which this is.
  const scored = confidence.plddt !== undefined;
  const data = {
    atom_chain_ids: atoms.chains,
    [scored ? "atom_plddts" : "atom_certainty"]: Array.from(atoms.bFactors, round2),
  };
  // ...before the PAE, where the server puts it, and only when the fold
  // actually produced one.
  if (confidence.contactProbs !== undefined) {
    data.contact_probs = matrix2(paeMatrix(confidence.contactProbs, tokens));
  }
  // ...and the PAE only where there is one, for the same reason as above. It
  // was written unconditionally while `contact_probs` beside it was guarded.
  if (confidence.predictedAlignedError !== undefined) {
    data.pae = matrix2(paeMatrix(confidence.predictedAlignedError, tokens));
  } else if (alignedError !== undefined) {
    // 🔴 A DIFFERENT KEY, BECAUSE IT IS A DIFFERENT PROVENANCE. This is
    // estimated from the distogram rather than predicted by a confidence head
    // (src/esmfold2/aligned-error.js), and `pae` is the key a reader parses
    // expecting the server's - the same care `atom_certainty` takes with the
    // B-factor column. It is still a predicted aligned error, and it is still
    // in angstroms; what it is not is this model's own head's opinion.
    data.estimated_aligned_error = matrix2(paeMatrix(alignedError, tokens));
  }
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
/**
 * The strongest contact the model predicts, within each chain and between each
 * pair - one number per chain pair, off the distogram alone.
 *
 * 🔴 IT IS THE ONE SCORE A MODEL WITH NO CONFIDENCE HEAD CAN STILL GIVE.
 * AlphaFold 3 splits intra- from cross-chain too, but only through ipTM
 * (`iptm_ichain` and `iptm_xchain` in its own code) and its contact-weighted
 * PDE summaries - all of which need the confidence head. This needs the trunk,
 * which every model here has: the diagonal says how sure the model is that the
 * chain touches itself at range, and an off-diagonal entry says whether it
 * believes in the interface at all.
 *
 * 🔴 AND SEQUENCE NEIGHBOURS ARE EXCLUDED OR THE DIAGONAL IS ALWAYS 1. A token
 * is in contact with itself and with the residue beside it whatever the fold,
 * so an unfiltered maximum reports 1.00 for every chain and says nothing. The
 * rule is the one used everywhere else here - the same chain and within six
 * RESIDUES, which drops a ligand's whole self-block because its atoms share a
 * residue number. See ../src/heads/contact-threshold.js for why a token index
 * would not do.
 */
export function chainPairMaxContact(contactProbs, tokenChainIds, tokenResIds, chains) {
  const tokens = tokenChainIds.length;
  const index = new Map(chains.map((letter, at) => [letter, at]));
  const out = chains.map(() => chains.map(() => null));
  for (let i = 0; i < tokens; i += 1) {
    const a = index.get(tokenChainIds[i]);
    if (a === undefined) continue;
    for (let j = i + 1; j < tokens; j += 1) {
      const b = index.get(tokenChainIds[j]);
      if (b === undefined) continue;
      if (a === b && Math.abs(tokenResIds[i] - tokenResIds[j]) <= 6) continue;
      const value = contactProbs[i * tokens + j];
      if (out[a][b] === null || value > out[a][b]) {
        out[a][b] = value;
        out[b][a] = value;
      }
    }
  }
  return out.map((row) => row.map((value) => (value === null ? null : round2(value))));
}

export function summaryConfidencesJson({ confidence, chainLengths, tokenChainIds,
                                         tokenResIds }) {
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
  // ...before the ipTM scores, because it is the one that exists without a
  // confidence head and a reader of an EF2-fast archive will find nothing else.
  if (confidence.contactProbs !== undefined && tokenResIds !== undefined) {
    summary.chain_pair_max_contact = chainPairMaxContact(
      confidence.contactProbs, tokenChainIds, tokenResIds, chains);
  }
  // ...and the per-chain certainty, for a model that has no pTM to report. The
  // two are kept apart for AF3's own reason: a chain can be folded well and
  // docked badly, and one mean over both says neither.
  if (confidence.chainCertainty !== undefined) {
    summary.chain_certainty = confidence.chainCertainty;
  }
  if (confidence.chainInterfaceCertainty !== undefined && chains.length > 1) {
    summary.chain_interface_certainty = confidence.chainInterfaceCertainty;
  }
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

/**
 * 🔴 A README THAT DESCRIBES A CONTROL THE MODEL DOES NOT HAVE IS WRONG, NOT
 * MERELY VERBOSE. EF2-fast folds from the sequence alone - the page hides its
 * MSA row for exactly that reason - and its archive still said
 * `max msa: 128` and `alignment: none (single sequence)`, which are the shared
 * dials' values reported as though they had been used. `msaOrigin` is
 * undefined for such a model, and that is what drops both the line and the
 * paragraph about `msas/` below, which would otherwise describe a directory the
 * archive does not contain.
 */
function readme({ stem, model, settings, msaOrigin, templateCount, scored = true,
                  alignedError, alignmentOmitted = false }) {
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
  if (msaOrigin !== undefined) {
    // 🔴 AND "IT IS NOT HERE" IS PART OF WHAT RAN. The saved session drops the
    // alignment because it is 96.8% of the bytes - measured on ubiquitin,
    // 3,001,450 of 3,101,347, against a fold's own ~100 KB - and a reader who
    // cannot tell an omitted alignment from an absent one will re-search and
    // quietly get a different fold.
    lines.push(`- alignment: ${msaOrigin}${alignmentOmitted ? " (not in this archive)" : ""}`);
  }
  if (templateCount !== undefined) {
    lines.push(`- templates: ${templateCount === 0 ? "none" : `${templateCount} used`}`);
  }
  if (!scored) {
    // ...said once, plainly, because the B-factor column and a missing summary
    // are both surprising on their own and neither explains itself.
    lines.push(
      "",
      "This checkpoint has no confidence head, so there is no pLDDT, no pTM and",
      "no PAE - those fields are absent rather than zero, which would read as",
      "the model's opinion. What the trunk can still say is in",
      "`_summary_confidences_0.json` as `chain_pair_max_contact`, and the",
      "structure's B-factor column carries a distogram-derived certainty; it is",
      "an ordering, not a calibrated score, and the PDB says so in a REMARK.",
    );
    // ...and the estimate, described where somebody reading the file will look
    // for it. It is named `estimated_aligned_error` rather than `pae` precisely
    // so a reader parsing the server's format does not pick it up unknowingly,
    // which means the README has to say it is there.
    if (alignedError !== undefined) {
      lines.push(
        "",
        "`_full_data_0.json` does carry `estimated_aligned_error`, which is a",
        "predicted aligned error in angstroms READ OFF THE DISTOGRAM rather than",
        "produced by a confidence head. It is under its own key, not `pae`, so",
        "that a reader expecting the server's field does not take it for one.",
        "",
        "It orders pairs WITHIN this fold - which parts are placed reliably",
        "against which, the question a PAE plot is read for - and it does not",
        "compare between folds: its absolute values regress toward the middle of",
        "the range, so a mean of it says little. Read the map, not the number.",
      );
    }
  }
  lines.push(
    "",
    "## Layout",
    "",
    "The AlphaFold 3 server's, with one difference: the structure is written as",
    "PDB rather than mmCIF.",
  );
  if (msaOrigin !== undefined && alignmentOmitted) {
    // 🔴 THE THIRD STATE. `msaOrigin` alone answered "does this MODEL take an
    // alignment", and the paragraph below assumed that taking one means
    // carrying one. A saved session takes one and carries none, so keying on
    // the origin alone described an `msas/` that is not in the file - the same
    // class of wrong README as the one that claimed EF2-fast reads the MSA
    // dial. A re-search reproduces a fold, not THIS fold.
    lines.push(
      "",
      "There is no `msas/`: this fold used an alignment, and it was left out to",
      "keep the file small. Folding this sequence again will search afresh and",
      "may find different hits, so the structure it produces will resemble this",
      "one without reproducing it. Use \"Download all\" on a live fold to get an",
      "archive that carries its alignment and restores exactly.",
    );
  } else if (msaOrigin !== undefined) {
    lines.push(
      "",
      "`msas/` holds one alignment per chain, split into the paired and unpaired",
      "blocks the model reads separately. Drop this whole .zip onto LocalFold's",
      "alignment upload box to fold again with exactly these alignments - the two",
      "blocks are not interchangeable, so re-uploading a single merged a3m would",
      "not reproduce this fold.",
    );
  } else {
    lines.push(
      "",
      "There is no `msas/`: this model folds from the sequence alone.",
    );
  }
  lines.push("");
  return lines.join("\n");
}

/**
 * Every member of the archive, ready for writeZip.
 *
 * @returns {Map<string, string>}
 */
export function buildFoldArchive({
  stem, model, settings, entities, prediction, msas = {}, templates,
  msaOrigin, alignmentOmitted = false,
}) {
  const name = safeJobName(stem);
  const { confidence, alignedError, pdb, chainLengths } = prediction;
  // 🔴 THE TOKEN COUNT DOES NOT COME FROM THE PAE, because a model can have no
  // confidence head at all. EF2-fast has none - `lastPrediction` carries no
  // `confidence` object on purpose, since an object of zeros would be read as
  // the model's opinion - and taking the square root of an absent matrix's
  // length threw "Cannot read properties of undefined (reading 'length')" on
  // the download button, which names neither the model nor the field.
  const tokens = prediction.tokens?.chainIds?.length
    ?? (confidence.predictedAlignedError !== undefined
      ? Math.round(Math.sqrt(confidence.predictedAlignedError.length))
      : chainLengths.reduce((total, length) => total + length, 0));
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
  // 🔴 AND THE SUMMARY IS OMITTED WHEN IT WOULD HOLD ONLY CHAIN LETTERS. Every
  // score in it is guarded on the field it needs, so a model with no confidence
  // head produced a file whose entire content was `chain_ids` - 35 copies of
  // "A" for a 35-residue fold. A field we do not compute is omitted, not
  // guessed; the same is true of a file.
  const summary = summaryConfidencesJson({
    confidence, chainLengths, tokenChainIds: chainIds, tokenResIds: resIds,
  });
  // 🔴 TWO DIFFERENT QUESTIONS, AND ONE FLAG WAS ANSWERING BOTH. Whether the
  // FILE has anything in it decides whether to write it; whether the MODEL has
  // a confidence head decides what the README explains. They came apart the
  // moment `chain_pair_max_contact` gave a head-less model something to say.
  const hasScores = Object.keys(JSON.parse(summary)).some((key) => key !== "chain_ids");
  if (hasScores) files.set(`${name}_summary_confidences_0.json`, summary);
  const scored = confidence.plddt !== undefined;
  files.set(`${name}_full_data_0.json`, fullDataJson({
    confidence, alignedError, pdb, tokenChainIds: chainIds, tokenResIds: resIds,
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

  (templates ?? []).forEach((template, index) => {
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
    // 🔴 UNDEFINED MEANS "THIS MODEL HAS NONE", AS `msaOrigin` DOES. EF2-fast
    // cannot take a template at all - `grep -rn template` over the whole
    // upstream package returns nothing - so "templates: none" reported a choice
    // where there was no control. An empty ARRAY still means "none were used".
    stem, model, settings, msaOrigin, scored, alignedError, alignmentOmitted,
    templateCount: templates === undefined ? undefined : templates.length,
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
