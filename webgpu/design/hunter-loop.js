/**
 * Protein Hunter's design loop: fold, redesign off the prediction, fold again.
 *
 * Transcribed from `_run_design_cycle` in `boltz_ph/pipeline.py` of
 * github.com/yehlincho/Protein-Hunter (biorxiv 2025.10.10.681530). The method
 * is short and the paper's contribution is that it works, not that it is
 * complicated:
 *
 *     cycle 0   fold a mostly-X starting sequence
 *     cycle k   MPNN off cycle k-1's backbone -> fold -> score
 *     keep      the cycle with the best interface score
 *
 * 🔴 `fold` AND `design` ARE ARGUMENTS, WHICH IS THE WHOLE REASON THIS FILE
 * EXISTS SEPARATELY. `npm run test:gpu` cannot load Dawn on the machine this
 * was written on, so a loop that reached for a GPUDevice would have no test
 * that runs here at all. Injected, every schedule in it - the ramp, the
 * first-cycle proline omission, the alanine guard, which cycle wins, where an
 * abort lands - is checked by the CPU suite against stub functions, and the
 * page supplies the real pair. See test/hunter-loop.test.js.
 *
 * 🔴 IT IS A GENERATOR BECAUSE A CYCLE IS A RESULT, NOT A STEP TOWARDS ONE.
 * Each fold is seconds of GPU time and the structure it produced is worth
 * looking at whether or not the run continues. Returning an array at the end
 * would mean a page that shows nothing for a minute and then everything, and
 * a stop button that throws away work that was already done.
 */
import { throwIfAborted } from "../runtime/abort.js";
import { uniformFrom } from "../af3/fold.js";
import {
  alanineBias, alanineFraction, omittedLetters, sampleSequence,
} from "../../cpu/design/sample-sequence.js";

/**
 * The most alanine a cycle may contain and still be allowed to win.
 *
 * MPNN's answer to a backbone it cannot read is a poly-alanine helix, which
 * folds confidently, scores well, and is not a design. The reference's guard,
 * and it is a guard on SELECTION only - the cycle still runs, still shows, and
 * the next cycle still designs off it.
 */
export const MAX_ALANINE = 0.20;

/** Chain letters, in the order `toPdb` in webgpu/af3/fold.js assigns them. */
export const CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/**
 * The number a run is optimised on.
 *
 * 🔴 ipTM FOR A COMPLEX, pLDDT FOR A MONOMER, AND THE REFERENCE HAS NO SECOND
 * CASE. Protein Hunter is a binder method: it reads `pair_chains_iptm` and
 * takes 0.0 when there is one chain, so a monomer run there optimises nothing
 * and every cycle ties. A monomer hallucination is a reasonable thing to ask
 * this page for, and mean pLDDT is what it has - so that is what it selects
 * on, divided by 100 so the run's `score` column is 0-1 either way. That
 * division is cosmetic: selection only ever compares cycles of ONE run, which
 * all carry the same objective. `objective` says which was used, because the
 * two numbers are not comparable across runs however they are scaled.
 *
 * @param {{meanPlddt: number, iptm: number, chainPairIptm?: object}} confidence
 * @param {number} chainIndex the designed chain, zero-based
 * @returns {{value: number, objective: "iptm"|"plddt"}}
 */
export function objectiveOf(confidence, chainIndex) {
  const pairs = confidence.chainPairIptm ?? {};
  // 🔴 THE INTERFACES THE DESIGNED CHAIN IS IN, NOT THE POOLED ipTM. With
  // three chains the pooled number averages the target's own interface with
  // the binder's, so a binder that misses can be carried by a target that
  // holds together. The reference takes max(pair[i][j], pair[j][i]) over
  // every j != i and means those, which is what this reproduces.
  // 🔴 THE KEYS ARE ASYM IDS JOINED BY "|", NOT CHAIN LETTERS. chainPairTmScores
  // in shared/heads/tm-score.js writes `${first}|${second}` over the numeric
  // asym_id, and toPdb turns asym n into letter n-1 - so chain index 0 is
  // asym 1 is "A". Matching on letters here silently found no interface at all
  // and every complex fell through to pLDDT.
  const asym = String(chainIndex + 1);
  const involved = [];
  for (const [key, value] of Object.entries(pairs)) {
    const ends = key.split("|");
    if (ends.length === 2 && (ends[0] === asym || ends[1] === asym)
        && Number.isFinite(value)) {
      involved.push(value);
    }
  }
  if (involved.length > 0) {
    let total = 0;
    for (const value of involved) total += value;
    return { value: total / involved.length, objective: "iptm" };
  }
  if (Number.isFinite(confidence.iptm)) {
    return { value: confidence.iptm, objective: "iptm" };
  }
  return { value: confidence.meanPlddt / 100, objective: "plddt" };
}

/**
 * Assemble the sequence `foldAf3` folds: the designed chain in its slot, the
 * fixed chains around it, colon-separated.
 *
 * @param {string[]} chains
 * @param {number} index
 * @param {string} sequence
 * @returns {string}
 */
export function withChain(chains, index, sequence) {
  const all = [...chains];
  all[index] = sequence;
  return all.join(":");
}

/**
 * One design run: a starting sequence and `cycles` rounds of redesign.
 *
 * @param {object} options
 * @param {(sequence: string, context: object) => Promise<object>} options.fold
 *   given the colon-joined complex, resolves to `foldAf3`'s result - it needs
 *   `pdb` and `confidence`
 * @param {(pdb: string, context: object) => Promise<object>|object}
 *   options.design given the prediction, resolves to `{sequence}` for the
 *   designed chain. See designChain() in ./mpnn-bridge.js.
 * @param {string[]} options.chains every chain's sequence; the designed one's
 *   entry is replaced each cycle and may start empty
 * @param {number} [options.chainIndex=0] which of them is designed. The
 *   reference fixes this at "A" and so does the page, but the loop does not
 *   need to.
 * @param {number} [options.cycles=5]
 * @param {number} [options.length] the designed chain's length when it has no
 *   starting sequence; drawn from [minLength, maxLength] when absent
 * @param {number} [options.minLength=100] the reference's
 *   --min_protein_length. The page always passes `length` and so never reaches
 *   these two; they are kept because a length RANGE is part of the method -
 *   the reference draws a new one per design - and a caller sweeping lengths
 *   would otherwise have to reimplement the draw.
 * @param {number} [options.maxLength=150]
 * @param {number} [options.percentX=90]
 * @param {boolean} [options.excludeP=false] see sampleSequence
 * @param {number} [options.temperature=0.1]
 * @param {string} [options.omit="C"]
 * @param {boolean} [options.alanineBias=false]
 * @param {number} [options.alanineBiasStart=-0.5]
 * @param {number} [options.alanineBiasEnd=-0.1]
 * @param {Iterable<number>} [options.hold] positions within the designed chain
 *   to keep as they are, every cycle. See designChain() in ./mpnn-bridge.js.
 * @param {(cycle: object) => (string|null|Promise<string|null>)} [options.veto]
 *   given a finished cycle, a reason it is not a candidate - or null to keep it.
 *   The reason lands on the record as `vetoed`.
 * @param {number} [options.beam=1] how many sequences to sample per cycle. Each
 *   is folded and the best carries forward, so this multiplies the folds.
 * @param {number} [options.seed=0]
 * @param {number} [options.run=0] carried onto every record, for the table
 * @param {AbortSignal} [options.signal]
 * @yields {object} one record per cycle
 */
export async function* runDesign(options) {
  const {
    fold, design, chains, chainIndex = 0, cycles = 5, signal, run = 0,
  } = options;
  if (typeof fold !== "function") throw new TypeError("runDesign needs a fold function");
  if (typeof design !== "function") throw new TypeError("runDesign needs a design function");
  if (!Array.isArray(chains) || chains.length === 0) {
    throw new TypeError("runDesign needs at least one chain");
  }
  if (chainIndex < 0 || chainIndex >= chains.length) {
    throw new RangeError(`chainIndex ${chainIndex} is outside ${chains.length} chains`);
  }

  // 🔴 ONE STREAM, SEEDED ONCE. The starting length, the starting sequence and
  // every design draw come off the same generator, so a seed names the whole
  // run - not the sequence but not the length, which is the shape a seed
  // silently stops meaning anything in.
  const random = uniformFrom(options.seed ?? 0);
  const temperature = options.temperature ?? 0.1;
  const beam = Math.max(1, Math.floor(options.beam ?? 1));
  const standingOmit = options.omit ?? "C";

  let sequence = chains[chainIndex] ?? "";
  if (sequence.length === 0) {
    const minLength = options.minLength ?? 100;
    const maxLength = options.maxLength ?? 150;
    const length = options.length
      ?? (minLength + Math.floor(random() * (maxLength - minLength + 1)));
    sequence = sampleSequence(length, {
      percentX: options.percentX ?? 90,
      excludeP: options.excludeP ?? false,
      random,
    });
  }

  let best = null;
  let previous = null;
  const chainLetter = CHAIN_IDS[chainIndex];

  for (let cycle = 0; cycle <= cycles; cycle += 1) {
    throwIfAborted(signal);

    // Cycle 0 folds the starting sequence; every later cycle designs first.
    let designScore;
    let bias;
    let omit;
    // The cycle's candidate sequences. Empty on cycle 0, which folds the
    // starting draw rather than designing anything.
    let drawn = [];
    if (cycle > 0) {
      omit = omittedLetters(cycle - 1, standingOmit);
      bias = options.alanineBias === true
        ? alanineBias(cycle - 1, cycles, {
          start: options.alanineBiasStart, end: options.alanineBiasEnd,
        })
        : 0;
      // ...off the previous cycle's structure, which is the loop's whole point.
      //
      // 🔴 SEVERAL SEQUENCES FROM ONE BACKBONE IS A BEAM, and `beam` is how
      // wide. MPNN samples from the same structure each time - the draws differ
      // because they come off the same seeded stream - so the candidates are
      // different sequences for the SAME backbone, which is the thing worth
      // searching over. The reference's method is a beam of one.
      const candidates = [];
      for (let draw = 0; draw < beam; draw += 1) {
        const designed = await design(previous.pdb, {
          chain: chainLetter, temperature, omit, alanineBias: bias, random, signal,
          cycle, run, draw,
          // 🔴 PASSED STRAIGHT THROUGH, because WHICH positions are held is the
          // caller's question and not the loop's: a framework, a motif, or
          // nothing. The loop's only stake is that it is the same every cycle -
          // held residues that moved between cycles would make the trajectory
          // meaningless.
          hold: options.hold,
        });
        throwIfAborted(signal);
        candidates.push(designed);
      }
      drawn = candidates;
    }

    // 🔴 EVERY CANDIDATE IS FOLDED AND THE BEST ONE BECOMES THE CYCLE. Scoring
    // them by MPNN's own likelihood instead would be cheaper and would measure
    // the wrong thing: the objective is what the FOLD says, and a sequence MPNN
    // likes is not the same as a sequence that folds to the target.
    let folded;
    let picked = null;
    if (drawn.length <= 1) {
      sequence = drawn.length === 1 ? drawn[0].sequence : sequence;
      designScore = drawn[0]?.score;
      folded = await fold(withChain(chains, chainIndex, sequence), { cycle, run, signal });
      throwIfAborted(signal);
    } else {
      for (const [draw, candidate] of drawn.entries()) {
        const attempt = await fold(withChain(chains, chainIndex, candidate.sequence),
          { cycle, run, draw, draws: drawn.length, signal });
        throwIfAborted(signal);
        const score = objectiveOf(attempt.confidence ?? {}, chainIndex);
        if (picked === null || score > picked.score) {
          picked = { score, candidate, attempt };
        }
      }
      sequence = picked.candidate.sequence;
      designScore = picked.candidate.score;
      folded = picked.attempt;
    }

    const confidence = folded.confidence ?? {};
    const { value, objective } = objectiveOf(confidence, chainIndex);
    const alanine = alanineFraction(sequence);
    // 🔴 THE GUARD IS ON WINNING, NOT ON CONTINUING. A cycle over the alanine
    // ceiling still runs and is still designed off; it just cannot be the
    // answer. Skipping it instead would end the run at its worst cycle.
    //
    // 🔴 AND CYCLE 0 CANNOT WIN EITHER, WHICH IS NOT AN EXTRA RULE. Its
    // sequence is the mostly-X draw the run STARTED from - nothing designed
    // it - so "best" naming it would be reporting the question as the answer.
    // The reference gets this by construction: it records cycle 0's metrics
    // and only ever compares inside the design loop. Measured here on a
    // 20-mer of pure X, cycle 0 folded to pLDDT 72.9 and won a run whose two
    // design cycles were both over the alanine ceiling - twenty unknown
    // residues offered as a design.
    // 🔴 VALIDITY IS A VETO, AND IT IS THE CALLER'S TO SUPPLY. The loop knows
    // two things that make a cycle ineligible - it is cycle 0, or it is over the
    // alanine ceiling - and both are about the SEQUENCE. Whether the molecule is
    // right (does the binder touch its target, does a ring close, do the bases
    // pair) is geometry, and the loop has no business parsing structures. A
    // caller that cares passes `veto`.
    //
    // 🔴 THE REASON IS KEPT, not just the verdict: a rejected cycle has to be
    // able to say why, with its measurement.
    let vetoed = null;
    if (cycle > 0 && typeof options.veto === "function") {
      vetoed = (await options.veto({ pdb: folded.pdb, confidence, sequence, cycle, run })) ?? null;
    }
    const eligible = cycle > 0 && alanine <= MAX_ALANINE && vetoed === null;
    const record = {
      run, cycle, sequence, alanine, objective, score: value, designScore, vetoed,
      // How many sequences this cycle tried, so a table can say what it cost.
      drawn: drawn.length,
      omit, alanineBias: bias, chainIndex, chain: chainLetter,
      complex: withChain(chains, chainIndex, sequence),
      pdb: folded.pdb,
      meanPlddt: confidence.meanPlddt,
      ptm: confidence.ptm,
      iptm: confidence.iptm,
      chainPairIptm: confidence.chainPairIptm,
      folded,
      best: false,
    };
    if (eligible && (best === null || value > best.score)) {
      if (best !== null) best.best = false;
      record.best = true;
      best = record;
    }
    previous = record;
    yield record;
  }
}
