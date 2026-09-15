/**
 * Fold a MULTI-CHAIN target and score every chain in ONE frame.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/fold-complex.js --target=1brs --chains=A,D
 *     ... --model=/model-af3-int5/manifest.json --mode=ode
 *
 * 🔴 AND PICK A TARGET WITH AN INTERFACE, WHICH IS NOT WHAT "TWO CHAINS IN THE
 * FILE" MEANS. The first target tried here was 5CAJ A:B, which this repository
 * already had as a fixture - and its two chains are two independent copies in
 * the asymmetric unit whose closest alpha carbons are **11.08 A** apart, with
 * ZERO contacts under 8 A and centroids 44 A apart. Their relative placement is
 * crystal packing, so a "complex RMSD" over them measures nothing a model
 * predicts, and the 9 to 24 A that produced read exactly like an interface
 * defect. `interface.nativeContacts` is in the report so that cannot happen
 * quietly again: 0 native contacts means the target is the wrong question.
 * 1BRS A:D (barnase and barstar) has 36, and 1TIM A:B - a real
 * 494-residue homodimer - has 101, which is the LARGE target.
 *
 * 🔴 THE AF3 LINEAGE HAD NO COMPLEX SCORER, WHICH IS WHY EVERY NUMBER IN THESE
 * DOCS IS A SINGLE CHAIN OF 68 TO 92 RESIDUES. `fold-opendde.js` reads ONE
 * chain out of the crystal and folds its sequence alone; the AF2 side has
 * `probe-recycles-on-complexes.js` and its neighbours, and this side has
 * nothing. So the whole sampler comparison this tool was written for - flow
 * against ODE against diffusion - had been decided on 6MRR, a 68-residue
 * DESIGNED protein that this repository's own notes call too easy to
 * discriminate ("folds to 0.5 A from its sequence alone").
 *
 * 🔴 AND AN INTERFACE IS THE PART A PER-CHAIN SCORE CANNOT SEE. Two chains can
 * each be perfect and still be in the wrong place relative to each other, which
 * is the only failure mode a complex has that a monomer does not. So the
 * superposition is fitted ONCE over every chain's alpha carbons together and
 * every number is reported in that shared frame - `complex` is the honest
 * score, and the per-chain rows below it say whether a bad one is a bad FOLD or
 * a bad PLACEMENT.
 *
 * 🔴 THE CHAINS ARE JOINED WITH ":" AND THAT IS THE MODEL'S OWN CONVENTION, not
 * this tool's - `af3BatchFromA3m` splits on it and `chainIdentity()` numbers
 * them, the same path the page takes when a user pastes two chains.
 */
import { foldBatch } from "../../src/af3/fold.js";
import { af3BatchFromA3m } from "../../src/af3/featurise/batch.js";
import { dialectFor, featuriserDialect } from "../../src/af3/dialect.js";
import { openAf3Store } from "../../src/af3/weights/weights.js";
import { foldWeights } from "../../src/af3/weights/diffusion-weights.js";
import { assertChainGeometry } from "./chain-geometry.js";
import { superpose, scoreSlice, modelAlphaCarbons, chainAssignments }
  from "./superpose.js";
import { buildTemplate } from "../../web/template-source.js";
import { mergeTemplateSlots } from "../../src/af3/featurise/template-input.js";
import { multichainMaskFor, coverageOf } from "../../src/af3/featurise/template-features.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const THREE_TO_ONE = {
  ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E", GLY: "G",
  HIS: "H", ILE: "I", LEU: "L", LYS: "K", MET: "M", PHE: "F", PRO: "P", SER: "S",
  THR: "T", TRP: "W", TYR: "Y", VAL: "V", MSE: "M",
};

/** One chain's sequence and alpha carbons, in residue-number order. */
function readChain(text, wanted) {
  const residues = new Map();
  for (const line of text.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    const alt = line[16];
    if (alt !== " " && alt !== "A") continue;
    if (line[21] !== wanted) continue;
    const code = THREE_TO_ONE[line.slice(17, 20).trim()];
    if (code === undefined) continue;
    const number = Number(line.slice(22, 26));
    if (!residues.has(number)) residues.set(number, { code, ca: null });
    if (line.slice(12, 16).trim() === "CA") {
      residues.get(number).ca = [Number(line.slice(30, 38)), Number(line.slice(38, 46)),
                                 Number(line.slice(46, 54))];
    }
  }
  const ordered = [...residues.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
  return { sequence: ordered.map((r) => r.code).join(""),
           alphaCarbons: ordered.map((r) => r.ca) };
}

export async function main(device, args) {
  const target = option(args, "target", "1brs");
  const wanted = option(args, "chains", "A,D").split(",").filter((c) => c !== "");
  const mode = option(args, "mode", "diffusion");
  const steps = Number(option(args, "steps", mode === "diffusion" ? "25" : "16"));
  const manifest = option(args, "model", "/model-af3-int5/manifest.json");

  const text = await (await fetch(`/tools/fixtures/${target}-crystal.pdb`)).text();
  const chains = wanted.map((id) => ({ id, ...readChain(text, id) }));
  for (const chain of chains) {
    if (chain.sequence.length === 0) throw new Error(`${target} has no chain ${chain.id}`);
  }
  const sequence = chains.map((c) => c.sequence).join(":");

  const store = await openAf3Store(manifest);
  const weights = await foldWeights(store);
  const dialect = weights.trunk?.dialect ?? dialectFor("alphafold3");
  const { batch } = af3BatchFromA3m(sequence, null,
    { ...featuriserDialect(dialect), chainKinds: chains.map(() => "protein") });

  // 🔴 A TEMPLATE, AND FOR A COMPLEX THERE ARE TWO WAYS TO GIVE ONE. The page
  // builds one SLOT PER CHAIN, which is AF3's own convention and is right for a
  // monomer; for a complex it throws the interface away, because
  // `multichainMaskFor` opens a cross-chain pair only where one slot covers
  // BOTH ends. `--per-chain-templates` is that arm; the default merges the
  // chains into ONE slot with `spanChains`, which is the only way the model is
  // told where the chains sit relative to each other. See
  // `mergeTemplateSlots` in src/af3/featurise/template-input.js.
  const templatePath = option(args, "template", "");
  const perChainTemplates = args.includes("--per-chain-templates");
  let templateSlots;
  let templateCoverage;
  let crossChainOpen = 0;
  if (templatePath !== "") {
    const templateText = await (async () => {
      const response = await fetch(templatePath);
      if (!response.ok) throw new Error(`failed to load ${templatePath}: ${response.status}`);
      return response.text();
    })();
    // Residue -> token, and the residues of each chain, off the batch itself -
    // the same construction web/af3-model.js uses, because a chain after a
    // modified residue does not start where a residue count says.
    const tokenOfResidue = new Int32Array(batch.sequence.length).fill(-1);
    for (let token = batch.tokens - 1; token >= 0; token -= 1) {
      const residue = batch.residueOfToken?.[token] ?? token;
      if (residue >= 0 && residue < tokenOfResidue.length) tokenOfResidue[residue] = token;
    }
    const residuesOfChain = [];
    for (let residue = 0; residue < batch.sequence.length; residue += 1) {
      const chain = batch.chainOfResidue?.[residue] ?? 0;
      (residuesOfChain[chain] ??= []).push(residue);
    }
    const built = chains.map((chain, index) => buildTemplate({
      text: templateText, chain: chain.id, query: chain.sequence,
      tokens: batch.tokens, minConfidence: 0,
      // Spanning is a property of the SLOT and only means anything on the
      // merged one; a per-chain slot covers one end of every cross pair.
      spanChains: !perChainTemplates,
      tokenOf: (residue) =>
        tokenOfResidue[(residuesOfChain[index] ?? [])[residue] ?? -1] ?? -1,
    }));
    templateCoverage = built.map((b, index) => ({
      chain: chains[index].id, residues: b.coverage.residues, of: b.coverage.of,
    }));
    const slots = built.map((b) => b.slot);
    // 🔴 `--no-span-chains` IS THE ARM THAT ISOLATES THE INTERFACE. Merged
    // against per-chain confounds two things at once under a dialect that
    // averages the features over PRESENT slots: the merged slot both opens the
    // cross-chain block AND carries full intra-chain weight, where two
    // per-chain slots each carry half. This arm is the SAME single merged slot
    // with the cross-chain block masked, so the only difference is the
    // interface.
    const spanChains = !args.includes("--no-span-chains");
    templateSlots = perChainTemplates ? slots : [(() => {
      const merged = mergeTemplateSlots(slots);
      merged.spanChains = spanChains;
      return merged;
    })()];
    // 🔴 AND THE TOOL COUNTS WHAT THE MASK ACTUALLY OPENS, because "the merged
    // slot lets the template speak across the boundary" is a claim about a
    // mask this tool never looks at - and a `spanChains` that failed to reach
    // `chainMaskFor` would produce exactly the per-chain answer while the
    // report said "merged". This is the same computation the embedder does,
    // over the same slots.
    crossChainOpen = templateSlots.reduce((most, slot) => {
      const mask = multichainMaskFor(batch.asymId, batch.tokens, {
        coverage: coverageOf(slot, batch.tokens), spanChains: slot.spanChains === true });
      let open = 0;
      for (let i = 0; i < batch.tokens; i += 1) {
        for (let j = 0; j < batch.tokens; j += 1) {
          if (batch.asymId[i] !== batch.asymId[j] && mask[i * batch.tokens + j] > 0) open += 1;
        }
      }
      return Math.max(most, open);
    }, 0);
  }

  const started = performance.now();
  const result = await foldBatch(device, batch, weights, {
    mode, steps, recycles: Number(option(args, "recycles", "3")),
    seed: Number(option(args, "seed", "20260831")),
    ...(templateSlots === undefined ? {} : { templateSlots }),
  });
  const seconds = (performance.now() - started) / 1000;

  // 🔴 GEOMETRY FIRST, because a complex that is not a chain scores like one
  // that is - see chain-geometry.js.
  const geometry = result.geometry;
  const verdict = assertChainGeometry(geometry,
    { plddt: result.meanPlddt, allow: args.includes("--allow-broken-geometry"),
      doc: "docs/AF3.md" });

  const model = modelAlphaCarbons(batch, result.positions);
  const round = (s) => (s === null || s === undefined ? null
    : { residues: s.residues ?? s.pairs, rmsd: Math.round(s.rmsd * 1000) / 1000,
        tm: Math.round(s.tm * 10000) / 10000 });

  // 🔴 A HOMODIMER'S CHAINS ARE INTERCHANGEABLE AND THE SCORE MUST SAY SO.
  // 5CAJ's A and B are the SAME 261-residue sequence, so nothing distinguishes
  // "the model's first chain" from either of the crystal's - a perfect
  // prediction with the two labels the other way round scores as a total
  // failure. AlphaFold 3 does this itself and calls it chain permutation
  // alignment. Only chains with an IDENTICAL sequence may swap: permuting two
  // different chains is not a relabelling, it is a different answer.
  const chainAt = chains.flatMap((chain, index) =>
    chain.alphaCarbons.map(() => index));
  const assignments = chainAssignments(chains.map((c) => c.sequence));
  const truthFor = (assignment) => assignment.flatMap((from) => chains[from].alphaCarbons);
  // 🔴 ONE FRAME FOR THE WHOLE COMPLEX. Fitting per chain would report two
  // perfect chains that are nowhere near each other as a perfect answer.
  const scoredAssignments = assignments.map((assignment) => {
    const truthHere = truthFor(assignment);
    const fit = superpose(model, truthHere);
    return { assignment, truth: truthHere, fit };
  });
  if (scoredAssignments[0].fit.pairs === 0) {
    throw new Error("no alpha carbons paired with the crystal");
  }
  const best = scoredAssignments.reduce((a, b) => (b.fit.rmsd < a.fit.rmsd ? b : a));
  const identity = scoredAssignments[0];
  const truth = best.truth;
  const fitted = best.fit;

  let at = 0;
  const perChain = {};
  for (const [slot, chain] of chains.entries()) {
    const against = chains[best.assignment[slot]];
    const count = chain.alphaCarbons.length;
    // 🔴 TWO SCORES A CHAIN, AND THE PAIR IS THE DIAGNOSTIC. `inComplex` is the
    // chain measured in the frame fitted over everything, which is what a
    // complex is judged on; `alone` re-fits that chain by itself. A chain that
    // is good ALONE and bad IN COMPLEX is folded right and placed wrong - the
    // interface failed - and a chain bad both ways did not fold. Reporting only
    // the first cannot tell those apart, and they want different fixes.
    perChain[chain.id] = {
      against: against.id,
      inComplex: round(scoreSlice(model, truth, fitted.place, fitted.d0, at, count)),
      alone: round(superpose(model.slice(at, at + count), truth.slice(at, at + count))),
    };
    at += count;
  }
  // 🔴 AND THE INTERFACE ITSELF, BECAUSE AN RMSD OVER A COMPLEX CANNOT NAME
  // WHAT WENT WRONG. Two chains 9 A out and two chains that never touched score
  // in the same range, and the question a complex asks is whether the model
  // found the CONTACTS. `fnat` is the fraction of the crystal's inter-chain
  // alpha-carbon contacts (under 8 A) that the model also makes - a
  // superposition-free number, so it says nothing about the fit and everything
  // about the interface.
  const contactsOf = (points) => {
    const set = new Set();
    for (let i = 0; i < points.length; i += 1) {
      if (points[i] === null || points[i] === undefined) continue;
      for (let j = i + 1; j < points.length; j += 1) {
        if (chainAt[i] === chainAt[j]) continue;
        const q = points[j];
        if (q === null || q === undefined) continue;
        const p = points[i];
        if (Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]) < 8) set.add(i * 100000 + j);
      }
    }
    return set;
  };
  const native = contactsOf(truth);
  const predicted = contactsOf(model);
  let shared = 0;
  for (const contact of predicted) if (native.has(contact)) shared += 1;
  // 🔴 AND RECALL ALONE IS FOOLED BY A COLLAPSE, WHICH IT DID WITHIN AN HOUR OF
  // BEING WRITTEN. `fnat` is shared/native - how many of the crystal's contacts
  // the model also makes - and a fold that has collapsed into a ball makes
  // EVERY pair a contact, so it recovers all of them: boltz2 in flow on 1TIM
  // seed 21 read **fnat 0.901 with its chains 16.8 A out of shape**, which
  // reads as a good interface on a fold that has none. `precision` is the other
  // half - how many of the model's contacts are real - and it is what a
  // collapse destroys. Report both; a high recall with a low precision is a
  // blob, not an interface.
  const interface_ = {
    nativeContacts: native.size,
    modelContacts: predicted.size,
    shared,
    // The crystal having no interface at all is a fact about the target, not a
    // score of zero - say null rather than dividing by nothing.
    fnat: native.size === 0 ? null : Math.round((shared / native.size) * 1000) / 1000,
    precision: predicted.size === 0 ? null
      : Math.round((shared / predicted.size) * 1000) / 1000,
  };

  return {
    target, chains: wanted, mode, steps, tokens: batch.tokens,
    interface: interface_,
    template: templatePath === "" ? null
      : { path: templatePath, slots: perChainTemplates ? "per chain"
            : (args.includes("--no-span-chains") ? "merged, chains masked" : "merged"),
          coverage: templateCoverage, crossChainPairsOpen: crossChainOpen },
    residues: truth.length, seconds: Math.round(seconds * 10) / 10,
    meanPlddt: result.meanPlddt, geometry, geometryOk: verdict.ok,
    // The number that counts: every chain in the shared frame, under the best
    // relabelling of interchangeable chains. `asLabelled` is the same score
    // without the permutation, so the difference between them is visible
    // rather than hidden inside a minimum.
    complex: round({ residues: fitted.pairs, rmsd: fitted.rmsd, tm: fitted.tm }),
    permutation: best.assignment.map((from) => chains[from].id).join(""),
    permutations: scoredAssignments.length,
    asLabelled: round({ residues: identity.fit.pairs, rmsd: identity.fit.rmsd,
                        tm: identity.fit.tm }),
    perChain,
  };
}
