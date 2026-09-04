/**
 * AF2 training set: distogram in, pLDDT out, one row per residue per recycle.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-af2-dgram-plddt.js --sample=10
 *     node tools/gpu-chrome.mjs tools/gpu/probe-af2-dgram-plddt.js --from=0 --to=12
 *
 * WHY AF2 AND NOT AF3. The question is how to turn a distogram into a pLDDT,
 * and AF2 answers it for free: its confidence head runs on EVERY recycle, so
 * each pass hands over a distogram, the structure it produced and the pLDDT
 * that structure was given, all labelled and all real. AF3 runs its head once,
 * at the end, so every label there has to be manufactured - which is the very
 * gap this map is being fitted to close.
 *
 * 🔴 THE DISTOGRAM COMES FROM THE HEAD THIS REPO ONLY JUST CONVERTED. AF2 has
 * always had one - it is what the contact map in every AlphaFold figure is
 * drawn from - and neither published bundle carried it until
 * tools/add_distogram_head.py added it. src/heads/distogram.js is the head;
 * one projection of the pair representation to 64 bins over 2-22 A,
 * symmetrised.
 *
 * 🔴 SINGLE SEQUENCE, DELIBERATELY. An alignment makes almost everything fold,
 * and a set where every label is 85 teaches a predictor that the answer is
 * always 85. Query-only gives the range: some of these resolve and some do not.
 *
 * WHAT IS SAVED, per target per recycle: the distogram as bytes, the alpha
 * carbons, the head's per-residue pLDDT, and lDDT against the native for
 * context. Everything a feature might be built from later, rather than the
 * values of features chosen now.
 */
import { AlphaFoldFixture } from "../../src/reference/alphafold-fixture.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";
import { AlphaFoldMonomerGpu } from "../../src/model/monomer.js";
import { distogramLogits } from "../../src/heads/distogram.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const AA3 = {
  ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E", GLY: "G",
  HIS: "H", ILE: "I", LEU: "L", LYS: "K", MET: "M", PHE: "F", PRO: "P", SER: "S",
  THR: "T", TRP: "W", TYR: "Y", VAL: "V",
};

function parseNative(text) {
  const sequence = [];
  const coords = [];
  let chain;
  for (const line of text.split("\n")) {
    if (!line.startsWith("ATOM") || line.slice(12, 16).trim() !== "CA") continue;
    const alt = line[16];
    if (alt !== " " && alt !== "A") continue;
    const code = AA3[line.slice(17, 20)];
    if (code === undefined) continue;
    if (chain === undefined) chain = line[21];
    if (line[21] !== chain) break;
    sequence.push(code);
    coords.push([Number(line.slice(30, 38)), Number(line.slice(38, 46)),
      Number(line.slice(46, 54))]);
  }
  return { sequence: sequence.join(""), coords };
}

const THRESHOLDS = [0.5, 1, 2, 4];

/** lDDT against the native. Distance-based, so no superposition anywhere. */
function trueLddt(model, native, radius = 15) {
  const n = native.length;
  const out = new Float64Array(n);
  const gap = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  for (let i = 0; i < n; i += 1) {
    let preserved = 0;
    let included = 0;
    for (let j = 0; j < n; j += 1) {
      if (i === j) continue;
      const reference = gap(native[i], native[j]);
      if (reference >= radius) continue;
      included += 1;
      const error = Math.abs(gap(model[i], model[j]) - reference);
      for (const threshold of THRESHOLDS) if (error < threshold) preserved += 1;
    }
    out[i] = included === 0 ? 0 : (100 * preserved) / (THRESHOLDS.length * included);
  }
  return out;
}

/**
 * The distogram as bytes: softmax, upper triangle, one byte a bin.
 *
 * 🔴 PROBABILITIES, NOT LOGITS. Logits are unbounded and their scale is
 * arbitrary; every feature is a sum over probabilities, and those quantise
 * into a byte with no scale factor to carry. Upper triangle only, because the
 * head is symmetrised by construction.
 */
function packDistogram(logits, length, bins) {
  const bytes = new Uint8Array((length * (length + 1) / 2) * bins);
  const probability = new Float64Array(bins);
  let out = 0;
  for (let i = 0; i < length; i += 1) {
    for (let j = i; j < length; j += 1) {
      const base = (i * length + j) * bins;
      let largest = -Infinity;
      for (let b = 0; b < bins; b += 1) {
        if (logits[base + b] > largest) largest = logits[base + b];
      }
      let total = 0;
      for (let b = 0; b < bins; b += 1) {
        probability[b] = Math.exp(logits[base + b] - largest);
        total += probability[b];
      }
      for (let b = 0; b < bins; b += 1) {
        bytes[out + b] = Math.round((probability[b] / total) * 255);
      }
      out += bins;
    }
  }
  let binary = "";
  const step = 0x8000;
  for (let at = 0; at < bytes.length; at += step) {
    binary += String.fromCharCode(...bytes.subarray(at, at + step));
  }
  return btoa(binary);
}

export async function main(device, args) {
  const recycles = Number(option(args, "recycles", "3"));
  const maxLength = Number(option(args, "max-length", "1000"));
  let names = await (await fetch("/natives/index.json")).json();
  const sample = Number(option(args, "sample", "0"));
  if (sample > 0) {
    let state = Number(option(args, "sample-seed", "1")) >>> 0;
    const random = () => {
      state = (state * 1664525 + 1013904223) >>> 0;
      return state / 4294967296;
    };
    const shuffled = [...names];
    for (let i = shuffled.length - 1; i > 0; i -= 1) {
      const j = Math.floor(random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    names = shuffled.slice(0, sample);
  } else {
    names = names.slice(Number(option(args, "from", "0")),
      Number(option(args, "to", "1000")));
  }

  // ...the LOCAL bundle by directory, as tools/gpu/fold-af2.js does: loadModel
  // would resolve the monomer family to its remote and pull 227 MB.
  const { MODEL_BUNDLES, loadManifest } = await import("../../src/reference/manifests/index.js");
  const store = await HttpTensorStore.fromManifest(
    MODEL_BUNDLES.monomer.directory, await loadManifest("monomer"));
  const fixture = AlphaFoldFixture.fromStore(store);
  const [embedding, template, extraStack, mainStack, structure, confidence,
         geometry, featureTables, paeBreaks, distogram] = await Promise.all([
    fixture.embeddingWeights(), fixture.templateWeights(), fixture.extraStackWeights(),
    fixture.mainStackWeights(), fixture.structureWeights(), fixture.confidenceWeights(),
    fixture.geometryTables(), fixture.queryOnlyFeatureTables(),
    fixture.tensor("confidencePaeBreaks"), fixture.distogramHeadWeights(),
  ]);
  if (distogram === undefined) {
    return { ok: false, error: "this bundle has no distogram head" };
  }
  const weights = { embedding, template, extraStack, mainStack, structure,
    lddt: confidence.lddt, pae: confidence.pae, geometry };

  const columns = ["target", "recycle", "residue", "plddt", "true"];
  const rows = [];
  const targets = {};
  const failures = [];
  for (const name of names) {
    try {
      const native = parseNative(await (await fetch(`/natives/${name}.pdb`)).text());
      const length = native.sequence.length;
      if (length < 30 || length > maxLength) continue;
      const passes = [];
      await new AlphaFoldMonomerGpu(device).predictA3m(
        `>query\n${native.sequence}\n`, weights, featureTables,
        { recycles, randomSeed: 0, maxMsaSequences: 1, maxExtraSequences: 1,
          chainLengths: [length] },
        paeBreaks,
        // 🔴 EVERY RECYCLE, WHICH IS THE WHOLE POINT. Each one carries its own
        // pair representation, its own structure and its own pLDDT.
        (recycle) => { passes.push(recycle); },
      );

      targets[name] = { sequence: native.sequence, bins: distogram.bins,
        firstBreak: distogram.firstBreak, lastBreak: distogram.lastBreak,
        native: native.coords.map((p) => p.map((v) => Number(v.toFixed(2)))),
        models: {}, distograms: {} };

      for (const [index, pass] of passes.entries()) {
        const logits = distogramLogits(pass.pair, distogram.halfLogitsWeights,
          distogram.halfLogitsBias, length, { channels: 128, bins: distogram.bins });
        targets[name].distograms[index] = packDistogram(logits, length, distogram.bins);
        const model = [];
        for (let residue = 0; residue < length; residue += 1) {
          const at = (residue * 37 + 1) * 3;
          model.push([pass.structure.atom37[at], pass.structure.atom37[at + 1],
            pass.structure.atom37[at + 2]]);
        }
        targets[name].models[index] = model.map((p) => p.map((v) => Number(v.toFixed(2))));
        const truth = trueLddt(model, native.coords);
        for (let residue = 0; residue < length; residue += 1) {
          rows.push([name, index, residue,
            Number(pass.confidence.plddt[residue].toFixed(2)),
            Number(truth[residue].toFixed(2))]);
        }
      }
    } catch (cause) {
      failures.push({ name, error: String(cause?.message ?? cause) });
    }
  }
  return { ok: true, columns, targets: Object.keys(targets).length, rows, data: targets,
    failures };
}
