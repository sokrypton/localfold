/**
 * Does moving the flow's starting sigma cost accuracy on real proteins?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-sigma0.js \
 *       --sigmas=2560,160 --steps=8 --seeds=1,2,3,4
 *
 * probe-ligand-flow.js found that a ligand's bonds resolve far sooner if the
 * walk starts low - sigma0 = 16 reaches in eight steps what AF3's own 2560
 * needs sixteen for - and that the same setting collapses a protein. sigma0 =
 * 160 looked like the setting that is safe for both, but "safe" there meant one
 * protein scored on pLDDT and CA-CA, which is not what this repo changes a
 * sampler on. This scores against the crystal structures instead.
 *
 * 🔴 pLDDT IS NOT AN ACCURACY MEASURE AND MUST NOT DECIDE THIS. It is the
 * model's own opinion, produced by a head that never saw the crystal; a
 * schedule change that made predictions more confident and less correct would
 * look like an improvement in it. RMSD and TM against the deposited structure
 * are the numbers docs/AF3.md records, so they are the numbers a default moves on.
 *
 * WHAT IT FOUND, at flow 8 over four seeds, mean RMSD (range) and TM, AFTER the
 * diffusion head's weight-name fix (see docs/AF3.md):
 *
 *     sigma0 | 6MRR (68 res)              | 1QYS (92 res)
 *       2560 | 0.701 [0.65-0.76]  TM .950 | 0.841 [0.82-0.87]  TM .951
 *        640 | 0.701 [0.66-0.78]  TM .951 | 0.851 [0.84-0.87]  TM .950
 *        160 | 0.708 [0.67-0.76]  TM .950 | 0.866 [0.85-0.88]  TM .948
 *         16 | 0.740 [0.66-0.83]  TM .949 | 1.101 [0.81-1.70]  TM .919
 *
 * 🔴 THE DEFAULT STILL DID NOT MOVE, AND THE PENALTY SHRANK. Against the wrong
 * weights sigma0 = 160 cost 1QYS 0.043 A with seed ranges that did not overlap
 * the default's; against the right ones it costs 0.025 A and they do overlap.
 * It is still monotonic and still a real regression paid by every protein fold
 * to help a ligand - so the flow keeps AF3's own top of schedule, and a caller
 * who wants a lower one passes `schedule` explicitly. sigma0 = 16, which is
 * where a lone ligand's bonds resolve fastest, is where a protein comes apart:
 * 0.26 A worse on average and one seed at 1.70.
 *
 * 🔴 AND EVERY NUMBER HERE MOVED WHEN A WEIGHT NAME CHANGED. The pre-fix table
 * ranked the same three starts in the same order at almost the same values,
 * which is exactly why a sampler sweep is no evidence that the model under it
 * is right. Run tools/gpu/probe-head-vs-af3-steps.js first.
 *
 * 🔴 AND THE COMPARISON IS CA-ONLY, BY RESIDUE INDEX. Both benchmark chains are
 * single-chain and complete, so the correspondence is the identity - no
 * alignment is needed and none is done, which keeps this from quietly scoring a
 * shifted register as a good fit.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { atomName, foldBatch } from "../../src/af3/fold.js";
import { confidenceWeights, openAf3Store, trunkWeights } from "../../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** sigmaMax is in units of sigmaData, so the walk starts at sigmaData*sigmaMax. */
const SIGMA_DATA = 16;

const ONE_LETTER = {
  ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E", GLY: "G",
  HIS: "H", ILE: "I", LEU: "L", LYS: "K", MET: "M", PHE: "F", PRO: "P", SER: "S",
  THR: "T", TRP: "W", TYR: "Y", VAL: "V", MSE: "M",
};

/**
 * The CA trace AND the sequence, both read from the deposited file.
 *
 * 🔴 THE SEQUENCE IS NOT TYPED IN. Folding a sequence that is not the one the
 * crystal contains scores a different protein against it, and the failure looks
 * like a mediocre RMSD rather than like a mistake - so both come from the same
 * ATOM records, and the residue correspondence is exact by construction.
 * Residues the model cannot fold (anything outside the 20) end the trace rather
 * than shifting it.
 */
function crystalChain(text) {
  const seen = new Map();
  for (const line of text.split("\n")) {
    if (line.startsWith("ENDMDL")) break;                 // the first model only
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    if (line.slice(12, 16).trim() !== "CA") continue;
    const altLoc = line[16];
    if (altLoc !== " " && altLoc !== "A") continue;        // one conformer
    const code = ONE_LETTER[line.slice(17, 20).trim()];
    if (code === undefined) continue;
    const key = `${line[21]}|${line.slice(22, 27).trim()}`;
    if (seen.has(key)) continue;
    seen.set(key, { code, chain: line[21], xyz: [
      Number(line.slice(30, 38)), Number(line.slice(38, 46)), Number(line.slice(46, 54)),
    ] });
  }
  // One chain only: the first that appears, so a crystal with two copies in the
  // asymmetric unit is not scored as a complex we did not fold.
  const first = [...seen.values()][0]?.chain;
  const residues = [...seen.values()].filter((r) => r.chain === first);
  return {
    sequence: residues.map((r) => r.code).join(""),
    alphaCarbons: residues.map((r) => r.xyz),
  };
}

/** Our own CA coordinates, in residue order. */
function predictedAlphaCarbons(batch, positions) {
  const { dense } = batch;
  const out = [];
  for (let token = 0; token < batch.tokens; token += 1) {
    for (let atom = 0; atom < dense; atom += 1) {
      const slot = token * dense + atom;
      if (!batch.predDenseAtomMask[slot]) continue;
      if (atomName(batch.refAtomNameChars, slot) !== "CA") continue;
      out.push([positions[slot * 3], positions[slot * 3 + 1], positions[slot * 3 + 2]]);
    }
  }
  return out;
}

/** Kabsch superposition, returning RMSD and TM-score over the common prefix. */
export function rmsdAndTm(a, b) {
  const n = Math.min(a.length, b.length);
  if (n < 3) return { rmsd: NaN, tm: NaN, residues: n };
  const centre = (points) => {
    const c = [0, 0, 0];
    for (let i = 0; i < n; i += 1) for (let k = 0; k < 3; k += 1) c[k] += points[i][k] / n;
    return points.slice(0, n).map((p) => [p[0] - c[0], p[1] - c[1], p[2] - c[2]]);
  };
  const x = centre(a);
  const y = centre(b);
  // Covariance, then a rotation from its SVD - done here by Jacobi on the 4x4
  // quaternion form, which needs no linear algebra dependency.
  const m = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < n; i += 1) {
    for (let r = 0; r < 3; r += 1) for (let c = 0; c < 3; c += 1) m[r][c] += x[i][r] * y[i][c];
  }
  const k = [
    [m[0][0] + m[1][1] + m[2][2], m[1][2] - m[2][1], m[2][0] - m[0][2], m[0][1] - m[1][0]],
    [m[1][2] - m[2][1], m[0][0] - m[1][1] - m[2][2], m[0][1] + m[1][0], m[2][0] + m[0][2]],
    [m[2][0] - m[0][2], m[0][1] + m[1][0], -m[0][0] + m[1][1] - m[2][2], m[1][2] + m[2][1]],
    [m[0][1] - m[1][0], m[2][0] + m[0][2], m[1][2] + m[2][1], -m[0][0] - m[1][1] + m[2][2]],
  ];
  // Power iteration for the dominant eigenvector: the optimal rotation quaternion.
  let q = [1, 0.1, 0.1, 0.1];
  for (let step = 0; step < 500; step += 1) {
    const next = [0, 0, 0, 0];
    for (let r = 0; r < 4; r += 1) for (let c = 0; c < 4; c += 1) next[r] += k[r][c] * q[c];
    const shift = 3 * Math.max(...k.map((row) => Math.max(...row.map(Math.abs))));
    for (let r = 0; r < 4; r += 1) next[r] += shift * q[r];
    const norm = Math.hypot(...next);
    q = next.map((v) => v / norm);
  }
  const [w, i0, j0, k0] = q;
  const rot = [
    [w * w + i0 * i0 - j0 * j0 - k0 * k0, 2 * (i0 * j0 - w * k0), 2 * (i0 * k0 + w * j0)],
    [2 * (i0 * j0 + w * k0), w * w - i0 * i0 + j0 * j0 - k0 * k0, 2 * (j0 * k0 - w * i0)],
    [2 * (i0 * k0 - w * j0), 2 * (j0 * k0 + w * i0), w * w - i0 * i0 - j0 * j0 + k0 * k0],
  ];
  const d0 = 1.24 * Math.cbrt(Math.max(n, 19) - 15) - 1.8;
  let square = 0;
  let tm = 0;
  for (let index = 0; index < n; index += 1) {
    const p = [0, 0, 0];
    for (let r = 0; r < 3; r += 1) {
      for (let c = 0; c < 3; c += 1) p[r] += rot[r][c] * x[index][c];
    }
    const d2 = (p[0] - y[index][0]) ** 2 + (p[1] - y[index][1]) ** 2 + (p[2] - y[index][2]) ** 2;
    square += d2;
    tm += 1 / (1 + d2 / (d0 * d0));
  }
  return { rmsd: Math.sqrt(square / n), tm: tm / n, residues: n };
}

export async function main(device, args) {
  const sigmas = option(args, "sigmas", "2560,160").split(",").map(Number);
  const steps = Number(option(args, "steps", "8"));
  const seeds = option(args, "seeds", "1,2,3,4").split(",").map(Number);
  const only = option(args, "targets", "6MRR,1QYS").split(",");

  const store = await openAf3Store(option(args, "model", "/model-af3-full-f32/manifest.json"));
  const weights = {
    trunk: await trunkWeights(store), diffusion: await diffusionWeights(store),
    confidence: await confidenceWeights(store), atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
  };

  const rows = [];
  for (const name of only) {
    const response = await fetch(`https://files.rcsb.org/download/${name}.pdb`);
    if (!response.ok) throw new Error(`could not fetch ${name}: ${response.status}`);
    const { sequence, alphaCarbons: crystal } = crystalChain(await response.text());
    const batch = featuriseProtein(sequence, {});
    for (const sigma0 of sigmas) {
      for (const seed of seeds) {
        const result = await foldBatch(device, batch, weights, {
          mode: "flow", steps, recycles: 0, seed,
          schedule: { sigmaMax: sigma0 / SIGMA_DATA },
        });
        const scored = rmsdAndTm(predictedAlphaCarbons(batch, result.positions), crystal);
        rows.push({
          target: name, sigma0, seed,
          rmsd: Number(scored.rmsd.toFixed(3)), tm: Number(scored.tm.toFixed(3)),
          residues: scored.residues,
          caca: Number(result.geometry.caca.toFixed(2)),
          plddt: Number(result.meanPlddt.toFixed(1)),
        });
      }
    }
  }
  return rows;
}
