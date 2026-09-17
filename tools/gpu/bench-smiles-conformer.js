/**
 * Where does refining a conformer on the device start paying? A batch, or never.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-smiles-conformer.js
 *
 * 🔴 FOR ONE SMALL LIGAND THE DEVICE LOSES, AND IT LOSES FOR A REASON THE
 * KERNEL CANNOT FIX. WGSL will not let a barrier sit inside control flow that
 * depends on a workgroup reduction, so the device cannot stop early: it runs
 * the full step budget and the full line search every time, where the host
 * breaks out the moment the bounds are satisfied or the search stalls. Biotin
 * is 16 ms on the host and 49 on the device for the same answer. That is not
 * overhead to be tuned away - it is strictly more arithmetic.
 *
 * What the device has instead is the batch axis. One conformer is one
 * workgroup, a dispatch carries as many as are asked for, and distance
 * geometry wants several random starts per molecule anyway. So the question
 * this answers is not "is the GPU faster" but "how many molecules before it
 * is", which is what a screen cares about and a single fold does not.
 */
import { parseSmiles } from "../../src/chem/smiles.js";
import { chiralCentres } from "../../src/chem/stereo.js";
import {
  distanceBounds, smoothBounds, embedBounds, refineCoordinates, planarQuadruples,
} from "../../src/chem/conformer.js";
import { refineOnDevice } from "../../src/chem/conformer-webgpu.js";
import { hashOf } from "../../src/chem/component.js";

const LIGAND = "CC(=O)Oc1ccccc1C(=O)O";           // aspirin: 13 atoms, drug-sized

export async function main(device, args) {
  const option = (name, fallback) => {
    const found = args.find((one) => one.startsWith(`--${name}=`));
    return found === undefined ? fallback : found.slice(name.length + 3);
  };
  const smiles = option("smiles", LIGAND);
  const counts = option("counts", "1,4,16,64,256").split(",").map(Number);

  const graph = parseSmiles(smiles);
  const bounds = distanceBounds(graph);
  smoothBounds(bounds);
  const centres = chiralCentres(graph);
  const planar = planarQuadruples(graph, bounds.rings);
  const seed = hashOf(smiles);

  const rows = [];
  for (const count of counts) {
    const starts = [];
    for (let index = 0; index < count; index += 1) {
      starts.push(embedBounds(bounds, seed + index * 7919));
    }

    // 🔴 BOTH ARMS ARE RUN TWICE AND THE SECOND IS READ. The first device call
    // compiles the pipeline, which is tens of milliseconds and is a one-time
    // cost a screen pays once; charging it to every row would make the batch
    // curve look flat when it is not.
    await refineOnDevice(device, bounds, starts.slice(0, 1), { planar, chiral: centres });

    const hostStart = performance.now();
    for (const start of starts) refineCoordinates(start, bounds, centres, { planar });
    const hostMs = performance.now() - hostStart;

    const deviceStart = performance.now();
    await refineOnDevice(device, bounds, starts, { planar, chiral: centres });
    const deviceMs = performance.now() - deviceStart;

    rows.push({
      conformers: count,
      hostMs: Number(hostMs.toFixed(1)),
      deviceMs: Number(deviceMs.toFixed(1)),
      speedup: Number((hostMs / deviceMs).toFixed(2)),
      hostPerEach: Number((hostMs / count).toFixed(2)),
      devicePerEach: Number((deviceMs / count).toFixed(2)),
    });
  }

  const crossover = rows.find((row) => row.speedup >= 1);
  return {
    smiles,
    atoms: graph.atoms.length,
    rows,
    crossover: crossover === undefined ? null : crossover.conformers,
    note: crossover === undefined
      ? "the host wins at every batch size measured"
      : `the device wins from ${crossover.conformers} conformers a dispatch`,
  };
}
