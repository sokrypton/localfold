/**
 * rosettafold3's chirality gradient in WebGL2 (`chiralGradientSource`, webgl/af3/diffusion/atom-encoder.js) against
 * its CPU twin, `chiralPositionGradients` in shared/af3/diffusion/chiral-gradient.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-chiral.js [--tokens=40] [--falsify]
 *
 * The centres are the featuriser's own (`chiralCentres` over every residue type, all 24 dense slots present), the
 * positions a random cloud, and the index the fold's (`chiralIndex`), so this holds the kernel AND the inversion. The
 * gradient is a float32 central difference of a squared angle error over a 1e-4 step, so the bar is a relative one
 * against the gradient's scale, not an absolute one.
 */
import { chiralCentres } from "../../shared/af3/featurise/template-features.js";
import { chiralPositionGradients } from "../../shared/af3/diffusion/chiral-gradient.js";
import { chiralGradientSource } from "../../webgl/af3/diffusion/atom-encoder.js";
import { chiralIndex } from "../../webgl/af3/model/fold.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(_device, args) {
  const tokens = Number(option(args, "tokens", "40"));
  const bar = Number(option(args, "bar", "2e-3"));
  const dense = 24;
  const atoms = tokens * dense;
  let seed = 5;
  const random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };
  const aatype = Int32Array.from({ length: tokens }, (_, t) => t % 20);
  const mask = new Float32Array(atoms).fill(1);
  const chirals = chiralCentres(aatype, mask, tokens, dense);
  // a residue-sized cloud per token, at the SCALE the encoder sees: the head divides the noisy coordinates by
  // sqrt(sigma^2 + 16) before this kernel (head.js `scale.in`), so a 3.8 A step is ~0.24 here. 🔴 At crystal scale (a
  // 40-token chain spans 150 A) float32 cannot hold a 1e-4 nudge - its spacing there is 1.5e-5 - and the comparison
  // read 1.3e-2 for the arithmetic, not the kernel.
  const unit = Number(option(args, "unit", String(1 / 16)));
  const positions = Float32Array.from({ length: atoms * 3 },
    (_, i) => unit * (3 * random() + Math.floor(i / (dense * 3)) * 3.8));
  const expected = chiralPositionGradients(positions, chirals.centers, chirals.angles, atoms);

  const execution = new Execution(createWebGL2Context());
  const index = chiralIndex(execution, chirals, atoms);
  if (args?.includes("--falsify")) index.angles = execution.upload(new Float32Array(chirals.count).fill(1), "falsified");
  const positionsTensor = execution.upload(positions, "positions");
  const out = execution.allocate(atoms * 4, "chiral");
  execution.dispatch({
    source: chiralGradientSource(atoms), output: out,
    inputs: { positionsTex: positionsTensor, centersTex: index.centers, anglesTex: index.angles,
      offsetsTex: index.offsets, entriesTex: index.entries },
    label: "check.chiral",
  });
  const got = Float32Array.from(execution.read(out));
  let worst = 0;
  let scale = 0;
  let squared = 0;
  let reference = 0;
  for (let atom = 0; atom < atoms; atom += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      const e = expected[atom * 3 + axis];
      const d = got[atom * 4 + axis] - e;
      worst = Math.max(worst, Math.abs(d));
      scale = Math.max(scale, Math.abs(e));
      squared += d * d;
      reference += e * e;
    }
  }
  const relativeRms = Math.sqrt(squared / reference);
  return {
    tokens, centres: chirals.count, atomsTouched: new Set(chirals.centers).size,
    relativeRms: Number(relativeRms.toPrecision(4)), worstOverScale: Number((worst / scale).toPrecision(4)),
    verdict: relativeRms <= bar ? `ok: within ${bar}` : `FAILED: relRMS ${relativeRms.toPrecision(4)} over ${bar}`,
  };
}
