/**
 * The outer product mean's two paths, on ONE input, compared against each other.
 *
 * 🔴 THE 64 MiB CAP MEANT THE FAST PATH NEVER RAN PAST 128 RESIDUES, so the two
 * were never compared at a size where the choice matters. They contract the
 * sequence axis in a different order, so they are numerically different by
 * construction and exact equality is the wrong bar - but a REORDERING is ~1e-6,
 * and anything larger is a different computation.
 *
 * 🔴 AND THE FAST PATH IS NOW PAIR-BLOCKED, so there is a third arm: the same
 * path with a cap small enough to force MANY blocks. Blocking changes no sum's
 * ORDER - a pair's contraction is untouched, only where it lands in the
 * intermediate - so that arm has to be **bit-exact** against the one-block one,
 * and a bar of 1e-6 there would pass an off-by-a-block index. It is the arm
 * that matters for a phone, where the cap is 128 MiB and every real protein
 * takes several blocks.
 */
import { OuterProductMeanGpu } from "../../src/evoformer/outer-product-mean.js";
import { relativeRms } from "./relative-rms.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const length = Number(option(args, "length", "200"));
  const sequences = Number(option(args, "sequences", "64"));
  const cM = Number(option(args, "cm", "32"));
  const cOuter = Number(option(args, "outer", "32"));
  const cZ = Number(option(args, "cz", "64"));

  let state = 7;
  const noise = (n, scale = 1) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      out[i] = ((state / 4294967296) * 2 - 1) * scale;
    }
    return out;
  };
  const input = {
    activations: noise(sequences * length * cM),
    mask: new Float32Array(sequences * length).fill(1),
    sequences, length, cM, cOuter, cZ,
    weights: {
      layerNormScale: noise(cM, 0.1).map((v) => v + 1), layerNormOffset: noise(cM, 0.1),
      leftWeight: noise(cM * cOuter, 0.2), leftBias: noise(cOuter, 0.1),
      rightWeight: noise(cM * cOuter, 0.2), rightBias: noise(cOuter, 0.1),
      outputWeight: noise(cOuter * cOuter * cZ, 0.05), outputBias: noise(cZ, 0.1),
    },
  };
  const bytes = length * length * cOuter * cOuter * 4;
  const runner = new OuterProductMeanGpu(device);
  // 0 forces the tiled fallback; a huge cap forces outer-first.
  const tiled = await runner.run({ ...input, outerFirstLimitBytes: 0 });
  const outerFirst = await runner.run({ ...input, outerFirstLimitBytes: bytes });
  // ...and the same path in blocks of 8 pairs, which at any real length is
  // thousands of them. Same arithmetic, different buffer offsets.
  const blocked = await runner.run({ ...input, outerFirstLimitBytes: 8 * cOuter * cOuter * 4 });
  // 🔴 AND THE HALF-PRECISION CONTRACTION, AT WHATEVER DEPTH THE CALLER ASKED
  // FOR, because depth is the ONLY axis its risk lives on. Its staged tile and
  // its chunk accumulator are f16 and its running total is f32, so the error
  // should be flat in `sequences` - and the failure mode it is guarding against
  // (alphafold2-webgpu's whole-contraction f16 accumulator, 96.80 pLDDT to
  // 69.94 at 508 rows) is one that only appears deep. Run this at --sequences=8
  // and it proves nothing.
  const half = await runner.run({ ...input, outerFirstLimitBytes: bytes, contractPrecision: "f16" });
  const rel = relativeRms(outerFirst.output, tiled.output);
  const relBlocked = relativeRms(blocked.output, outerFirst.output);
  const relHalf = relativeRms(half.output, outerFirst.output);
  const finite = half.output.every((v) => Number.isFinite(v));
  const blocks = Math.ceil((length * length) / 8);
  return {
    length, sequences, cOuter, cZ, intermediateMiB: Number((bytes / 1048576).toFixed(1)),
    tiledMs: Number(tiled.elapsedMilliseconds.toFixed(1)),
    outerFirstMs: Number(outerFirst.elapsedMilliseconds.toFixed(1)),
    blockedMs: Number(blocked.elapsedMilliseconds.toFixed(1)),
    blocks,
    speedup: Number((tiled.elapsedMilliseconds / outerFirst.elapsedMilliseconds).toFixed(2)),
    relRms: Number(rel.toPrecision(3)),
    relRmsBlocked: relBlocked,
    halfMs: Number(half.elapsedMilliseconds.toFixed(1)),
    relRmsHalf: Number(relHalf.toPrecision(3)),
    halfFinite: finite,
    // A reordering of the same sum; not exact, and not different either.
    // The half-precision bar is the block's own dense kernels, which
    // docs/A100.md prices at 1.7e-3 to 2.4e-3 - this must not be the largest
    // single contribution.
    ok: rel < 1e-4 && relBlocked === 0 && finite && relHalf < 2e-3,
  };
}
