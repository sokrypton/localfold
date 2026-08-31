/**
 * The diffusion sampler's arithmetic and its rigid augmentation.
 *
 * tools/oracle/check_af3_sampler.js checks the schedule and the step update
 * against AF3 itself. What it cannot check is the AUGMENTATION, because that is
 * a random rigid motion and AF3's draw comes from a PRNG this does not
 * implement - so its defining property is tested here instead: a rigid motion
 * moves atoms without changing any distance between them.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  noiseLevels, noiseSchedule, randomAugmentation, randomRotation, sample, samplerStep,
} from "../src/af3/diffusion-sampler-reference.js";

/** A deterministic gaussian, so a failure is reproducible. */
function gaussians(seed = 1) {
  let state = seed >>> 0;
  const uniform = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return (state + 1) / 4294967297;
  };
  return () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
}

describe("AF3 noise schedule", () => {
  it("starts at sigmaData * sigmaMax and ends at sigmaData * sigmaMin", () => {
    assert.ok(Math.abs(noiseSchedule(0) - 16 * 160) < 1e-6, `${noiseSchedule(0)}`);
    assert.ok(Math.abs(noiseSchedule(1) - 16 * 0.0004) < 1e-9, `${noiseSchedule(1)}`);
  });

  it("descends monotonically, and steeply", () => {
    const levels = noiseLevels(20);
    for (let index = 1; index < levels.length; index += 1) {
      assert.ok(levels[index] < levels[index - 1], `level ${index} did not descend`);
    }
    // 🔴 rho = 7 MAKES THIS FAR FROM LINEAR. Half way down the schedule the
    // noise is 56 A, not the 1280 a linear interpolation would give, so most of
    // the steps are spent at low noise where the structure actually forms.
    assert.ok(levels[10] > 50 && levels[10] < 60, `midpoint was ${levels[10]}`);
  });
});

describe("AF3 sampler step", () => {
  it("scales the gradient by 1.5, not 1", () => {
    // One atom, one axis: noisy at 10, denoised at 0, stepping 100 -> 50.
    const out = samplerStep(Float32Array.from([10, 0, 0]),
                            Float32Array.from([0, 0, 0]), 100, 50);
    // gradient = (10 - 0) / 100 = 0.1; d_t = -50; 10 + 1.5 * -50 * 0.1 = 2.5
    assert.ok(Math.abs(out[0] - 2.5) < 1e-5, `${out[0]}`);
  });

  it("leaves an already-denoised atom where it is", () => {
    // ...WHERE IT IS, not at the origin: a zero gradient means the step moves
    // nothing, so the output is the input rather than nothing.
    const input = Float32Array.from([7, -3, 2]);
    const out = samplerStep(input, Float32Array.from([7, -3, 2]), 100, 50);
    for (let index = 0; index < out.length; index += 1) {
      assert.ok(Math.abs(out[index] - input[index]) < 1e-6, `[${index}] = ${out[index]}`);
    }
  });
});

describe("AF3 random augmentation", () => {
  it("builds an orthonormal rotation", () => {
    const rotation = randomRotation(gaussians(7));
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    for (let i = 0; i < 3; i += 1) {
      assert.ok(Math.abs(dot(rotation[i], rotation[i]) - 1) < 1e-6, `row ${i} not unit`);
      for (let j = i + 1; j < 3; j += 1) {
        assert.ok(Math.abs(dot(rotation[i], rotation[j])) < 1e-6, `${i},${j} not orthogonal`);
      }
    }
    // ...and right-handed, since e2 is the cross product of the other two.
    const determinant =
      rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
      - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
      + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]);
    assert.ok(Math.abs(determinant - 1) < 1e-6, `determinant ${determinant}`);
  });

  it("preserves every distance, which is what makes it an augmentation", () => {
    const atoms = 5;
    const positions = new Float32Array(atoms * 3);
    const draw = gaussians(3);
    for (let index = 0; index < positions.length; index += 1) positions[index] = draw() * 10;
    const mask = Float32Array.from([1, 1, 1, 1, 1]);
    const moved = randomAugmentation(positions, mask, atoms, gaussians(11));
    const distance = (p, a, b) => Math.hypot(p[a * 3] - p[b * 3],
                                             p[a * 3 + 1] - p[b * 3 + 1],
                                             p[a * 3 + 2] - p[b * 3 + 2]);
    for (let a = 0; a < atoms; a += 1) {
      for (let b = a + 1; b < atoms; b += 1) {
        assert.ok(Math.abs(distance(positions, a, b) - distance(moved, a, b)) < 1e-3,
                  `distance ${a}-${b} changed`);
      }
    }
  });

  it("centres on the REAL atoms, ignoring padded slots", () => {
    // 🔴 PADDED SLOTS HOLD ZEROS AND MUST NOT VOTE. Two real atoms at x = 100
    // and 102 centre at 101; averaging in two padded zeros would centre at
    // 50.5 and translate the structure 50 A off. With the rotation and
    // translation suppressed, the centred coordinates are +-1.
    const positions = Float32Array.from([100, 0, 0, 102, 0, 0, 0, 0, 0, 0, 0, 0]);
    const mask = Float32Array.from([1, 1, 0, 0]);
    let call = 0;
    // six draws build the rotation, three the translation: make it the identity
    // with no shift.
    const scripted = () => {
      const values = [1, 0, 0, 0, 1, 0, 0, 0, 0];
      return values[call++] ?? 0;
    };
    const moved = randomAugmentation(positions, mask, 4, scripted);
    assert.ok(Math.abs(moved[0] + 1) < 1e-4, `first atom at ${moved[0]}`);
    assert.ok(Math.abs(moved[3] - 1) < 1e-4, `second atom at ${moved[3]}`);
    // ...and padded slots stay exactly zero.
    for (let index = 6; index < 12; index += 1) assert.equal(moved[index], 0);
  });
});

describe("AF3 sampler trajectory", () => {
  const atoms = 4;
  const target = Float32Array.from([0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3]);
  const mask = Float32Array.from([1, 1, 1, 1]);
  // A denoiser that always names the same answer, so the trajectory's shape is
  // the sampler's doing and not the model's.
  const denoise = () => target;

  const collect = (steps) => {
    const frames = [];
    const final = sample(denoise, {
      atoms, mask, steps, normal: gaussians(5),
      onStep: (event) => frames.push(event),
    });
    return { frames, final };
  };

  it("reports one frame per step, numbered from one", () => {
    const { frames } = collect(12);
    assert.equal(frames.length, 12);
    assert.deepEqual(frames.map((frame) => frame.step),
                     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    for (const frame of frames) assert.equal(frame.steps, 12);
  });

  it("descends in noise level, so a frame index is a progress bar", () => {
    const { frames } = collect(12);
    for (let index = 1; index < frames.length; index += 1) {
      assert.ok(frames[index].noiseLevel < frames[index - 1].noiseLevel,
                `frame ${index} did not descend`);
    }
  });

  it("hands out copies, not views of the live buffer", () => {
    // 🔴 THE BUG THIS EXISTS FOR. Passing the sampler's own arrays would make
    // every frame alias the same memory: the animation would replay the FINAL
    // structure `steps` times and look like a still. Nothing about that reads
    // as a bug in the viewer.
    const { frames } = collect(6);
    for (let index = 1; index < frames.length; index += 1) {
      assert.notEqual(frames[index].positions, frames[index - 1].positions,
                      "consecutive frames share a buffer");
    }
    const early = frames[0].positions;
    const snapshot = Float32Array.from(early);
    collect(6);
    assert.deepEqual(early, snapshot, "an earlier frame was mutated later");
  });

  it("converges on the denoiser's answer, which is what makes it watchable", () => {
    // The last frame's guess is the target; the first is dominated by noise.
    const { frames, final } = collect(40);
    const spread = (positions) => {
      let total = 0;
      for (let index = 0; index < positions.length; index += 1) {
        const difference = positions[index] - target[index];
        total += difference * difference;
      }
      return Math.sqrt(total / positions.length);
    };
    assert.ok(spread(frames[0].positions) > spread(final),
              "the trajectory did not move toward the answer");
    // ...and `denoised` is the target exactly, every frame, since that is what
    // this denoiser returns. That is the frame an animation should show.
    for (const frame of frames) assert.deepEqual(frame.denoised, target);
  });

  it("runs without an onStep, which is the normal path", () => {
    assert.doesNotThrow(() => sample(denoise, { atoms, mask, steps: 5, normal: gaussians(2) }));
  });
});
