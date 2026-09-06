// Does src/esmfold2/diffusion-webgpu.js compute ESMFold2's whole denoise step?
//
//     .venv-esm/bin/python tools/esmc/dump-esmfold2-trunk.py \
//         --sequence-length 40 --esmc esmc-600m --out oracle-dumps/esmfold2-trunk-40-lm.json
//     python3 tools/export_esmfold2_trunk.py
//     node tools/gpu-chrome.mjs tools/gpu/check-esmfold2-diffusion-gpu.js
//
// 🔴 IT TAKES THE MODULE'S OWN ARGUMENTS, so this is `f(x) == y` for the
// denoiser alone: `z_trunk`, `rel_pos`, `s_inputs`, `x_noisy` and `t_hat` are
// all recorded, and nothing upstream can break it or be blamed for it. The CPU
// reference passes the same comparison at 1.51e-4 (tools/check-esmfold2-diffusion.js)
// and this runs the SAME arithmetic through twelve token blocks, two atom
// stacks and about three hundred dispatches.
//
// 🔴 AND THE BOUND IS THE ATOM ATTENTION'S bfloat16, NOT A TOLERANCE CHOSEN TO
// PASS. `SWA3DRoPEAttention.forward` downcasts q, k and v unconditionally, so
// six blocks of it put a floor at about 2e-4 whatever the port does; the rope
// TABLE is bfloat16 in the checkpoint too. `--precision=f32` turns the
// downcast off, which should make the answer WORSE against this dump - that is
// the control, and it is what says the narrowing is the model rather than a
// concession.
import { readTensor } from "../../src/reference/dtype.js";
import { GpuBufferAllocator } from "../../src/runtime/allocator.js";
import { pipelineCacheForDevice } from "../../src/runtime/pipeline-cache.js";
import { Esmfold2DenoiserGpu } from "../../src/esmfold2/diffusion-webgpu.js";
import { atomDecoderWeights, atomEncoderWeights, denoiserWeights }
  from "../../src/esmfold2/weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const relative = (got, want) => {
  let error = 0, total = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i] - want[i];
    error += d * d;
    total += want[i] * want[i];
  }
  return Math.sqrt(error / total);
};

export async function main(device, args = []) {
  const bundle = option(args, "bundle", "/model-esmfold2-trunk-f32");
  const dumpPath = option(args, "dump", "/oracle-dumps/esmfold2-trunk-40-lm.json");
  const bound = Number(option(args, "bound", "1.5e-4"));
  const precisions = option(args, "precision", "bf16,f32").split(",");

  const dump = await (await fetch(dumpPath)).json();
  const manifest = await (await fetch(`${bundle}/manifest.json`)).json();
  if (manifest.trunk?.source !== dump.esmfold2) {
    throw new Error(`bundle is ${manifest.trunk?.source} and the dump is ${dump.esmfold2}`);
  }
  if (dump.denoiser == null) {
    throw new Error(`${dumpPath} has no denoiser record; re-dump with a build that hooks it`);
  }

  const shards = new Map();
  const read = async (name) => {
    const record = manifest.tensors[name];
    if (record === undefined) throw new Error(`bundle has no tensor ${name}`);
    if (!shards.has(record.file)) {
      shards.set(record.file, await (await fetch(`${bundle}/${record.file}`)).arrayBuffer());
    }
    return readTensor(record, shards.get(record.file), record.byteOffset ?? 0, true);
  };

  const M = manifest.trunk;
  const record = dump.denoiser;
  const value = (name) => Float32Array.from(record.arguments[name].values);
  const integers = (name) => Int32Array.from(record.arguments[name].values);
  const tokens = record.arguments.s_inputs.shape[1];
  const atoms = record.arguments.ref_pos.shape[1];
  const tHat = record.arguments.t_hat.values[0];

  // 🔴 THE ELEMENT AND NAME FEATURES COME FROM THE FEATURISER'S ARRAYS, NOT THE
  // DENOISER'S ARGUMENTS. The module is handed them ALREADY ONE-HOT -
  // `ref_element` at [320, 128] - and `atomConditioning` builds the one-hot
  // itself, so passing the module's copy makes it read 0s and 1s as element
  // indices. Every shape conforms; the encoder came back at relRMS 0.89 with
  // corr 0.72, which reads exactly like a wrong convention in three SWA blocks.
  const features = {
    refPos: value("ref_pos"), refCharge: value("ref_charge"), mask: value("ref_mask"),
    refElement: Int32Array.from(dump.features.ref_element.values),
    refAtomNameChars: Int32Array.from(dump.features.ref_atom_name_chars.values),
    refSpaceUid: value("ref_space_uid"), atomToToken: integers("tok_idx"),
  };

  const [weights, encoder, decoder] = await Promise.all([
    denoiserWeights(read, { tokenBlocks: M.tokenBlocks }),
    atomEncoderWeights(read, "diffusionAtomEncoder", M.atomBlocks, { withCoordinates: true }),
    atomDecoderWeights(read, "diffusionAtomDecoder", M.atomBlocks),
  ]);
  weights.encoder = encoder;
  weights.decoder = decoder;

  const want = Float32Array.from(record.output);
  const noisy = value("x_noisy");
  const allocator = new GpuBufferAllocator(device);
  const cache = pipelineCacheForDevice(device);
  const storage = GPUBufferUsage.STORAGE;
  const pair = allocator.upload("check.z-trunk", value("z_trunk"), storage);
  const relPos = allocator.upload("check.rel-pos",
    value("relative_position_encoding"), storage);

  const results = [];
  let failures = 0;
  for (const precision of precisions) {
    const denoiser = new Esmfold2DenoiserGpu(device, allocator, cache);
    await denoiser.prepare({
      shape: {
        tokens, atoms,
        pairChannels: M.pairChannels, singleInputs: M.singleInputs,
        tokenChannels: M.tokenChannels2, tokenHeads: M.tokenHeads,
        multiplier: M.transitionMultiplier, sigmaData: M.sigmaData,
        atomChannels: M.atomChannels, atomHeads: M.atomHeads, atomBlocks: M.atomBlocks,
        atomHidden: M.atomChannels * 2, window: M.atomWindow,
        attentionPrecision: precision,
      },
      weights, features, sInputs: value("s_inputs"), pair, relPos,
    });
    const started = performance.now();
    const got = await denoiser.denoise(noisy, tHat);
    const elapsed = performance.now() - started;
    // ...and a second step, to price a warm one: everything but the two
    // writeBuffers and the readback is recorded and does not run again.
    const warmStarted = performance.now();
    await denoiser.denoise(noisy, tHat);
    const warm = performance.now() - warmStarted;
    const memory = allocator.snapshot();
    denoiser.release();
    const score = relative(got, want);
    // 🔴 ONLY THE SHIPPING ARM IS HELD TO THE BOUND. The f32 arm computes
    // something the checkpoint does not - see the control below - so scoring it
    // against a dump of the bf16 module and calling the difference a failure
    // would be a checker reporting a fault in its own setup. Its ceiling is
    // three times the bound, which is loose enough to be about the arithmetic
    // being right at all and tight enough to catch a broken kernel.
    const limit = precision === "f32" ? bound * 3 : bound;
    const ok = score <= limit;
    if (!ok) failures += 1;
    results.push({ precision, relRMS: score, ok, bound: limit,
                   firstStepMilliseconds: elapsed, warmStepMilliseconds: warm,
                   peakMebibytes: memory.peakBytes / 1048576 });
    console.log(`  ${precision.padEnd(6)} relRMS ${score.toExponential(3)}   `
      + `bound ${limit.toExponential(1)}   ${ok ? "ok" : "FAILED"}   `
      + `${elapsed.toFixed(0)} ms cold, ${warm.toFixed(0)} ms warm`);
  }
  pair.release();
  relPos.release();

  // 🔴 THE CONTROL IS THAT f32 IS WORSE, WHICH IS NOT WHAT A PRECISION ARM
  // USUALLY MEANS. Everywhere else in this tree a narrower arm is a trade; here
  // the narrowing IS the model, so an f32 attention is a more accurate
  // computation of something the shipping checkpoint does not do. If the two
  // arms ever agree, the downcast has stopped reaching the kernel - which is
  // exactly the failure a bound alone cannot see.
  const narrow = results.find((r) => r.precision === "bf16");
  const wide = results.find((r) => r.precision === "f32");
  let control;
  if (narrow !== undefined && wide !== undefined) {
    control = wide.relRMS > narrow.relRMS * 1.5;
    if (!control) failures += 1;
    console.log(`  control: f32 attention is ${(wide.relRMS / narrow.relRMS).toFixed(2)}x `
      + `the bf16 arm's error   ${control ? "ok" : "FAILED - the downcast is not reaching it"}`);
  }

  return { tokens, atoms, tHat, bound, failures, control, results,
           message: failures === 0
             ? "LocalFold's GPU denoiser computes ESMFold2's denoiser"
             : `${failures} arm(s) out of bound` };
}
