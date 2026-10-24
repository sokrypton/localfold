/**
 * End-to-end RFD4-Proteina flow-matching sampling loop on WebGPU (`generateRfd4p`).
 *
 * Per step:
 *   1. Assemble trunk features on CPU (`trunkSeqFeatures`, `trunkCondFeatures` with `recycleRemaining`
 *      null and 1, `pairFeatures`, `pairCondFeatures`).
 *   2. Evaluate `Rfd4pTrunkGpu.forward(n, f, depth[s])`.
 *   3. Apply `flowEmitMasks` to predicted velocities `v`, then advance `bb_ca` and `local_latents`
 *      with `simulationStep` (which also returns the clean estimate `x1 = x_t + (1 - t) v`).
 *   4. Stream live frames (`x1` and optional decoded all-atom coordinates) and yield to the browser
 *      via `MessageChannel` (unclamped in background tabs).
 *   5. After the final step, compute `sequenceForDecoder(batch, n, lastLogits)`, run `Rfd4pDecoderGpu`,
 *      and format into Angstrom `coors [n, 37, 3]`, `atomMask [n, 37]`, and `residueType [n]`.
 */

import { formatSampleLocalLatents, sequenceForDecoder } from '../../cpu/rfd4p/decoder.js';
import { ConvergenceMonitor } from '../../shared/rfd4p/convergence.js';
import { pairCondFeatures, pairFeatures, trunkCondFeatures, trunkSeqFeatures } from '../../shared/rfd4p/features.js';
import {
  DEFAULT_SAMPLING,
  flowEmitMasks,
  recycleDepths,
  sampleInitialNoise,
  sampleSchedule,
  seededRng,
  simulationStep,
  tsrRatio,
} from '../../shared/rfd4p/flow.js';
import { processSampleToAtoms, toCifString, toPdbString } from '../../shared/rfd4p/output.js';
import {
  RESAMPLED_FORWARD_MARGINAL,
  buildPartialSchedule,
  partialStepCount,
  corruptBatch,
  resolveInverseFoldingMethod,
  resolvePartialT,
  sampleNoisedTargetBackbone,
} from '../../shared/rfd4p/partial.js';
import { Rfd4pDecoderGpu } from './decoder.js';
import { Rfd4pEncoderGpu } from './encoder.js';
import { Rfd4pTrunkGpu } from './trunk.js';

let yieldChannel;

/** Yield to the browser event loop via MessageChannel (avoids >=1s background-tab timer clamping). */
export function yieldToBrowser() {
  if (typeof MessageChannel === 'undefined') {
    return new Promise((resolve) => setTimeout(resolve, 0));
  }
  if (yieldChannel === undefined) yieldChannel = new MessageChannel();
  return new Promise((resolve) => {
    yieldChannel.port1.onmessage = () => resolve();
    yieldChannel.port2.postMessage(undefined);
  });
}

function tensorData(batch, key) {
  const v = batch[key];
  if (!v) throw new Error(`batch is missing '${key}'`);
  return v.data ?? v;
}

export class Rfd4pGeneratorGpu {
  /**
   * @param {GPUDevice} device
   * @param {object} weights name -> {data: Float32Array, shape}
   * @param {object} cfg bundle `config` (`{ nn, autoencoder_config }`)
   * @param {{half?: boolean, chunkFloats?: number}} [opts]
   */
  constructor(device, weights, cfg, opts = {}) {
    const resolvedWeights = weights?.tensors ?? weights;
    const resolvedCfg = cfg?.config ?? cfg;
    this.device = device;
    this.weights = resolvedWeights;
    this.cfg = resolvedCfg;
    this.opts = opts;
    this.half = opts.half ?? false;
    this.trunk = new Rfd4pTrunkGpu(device, resolvedWeights, resolvedCfg.nn, opts);
    this.decoder = new Rfd4pDecoderGpu(device, resolvedWeights, resolvedCfg.autoencoder_config.nn_ae.decoder, opts);
    this.encoder = opts.withEncoder
      ? new Rfd4pEncoderGpu(device, resolvedWeights, resolvedCfg.autoencoder_config.nn_ae.encoder, opts)
      : null;
  }

  /** Lazily instantiate and return the WebGPU Autoencoder Encoder. */
  getEncoder() {
    if (!this.encoder) {
      this.encoder = new Rfd4pEncoderGpu(
        this.device,
        this.weights,
        this.cfg.autoencoder_config.nn_ae.encoder,
        this.opts,
      );
    }
    return this.encoder;
  }

  /**
   * Extract clean `{ bb_ca, local_latents }` from `batch` using the WebGPU Autoencoder Encoder
   * (`Proteina.add_clean_samples`), needed for partial diffusion and inverse folding.
   */
  async extractCleanSamples(batch, { n, rng, deterministic = false }) {
    const caNm = new Float32Array(n * 3);
    const coorsNm = batch.coords_nm ? tensorData(batch, 'coords_nm') : null;
    const coorsA = !coorsNm && batch.coords ? tensorData(batch, 'coords') : null;
    if (!coorsNm && !coorsA) throw new Error('Partial diffusion / inverse folding requires batch.coords_nm or batch.coords');
    for (let i = 0; i < n; i += 1) {
      const base = (i * 37 + 1) * 3;
      for (let k = 0; k < 3; k += 1) {
        const v = coorsNm ? coorsNm[base + k] : coorsA[base + k] / 10;
        caNm[i * 3 + k] = Number.isFinite(v) ? v : 0;
      }
    }
    const enc = await this.getEncoder().encode(batch, { n, rng, deterministic });
    return { bb_ca: caNm, local_latents: enc.zLatent };
  }

  /**
   * Run flow-matching generation (full, partial diffusion, or inverse folding) on an upstream-layout `batch`.
   * @param {object} batch
   * @param {object} [opts]
   * @param {number} [opts.seed]
   * @param {{randn: (k: number) => ArrayLike<number>, randperm: (k: number, call?: number) => ArrayLike<number>}} [opts.rng]
   * @param {object} [opts.sampling]
   * @param {number} [opts.nsteps]
   * @param {'low'|'medium'|number|Array<[number, number]>} [opts.recycleSchedule]
   * @param {number} [opts.fullNsteps]
   * @param {boolean} [opts.inverseFolding]
   * @param {number} [opts.partialTrajectoryFractionSteps]
   * @param {number} [opts.partialTBackbone]
   * @param {number} [opts.partialTLatent]
   * @param {boolean} [opts.couplePartialT]
   * @param {{bb_ca: Float32Array, local_latents: Float32Array}} [opts.x1] pre-extracted clean samples
   * @param {number} [opts.decodeEvery] decode live all-atom frame every N steps (0 = only CA `x1`)
   * @param {(step: number, nsteps: number) => boolean} [opts.decodeStep] also decode a live all-atom frame at
   *   `step` when this returns true (asked before the step's trunk pass, so its logits are read back)
   * @param {boolean|{patience?: number, caTolA?: number, minFrac?: number}} [opts.earlyStop] stop once the x1 Cα and
   *   the designed sequence have both been still for `patience` steps (`shared/rfd4p/convergence.js`); the stopping
   *   step's x1 is the sample. `onStep` info carries `convergence`; the result reports `stepsRun`.
   * @param {(info: object) => void|Promise<void>} [opts.onStep]
   * @param {AbortSignal} [opts.signal]
   */
  async generate(batch, opts = {}) {
    const pad = tensorData(batch, 'padding_mask');
    const n = batch.padding_mask?.shape ? batch.padding_mask.shape.at(-1) : pad.length;
    const sampling = opts.sampling ?? DEFAULT_SAMPLING;
    const baseNsteps = opts.nsteps ?? sampling.nsteps;
    const rng = opts.rng ?? seededRng(opts.seed ?? 0);
    const decodeEvery = opts.decodeEvery ?? 0;
    const centeringScale = sampling.centering?.scale ?? 0.2;
    const coordCondMask = tensorData(batch, 'coord_cond_mask');
    const moleculeIid = tensorData(batch, 'molecule_iid');

    const emit = flowEmitMasks(
      {
        padding_mask: pad,
        is_in_context_conditioning: tensorData(batch, 'is_in_context_conditioning'),
        supports_local_latent: tensorData(batch, 'supports_local_latent'),
      },
      n,
    );

    const inverseFoldingMethod = resolveInverseFoldingMethod(opts);
    const partialTDict = resolvePartialT({ ...opts, nsteps: baseNsteps }, sampling);
    const isPartial = Object.keys(partialTDict).length > 0;
    const frac = opts.partialTrajectoryFractionSteps ?? opts.partial_trajectory_fraction_steps ?? null;
    const nsteps = isPartial && frac != null ? partialStepCount(frac, baseNsteps) : baseNsteps;
    const depths = recycleDepths(opts.recycleSchedule ?? 'medium', nsteps, opts.fullNsteps ?? baseNsteps);

    let sched;
    let x;
    let targetBackbone = null;
    const resampleTargetBackbone = inverseFoldingMethod === RESAMPLED_FORWARD_MARGINAL;

    if (isPartial) {
      if (partialTDict.bb_ca === undefined) partialTDict.bb_ca = 0.0;
      if (partialTDict.local_latents === undefined) partialTDict.local_latents = 0.0;
      const cleanX1 = opts.x1 ?? (await this.extractCleanSamples(batch, { n, rng }));
      const corrupted = corruptBatch({
        n,
        x1: cleanX1,
        partialTDict,
        emit,
        coordCondMask,
        moleculeIid,
        rng,
        centeringScale,
      });
      x = corrupted.xt;
      sched = buildPartialSchedule(partialTDict, nsteps, sampling, {
        fullNsteps: frac != null ? baseNsteps : null,
      });
      if (resampleTargetBackbone) {
        targetBackbone = Float32Array.from(corrupted.x1.bb_ca);
        x.bb_ca = sampleNoisedTargetBackbone({
          n,
          targetBackbone,
          tBb: sched.bb_ca.ts[0],
          emitBbCa: emit.bb_ca,
          coordCondMask,
          moleculeIid,
          rng,
          centeringScale,
        });
      }
    } else {
      sched = sampleSchedule({ ...sampling, nsteps });
      x = sampleInitialNoise({
        n,
        emit,
        coordCondMask,
        moleculeIid,
        rng,
        centeringScale,
      });
    }

    const nnCfg = this.cfg.nn;
    const distCfg = { dim: nnCfg.xt_pair_dist_dim, min: nnCfg.xt_pair_dist_min, max: nnCfg.xt_pair_dist_max };
    const featCache = {};
    const simBuf = new Float32Array(16 + n * 35);
    let gpuStepLoop = false;
    let lastLogits = null;
    let lastInsertion = null;
    const monitor = opts.earlyStop
      ? new ConvergenceMonitor(nsteps, opts.earlyStop === true ? {} : opts.earlyStop, emit.bb_ca)
      : null;
    let stepsRun = nsteps;
    // Stop after step s: x1(s) becomes the sample (one Euler step from t(s) to 1).
    const converged = (s, x1, convergence) => {
      if (!convergence?.stop) return false;
      x.bb_ca.set(x1.bb_ca);
      x.local_latents.set(x1.local_latents);
      stepsRun = s + 1;
      return true;
    };

    for (let s = 0; s < nsteps; s += 1) {
      if (opts.signal?.aborted) throw new Error('Generation aborted');
      const tt = {
        bb_ca: Math.fround(sched.bb_ca.ts[s]),
        local_latents: Math.fround(sched.local_latents.ts[s]),
      };
      const isLast = s === nsteps - 1;
      const wantDecodeStep = Boolean(opts.onStep) && (
        (decodeEvery > 0 && ((s + 1) % decodeEvery === 0 || s + 1 === nsteps)) || Boolean(opts.decodeStep?.(s, nsteps)));
      const wantLogits = isLast || Boolean(opts.wantStepLogits) || wantDecodeStep || Boolean(monitor);
      // The insertion head normally runs on the last step only; run it on any step that could end the run, and on
      // decoded live frames when asked (so the caller can place an unindexed motif in each frame).
      const wantInsertion = isLast || Boolean(monitor?.mayStopAt(s)) || (wantDecodeStep && Boolean(opts.liveInsertion));
      let seqData = null;
      if (s === 0 || !gpuStepLoop) {
        seqData = trunkSeqFeatures(batch, {
          n,
          xt: x,
          feats: nnCfg.feats_seq,
          weights: this.weights,
          randperm: (len) => rng.randperm(len, s),
          noiseAlignChirality: nnCfg.noise_align_chirality,
          cache: featCache,
          rowStride: this.trunk.w.seqLin.stride || 0,
        }).data;
        if (s === 0) {
          gpuStepLoop =
            !resampleTargetBackbone &&
            featCache.caOff === 0 &&
            featCache.latOff === 3 &&
            (featCache.crdUniq?.length ?? 0) <= 1 &&
            (!nnCfg.noise_align_chirality || (featCache.activeFrames?.length ?? 0) === 0);
        }
      } else if (featCache.grpOff >= 0) {
        rng.randperm(featCache.crdUniq.length, s);
      }
      const f = {
        seq: seqData,
        cond0: trunkCondFeatures(batch, {
          n, t: tt, recycleRemaining: null, feats: nnCfg.feats_cond_seq, weights: this.weights, cache: featCache, singleRow: true,
        }).data,
        cond1: depths[s] > 1
          ? trunkCondFeatures(batch, {
              n, t: tt, recycleRemaining: 1, feats: nnCfg.feats_cond_seq, weights: this.weights, cache: featCache, singleRow: true,
            }).data
          : null,
        pair: s === 0
          ? pairFeatures(batch, {
              n, caNm: x.bb_ca, feats: nnCfg.feats_pair_repr, dist: distCfg, seqSepDim: nnCfg.seq_sep_dim, cache: featCache,
              rowStride: this.trunk.w.pairLin.stride || 0, skipPairDists: true,
            }).data
          : null,
        pairCond: pairCondFeatures(tt, nnCfg.feats_pair_cond),
      };

      if (gpuStepLoop) {
        simBuf.fill(0);
        let pOff = 0;
        let epsOff = 16;
        for (const dm of ['bb_ca', 'local_latents']) {
          const ts = sched[dm].ts;
          const t = Math.fround(ts[s]);
          const dt = ts[s + 1] - ts[s];
          const gt = sched[dm].gt[s];
          const stepCfg = sched[dm].step;
          const r = t === 0 ? 0 : tsrRatio(t, stepCfg.tsr_k, stepCfg.tsr_sigma);
          const stochastic = t !== 0 && !(t > stepCfg.t_lim_ode || t < stepCfg.t_lim_ode_below);
          const std = stochastic ? Math.sqrt(2 * gt * stepCfg.sc_scale_noise * dt) : 0;
          const len = dm === 'bb_ca' ? n * 3 : n * 32;
          simBuf[pOff] = t;
          simBuf[pOff + 1] = dt;
          simBuf[pOff + 2] = gt;
          simBuf[pOff + 3] = r;
          simBuf[pOff + 4] = std;
          simBuf[pOff + 5] = stochastic ? 1 : 0;
          if (stochastic) {
            simBuf.set(rng.randn(len), epsOff);
          }
          pOff += 6;
          epsOff += len;
        }
        const needReadback = isLast || Boolean(opts.onStep) || Boolean(monitor);
        const out = await this.trunk.forward(n, f, depths[s], {
          insertion: wantInsertion,
          logits: wantLogits,
          emit: s === 0 ? emit : null,
          simBuf,
          readback: needReadback,
        });
        if (needReadback) {
          x.bb_ca.set(out.bb_ca);
          x.local_latents.set(out.local_latents);
          if (out.logits) lastLogits = out.logits;
          if (out.insertion) lastInsertion = out.insertion;
        }
        const convergence = monitor?.observe(s, out.x1.bb_ca, sequenceForDecoder(batch, n, lastLogits)) ?? null;
        if (opts.onStep) {
          let liveDecoded = null;
          if (wantDecodeStep && (!convergence?.stop || decodeEvery > 0)) {
            const liveSeq = sequenceForDecoder(batch, n, lastLogits);
            const dec = await this.decoder.decode(batch, { n, caNm: out.x1.bb_ca, z: out.x1.local_latents, seq: liveSeq });
            liveDecoded = formatSampleLocalLatents(batch, { n, caNm: out.x1.bb_ca, decoded: dec });
          }
          await opts.onStep({
            step: s,
            nsteps,
            t: tt,
            depth: depths[s],
            x1: out.x1,
            xt: { bb_ca: Float32Array.from(x.bb_ca), local_latents: Float32Array.from(x.local_latents) },
            logits: lastLogits,
            decoded: liveDecoded,
            // 🔴 THE PADDING MASK TRAVELS WITH THE FRAME. `processSampleToAtoms`
            // needs it, and without it a caller drawing a live frame would
            // either rebuild it from the batch - a second copy of this
            // function's own input - or draw the padded positions as atoms.
            paddingMask: pad,
            insertionLogits: liveDecoded ? lastInsertion : null,
            convergence,
          });
        }
        if (convergence && converged(s, out.x1, convergence)) break;
        continue;
      }

      const out = await this.trunk.forward(n, f, depths[s], { insertion: wantInsertion, logits: wantLogits });
      for (let i = 0; i < n; i += 1) {
        if (!emit.bb_ca[i]) out.bb_ca.fill(0, i * 3, i * 3 + 3);
        if (!emit.local_latents[i]) out.local_latents.fill(0, i * 32, i * 32 + 32);
      }
      if (out.logits) lastLogits = out.logits;
      if (out.insertion) lastInsertion = out.insertion;

      const x1 = {};
      for (const dm of ['bb_ca', 'local_latents']) {
        const ts = sched[dm].ts;
        x1[dm] = simulationStep({
          x: x[dm],
          v: out[dm],
          t: Math.fround(ts[s]),
          dt: ts[s + 1] - ts[s],
          gt: sched[dm].gt[s],
          emit: emit[dm],
          dim: dm === 'bb_ca' ? 3 : 32,
          step: sched[dm].step,
          rng,
        });
      }

      if (resampleTargetBackbone) {
        if (s === nsteps - 1) {
          x.bb_ca = Float32Array.from(targetBackbone);
        } else {
          x.bb_ca = sampleNoisedTargetBackbone({
            n,
            targetBackbone,
            tBb: sched.bb_ca.ts[s + 1],
            emitBbCa: emit.bb_ca,
            coordCondMask,
            moleculeIid,
            rng,
            centeringScale,
          });
        }
        x1.bb_ca = Float32Array.from(targetBackbone);
      }

      const convergence = monitor?.observe(s, x1.bb_ca, sequenceForDecoder(batch, n, lastLogits)) ?? null;
      if (opts.onStep) {
        let liveDecoded = null;
        if (wantDecodeStep && (!convergence?.stop || decodeEvery > 0)) {
          const liveSeq = sequenceForDecoder(batch, n, lastLogits);
          const dec = await this.decoder.decode(batch, { n, caNm: x1.bb_ca, z: x1.local_latents, seq: liveSeq });
          liveDecoded = formatSampleLocalLatents(batch, { n, caNm: x1.bb_ca, decoded: dec });
        }
        await opts.onStep({
          step: s,
          nsteps,
          t: tt,
          depth: depths[s],
          x1,
          xt: { bb_ca: Float32Array.from(x.bb_ca), local_latents: Float32Array.from(x.local_latents) },
          logits: lastLogits,
          decoded: liveDecoded,
          insertionLogits: liveDecoded ? lastInsertion : null,
          convergence,
        });
        await yieldToBrowser();
      }
      if (converged(s, x1, convergence)) break;
    }

    const seq = sequenceForDecoder(batch, n, lastLogits);
    const decoded = await this.decoder.decode(batch, { n, caNm: x.bb_ca, z: x.local_latents, seq });
    const formatted = formatSampleLocalLatents(batch, { n, caNm: x.bb_ca, decoded });
    const inputAtoms = opts.inputAtoms ?? batch.__atoms ?? null;
    const structure = processSampleToAtoms({
      coors: formatted.coors,
      atomMask: formatted.atomMask,
      residueType: formatted.residueType,
      paddingMask: pad,
      insertionLogits: lastInsertion,
      inputAtoms,
      batch,
    });
    return {
      n,
      nsteps,
      stepsRun,
      x,
      seq,
      logits: lastLogits,
      insertionLogits: lastInsertion,
      decoded,
      coors: formatted.coors,
      atomMask: formatted.atomMask,
      residueType: formatted.residueType,
      structure,
      cif: toCifString(structure),
      pdb: toPdbString(structure),
    };
  }

  destroy() {
    this.trunk.destroy();
    this.decoder.destroy();
    if (this.encoder) this.encoder.destroy();
  }
}

/** One-shot helper that creates an `Rfd4pGeneratorGpu`, runs `generate`, and destroys GPU buffers. */
export async function generateRfd4p(device, weights, cfg, batch, opts = {}) {
  const gen = new Rfd4pGeneratorGpu(device, weights, cfg, opts);
  try {
    return await gen.generate(batch, opts);
  } finally {
    gen.destroy();
  }
}
