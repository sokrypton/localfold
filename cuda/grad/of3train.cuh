// OpenFold3's optimisation (openfold3/projects/of3_all_atom/runner.py and core/utils): what turns a step's gradients
// into an update.
//
//   per-sample clipping   each example's gradient clipped to a global norm of 10 (grad_manager.py _clip_grads: the
//                         norm over the parameters that example ENABLES - a sample whose confidence losses all weigh
//                         zero leaves the confidence head out, of the norm and of the count) and added to an
//                         accumulator; at the step each parameter's sum divided by how many examples it took part in
//   Adam                  torch.optim.Adam, betas (0.9, 0.95), eps 1e-8, no weight decay
//   AlphaFoldLRScheduler  lr = base + step / warmup * max through the warmup (1000 steps from 0), max on the plateau,
//                         max * 0.95^(floor((step - 50000) / 50000) + 1) past step 50000; max 1.8e-3
//   EMA                   after each step, ema -= (1 - 0.999) (ema - w), every parameter
//
// And a checkpoint of the whole of it (weights, moments, the EMA, the accumulator's step), by parameter name, so a
// run resumes where it stopped.
#pragma once
#include <fstream>
#include <set>
#include "adam.cuh"
#include "of3dist.cuh"

namespace grad {

__global__ void scaleAddK(real* __restrict__ acc, const real* __restrict__ g, real s, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) acc[t] += s * g[t];
}
__global__ void adamOf3K(real* __restrict__ w, const real* __restrict__ acc, real* __restrict__ m, real* __restrict__ v, size_t n,
                         real lr, real b1, real b2, real eps, real c1, real c2, real inv) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  const real g = acc[t] * inv;
  const real mt = b1 * m[t] + (1 - b1) * g, vt = b2 * v[t] + (1 - b2) * g * g;
  m[t] = mt; v[t] = vt;
  w[t] -= lr * (mt / c1) / (sqrt(vt / c2) + eps);
}
__global__ void emaK(real* __restrict__ e, const real* __restrict__ w, real decay, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) e[t] -= (1 - decay) * (e[t] - w[t]);
}

struct Of3Optimizer {
  // the run's settings (model_config.py's optimizer, lr_scheduler, ema, gradient_clipping)
  double maxLr = 1.8e-3, baseLr = 0, b1 = 0.9, b2 = 0.95, eps = 1e-8, clip = 10, emaDecay = 0.999, decayFactor = 0.95;
  long warmup = 1000, decayAfter = 50000, decayEvery = 50000;
  long step = 0;                         // optimiser steps taken (the scheduler's last_epoch)
  std::vector<real *> m, v, acc, ema;
  std::vector<int> participation;        // examples a parameter took part in since the last step
  std::vector<char> frozen;              // a buffer in OpenFold3 (the Fourier noise embedding): never updated, never clipped
  int examples = 0;
  double* norm = nullptr;
  double lastNorm = 0, lastLr = 0;
  // m, v, the accumulator and the EMA each one allocation, every parameter's at its offset (64-element aligned): four
  // slabs instead of a cudaMalloc a tensor (3760 of them in OpenBind-0, each rounded up - 2 GB of rounding over the
  // four), and the accumulators one contiguous range the data-parallel sum reads in place
  real *mSlab = nullptr, *vSlab = nullptr, *accSlab = nullptr, *emaSlab = nullptr; size_t slabN = 0;
  void init(const Params& ps) {
    std::vector<size_t> off;
    for (auto* p : ps.all) { off.push_back(slabN); slabN += (p->n + 63) / 64 * 64; }
    auto zeros = [&] { real* p; GK(cudaMalloc(&p, std::max<size_t>(slabN, 1) * sizeof(real))); GK(cudaMemset(p, 0, std::max<size_t>(slabN, 1) * sizeof(real))); return p; };
    mSlab = zeros(); vSlab = zeros(); accSlab = zeros(); emaSlab = zeros();
    for (size_t i = 0; i < ps.all.size(); ++i) {
      m.push_back(mSlab + off[i]); v.push_back(vSlab + off[i]); acc.push_back(accSlab + off[i]); ema.push_back(emaSlab + off[i]);
      GK(cudaMemcpy(ema[i], ps.all[i]->w, ps.all[i]->n * sizeof(real), cudaMemcpyDeviceToDevice));
    }
    participation.assign(ps.all.size(), 0);
    frozen.assign(ps.all.size(), 0);
    GK(cudaMalloc(&norm, sizeof(double)));
  }
  double lr(long s) const {
    if (s <= warmup) return baseLr + (double)s / warmup * maxLr;
    if (s > decayAfter) return maxLr * std::pow(decayFactor, (double)((s - decayAfter) / decayEvery + 1));
    return maxLr;
  }
  // one example's gradients (in the parameters' g): clipped by their global norm over the enabled parameters, added to
  // the accumulator. `disabled(i)` true leaves parameter i out of the norm and the count.
  template <class F> double clipAndAccumulate(const Params& ps, F disabled) {
    GK(cudaMemsetAsync(norm, 0, sizeof(double), STREAM));
    for (size_t i = 0; i < ps.all.size(); ++i) {
      if (disabled(i) || frozen[i]) continue;
      Param* p = ps.all[i];
      sqSumK<<<(unsigned)std::min<size_t>(blocks(p->n), 1024), 256, 0, STREAM>>>(p->g, p->n, norm);
    }
    double h; GK(cudaMemcpyAsync(&h, norm, sizeof(double), cudaMemcpyDeviceToHost, STREAM));
    GK(cudaStreamSynchronize(STREAM));
    lastNorm = std::sqrt(h);
    const double coef = clip / std::max(lastNorm, clip);
    for (size_t i = 0; i < ps.all.size(); ++i) {
      if (disabled(i) || frozen[i]) continue;
      Param* p = ps.all[i];
      scaleAddK<<<blocks(p->n), 256, 0, STREAM>>>(acc[i], p->g, (real)coef, p->n);
      ++participation[i];
    }
    ++examples;
    return lastNorm;
  }
  // the optimiser step over the accumulated examples, then the EMA; the accumulator cleared
  void apply(Params& ps, Comm* comm = nullptr) {
    // (data-parallel: every rank's accumulators and participation counts summed first - the same global average, the
    // same step, on every rank; the slab's alignment padding is zero on every rank, so it sums to zero)
    if (comm && comm->world > 1) {
      comm->sumReals(accSlab, slabN);
      comm->sumInts(participation);
    }
    lastLr = lr(step);             // (the scheduler's lr for this step, then stepped)
    ++step;
    const real c1 = (real)(1 - std::pow(b1, (double)step)), c2 = (real)(1 - std::pow(b2, (double)step));
    for (size_t i = 0; i < ps.all.size(); ++i) {
      Param* p = ps.all[i];
      if (frozen[i]) continue;
      const real inv = (real)(1.0 / std::max(1, participation[i]));
      adamOf3K<<<blocks(p->n), 256, 0, STREAM>>>(p->w, acc[i], m[i], v[i], p->n, (real)lastLr, (real)b1, (real)b2, (real)eps, c1, c2, inv);
      emaK<<<blocks(p->n), 256, 0, STREAM>>>(ema[i], p->w, (real)emaDecay, p->n);
      GK(cudaMemsetAsync(acc[i], 0, p->n * sizeof(real), STREAM));
      participation[i] = 0;
    }
    examples = 0;
  }
  // ---- a checkpoint: "LFCKPT1", the step, then per parameter its name, length and four arrays (w, m, v, ema)
  void save(const Params& ps, const std::string& path) const {
    std::ofstream f(path + ".tmp", std::ios::binary);
    f.write("LFCKPT1", 7);
    f.write((const char*)&step, sizeof(step));
    const uint64_t count = ps.all.size(); f.write((const char*)&count, 8);
    std::vector<real> h;
    for (size_t i = 0; i < ps.all.size(); ++i) {
      const Param* p = ps.all[i];
      const uint32_t len = (uint32_t)p->name.size(); f.write((const char*)&len, 4); f.write(p->name.data(), len);
      const uint64_t n = p->n; f.write((const char*)&n, 8);
      h.resize(p->n);
      for (const real* d : {(const real*)p->w, (const real*)m[i], (const real*)v[i], (const real*)ema[i]}) {
        GK(cudaMemcpy(h.data(), d, p->n * sizeof(real), cudaMemcpyDeviceToHost));
        f.write((const char*)h.data(), p->n * sizeof(real));
      }
    }
    f.close();
    if (!f) { fprintf(stderr, "could not write %s\n", path.c_str()); exit(1); }
    std::rename((path + ".tmp").c_str(), path.c_str());
  }
  // every parameter by name, and nothing missing or extra (a checkpoint from another model is refused, not half read)
  void load(Params& ps, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot read checkpoint %s\n", path.c_str()); exit(1); }
    char tag[7]; f.read(tag, 7);
    if (std::string(tag, 7) != "LFCKPT1") { fprintf(stderr, "%s: not a checkpoint\n", path.c_str()); exit(1); }
    f.read((char*)&step, sizeof(step));
    uint64_t count; f.read((char*)&count, 8);
    if (count != ps.all.size()) { fprintf(stderr, "%s: %zu parameters, this model has %zu\n", path.c_str(), (size_t)count, ps.all.size()); exit(1); }
    std::map<std::string, size_t> at; for (size_t i = 0; i < ps.all.size(); ++i) at[ps.all[i]->name] = i;
    std::vector<real> h;
    for (uint64_t k = 0; k < count; ++k) {
      uint32_t len; f.read((char*)&len, 4); std::string name(len, ' '); f.read(&name[0], len);
      uint64_t n; f.read((char*)&n, 8);
      auto it = at.find(name);
      if (it == at.end() || ps.all[it->second]->n != n) { fprintf(stderr, "%s: %s does not match this model\n", path.c_str(), name.c_str()); exit(1); }
      const size_t i = it->second;
      h.resize(n);
      for (real* d : {ps.all[i]->w, m[i], v[i], ema[i]}) {
        f.read((char*)h.data(), n * sizeof(real));
        GK(cudaMemcpy(d, h.data(), n * sizeof(real), cudaMemcpyHostToDevice));
      }
    }
    if (!f) { fprintf(stderr, "%s is truncated\n", path.c_str()); exit(1); }
  }
  // swap the live weights with the EMA's (evaluation reads the EMA, as OpenFold3's validation does); call again to swap back
  void swapEma(Params& ps) {
    for (size_t i = 0; i < ps.all.size(); ++i) std::swap(ps.all[i]->w, ema[i]);
  }
};

}  // namespace grad
