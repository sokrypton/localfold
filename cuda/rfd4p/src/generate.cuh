// The whole generation: noise in, a structure out. Drives `trunk.cuh` over the flow schedule in
// `sampler.cuh`, decodes the last step's clean estimate with `decoder.cuh`, and writes a PDB.
//
// 🔴 ONLY 35 OF THE 897 SEQUENCE COLUMNS MOVE BETWEEN STEPS - `xt_bb_ca` at offset 0 and
// `xt_local_latents` at offset 3 - so the features are exported once and those columns are
// overwritten on the device. The pair track's first 30 columns are its CA distance bins, refilled
// the same way. Everything else about a step is the conditioning, which depends on t alone and is
// exported for every step.
#pragma once
#include "decoder.cuh"
#include "sampler.cuh"

namespace rfd4p {

// x_t into the sequence features: the three CA columns and the thirty-two latent columns.
__global__ void writeStateK(float* __restrict__ feats, const float* __restrict__ bb,
                            const float* __restrict__ ll, int n, int fdim, int latent) {
  const int i = blockIdx.x;
  if (i >= n) return;
  float* row = feats + (size_t)i * fdim;
  for (int c = threadIdx.x; c < 3 + latent; c += blockDim.x) {
    row[c] = c < 3 ? bb[i * 3 + c] : ll[(size_t)i * latent + (c - 3)];
  }
}

// `xt_bb_ca_pair_dists`: a one-hot over `dim` buckets of the CA-CA distance, bucketed right-open
// against `limits[k] = min + (max - min) * k / (dim - 2)`, which is `torch.bucketize(right=False)`.
__global__ void pairDistBinsK(float* __restrict__ pair, const float* __restrict__ caNm, int n,
                              int pdim, int dim, float mn, float mx) {
  size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (size_t)n * n) return;
  const int i = (int)(p / n), j = (int)(p % n);
  const float dx = caNm[i * 3] - caNm[j * 3];
  const float dy = caNm[i * 3 + 1] - caNm[j * 3 + 1];
  const float dz = caNm[i * 3 + 2] - caNm[j * 3 + 2];
  const float r = sqrtf(dx * dx + dy * dy + dz * dz);
  int bin = 0;
  while (bin < dim - 1 && mn + (mx - mn) * bin / (float)(dim - 2) < r) ++bin;
  float* row = pair + p * pdim;
  for (int c = 0; c < dim; ++c) row[c] = 0.f;
  row[bin] = 1.f;
}

// The emit masks applied to the trunk's two predictions, as `flowEmitMasks` does.
__global__ void maskEmitK(float* __restrict__ x, const float* __restrict__ emit, size_t rows,
                          int dim) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * (size_t)dim && emit[i / dim] == 0.f) x[i] = 0.f;
}

struct GenOut {
  std::vector<float> coors;      // [n, 37, 3] in Angstrom
  std::vector<uint8_t> atomMask; // [n, 37]
  std::vector<int> residueType;  // [n], -1 where nothing was emitted
};

/**
 * `_format_sample_local_latents`: the decoder's three heads into a structure. Per token, so it runs
 * on the host - the arrays are [n, 147] and [n, 111] and the work is an argmax and a table lookup.
 */
inline GenOut postprocess(int n, int nTypes, const std::vector<float>& logits,
                          const std::vector<float>& coorsFlat, const std::vector<float>& presence,
                          const std::vector<float>& caNm, const std::vector<int>& seq,
                          const std::vector<int>& tokenType, const std::vector<int>& coordGroup,
                          const std::vector<int>& allowed, const std::vector<int>& legal,
                          const std::vector<float>& emit, const std::vector<float>& pad) {
  GenOut out;
  out.coors.assign((size_t)n * 111, 0.f);
  out.atomMask.assign((size_t)n * 37, 0);
  out.residueType.assign(n, -1);
  for (int i = 0; i < n; ++i) {
    if (emit[i] == 0.f) {
      // Not emitted: no residue, but a padded token still carries the sampled CA in slot 1.
      if (pad[i] != 0.f) {
        out.atomMask[(size_t)i * 37 + 1] = 1;
        for (int k = 0; k < 3; ++k) out.coors[(size_t)i * 111 + 3 + k] = caNm[i * 3 + k] * 10.f;
      }
      continue;
    }
    const int tt = std::max(0, std::min(6, tokenType[i]));
    int best = 0;
    float bestV = -3.4028234663852886e38f;
    for (int r = 0; r < nTypes; ++r) {
      const float v = allowed[(size_t)tt * nTypes + r] ? logits[(size_t)i * nTypes + r]
                                                       : -3.4028234663852886e38f;
      if (v > bestV) { bestV = v; best = r; }
    }
    const int rt = seq[i] != 0 ? seq[i] : best;    // 0 is `<M>`
    out.residueType[i] = rt;
    for (int a = 0; a < 37; ++a) {
      const bool required = coordGroup[(size_t)i * 37 + a] >= 0;
      const bool present = presence[(size_t)i * 37 + a] > 0.f || a == 1 || required;
      if (!(legal[(size_t)rt * 37 + a] && present)) continue;
      out.atomMask[(size_t)i * 37 + a] = 1;
      for (int k = 0; k < 3; ++k) {
        const float rel = a == 1 ? 0.f : coorsFlat[(size_t)i * 111 + a * 3 + k];
        out.coors[(size_t)i * 111 + a * 3 + k] = (rel + caNm[i * 3 + k]) * 10.f;
      }
    }
  }
  return out;
}

/** The structure as PDB records. `atomNames` is 37 long, `resNames` 147. */
inline std::string toPdb(const GenOut& g, int n, const std::vector<std::string>& atomNames,
                         const std::vector<std::string>& resNames) {
  std::string out;
  char line[96];
  int serial = 1;
  for (int i = 0; i < n; ++i) {
    const int rt = g.residueType[i];
    const std::string res = rt >= 0 && rt < (int)resNames.size() ? resNames[rt] : "UNK";
    for (int a = 0; a < 37; ++a) {
      if (!g.atomMask[(size_t)i * 37 + a]) continue;
      const std::string& name = atomNames[a];
      // PDB columns: a one- or two-letter element name starts at 14, not 13.
      char field[5];
      snprintf(field, sizeof field, name.size() >= 4 ? "%-4s" : " %-3s", name.c_str());
      snprintf(line, sizeof line,
               "ATOM  %5d %4s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n", serial++,
               field, res.c_str(), i + 1, g.coors[(size_t)i * 111 + a * 3],
               g.coors[(size_t)i * 111 + a * 3 + 1], g.coors[(size_t)i * 111 + a * 3 + 2],
               name.substr(0, 1).c_str());
      out += line;
    }
  }
  out += "TER\nEND\n";
  return out;
}

}  // namespace rfd4p
