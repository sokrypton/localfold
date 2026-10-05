// Synthyra's ESMFold2 confidence head (biohub ships the checkpoint without one; Synthyra trained it on
// the same frozen trunk). src/esmfold2/confidence-reference.js and their ConfidenceHead are the reading:
//
//   s = LN(s_inputs);  z = LN(z_trunk) + rel_pos + bonds + s_to_z(s)_i + s_to_z_T(s)_j + prod_out(in1(s)_i * in2(s)_j)
//   z += distance_embed[#(|x_rep_i - x_rep_j| > boundaries)]
//   z = z + trunk_4_blocks(z)                       (the trunk's own pair block, its own weights)
//   single = (softmax_j(z @ attn) z) @ out          (row-attention pooling)
//   pLDDT per atom = E[bin] over [0, 1] of LN(single[token]) @ plddtWeight[slot in token]
//   PAE = E[bin] over [0, 32] of z @ pae;  pTM / ipTM from the PAE logits (d0 from the live token count)
#pragma once
#include "trunk.cuh"

__global__ void confZK(float* z, RelIdx rel, const float* bonds, const float* wBond, const float* rows,
                       const float* cols, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / T), j = (int)(ij % T);
  z[t] += relPosAt(rel, i, j, c, C) + bonds[ij] * wBond[c] + rows[(size_t)i * C + c] + cols[(size_t)j * C + c];
}
__global__ void outerProductK(const float* a, const float* b, float* out, size_t p0, size_t n, int T, int C) {   // a_i * b_j, rows p0..
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * C) return;
  int c = (int)(t % C); size_t ij = p0 + t / C; int i = (int)(ij / T), j = (int)(ij % T);
  out[t] = a[(size_t)i * C + c] * b[(size_t)j * C + c];
}
// a warp a pair: its distance and bucket once (a ballot over the edges), then the row 16 bytes a lane - an
// element a thread took the distance and walked every edge for each of the C channels (12 ms of a 1,044-token
// confidence head's 194; C a multiple of 4)
__global__ void distanceEmbedK(float* z, const float* x, const int* rep, const float* edges, int nEdges,
                               const float* table, int T, int C) {
  size_t ij = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5; int lane = threadIdx.x & 31;
  if (ij >= (size_t)T * T) return;
  int i = (int)(ij / T), j = (int)(ij % T);
  const float* a = x + (size_t)rep[i] * 3; const float* b = x + (size_t)rep[j] * 3;
  float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  float d = sqrtf(dx * dx + dy * dy + dz * dz);
  int bucket = 0;
  for (int e0 = 0; e0 < nEdges; e0 += 32) bucket += __popc(__ballot_sync(~0u, e0 + lane < nEdges && d > edges[e0 + lane]));
  float4* row = reinterpret_cast<float4*>(z + ij * C);
  const float4* tb = reinterpret_cast<const float4*>(table + (size_t)bucket * C);
  for (int c = lane; c < C / 4; c += 32) { float4 v = row[c], w = tb[c]; v.x += w.x; v.y += w.y; v.z += w.z; v.w += w.w; row[c] = v; }
}
// row-attention pooling: pooled[i] = sum_j softmax_j(score[i, j]) z[i, j]   (a block per row, C <= 1024)
__global__ void rowPoolK(const float* z, const float* score, float* pooled, int T, int C) {
  int i = blockIdx.x;
  __shared__ float m, s;
  if (threadIdx.x == 0) {
    float mx = -INFINITY;
    for (int j = 0; j < T; ++j) mx = fmaxf(mx, score[(size_t)i * T + j]);
    float sum = 0;
    for (int j = 0; j < T; ++j) sum += expf(score[(size_t)i * T + j] - mx);
    m = mx; s = sum;
  }
  __syncthreads();
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float acc = 0;
    for (int j = 0; j < T; ++j) acc += expf(score[(size_t)i * T + j] - m) / s * z[((size_t)i * T + j) * C + c];
    pooled[(size_t)i * C + c] = acc;
  }
}
__global__ void gatherAtomsK(const float* tok, const int* atomToToken, float* out, int A, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < (size_t)A * C) out[t] = tok[(size_t)atomToToken[t / C] * C + t % C];
}
// per atom: logits over bins from its slot's table, then the expectation over [0, 1]
__global__ void plddtAtomK(const float* s, const int* slot, const float* table, float* plddt, int A, int C, int bins) {
  int a = blockIdx.x;
  __shared__ float logit[64];
  const float* tb = table + (size_t)slot[a] * C * bins;
  for (int b = threadIdx.x; b < bins; b += blockDim.x) {
    float acc = 0;
    for (int c = 0; c < C; ++c) acc += s[(size_t)a * C + c] * tb[(size_t)c * bins + b];
    logit[b] = acc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float mx = -INFINITY; for (int b = 0; b < bins; ++b) mx = fmaxf(mx, logit[b]);
    float tot = 0, e = 0;
    for (int b = 0; b < bins; ++b) { float w = expf(logit[b] - mx); tot += w; e += w * (b + 0.5f) / bins; }
    plddt[a] = e / tot;
  }
}

// PAE and the pTM terms from the logits, a thread a pair: pae = sum p_b centre_b, tm = sum p_b / (1 + (centre_b / d0)^2);
// then a block a row: the row's mean tm over every column and over the other chains' columns (double)
__global__ void paeK(const float* logits, float* pae, float* tm, size_t P, int bins, double width, double d0) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= P) return;
  const float* lg = logits + ij * bins;
  float mx = -INFINITY;
  for (int b = 0; b < bins; ++b) mx = fmaxf(mx, lg[b]);
  double tot = 0, mean = 0, t = 0;
  for (int b = 0; b < bins; ++b) {
    double w = exp((double)lg[b] - mx), centre = width * (b + 0.5), r = centre / d0;
    tot += w; mean += w * centre; t += w / (1 + r * r);
  }
  pae[ij] = (float)(mean / tot); tm[ij] = (float)(t / tot);
}
__global__ void tmRowsK(const float* tm, const int* asym, double* rows, int T) {   // rows [T][2]: mean, inter-chain mean
  int i = blockIdx.x;
  __shared__ double s[2][256], c[256];
  double sum = 0, isum = 0, icnt = 0;
  for (int j = threadIdx.x; j < T; j += blockDim.x) {
    double v = tm[(size_t)i * T + j]; sum += v;
    if (asym[i] != asym[j]) { isum += v; icnt += 1; }
  }
  s[0][threadIdx.x] = sum; s[1][threadIdx.x] = isum; c[threadIdx.x] = icnt;
  __syncthreads();
  for (int o = blockDim.x / 2; o; o >>= 1) {
    if (threadIdx.x < o) { s[0][threadIdx.x] += s[0][threadIdx.x + o]; s[1][threadIdx.x] += s[1][threadIdx.x + o]; c[threadIdx.x] += c[threadIdx.x + o]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { rows[2 * i] = s[0][0] / (T + 1e-8); rows[2 * i + 1] = s[1][0] / (c[0] + 1e-8); }
}

struct Confidence { std::vector<float> plddtAtom, plddtToken, pae; double ptm, iptm, meanPlddt; };

// consume: the trunk's z is read by nothing after this call - on a card short of room the head then works in it
inline Confidence confidenceHead(int T, int A, const float* zTrunk, const float* sInputs, int Si, const float* xDevice,
                                 bool check, bool consume = false) {
  int C = (int)dimOf("f/confidence/sToZ", 1), Cs = (int)dimOf("f/confidence/poolingOutput", 1);
  size_t P = (size_t)T * T;
  float* s = scratch<float>("cf.s", (size_t)T * Si);
  layerNorm(sInputs, s, T, Si, F("confidence/sInputsNorm/scale"), F("confidence/sInputsNorm/offset"));
  // on a card short of room, a last call normalises the trunk's z in place (each lane reads its channels before
  // writing them): a 256-channel f32 pair fewer, 4.1 GB at 2000 tokens
  // (tight only where the device lacks the room for the two pair-sized tensors the roomy form holds: parking
  // the residual on the host was 1.17 s of a 1,044-token fold on a 40 GB A100 - a 1.1 GB pinned allocation
  // and a copy each way - with 30 GB free; LOCALFOLD_BIG=1 still takes the parked form)
  bool tight = shortPair(P, C) && !roomFor(2 * P * C * 4), inPlace = consume && tight;
  float* z = inPlace ? const_cast<float*>(zTrunk) : dalloc(P * C);
  layerNorm(zTrunk, z, P, C, F("confidence/zNorm/scale"), F("confidence/zNorm/offset"));
  float* r = scratch<float>("cf.r", (size_t)T * C); float* c = scratch<float>("cf.c", (size_t)T * C);
  float* l = scratch<float>("cf.l", (size_t)T * C); float* rr = scratch<float>("cf.rr", (size_t)T * C);
  gemm(s, F("confidence/sToZ"), r, T, Si, C);
  gemm(s, F("confidence/sToZTranspose"), c, T, Si, C);
  gemm(s, F("confidence/sToZProdIn1"), l, T, Si, C);
  gemm(s, F("confidence/sToZProdIn2"), rr, T, Si, C);
  confZK<<<blocks(P * C), 256, 0, STREAM>>>(z, relIdx(), W("token_bonds"), F("featuriser/tokenBonds"), r, c, T, C);
  size_t chunk = std::min<size_t>(P, ((size_t)64 << 20) / (4 * (size_t)C));
  float* prod = scratch<float>("cf.prod", chunk * C);
  for (size_t p0 = 0; p0 < P; p0 += chunk) {
    size_t n = std::min(chunk, P - p0);
    outerProductK<<<blocks(n * C), 256, 0, STREAM>>>(l, rr, prod, p0, n, T, C);
    gemm(prod, F("confidence/sToZProdOut"), z + p0 * C, n, C, C, 1.f);
  }
  distanceEmbedK<<<blocks(P * 32), 256, 0, STREAM>>>(z, xDevice, Idev("distogram_atom_idx"), F("confidence/boundaries"),
    (int)M.len("f/confidence/boundaries"), F("confidence/distanceEmbedding"), T, C);
  // z + trunk(z) - on a card short of room z waits in pinned host memory while the blocks run on it in place,
  // then comes back a chunk at a time and is added (the same sum: one more pair-sized tensor not on the card)
  float* mask = scratch<float>("trunk.mask", P);
  fillK<<<blocks(P), 256, 0, STREAM>>>(mask, 1.f, P);
  float* stack = tight ? z : dalloc(P * C);
  static float* heldBuf = nullptr; static size_t heldHave = 0;     // (pinned, kept for the process, grown as needed)
  if (tight && heldHave < P * C) {
    if (heldBuf) CK(cudaFreeHost(heldBuf));
    if (cudaMallocHost(&heldBuf, P * C * 4) != cudaSuccess) {
      fprintf(stderr, "cannot pin %.2f GB of host memory for the confidence head's residual\n", P * C * 4 / 1e9); exit(1);
    }
    heldHave = P * C;
  }
  float* held = tight ? heldBuf : nullptr;
  if (tight) CK(cudaMemcpyAsync(held, z, P * C * 4, cudaMemcpyDeviceToHost, STREAM));
  else CK(cudaMemcpyAsync(stack, z, P * C * 4, cudaMemcpyDeviceToDevice, STREAM));
  for (int b = 0; M.has("f/confidence/blocks/" + std::to_string(b) + "/pairTransition/transition1"); ++b)
    trunkBlock(stack, mask, T, C, "confidence/blocks", b);
  if (tight) {
    releaseScratch({ "ftri.", "ftr.", "trib." });
    float* back = scratch<float>("cf.back", chunk * C);
    for (size_t p0 = 0; p0 < P; p0 += chunk) {
      size_t n = std::min(chunk, P - p0);
      CK(cudaMemcpyAsync(back, held + p0 * C, n * C * 4, cudaMemcpyHostToDevice, STREAM));
      addK<<<blocks(n * C), 256, 0, STREAM>>>(z + p0 * C, back, n * C);
    }
  } else {
    addK<<<blocks(P * C), 256, 0, STREAM>>>(z, stack, P * C);
    CK(cudaFree(stack));
  }
  // single by row-attention pooling
  float* score = scratch<float>("cf.score", P);
  gemm(z, F("confidence/poolingAttention"), score, P, C, 1);
  float* pooled = scratch<float>("cf.pooled", (size_t)T * C); float* single = scratch<float>("cf.single", (size_t)T * Cs);
  rowPoolK<<<T, 256, 0, STREAM>>>(z, score, pooled, T, C);
  gemm(pooled, F("confidence/poolingOutput"), single, T, C, Cs);
  // pLDDT per atom (a table a slot: the atom's position within its token, clamped)
  int bins = (int)dimOf("f/confidence/plddtWeight", 2), slots = (int)dimOf("f/confidence/plddtWeight", 0);
  std::vector<int> a2t(A), slot(A);
  CK(cudaMemcpy(a2t.data(), Idev("atom_to_token"), A * 4, cudaMemcpyDeviceToHost));
  for (int a = 0, k = 0; a < A; ++a) { if (a > 0 && a2t[a] != a2t[a - 1]) k = 0; slot[a] = std::min(k++, slots - 1); }
  int* dSlot = upload(slot.data(), A);
  float* sa = scratch<float>("cf.sa", (size_t)A * Cs); float* pa = scratch<float>("cf.pa", A);
  gatherAtomsK<<<blocks((size_t)A * Cs), 256, 0, STREAM>>>(single, Idev("atom_to_token"), sa, A, Cs);
  layerNorm(sa, sa, A, Cs, F("confidence/plddtNorm/scale"), F("confidence/plddtNorm/offset"));
  plddtAtomK<<<A, 64, 0, STREAM>>>(sa, dSlot, F("confidence/plddtWeight"), pa, A, Cs, bins);
  // PAE logits
  int pb = (int)dimOf("f/confidence/pae", 1);
  float* paeL = scratch<float>("cf.pae", P * pb);
  gemm(z, F("confidence/pae"), paeL, P, C, pb);
  Confidence out;
  out.plddtAtom = download(pa, A);
  std::vector<float> mask_ = download(W("atom_mask"), A);
  std::vector<int> asym(T); CK(cudaMemcpy(asym.data(), Idev("asym_id"), T * 4, cudaMemcpyDeviceToHost));
  out.plddtToken.assign(T, 0.f);
  std::vector<float> cnt(T, 0.f);
  double wsum = 0, wtot = 0;
  for (int a = 0; a < A; ++a) {
    out.plddtToken[a2t[a]] += out.plddtAtom[a] * mask_[a]; cnt[a2t[a]] += mask_[a];
    wsum += out.plddtAtom[a] * mask_[a]; wtot += mask_[a];
  }
  for (int t = 0; t < T; ++t) out.plddtToken[t] /= std::max(cnt[t], 1e-6f);
  out.meanPlddt = wsum / (wtot + 1e-8);
  // PAE and pTM / ipTM, on the device (the logits were 17 MB to download at 261 tokens and 4.4 M
  // exponentials on the host: 50 of the head's 62 ms)
  double width = 32.0 / pb, d0 = 1.24 * cbrt(std::max(T, 19) - 15.0) - 1.8;
  float* paeD = scratch<float>("cf.paeOut", P); float* tmD = scratch<float>("cf.tm", P);
  double* rowsD = scratch<double>("cf.tmRows", (size_t)2 * T);
  paeK<<<blocks(P), 256, 0, STREAM>>>(paeL, paeD, tmD, P, pb, width, d0);
  tmRowsK<<<T, 256, 0, STREAM>>>(tmD, Idev("asym_id"), rowsD, T);
  out.pae = download(paeD, P);
  std::vector<double> rows(2 * T);
  CK(cudaMemcpy(rows.data(), rowsD, rows.size() * 8, cudaMemcpyDeviceToHost));
  double ptm = -1e30, iptm = -1e30;
  for (int i = 0; i < T; ++i) { ptm = std::max(ptm, rows[2 * i]); iptm = std::max(iptm, rows[2 * i + 1]); }
  out.ptm = ptm; out.iptm = iptm;
  if (check) {
    checkOracle("confidence pLDDT per atom", pa, A, "o/conf/plddt_per_atom");
    checkOracle("confidence PAE logits", paeL, P * pb, "o/conf/pae_logits");
    if (M.has("o/conf/ptm")) printf("  pTM %.6f against the oracle's %.6f\n", out.ptm, M.f("o/conf/ptm")[0]);
  }
  if (!inPlace) CK(cudaFree(z));
  CK(cudaFree(dSlot));
  return out;
}

// AlphaFold 3's confidence files beside the PDB, in native/af3's layout (its sampler.cuh) so one reader
// takes all three ports: <stem>_confidences.json - the expected PAE, token_plddts (the head's own pooling
// of its atoms), the token layout (chain letters from asym_id, residue numbers from residue_index) - and
// <stem>_summary_confidences.json (pTM, ipTM for a complex, mean pLDDT). Atom pLDDTs are the PDB's B factors.
inline void writeConfidences(const std::string& pdb, int T, const Confidence& conf, const std::vector<float>& contacts = {}) {
  std::string stem = pdb.size() > 4 && pdb.substr(pdb.size() - 4) == ".pdb" ? pdb.substr(0, pdb.size() - 4) : pdb;
  std::vector<int> asym(T), res(T);
  CK(cudaMemcpy(asym.data(), Idev("asym_id"), T * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(res.data(), Idev("residue_index"), T * 4, cudaMemcpyDeviceToHost));
  auto chainId = [](int a) {
    std::string id; for (a = a - 1; ; a = a / 26 - 1) { id.insert(id.begin(), (char)('A' + a % 26)); if (a < 26) break; }
    return id;
  };
  bool chains = false; for (int t = 1; t < T; ++t) chains |= asym[t] != asym[0];
  int first = *std::min_element(asym.begin(), asym.end());     // (EF2 counts chains and residues from 0;
  FILE* f = fopen((stem + "_confidences.json").c_str(), "w");    //  the PDB from A and 1)
  fprintf(f, "{\"pae\": [");
  for (int i = 0; i < T; ++i) {
    fprintf(f, "%s[", i ? ",\n  " : "");
    for (int j = 0; j < T; ++j) fprintf(f, "%s%.2f", j ? ", " : "", conf.pae[(size_t)i * T + j]);
    fprintf(f, "]");
  }
  fprintf(f, "]");
  if (!contacts.empty()) {
    fprintf(f, ",\n \"contact_probs\": [");
    for (int i = 0; i < T; ++i) {
      fprintf(f, "%s[", i ? ",\n  " : "");
      for (int j = 0; j < T; ++j) fprintf(f, "%s%.2f", j ? ", " : "", contacts[(size_t)i * T + j]);
      fprintf(f, "]");
    }
    fprintf(f, "]");
  }
  fprintf(f, ",\n \"token_plddts\": [");
  for (int i = 0; i < T; ++i) fprintf(f, "%s%.2f", i ? ", " : "", 100.f * conf.plddtToken[i]);
  fprintf(f, "],\n \"token_chain_ids\": [");
  for (int i = 0; i < T; ++i) fprintf(f, "%s\"%s\"", i ? ", " : "", chainId(asym[i] - first + 1).c_str());
  fprintf(f, "],\n \"token_res_ids\": [");
  for (int i = 0; i < T; ++i) fprintf(f, "%s%d", i ? ", " : "", res[i] + 1);
  fprintf(f, "]}\n");
  fclose(f);
  f = fopen((stem + "_summary_confidences.json").c_str(), "w");
  if (chains) fprintf(f, "{\"ptm\": %.4f, \"iptm\": %.4f, \"mean_plddt\": %.2f}\n", conf.ptm, conf.iptm, 100 * conf.meanPlddt);
  else fprintf(f, "{\"ptm\": %.4f, \"iptm\": null, \"mean_plddt\": %.2f}\n", conf.ptm, 100 * conf.meanPlddt);
  fclose(f);
}
