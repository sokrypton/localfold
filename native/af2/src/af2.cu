// AlphaFold 2 in CUDA: af3-any-model's AF2 multimer graph (one graph for monomer and multimer
// checkpoints, as the reference runs them), from native/af2/export_input.mjs's features and
// native/af2/export_weights.py's weights.
//
//   af2 <input dir> --weights=<dir> [--oracle=<dir>] [--out=fold.pdb] [--recycles=N]
//
// With --oracle (native/af2/oracle.py's dump of the reference on this same input), pass 0 is checked
// against it stage by stage.
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include "templates.cuh"
#include "structure.cuh"
#include "../../af3/src/profile.cuh"

static const char* RESTYPE3[21] = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
                                   "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK"};
static const char* ATOM37[37] = {"N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2",
                                 "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1",
                                 "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT"};

// mean over bins of softmax(logits) * centres, per row
static std::vector<float> expectation(const std::vector<float>& logits, size_t rows, int bins, const std::vector<float>& centres) {
  std::vector<float> out(rows);
  for (size_t r = 0; r < rows; ++r) {
    const float* l = logits.data() + r * bins;
    float mx = -INFINITY; for (int b = 0; b < bins; ++b) mx = std::max(mx, l[b]);
    double s = 0, e = 0;
    for (int b = 0; b < bins; ++b) { double p = std::exp(l[b] - mx); s += p; e += p * centres[b]; }
    out[r] = (float)(e / s);
  }
  return out;
}
// AF2's predicted TM-score from the PAE logits (confidence.compute_tm over every residue)
static float predictedTm(const std::vector<float>& logits, int L, int bins, const std::vector<int>* asym, bool interface) {
  float maxBin = 31.f, step = maxBin / (bins - 2);
  std::vector<float> centres(bins);
  for (int b = 0; b < bins - 1; ++b) centres[b] = b * step + step / 2;
  centres[bins - 1] = centres[bins - 2] + step;
  float d0 = 1.24f * std::cbrt((float)std::max(L, 19) - 15.f) - 1.8f;
  float best = 0;
  for (int i = 0; i < L; ++i) {
    double sum = 0; double count = 0;
    for (int j = 0; j < L; ++j) {
      if (interface && asym && (*asym)[i] == (*asym)[j]) continue;
      const float* l = logits.data() + ((size_t)i * L + j) * bins;
      float mx = -INFINITY; for (int b = 0; b < bins; ++b) mx = std::max(mx, l[b]);
      double s = 0, e = 0;
      for (int b = 0; b < bins; ++b) { double p = std::exp(l[b] - mx); s += p; e += p / (1 + (centres[b] / d0) * (centres[b] / d0)); }
      sum += e / s; count += 1;
    }
    if (count > 0) best = std::max(best, (float)(sum / (count + 1e-8)));
  }
  return best;
}
// logits[i, j] = half[i, j] + half[j, i]
__global__ void symmetriseK(const float* half_, float* out, int L, int B) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * B) return;
  int b = (int)(t % B); size_t ij = t / B; int i = (int)(ij / L), j = (int)(ij % L);
  out[t] = half_[t] + half_[((size_t)j * L + i) * B + b];
}

// a synthetic input of the given shapes (what export_input.mjs writes, its values arbitrary), in
// /dev/shm, for --warm: folding it loads what the real input's fold will need while that is exported
static std::string writeWarmInput(int L, int N, int E, int T) {
  std::string dir = "/dev/shm/af2-warm-" + std::to_string(getpid());
  if (mkdir(dir.c_str(), 0755) && errno != EEXIST) { fprintf(stderr, "cannot make %s\n", dir.c_str()); exit(1); }
  FILE* bin = fopen((dir + "/model.bin").c_str(), "wb");
  std::string idx; size_t offset = 0;
  auto put = [&](char kind, const std::string& name, size_t n, float fv, bool cycle = false) {
    std::vector<uint32_t> v(n);
    for (size_t i = 0; i < n; ++i) {
      if (kind == 'i') { int x = cycle ? (int)(i % 20) : (int)fv; memcpy(&v[i], &x, 4); }
      else { float x = fv; memcpy(&v[i], &x, 4); }
    }
    fwrite(v.data(), 4, n, bin);
    idx += std::string(1, kind) + " " + name + " " + std::to_string(offset) + " " + std::to_string(n) + "\n";
    offset += n;
  };
  put('i', "aatype", L, 0, true);
  { std::vector<int> ri(L); for (int i = 0; i < L; ++i) ri[i] = i;
    fwrite(ri.data(), 4, L, bin); idx += "i residue_index " + std::to_string(offset) + " " + std::to_string(L) + "\n"; offset += L; }
  put('t', "seq_mask", L, 1);
  put('i', "asym_id", L, 0); put('i', "entity_id", L, 0); put('i', "sym_id", L, 0);
  put('t', "f0/msa_feat", (size_t)N * L * 49, 0); put('t', "f0/msa_mask", (size_t)N * L, 1);
  put('i', "f0/extra_msa", (size_t)E * L, 0); put('t', "f0/extra_has_deletion", (size_t)E * L, 0);
  put('t', "f0/extra_deletion_value", (size_t)E * L, 0); put('t', "f0/extra_msa_mask", (size_t)E * L, 1);
  if (T > 0) {
    put('i', "t/aatype", (size_t)T * L, 0, true); put('t', "t/positions", (size_t)T * L * 37 * 3, 0);
    put('t', "t/mask", (size_t)T * L * 37, 1);
    idx += "m meta/templates " + std::to_string(T) + "\n";
  }
  idx += "m meta/tokens " + std::to_string(L) + "\nm meta/msa_rows " + std::to_string(N) + "\nm meta/extra_rows " +
         std::to_string(E) + "\nm meta/passes 1\n";
  fclose(bin);
  FILE* f = fopen((dir + "/model.idx").c_str(), "w"); fputs(idx.c_str(), f); fclose(f);
  return dir;
}
// one input, already loaded: the passes, the heads and the PDB. warm: the embedder, ONE block of
// each stack, the structure module and the heads, nothing written - every kernel loaded, every
// cuBLASLt plan and scratch buffer made, at this input's shapes
static int foldInput(const std::string& oracle, const std::string& out, int recycles, bool profile, bool warm,
                     std::chrono::steady_clock::time_point t0) {
  Trunk t{};
  t.L = (int)M.meta("meta/tokens"); t.N = (int)M.meta("meta/msa_rows"); t.E = (int)M.meta("meta/extra_rows");
  t.opmFirst = M.flag("meta/opm_first");
  int templates = (int)M.meta("meta/templates", 0);
  bool multimer = M.flag("meta/multimer");
  bool monomerTemplates = !multimer && templates > 0 && M.has("w/evoformer/template_embedding/attention/query_w");
  if (!multimer && templates > 0 && !monomerTemplates) {
    fprintf(stderr, "this checkpoint has no template embedder (models 3 to 5 are template-free): fold without --template\n");
    return 1;
  }
  if (multimer || monomerTemplates) t.T = templates;   // a template's single features become MSA rows
  int passes = (int)M.meta("meta/passes");
  if (recycles >= 0) passes = std::min(passes, recycles + 1);
  float positionScale = (float)M.meta("meta/position_scale");
  int L = t.L; size_t pairs = (size_t)L * L;
  t.msa = dalloc((size_t)(t.N + t.T) * L * 256); t.extra = dalloc((size_t)t.E * L * 64);
  t.pair = dalloc(pairs * 128); t.pairMask = dalloc(pairs);
  float* prevRow = dalloc((size_t)L * 256); float* prevPair = dalloc(pairs * 128); float* prevPos = dalloc((size_t)L * 37 * 3);
  CK(cudaMemset(prevRow, 0, (size_t)L * 256 * 4)); CK(cudaMemset(prevPair, 0, pairs * 128 * 4));
  CK(cudaMemset(prevPos, 0, (size_t)L * 37 * 3 * 4));
  float* single = dalloc((size_t)L * 384);
  if (!warm) printf("AF2 %s: %d residues, %d MSA rows, %d extra, %d passes (loaded in %.1f s)\n", M.flag("meta/multimer") ? "multimer" : "monomer",
         L, t.N, t.E, passes, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  int extraBlocks = warm ? 1 : (int)dimW("evoformer/extra_msa_stack/msa_transition/transition1/weights", 0);
  int mainBlocks = warm ? 1 : (int)dimW("evoformer/evoformer_iteration/msa_transition/transition1/weights", 0);
  if (warm) passes = 1;
  StructureOut so{};
  float* plddtLogits = nullptr; float* paeLogits = nullptr;
  if (profile && !warm) { prof::init(); prof::start(); }
  auto tf = std::chrono::steady_clock::now();
  // everything after the embedder: the same launches on the same buffers every pass, so from pass 1
  // on it is captured once as a CUDA graph and replayed (a pass is ~6000 launches)
  auto rest = [&](bool check, int pass) {
    auto mark = [&](const char* what) {
      if (!getenv("AF2_STAGE_TIMES")) return;
      CK(cudaStreamSynchronize(STREAM));
      printf("    pass %d %-14s %.1f ms\n", pass, what, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tf).count());
    };
    mark("embed");
    if (multimer) templateEmbedding(t.pair, t.pairMask, L);
    else if (monomerTemplates) templateEmbeddingMonomer(t.pair, t.pairMask, L, templates);
    if (t.T > 0) {
      templateRows(L, t.T, multimer, t.msa + (size_t)t.N * L * 256, const_cast<float*>(t.msaMask) + (size_t)t.N * L);
      if (check && !multimer) {
        checkOracle("template torsion features", scratch<float>("trow.feat", 1), (size_t)t.T * L * 57, "o/tmpl/feat");
        if (getenv("AF2_DUMP_TFEAT")) {
          auto h = download(scratch<float>("trow.feat", 1), (size_t)t.T * L * 57);
          FILE* df = fopen(getenv("AF2_DUMP_TFEAT"), "wb"); fwrite(h.data(), 4, h.size(), df); fclose(df);
        }
      }
    }
    if (check) {
      printf("pass 0 against the reference:\n");
      checkOracle("embed msa", t.msa, (size_t)t.N * L * 256, "o/embed/msa");
      checkOracle("embed pair", t.pair, pairs * 128, "o/embed/pair");
    }
    for (int b = 0; b < extraBlocks; ++b) {
      evoformerBlock(t, true, b);
      if (b == 0) mark("extra 0");
      if (check && b == 0) checkOracle("extra block 0 pair", t.pair, pairs * 128, "o/extra1/pair");
    }
    if (check) checkOracle("extra stack pair", t.pair, pairs * 128, "o/extra/pair");
    mark("extra stack");
    for (int b = 0; b < mainBlocks; ++b) {
      if (b == 1) mark("evoformer 0");
      if (b == 2) mark("evoformer 1");
      evoformerBlock(t, false, b);
      if (check && b == 0) {
        checkOracle("evoformer block 0 msa", t.msa, (size_t)t.N * L * 256, "o/evo1/msa");
        checkOracle("evoformer block 0 pair", t.pair, pairs * 128, "o/evo1/pair");
      }
    }
    linearB(t.msa, "evoformer/single_activations", -1, single, L, 256, 384);
    if (check) {
      checkOracle("evoformer msa first row", t.msa, (size_t)L * 256, "o/full/msa_first_row");
      checkOracle("evoformer pair", t.pair, pairs * 128, "o/full/pair");
      checkOracle("single", single, (size_t)L * 384, "o/full/single");
    }
    mark("evoformer"); memReport("evoformer"); so = structureModule(single, t.pair, L, positionScale);
    if (check) {
      checkOracle("structure act", so.act, (size_t)L * 384, "o/full/structure_act");
      checkOracle("angles", so.angles, (size_t)L * 14, "o/full/angles");
      checkOracle("atom14 positions", so.pos14, (size_t)L * 14 * 3, "o/full/final_atom14_positions");
      checkOracle("atom37 positions", so.pos37, (size_t)L * 37 * 3, "o/full/final_atom_positions");
    }
    {   // a multi-pass oracle (oracle.py --passes): every pass's pair and structure
      std::string o = "o/pass" + std::to_string(pass) + "/";
      if (!oracle.empty() && M.has(o + "final_atom_positions")) {
        printf("pass %d:\n", pass);
        checkOracle("pair", t.pair, pairs * 128, o + "pair");
        checkOracle("atom37 positions", so.pos37, (size_t)L * 37 * 3, o + "final_atom_positions");
      }
    }
    mark("structure");
    // the recycled state: the evoformer's first MSA row and pair, the final atom37 positions
    CK(cudaMemcpyAsync(prevRow, t.msa, (size_t)L * 256 * 4, cudaMemcpyDeviceToDevice, STREAM));
    CK(cudaMemcpyAsync(prevPair, t.pair, pairs * 128 * 4, cudaMemcpyDeviceToDevice, STREAM));
    CK(cudaMemcpyAsync(prevPos, so.pos37, (size_t)L * 37 * 3 * 4, cudaMemcpyDeviceToDevice, STREAM));
    // heads, on this pass's representations
    const std::string PL = "predicted_lddt_head/";
    float* a = scratch<float>("head.a", (size_t)L * 384); float* h1 = scratch<float>("head.h1", (size_t)L * 128);
    float* h2 = scratch<float>("head.h2", (size_t)L * 128);
    plddtLogits = scratch<float>("head.plddt", (size_t)L * 50);
    layerNorm(so.act, a, L, 384, PL + "input_layer_norm");
    linearB(a, PL + "act_0", -1, h1, L, 384, 128, true);
    linearB(h1, PL + "act_1", -1, h2, L, 128, 128, true);
    linearB(h2, PL + "logits", -1, plddtLogits, L, 128, 50);
    paeLogits = scratch<float>("head.pae", pairs * 64);
    linearB(t.pair, "predicted_aligned_error_head/logits", -1, paeLogits, pairs, 128, 64);
    if (check) {
      checkOracle("pLDDT logits", plddtLogits, (size_t)L * 50, "o/full/plddt_logits");
      checkOracle("PAE logits", paeLogits, pairs * 64, "o/full/pae_logits");
      float* dh = scratch<float>("head.dgramHalf", pairs * 64); float* dg = scratch<float>("head.dgram", pairs * 64);
      linearB(t.pair, "distogram_head/half_logits", -1, dh, pairs, 128, 64);
      symmetriseK<<<blocks(pairs * 64), 256, 0, STREAM>>>(dh, dg, L, 64);
      checkOracle("distogram logits", dg, pairs * 64, "o/full/distogram_logits");
    }
  };
  cudaGraphExec_t graph = nullptr;
  // capturing costs about half a replayed pass and a replay saves a few percent of one (59 residues:
  // 100 against 107 ms), so only a long recycle run gains - as native/af3's trunk graph rule
  bool graphs = !warm && !getenv("AF2_NO_GRAPHS") && oracle.empty() && (passes >= 8 || getenv("AF2_GRAPHS"));
  for (int pass = 0; pass < passes; ++pass) {
    bool check = pass == 0 && !oracle.empty();
    embed(t, pass, prevRow, prevPair, prevPos);
    if (!graphs || pass == 0) {
      rest(check, pass);
      if (getenv("AF2_PASS_TIMES")) {
        CK(cudaStreamSynchronize(STREAM));
        printf("  pass %d done at %.1f ms\n", pass, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tf).count());
      }
      continue;
    }
    if (!graph) {
      cudaGraph_t g;
      CK(cudaStreamBeginCapture(STREAM, cudaStreamCaptureModeThreadLocal));
      rest(false, pass);
      CK(cudaStreamEndCapture(STREAM, &g));
      CK(cudaGraphInstantiate(&graph, g, 0));
      CK(cudaGraphDestroy(g));
    }
    CK(cudaGraphLaunch(graph, STREAM));
    if (getenv("AF2_PASS_TIMES")) {
      CK(cudaStreamSynchronize(STREAM));
      printf("  pass %d done at %.1f ms\n", pass, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tf).count());
    }
  }
  CK(cudaStreamSynchronize(STREAM));
  double foldMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tf).count();
  if (warm) {
    for (float* p : {t.msa, t.extra, t.pair, t.pairMask, prevRow, prevPair, prevPos, single}) CK(cudaFree(p));
    return 0;
  }
  if (profile) prof::stop(30);
  std::vector<float> pl = download(plddtLogits, (size_t)L * 50);
  std::vector<float> centres(50); for (int b = 0; b < 50; ++b) centres[b] = (b + 0.5f) * 2.f;
  std::vector<float> plddt = expectation(pl, L, 50, centres);
  std::vector<float> pae = download(paeLogits, pairs * 64);
  std::vector<int> asym(L); CK(cudaMemcpy(asym.data(), Idev("asym_id"), L * 4, cudaMemcpyDeviceToHost));
  float ptm = predictedTm(pae, L, 64, &asym, false);
  bool chains = false; for (int i = 1; i < L; ++i) chains |= asym[i] != asym[0];
  float iptm = chains ? predictedTm(pae, L, 64, &asym, true) : NAN;
  double mean = 0; for (float v : plddt) mean += v; mean /= L;
  // the PDB
  std::vector<float> pos = download(so.pos37, (size_t)L * 37 * 3);
  std::vector<int> aatype(L), ri(L);
  CK(cudaMemcpy(aatype.data(), Idev("aatype"), L * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(ri.data(), Idev("residue_index"), L * 4, cudaMemcpyDeviceToHost));
  int firstAsym = *std::min_element(asym.begin(), asym.end());
  const float* mask37 = M.f("c/atom37_mask");
  FILE* f = fopen(out.c_str(), "w");
  int serial = 1;
  for (int i = 0; i < L; ++i) {
    int aa = std::min(std::max(aatype[i], 0), 19);
    char chain = (char)('A' + std::min(asym[i] - firstAsym, 25));     // asym ids count from the first chain's
    for (int a = 0; a < 37; ++a) {
      if (mask37[aa * 37 + a] == 0) continue;
      const float* p = &pos[((size_t)i * 37 + a) * 3];
      fprintf(f, "ATOM  %5d %-4s %3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f           %c\n", serial++,
              strlen(ATOM37[a]) < 4 ? (std::string(" ") + ATOM37[a]).c_str() : ATOM37[a], RESTYPE3[aatype[i] > 19 ? 20 : aa],
              chain, ri[i] + 1, p[0], p[1], p[2], 1.0, plddt[i], ATOM37[a][0]);
    }
  }
  fprintf(f, "END\n"); fclose(f);
  printf("mean pLDDT %.2f  pTM %.4f", mean, ptm);
  if (chains) printf("  ipTM %.4f", iptm);
  printf("  -> %s  (%d passes, %.1f ms)\n", out.c_str(), passes, foldMs);
  return 0;
}

// --detach-output: on success the last line is "af2: done" and stdout closes, so a caller reading it to
// its end returns while the driver releases this process's device (0.16 s of exit; native/af2/fold does)
static bool DETACH = false;
int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: af2 <input dir> --weights=<dir> [--oracle=<dir>] [--out=fold.pdb] [--recycles=N]\n"); return 1; }
  std::string weights, oracle, out = "fold.pdb", warmShape; int recycles = -1; bool profile = false, waitInput = false;
  for (int i = 2; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--oracle=", 9)) oracle = argv[i] + 9;
    else if (!strncmp(argv[i], "--out=", 6)) out = argv[i] + 6;
    else if (!strncmp(argv[i], "--recycles=", 11)) recycles = atoi(argv[i] + 11);
    else if (!strcmp(argv[i], "--profile")) profile = true;
    else if (!strcmp(argv[i], "--fast")) FAST = true;
    else if (!strcmp(argv[i], "--wait-input")) waitInput = true;     // start up while the input is still being exported
    else if (!strcmp(argv[i], "--detach-output")) DETACH = true;
    else if (!strncmp(argv[i], "--warm=", 7)) warmShape = argv[i] + 7;   // L,N,E,T: warm up at those shapes meanwhile
    else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 1; }
  }
  if (weights.empty()) { fprintf(stderr, "--weights=<dir> (native/af2/export_weights.py)\n"); return 1; }
  auto t0 = std::chrono::steady_clock::now();
  M.load(weights);
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  bool tf32 = FAST && !getenv("AF2_NO_TF32");
  if (getenv("AF2_NO_FLASH")) FAST = false;
  CB(cublasSetMathMode(H, tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));   // float32 means float32 unless --fast
  M.upload(0);
  if (!warmShape.empty()) {
    int wl = 0, wn = 1, we = 1, wt = 0;
    if (sscanf(warmShape.c_str(), "%d,%d,%d,%d", &wl, &wn, &we, &wt) < 1 || wl < 1) { fprintf(stderr, "--warm=L,N,E,T\n"); return 1; }
    std::string dir = writeWarmInput(wl, std::max(wn, 1), std::max(we, 1), wt);
    int seg = (int)M.segs.size();
    M.load(dir);
    foldInput("", "", 0, false, true, t0);
    CK(cudaStreamSynchronize(STREAM));
    forgetEntries(M.unload(seg));
    std::string rm = "rm -rf '" + dir + "'"; if (system(rm.c_str())) {}
  }
  if (waitInput) {          // the exporter writes model.idx last, by a rename
    std::string idx = std::string(argv[1]) + "/model.idx", failed = std::string(argv[1]) + "/model.failed";
    for (int k = 0; access(idx.c_str(), R_OK) != 0; ++k) {
      if (access(failed.c_str(), F_OK) == 0) { fprintf(stderr, "af2: the input's export failed\n"); return 1; }
      if (k > 600000) { fprintf(stderr, "no %s after ten minutes\n", idx.c_str()); return 1; }
      usleep(1000);
    }
  }
  M.load(argv[1]);
  if (!oracle.empty()) M.load(oracle);
  int rc = foldInput(oracle, out, recycles, profile, false, t0);
  if (DETACH && rc == 0) { printf("af2: done\n"); fflush(stdout); fflush(stderr); fclose(stdout); }
  finish(rc);
}
