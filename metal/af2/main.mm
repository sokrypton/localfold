// localfold-af2 on Metal: AlphaFold 2 (monomer and multimer checkpoints, models 1-5 - 2 to 5 as the page's deltas on
// model 1) natively over cuda/featurise's input, the page's bundle read through its weight walk.
//
//   localfold-af2 <input dir> --bundle=<dir> [--delta=<dir>] [--out=fold.pdb] [--recycles=N] [--tolerance=A]
#include "af2.h"
#include "af2_weights.h"
#include "host.h"
#include "standalone_api.h"
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <cstring>
#include <fstream>

extern const char* PORT_SOURCE;

namespace {
const char* RESTYPE3[21] = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
                            "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK"};
const char* ATOM37[37] = {"N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2",
                          "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1",
                          "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT"};
double ms(double since) { return (now() - since) * 1e3; }
std::string FRAMES_DIR;
double TOLERANCE = 0;

void appendMatrix2(std::string& j, const float* m, int L) {
  j.reserve(j.size() + (size_t)L * L * 7 + 16);
  j += "[";
  for (int i = 0; i < L; ++i) {
    j += i ? ",\n  [" : "[";
    for (int c = 0; c < L; ++c) {
      if (c) j += ", ";
      char b[32]; int n = snprintf(b, sizeof b, "%.2f", m[(size_t)i * L + c]); j.append(b, n);
    }
    j += "]";
  }
  j += "]";
}
// the early stop's measure (ColabFold's compute_tol): the RMS change of every C-alpha pair distance between two passes
double caPairChange(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& mask, int L) {
  double sum = 0, weights = 0;
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j) {
      double w = (double)mask[i] * mask[j];
      if (w == 0) continue;
      double pa = 0, pb = 0;
      for (int x = 0; x < 3; ++x) {
        double da = a[((size_t)i * 37 + 1) * 3 + x] - a[((size_t)j * 37 + 1) * 3 + x];
        double db = b[((size_t)i * 37 + 1) * 3 + x] - b[((size_t)j * 37 + 1) * 3 + x];
        pa += da * da; pb += db * db;
      }
      double d = std::sqrt(pa) - std::sqrt(pb);
      sum += d * d * w; weights += w;
    }
  return std::sqrt(sum / weights + 1e-8);
}
std::vector<float> plddtOf(const float* logits, int L) {
  std::vector<float> out(L);
  for (int i = 0; i < L; ++i) {
    const float* l = logits + i * 50; float mx = -INFINITY; for (int b = 0; b < 50; ++b) mx = std::max(mx, l[b]);
    double s = 0, e = 0; for (int b = 0; b < 50; ++b) { double p = std::exp(l[b] - mx); s += p; e += p * (b + 0.5) * 2; }
    out[i] = (float)(e / s);
  }
  return out;
}
double tmScore(const float* tm, int L, const std::vector<int>& asym, bool interface) {
  double best = 0;
  for (int i = 0; i < L; ++i) {
    double sum = 0, n = 0;
    for (int j = 0; j < L; ++j) { if (interface && asym[i] == asym[j]) continue; sum += tm[(size_t)i * L + j]; n += 1; }
    if (n > 0) best = std::max(best, sum / (n + 1e-8));
  }
  return best;
}

std::string pdbText(const std::vector<float>& pos, const std::vector<float>& plddt, int L, const std::vector<int>& aatype,
                    const std::vector<int>& asym, const std::vector<int>& ri, const float* mask37, const double* R = nullptr,
                    const double* centre = nullptr, const double* ref = nullptr) {
  int firstAsym = *std::min_element(asym.begin(), asym.end());
  std::string out; char line[128]; int serial = 1;
  for (int i = 0; i < L; ++i) {
    int aa = std::min(std::max(aatype[i], 0), 19);
    char chain = (char)('A' + std::min(asym[i] - firstAsym, 25));
    for (int a = 0; a < 37; ++a) {
      if (mask37[aa * 37 + a] == 0) continue;
      const float* p = &pos[((size_t)i * 37 + a) * 3];
      double q[3] = {p[0], p[1], p[2]};
      if (R) {
        double c[3] = {p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]};
        for (int d = 0; d < 3; ++d) q[d] = R[d * 3] * c[0] + R[d * 3 + 1] * c[1] + R[d * 3 + 2] * c[2] + ref[d];
      }
      snprintf(line, sizeof line, "ATOM  %5d %-4s %3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f           %c\n", serial++,
               strlen(ATOM37[a]) < 4 ? (std::string(" ") + ATOM37[a]).c_str() : ATOM37[a], RESTYPE3[aatype[i] > 19 ? 20 : aa],
               chain, ri[i] + 1, q[0], q[1], q[2], 1.0, plddt[i], ATOM37[a][0]);
      out += line;
    }
  }
  out += "END\n";
  return out;
}

int foldInput(const std::string& dir, const std::string& out, int recycles) {
  double t0 = now();
  Trunk t{};
  t.L = (int)M.meta("meta/tokens"); t.N = (int)M.meta("meta/msa_rows"); t.E = (int)M.meta("meta/extra_rows");
  t.opmFirst = M.meta("meta/opm_first", 0) != 0;
  int templates = (int)M.meta("meta/templates", 0);
  bool multimer = M.meta("meta/multimer", 0) != 0;
  bool monomerTemplates = !multimer && templates > 0 && M.has("w/evoformer/template_embedding/attention/query_w");
  if (!multimer && templates > 0 && !monomerTemplates)
    die("this checkpoint has no template embedder (models 3 to 5 are template-free): fold without --template");
  if ((multimer || monomerTemplates) && M.has("w/evoformer/template_single_embedding/weights")) t.T = templates;
  int passes = (int)M.meta("meta/passes");
  if (recycles >= 0) passes = std::min(passes, recycles + 1);
  float positionScale = (float)M.meta("meta/position_scale");
  int L = t.L; size_t pairs = (size_t)L * L;
  t.msa = allocT<float>((size_t)(t.N + t.T) * L * 256); t.extra = allocT<float>((size_t)std::max(t.E, 1) * L * 64);
  t.pair = allocT<float>(pairs * 128); t.pairMask = allocT<float>(pairs);
  t.msaMask = allocT<float>((size_t)(t.N + t.T) * L); t.extraMask = allocT<float>((size_t)std::max(t.E, 1) * L);
  float* prevRow = allocT<float>((size_t)L * 256); float* prevPair = allocT<float>(pairs * 128);
  float* prevPos = allocT<float>((size_t)L * 37 * 3);
  float* single = allocT<float>((size_t)L * 384);
  printf("AF2 %s: %d residues, %d MSA rows, %d extra, %d passes\n", multimer ? "multimer" : "monomer", L, t.N, t.E, passes);
  int extraBlocks = (int)dimW("evoformer/extra_msa_stack/msa_transition/transition1/weights", 0);
  int mainBlocks = (int)dimW("evoformer/evoformer_iteration/msa_transition/transition1/weights", 0);
  const bool distogram = M.has("w/distogram_head/half_logits/weights");
  std::vector<int> aatype(M.hostI("aatype"), M.hostI("aatype") + L), ri(M.hostI("residue_index"), M.hostI("residue_index") + L),
                   asym(M.hostI("asym_id"), M.hostI("asym_id") + L);
  std::vector<float> seqMask(M.hostF("seq_mask"), M.hostF("seq_mask") + L);
  const float* mask37 = M.hostF("c/atom37_mask");
  bool chains = false; for (int i = 1; i < L; ++i) chains |= asym[i] != asym[0];
  float d0 = 1.24f * std::cbrt((float)std::max(L, 19) - 15.f) - 1.8f;
  StructureOut so{};
  float* plddtLogits = allocT<float>((size_t)L * 50);
  float* paeLogits = allocT<float>(pairs * 64);
  float* paeD = allocT<float>(pairs); float* tmD = allocT<float>(pairs);
  std::vector<float> previous37, lastPos;
  std::vector<double> passRef; double passCentre[3] = {};
  int ran = passes; double converged = -1;
  double tf = now();
  for (int pass = 0; pass < passes; ++pass) {
    const double passStart = now();
    if (pass == 0 && profiling()) profileStart();
    embed(t, pass, prevRow, prevPair, prevPos);
    if (multimer) templateEmbedding(t.pair, t.pairMask, L);
    else if (monomerTemplates) templateEmbeddingMonomer(t.pair, t.pairMask, L, templates);
    if (t.T > 0) templateRows(L, t.T, multimer, t.msa + (size_t)t.N * L * 256, t.msaMask + (size_t)t.N * L);
    for (int b = 0; b < extraBlocks; ++b) evoformerBlock(t, true, b);
    for (int b = 0; b < mainBlocks; ++b) evoformerBlock(t, false, b);
    linearB(t.msa, "evoformer/single_activations", -1, single, L, 256, 384);
    so = structureModule(single, t.pair, L, positionScale);
    // the recycled state: the evoformer's first MSA row and pair, the final atom37 positions
    copy(prevRow, t.msa, (size_t)L * 256 * 4);
    copy(prevPair, t.pair, pairs * 128 * 4);
    copy(prevPos, so.pos37, (size_t)L * 37 * 3 * 4);
    // the heads, on this pass's representations
    const std::string PL = "predicted_lddt_head/";
    float* a = scratch<float>("head.a", (size_t)L * 384); float* h1 = scratch<float>("head.h1", (size_t)L * 128);
    float* h2 = scratch<float>("head.h2", (size_t)L * 128);
    layerNormW(so.act, a, L, 384, PL + "input_layer_norm");
    linearB(a, PL + "act_0", -1, h1, L, 384, 128, true);
    linearB(h1, PL + "act_1", -1, h2, L, 128, 128, true);
    linearB(h2, PL + "logits", -1, plddtLogits, L, 128, 50);
    linearB(t.pair, "predicted_aligned_error_head/logits", -1, paeLogits, pairs, 128, 64);
    run1d("af2_pae_tm", pairs, PaeTmArgs{paeLogits, paeD, tmD, pairs, d0, 0});
    if (pass == 0 && profiling()) profileReport("pass 0", 30);
    // this pass's line (and its files, with --frames)
    std::vector<float> pos = download(so.pos37, (size_t)L * 37 * 3);
    std::vector<float> plddt = plddtOf(download(plddtLogits, (size_t)L * 50).data(), L);
    std::vector<float> tm = download(tmD, pairs);
    double mean = 0; for (float v : plddt) mean += v / L;
    double ptm = tmScore(tm.data(), L, asym, false), iptm = chains ? tmScore(tm.data(), L, asym, true) : -1;
    char said[200];
    int n = snprintf(said, sizeof said, "  pass %d/%d: mean pLDDT %.2f  pTM %.4f", pass + 1, passes, mean, ptm);
    if (iptm >= 0) n += snprintf(said + n, sizeof said - n, "  ipTM %.4f", iptm);
    if (!lastPos.empty()) n += snprintf(said + n, sizeof said - n, "  moved %.2f A", caPairChange(lastPos, pos, seqMask, L));
    snprintf(said + n, sizeof said - n, "  %.1f ms", ms(passStart));      // (the pass's own time: the dev report reads it)
    printf("%s\n", said); fflush(stdout);
    if (!FRAMES_DIR.empty()) {
      char tag[32]; snprintf(tag, sizeof tag, "%02d-of-%02d", pass, passes);
      std::string base = FRAMES_DIR + "/", id = tag;
      std::vector<double> pts; double c[3] = {};
      for (int i = 0; i < L; ++i) {
        int aa = std::min(std::max(aatype[i], 0), 19);
        for (int q = 0; q < 37; ++q) if (mask37[aa * 37 + q] != 0) for (int d = 0; d < 3; ++d) pts.push_back(pos[((size_t)i * 37 + q) * 3 + d]);
      }
      size_t na = pts.size() / 3;
      for (size_t k = 0; k < na; ++k) for (int d = 0; d < 3; ++d) c[d] += pts[k * 3 + d] / na;
      for (size_t k = 0; k < na; ++k) for (int d = 0; d < 3; ++d) pts[k * 3 + d] -= c[d];
      if (passRef.empty()) { passRef = pts; for (int d = 0; d < 3; ++d) passCentre[d] = c[d]; }
      double R[9]; bestRotation(pts, passRef, R);
      std::vector<float> pae = download(paeD, pairs);
      std::vector<unsigned char> pae8(pairs);
      for (size_t k = 0; k < pairs; ++k) pae8[k] = (unsigned char)std::min(255.f, std::max(0.f, std::round(pae[k] / 0.125f)));
      writeWhole(base + "pae-" + id + ".u8", pae8.data(), pairs);
      if (distogram) {
        float* dh = scratch<float>("head.dgramHalf", pairs * 64); float* dg = scratch<float>("head.dgram", pairs * 64);
        float* cp = scratch<float>("head.contact", pairs);
        linearB(t.pair, "distogram_head/half_logits", -1, dh, pairs, 128, 64);
        run1d("af2_symmetrise", pairs * 64, SymmetriseArgs{dh, dg, (uint)L, 64});
        run1d("af2_contact8", pairs, Contact8Args{dg, cp, pairs});
        std::vector<float> ch = download(cp, pairs);
        std::vector<unsigned char> c8(pairs);
        for (size_t k = 0; k < pairs; ++k) c8[k] = (unsigned char)std::min(255.f, std::max(0.f, std::round(ch[k] * 255.f)));
        writeWhole(base + "contacts-" + id + ".u8", c8.data(), pairs);
      }
      std::string js = "{\"meanPlddt\": " + std::to_string(mean) + ", \"ptm\": " + std::to_string(ptm)
                       + (iptm >= 0 ? ", \"iptm\": " + std::to_string(iptm) : std::string()) + ", \"plddt\": [";
      for (int i = 0; i < L; ++i) { char v[16]; snprintf(v, sizeof v, "%s%.2f", i ? ", " : "", plddt[i]); js += v; }
      js += "]}\n";
      writeWhole(base + "pass-" + id + ".json", js.data(), js.size());
      std::string text = pdbText(pos, plddt, L, aatype, asym, ri, mask37, R, c, passCentre);
      writeWhole(base + "pass-" + id + ".pdb", text.data(), text.size());     // (last: a reader keys on it)
    }
    lastPos = pos;
    if (TOLERANCE > 0) {           // the early stop
      if (!previous37.empty() && pass > 0) {
        double change = caPairChange(previous37, pos, seqMask, L);
        if (change < TOLERANCE) { converged = change; ran = pass + 1; previous37 = pos; break; }
      }
      previous37 = pos;
    }
  }
  double foldMs = ms(tf);
  // the final pass's results
  std::vector<float> plddt = plddtOf(download(plddtLogits, (size_t)L * 50).data(), L);
  std::vector<float> pae = download(paeD, pairs), tm = download(tmD, pairs);
  float ptm = (float)tmScore(tm.data(), L, asym, false), iptm = chains ? (float)tmScore(tm.data(), L, asym, true) : NAN;
  double mean = 0; for (float v : plddt) mean += v; mean /= L;
  std::vector<float> pos = download(so.pos37, (size_t)L * 37 * 3);
  std::string text = pdbText(pos, plddt, L, aatype, asym, ri, mask37);
  writeWhole(out, text.data(), text.size());
  // the confidence files (AlphaFold 3's layout): atom pLDDTs in the PDB's order, the expected PAE, the contact map where
  // the weights carry the distogram head, the token layout; the summary
  {
    std::string stem = out.size() > 4 && out.substr(out.size() - 4) == ".pdb" ? out.substr(0, out.size() - 4) : out;
    int firstAsym = *std::min_element(asym.begin(), asym.end());
    auto chainId = [&](int i) { return (char)('A' + std::min(asym[i] - firstAsym, 25)); };
    std::vector<float> contact;
    if (distogram) {
      float* dh = scratch<float>("head.dgramHalf", pairs * 64); float* dg = scratch<float>("head.dgram", pairs * 64);
      float* cp = scratch<float>("head.contact", pairs);
      linearB(t.pair, "distogram_head/half_logits", -1, dh, pairs, 128, 64);
      run1d("af2_symmetrise", pairs * 64, SymmetriseArgs{dh, dg, (uint)L, 64});
      run1d("af2_contact8", pairs, Contact8Args{dg, cp, pairs});
      contact = download(cp, pairs);
    }
    std::string j = "{\"atom_chain_ids\": [", values;
    bool firstAtom = true;
    for (int i = 0; i < L; ++i) {
      int aa = std::min(std::max(aatype[i], 0), 19);
      for (int a = 0; a < 37; ++a) {
        if (mask37[aa * 37 + a] == 0) continue;
        char v[24]; snprintf(v, sizeof v, "%.2f", plddt[i]);
        j += std::string(firstAtom ? "\"" : ", \"") + chainId(i) + "\"";
        values += std::string(firstAtom ? "" : ", ") + v;
        firstAtom = false;
      }
    }
    j += "],\n \"atom_plddts\": [" + values + "],\n";
    if (!contact.empty()) { j += " \"contact_probs\": "; appendMatrix2(j, contact.data(), L); j += ",\n"; }
    j += " \"pae\": "; appendMatrix2(j, pae.data(), L);
    j += ",\n \"token_chain_ids\": [";
    for (int i = 0; i < L; ++i) { j += i ? ", \"" : "\""; j += chainId(i); j += "\""; }
    j += "],\n \"token_res_ids\": [";
    for (int i = 0; i < L; ++i) { j += i ? ", " : ""; j += std::to_string(ri[i] + 1); }
    j += "]}\n";
    writeWhole(stem + "_confidences.json", j.data(), j.size());
    char s[160];
    int n = snprintf(s, sizeof s, "{\"ptm\": %.4f, \"iptm\": %s, \"mean_plddt\": %.2f}\n", ptm,
                     std::isfinite(iptm) ? std::to_string(iptm).c_str() : "null", mean);
    writeWhole(stem + "_summary_confidences.json", s, n);
  }
  printf("mean pLDDT %.2f  pTM %.4f", mean, ptm);
  if (chains) printf("  ipTM %.4f", iptm);
  printf("  -> %s  (%d passes, %.1f ms)\n", out.c_str(), ran, foldMs);
  if (converged >= 0) printf("converged at %.2f A after %d passes\n", converged, ran);
  (void)t0;
  mt::sync();
  for (const void* p : {(const void*)t.msa, (const void*)t.extra, (const void*)t.pair, (const void*)t.pairMask, (const void*)t.msaMask,
                        (const void*)t.extraMask, (const void*)prevRow, (const void*)prevPair, (const void*)prevPos, (const void*)single,
                        (const void*)plddtLogits, (const void*)paeLogits, (const void*)paeD, (const void*)tmD})
    release(p);
  releaseScratch();
  return 0;
}


bool DETACH = false;
int foldMain(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: localfold-af2 <input dir> --bundle=<dir> [--delta=<dir>] [--out=fold.pdb] [--recycles=N]\n"); return 1; }
  std::string bundleDir, deltaDir, out = "fold.pdb", serveDir;
  int recycles = -1; bool waitInput = false, profile = false;
  for (int i = 2; i < argc; ++i) {
    const char* a = argv[i];
    if (!strncmp(a, "--bundle=", 9)) bundleDir = a + 9;
    else if (!strncmp(a, "--delta=", 8)) deltaDir = a + 8;
    else if (!strncmp(a, "--out=", 6)) out = a + 6;
    else if (!strncmp(a, "--recycles=", 11)) recycles = atoi(a + 11);
    else if (!strncmp(a, "--tolerance=", 12)) TOLERANCE = atof(a + 12);
    else if (!strncmp(a, "--frames=", 9)) FRAMES_DIR = a + 9;
    else if (!strncmp(a, "--serve=", 8)) serveDir = a + 8;
    else if (!strncmp(a, "--wait-input", 12)) waitInput = true;
    else if (!strcmp(a, "--detach-output")) DETACH = true;
    else if (!strcmp(a, "--profile")) profile = true;
    else if (!strcmp(a, "--fast") || !strncmp(a, "--warm=", 7)) {}
    else if (!strncmp(a, "--weights=", 10)) die("--weights (DeepMind's float32 export) is not read here: --bundle");
    else { fprintf(stderr, "unknown flag %s\n", a); return 1; }
  }
  if (bundleDir.empty()) { fprintf(stderr, "--bundle=<dir> [--delta=<dir>]\n"); return 1; }
  if (profile) setenv("LOCALFOLD_PROFILE", "1", 1);
  double tStart = now();
  setSource("af2", PORT_SOURCE);
  {   // the page's bundle read as published, through the weight walk (cuda/featurise/af2_weights.h)
    lf::Json manifest = lf::parseJson(lf::weights::readText(bundleDir + "/manifest.json"));
    std::string deltaModel;
    if (!deltaDir.empty()) {
      const lf::Json dm = lf::parseJson(lf::weights::readText(deltaDir + "/manifest.json"));
      const lf::Json* h = dm.get("delta");
      const lf::Json* name = h ? h->get("model") : nullptr;
      deltaModel = name && name->isString() ? name->s : "";
    }
    lf::weights::Shapes S; S.shape = Model::bundleShapes(bundleDir, deltaDir);
    std::vector<std::string> lines = lf::weights::af2WeightLines(manifest, S, deltaModel);
    // the Evoformer's and the templates' matrices decoded straight to float16; everything else float32
    M.loadBundleWalk(bundleDir, lines, deltaDir, [](const std::string& n, size_t elements) {
      return elements >= 8192 && !n.rfind("w/evoformer/", 0);
    });
  }
  if (getenv("AF2_STARTUP")) { mt::sync(); printf("weights up %.0f ms\n", ms(tStart)); }
  if (!serveDir.empty()) {
    serveJobs("af2", serveDir, [&](const std::string& input, const std::vector<std::string>& flags) {
      std::string jobOut = "fold.pdb"; int jobRecycles = -1; double tol = TOLERANCE; FRAMES_DIR.clear();
      for (auto& f : flags) {
        if (!f.compare(0, 6, "--out=")) jobOut = f.substr(6);
        else if (!f.compare(0, 11, "--recycles=")) jobRecycles = atoi(f.c_str() + 11);
        else if (!f.compare(0, 12, "--tolerance=")) TOLERANCE = atof(f.c_str() + 12);
        else if (!f.compare(0, 9, "--frames=")) FRAMES_DIR = f.substr(9);
      }
      M.loadInput(input);
      int code = foldInput(input, jobOut, jobRecycles);
      M.unloadInput();
      TOLERANCE = tol;
      return code;
    });
    exit(0);
  }
  if (waitInput) {
    std::string idx = std::string(argv[1]) + "/model.idx", failed = std::string(argv[1]) + "/model.failed";
    while (access(idx.c_str(), R_OK) != 0) {
      if (access(failed.c_str(), F_OK) == 0) { fprintf(stderr, "af2: the input's export failed\n"); return 1; }
      usleep(1000);
    }
  }
  M.loadInput(argv[1]);
  printf("loaded in %.2f s\n", now() - tStart);
  int rc = foldInput(argv[1], out, recycles);
  if (getenv("LOCALFOLD_METAL_STATS")) printStats();
  if (DETACH && rc == 0) { printf("af2: done\n"); fflush(stdout); fflush(stderr); fclose(stdout); }
  fflush(stdout);
  exit(rc);
}
}  // namespace

int main(int argc, char** argv) {
  if (getenv("LOCALFOLD_METAL_SPECS_NAME")) { printf("%s\n", specsName("af2").c_str()); return 0; }
  if (argc < 2 || !strncmp(argv[1], "--", 2) || !strcmp(argv[1], "-h")) return lf::standalone::main("af2", argc, argv, foldMain);
  return foldMain(argc, argv);
}
