// OpenFold3's losses as of3loss.h computes them, on a real batch's layout (any featurised AF3-lineage input - the
// kitchen sink reaches every branch: modified residues, DNA, RNA, ligands, token bonds) with a synthetic truth, a
// perturbed prediction and random logits: writes the case and its values and gradients for of3_loss_oracle.py, a
// line-by-line transcription of openfold3/core/loss in torch, to hold them to.
//
//   nvcc -O2 -std=c++17 -arch=sm_80 -o cuda/grad/test-of3loss cuda/grad/test_of3loss.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-of3loss --input=<featurised dir> --out=<dir>
//   python cuda/grad/of3_loss_oracle.py <dir>
#include <fstream>
#include <random>
#include "../af3/src/common.cuh"
#include "of3loss.h"

static void put(const std::string& path, const std::vector<double>& v) {
  std::ofstream f(path, std::ios::binary); f.write((const char*)v.data(), v.size() * 8);
}

int main(int argc, char** argv) {
  std::string input, out;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--input=", 8)) input = argv[i] + 8;
    else if (!strncmp(argv[i], "--out=", 6)) out = argv[i] + 6;
  }
  if (input.empty() || out.empty()) { fprintf(stderr, "--input= and --out= are required\n"); return 1; }
  M.load(input);
  const int n = (int)M.meta("batch.tokens"), dense = (int)M.meta("batch.dense");
  const size_t A = (size_t)n * dense;
  of3::Layout L = of3::makeLayout(n, dense, M.f("batch.predDenseAtomMask"), M.i("batch.refAtomNameChars"), M.i("batch.isDna"),
                                  M.i("batch.isRna"), M.i("batch.isLigand"), M.i("batch.isModified"),
                                  M.i("batch.features.asymId"), M.has("batch.bondMatrix") ? M.f("batch.bondMatrix") : nullptr);
  // a synthetic truth: each token's conformer (ref_pos) about a 3.8 A random walk; a prediction rotated, moved and
  // perturbed (1 A); 5% of the atoms unresolved
  std::mt19937_64 rng(7); std::normal_distribution<double> N(0, 1); std::uniform_real_distribution<double> U(0, 1);
  const float* ref = M.f("batch.refPos");
  std::vector<double> xt(A * 3, 0), x(A * 3, 0), gm(A, 0);
  double c[3] = {0, 0, 0};
  for (int t = 0; t < n; ++t) {
    double s[3] = {N(rng), N(rng), N(rng)}, sn = sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    for (int p = 0; p < 3; ++p) c[p] += 3.8 * s[p] / sn;
    for (int k = 0; k < dense; ++k) {
      const size_t i = (size_t)t * dense + k;
      if (!L.exists[i]) continue;
      for (int p = 0; p < 3; ++p) xt[i * 3 + p] = c[p] + 0.3 * ref[i * 3 + p];
      gm[i] = U(rng) > 0.05;
    }
  }
  const double th = 0.7, R[3][3] = {{cos(th), -sin(th), 0}, {sin(th), cos(th), 0}, {0, 0, 1}};
  for (size_t i = 0; i < A; ++i) {
    if (!L.exists[i]) continue;
    for (int p = 0; p < 3; ++p) { double v = 4.0 * (p + 1); for (int q = 0; q < 3; ++q) v += R[p][q] * xt[i * 3 + q]; x[i * 3 + p] = v + 1.0 * N(rng); }
  }
  auto gauss = [&](size_t len) { std::vector<double> v(len); for (double& e : v) e = 2 * N(rng); return v; };
  const int K = 64;
  auto lDg = gauss((size_t)n * n * K), lPl = gauss(A * 50), lPde = gauss((size_t)n * n * K), lPae = gauss((size_t)n * n * K), lRs = gauss(A * 2);
  // our values and gradients
  of3::DiffusionWeights W; W.bond = 1;          // (the bond term on, so the oracle checks it)
  const double t = 5.0;
  std::vector<double> dx(A * 3, 0);
  auto D = of3::diffusionLoss(L, x, xt, gm, t, W, &dx);
  {
    // the device's smooth LDDT against the host's (the trainer's path)
    of3dev::SmoothLddt sl; sl.init(L, xt, gm);
    std::vector<double> dx2(A * 3, 0);
    auto D2 = of3::diffusionLoss(L, x, xt, gm, t, W, &dx2, 1,
                                 [&](double sc, std::vector<double>* g) { return sl.run(x, sc, g, 0); });
    double num = 0, den = 0; for (size_t i = 0; i < A * 3; ++i) { num += (dx2[i] - dx[i]) * (dx2[i] - dx[i]); den += dx[i] * dx[i]; }
    const double ev = fabs(D2.lddt - D.lddt) / D.lddt, eg = sqrt(num / den);
    printf("device smooth LDDT against the host's: value %.2e, the whole diffusion gradient %.2e  %s\n", ev, eg, ev < 1e-12 && eg < 1e-12 ? "PASS" : "FAIL");
    if (!(ev < 1e-12 && eg < 1e-12)) return 1;
    sl.release();
  }
  std::vector<double> gDg(lDg.size(), 0), gPl(lPl.size(), 0), gPde(lPde.size(), 0), gPae(lPae.size(), 0), gRs(lRs.size(), 0);
  const double dg = of3::distogramLoss(L, xt, gm, lDg.data(), K, 1, gDg.data());
  const double pl = of3::plddtLoss(L, x, xt, gm, lPl.data(), 50, 1, gPl.data());
  const double pde = of3::pdeLoss(L, x, xt, gm, lPde.data(), K, 1, gPde.data());
  const double pae = of3::paeLoss(L, x, xt, gm, lPae.data(), K, 1, gPae.data());
  const double rs = of3::resolvedLoss(L, gm, lRs.data(), 1, gRs.data());
  size_t nb = 0; for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) nb += L.ligand[i] && !L.ligand[j] && !L.bondTok.empty() && L.bondTok[(size_t)i * n + j];
  int at = 0, nu = 0, li = 0; for (int i = 0; i < n; ++i) { at += L.atomized[i]; nu += L.nucleotide(i); li += L.ligand[i]; }
  printf("%d tokens (%d atomised, %d nucleotide, %d ligand), %zu ligand-polymer token bonds\n", n, at, nu, li, nb);
  printf("mse %.6f bond %.6f smooth_lddt %.6f total %.6f | distogram %.6f plddt %.6f pde %.6f pae %.6f resolved %.6f\n",
         D.mse, D.bond, D.lddt, D.total, dg, pl, pde, pae, rs);
  // the case and our answers
  std::vector<double> exists(A), names(A * 4), tok(n * 7);
  const int* nm = M.i("batch.refAtomNameChars"); const int* aat = M.i("batch.aatype");
  for (size_t i = 0; i < A; ++i) { exists[i] = L.exists[i]; for (int c4 = 0; c4 < 4; ++c4) names[i * 4 + c4] = nm[i * 4 + c4]; }
  for (int i = 0; i < n; ++i) {
    tok[i * 7 + 0] = M.i("batch.isDna")[i]; tok[i * 7 + 1] = M.i("batch.isRna")[i]; tok[i * 7 + 2] = M.i("batch.isLigand")[i];
    tok[i * 7 + 3] = M.i("batch.isModified")[i]; tok[i * 7 + 4] = M.i("batch.features.asymId")[i]; tok[i * 7 + 5] = aat[i];
    tok[i * 7 + 6] = 0;
  }
  std::vector<double> bonds((size_t)n * n); for (size_t i = 0; i < bonds.size(); ++i) bonds[i] = L.bondTok.empty() ? 0 : L.bondTok[i];
  put(out + "/x.bin", x); put(out + "/xt.bin", xt); put(out + "/gm.bin", gm); put(out + "/exists.bin", exists);
  put(out + "/names.bin", names); put(out + "/tok.bin", tok); put(out + "/bonds.bin", bonds);
  put(out + "/lDg.bin", lDg); put(out + "/lPl.bin", lPl); put(out + "/lPde.bin", lPde); put(out + "/lPae.bin", lPae); put(out + "/lRs.bin", lRs);
  put(out + "/dx.bin", dx); put(out + "/gDg.bin", gDg); put(out + "/gPl.bin", gPl); put(out + "/gPde.bin", gPde); put(out + "/gPae.bin", gPae); put(out + "/gRs.bin", gRs);
  put(out + "/values.bin", {D.mse, D.bond, D.lddt, D.total, dg, pl, pde, pae, rs, t, (double)n, (double)dense});
  return 0;
}
