// The two structure terms of AlphaFold 3's ranking score (alphafold3/model/confidences.py), on the
// atoms this port writes - checked against AF3's own functions by scores_oracle.py:
//
//   has_clash: some polymer chain has more than 100 atoms, or more than half its atoms, within
//     1.1 A of another polymer atom that is not its own residue or a sequence neighbour's;
//   fraction_disordered: the fraction of protein residues whose relative solvent accessibility,
//     DSSP's (each protein chain alone) averaged over a 25-residue window, exceeds 0.581.
//
//   ranking_score = 0.8 ipTM + 0.2 pTM (pTM alone for one chain) + 0.5 disordered - 100 clash
//
// The accessibility is DSSP's own method (Kabsch & Sander's dot surface as mkdssp computes it): 401
// golden-spiral dots on each atom's sphere grown by a 1.4 A probe, DSSP's radii (N 1.65, CA 1.87,
// C 1.76, O 1.4, every other heavy atom 1.8), float arithmetic, and the residue's total rounded to
// an integer before it is divided by the residue's maximum (Sander & Rost 1994).
#pragma once
#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct ScoreAtom {
  size_t slot;                 // the atom's dense slot (its coordinates: x[3 * slot ..])
  std::string chain, res, name, element;
  int seq; char icode; bool het;
};

// The atoms as the PDB writes them: template.pdb's records when the exporter wrote one, else the
// batch's own (writePdb's fallback)
// A PDB's records: the slot from the x column (template.pdb), or (coords given) each record's own
// coordinates, its slot its index
inline std::vector<ScoreAtom> pdbAtoms(std::istream& in, std::vector<float>* coords) {
  std::vector<ScoreAtom> out;
  auto trim = [](std::string s) { size_t a = s.find_first_not_of(' '), b = s.find_last_not_of(' ');
                                   return a == std::string::npos ? std::string() : s.substr(a, b - a + 1); };
  std::string line;
  while (std::getline(in, line)) {
    if (line.size() < 54 || (line.compare(0, 4, "ATOM") && line.compare(0, 6, "HETATM"))) continue;
    ScoreAtom a;
    if (coords) {
      a.slot = out.size();
      for (int k = 0; k < 3; ++k) coords->push_back(std::stof(line.substr(30 + 8 * k, 8)));
    } else {
      a.slot = (size_t)std::lround(std::stod(line.substr(30, 8)));
    }
    a.het = !line.compare(0, 6, "HETATM");
    a.name = trim(line.substr(12, 4)); a.res = trim(line.substr(17, 3)); a.chain = line.substr(21, 1);
    a.seq = std::stoi(line.substr(22, 4)); a.icode = line[26];
    a.element = line.size() >= 78 ? trim(line.substr(76, 2)) : std::string();
    out.push_back(a);
  }
  return out;
}
// A chain's label from its asymId (1-based), A ... Z, AA, AB ... - the confidence files' and AF3's
inline std::string chainLabel(int asym) {
  std::string s;
  for (int at = asym - 1; at >= 0; at = at / 26 - 1) s.insert(s.begin(), (char)('A' + at % 26));
  return s;
}
inline std::vector<ScoreAtom> scoreAtoms(const std::string& dataDir) {
  std::vector<ScoreAtom> out;
  std::ifstream tf(dataDir + "/template.pdb");
  if (tf) {
    // 🔴 THE CHAIN FROM THE BATCH, NOT THE PDB's ONE CHARACTER: past 62 chains (26 before) the PDB
    // wraps, and two chains sharing a letter were scored - and written to the mmCIF - as one
    out = pdbAtoms(tf, nullptr);
    const int* asym = M.i("batch.asymId"); int dense = (int)M.meta("batch.dense");
    for (auto& a : out) a.chain = chainLabel(asym[a.slot / dense]);
    return out;
  }
  static const char* RES[20] = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                                "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"};
  int n = (int)M.meta("batch.tokens"), dense = (int)M.meta("batch.dense");
  const int* aatype = M.i("batch.aatype"); const int* names = M.i("batch.refAtomNameChars");
  const int* elem = M.i("batch.refElement"); const float* mask = M.f("batch.refMask");
  const int* chain = M.i("batch.asymId"); const int* resIdx = M.i("batch.residueIndex");
  for (int t = 0; t < n; ++t) for (int k = 0; k < dense; ++k) {
    size_t i = (size_t)t * dense + k;
    if (!mask[i]) continue;
    ScoreAtom a;
    a.slot = i; a.het = false; a.icode = ' ';
    for (int c = 0; c < 4; ++c) { int ch = names[i * 4 + c]; if (ch > 0) a.name += (char)(ch + 32); }
    a.res = aatype[t] >= 0 && aatype[t] < 20 ? RES[aatype[t]] : "UNK";
    a.chain = chainLabel(chain[t]); a.seq = resIdx[t];
    a.element = elem[i] == 1 ? "H" : "C";
    out.push_back(a);
  }
  return out;
}

inline const std::map<std::string, char>& aminoAcids() {
  static const std::map<std::string, char> m = {
    {"ALA", 'A'}, {"ARG", 'R'}, {"ASN", 'N'}, {"ASP", 'D'}, {"CYS", 'C'}, {"GLN", 'Q'}, {"GLU", 'E'},
    {"GLY", 'G'}, {"HIS", 'H'}, {"ILE", 'I'}, {"LEU", 'L'}, {"LYS", 'K'}, {"MET", 'M'}, {"PHE", 'F'},
    {"PRO", 'P'}, {"SER", 'S'}, {"THR", 'T'}, {"TRP", 'W'}, {"TYR", 'Y'}, {"VAL", 'V'}};
  return m;
}
inline bool isNucleotide(const std::string& r) {
  return r == "A" || r == "C" || r == "G" || r == "U" || r == "DA" || r == "DC" || r == "DG" || r == "DT";
}

// Sander & Rost's maximum accessible areas (AF3's table; anything else counts as alanine)
inline float maxAccessible(char aa) {
  static const std::map<char, float> m = {
    {'A', 106}, {'R', 248}, {'N', 157}, {'D', 163}, {'C', 135}, {'Q', 198}, {'E', 194}, {'G', 84}, {'H', 184},
    {'I', 169}, {'L', 164}, {'K', 205}, {'M', 188}, {'F', 197}, {'P', 136}, {'S', 130}, {'T', 142}, {'W', 227},
    {'Y', 222}, {'V', 142}};
  auto it = m.find(aa);
  return it == m.end() ? 106.f : it->second;
}

struct Residue { std::vector<size_t> atoms; char aa; std::string key; };

// chains in file order; a chain is a polymer if any residue is an amino acid or a nucleotide, and
// protein if any is an amino acid (a modified residue rides along with its chain, as in AF3)
struct ScoreChain { std::string id; std::vector<Residue> residues; bool protein = false, polymer = false; };
inline std::vector<ScoreChain> scoreChains(const std::vector<ScoreAtom>& atoms) {
  std::vector<ScoreChain> chains;
  for (size_t i = 0; i < atoms.size(); ++i) {
    const ScoreAtom& a = atoms[i];
    if (chains.empty() || chains.back().id != a.chain) { chains.push_back({}); chains.back().id = a.chain; }
    ScoreChain& c = chains.back();
    std::string key = std::to_string(a.seq) + a.icode + a.res;
    if (c.residues.empty() || c.residues.back().key != key) {
      auto it = aminoAcids().find(a.res);
      c.residues.push_back({ {}, it == aminoAcids().end() ? 'X' : it->second, key });
      if (it != aminoAcids().end() && !a.het) c.protein = c.polymer = true;
      if (isNucleotide(a.res) && !a.het) c.polymer = true;
    }
    c.residues.back().atoms.push_back(i);
  }
  return chains;
}

// coordinates as the PDB carries them (three decimals), which is what AF3 scores
inline float pdbRound(float v) { char b[32]; snprintf(b, sizeof b, "%.3f", v); return std::strtof(b, nullptr); }

inline bool hasClash(const std::vector<ScoreAtom>& atoms, const std::vector<ScoreChain>& chains, const float* x) {
  struct P { float x, y, z; int chain, seq; };
  std::vector<P> pts;
  std::vector<int> chainOf;
  for (size_t c = 0; c < chains.size(); ++c) {
    if (!chains[c].polymer) continue;
    for (auto& r : chains[c].residues) for (size_t i : r.atoms) {
      const ScoreAtom& a = atoms[i];
      pts.push_back({ pdbRound(x[a.slot * 3]), pdbRound(x[a.slot * 3 + 1]), pdbRound(x[a.slot * 3 + 2]), (int)c, a.seq });
    }
  }
  if (pts.empty()) return false;
  const float R = 1.1f;
  auto cell = [&](float v) { return (long)std::floor(v / R); };
  std::unordered_map<long long, std::vector<int>> grid;
  auto key = [](long a, long b, long c) { return ((a & 0x1FFFFF) << 42) | ((b & 0x1FFFFF) << 21) | (c & 0x1FFFFF); };
  for (int i = 0; i < (int)pts.size(); ++i) grid[key(cell(pts[i].x), cell(pts[i].y), cell(pts[i].z))].push_back(i);
  std::vector<char> clash(pts.size(), 0);
  for (int i = 0; i < (int)pts.size(); ++i) {
    long cx = cell(pts[i].x), cy = cell(pts[i].y), cz = cell(pts[i].z);
    for (long dx = -1; dx <= 1 && !clash[i]; ++dx) for (long dy = -1; dy <= 1 && !clash[i]; ++dy)
      for (long dz = -1; dz <= 1 && !clash[i]; ++dz) {
        auto it = grid.find(key(cx + dx, cy + dy, cz + dz));
        if (it == grid.end()) continue;
        for (int j : it->second) {
          double ex = (double)pts[i].x - pts[j].x, ey = (double)pts[i].y - pts[j].y, ez = (double)pts[i].z - pts[j].z;
          if (ex * ex + ey * ey + ez * ez > (double)R * R) continue;    // query_ball_point: distance <= r
          if (std::abs(pts[i].seq - pts[j].seq) > 1 || pts[i].chain != pts[j].chain) { clash[i] = 1; break; }
        }
      }
  }
  std::map<int, std::pair<size_t, size_t>> perChain;      // chain -> (atoms, clashing)
  for (size_t i = 0; i < pts.size(); ++i) { auto& v = perChain[pts[i].chain]; v.first++; v.second += clash[i]; }
  for (auto& [c, v] : perChain)
    if (v.second > 100 || (double)v.second / v.first > 0.5) return true;
  return false;
}

// DSSP's accessibility of every residue of one chain (the chain alone), rounded as DSSP prints it
inline std::vector<int> dsspAccessibility(const std::vector<ScoreAtom>& atoms, const ScoreChain& chain, const float* x) {
  const float kWater = 1.4f;
  // DSSP's MSurfaceDots(200): 401 points on the unit sphere (built once, thread-safely: samples score in parallel)
  static const std::vector<float> dots = [] {
    std::vector<float> d;
    const int N = 200, P = 2 * N + 1;
    const float golden = (1 + std::sqrt(5.0f)) / 2, kPI = 3.141592653589793238462643383279502884f;
    for (int i = -N; i <= N; ++i) {
      float lat = std::asin((2.0f * i) / P);
      float lon = static_cast<float>(std::fmod(i, golden) * 2 * kPI / golden);
      d.insert(d.end(), { std::sin(lon) * std::cos(lat), std::cos(lon) * std::cos(lat), std::sin(lat) });
    }
    return d;
  }();
  const float weight = (4 * 3.141592653589793238462643383279502884f) / 401;
  struct A { float x, y, z, r; int res; };
  std::vector<A> pts;
  for (size_t ri = 0; ri < chain.residues.size(); ++ri)
    for (size_t i : chain.residues[ri].atoms) {
      const ScoreAtom& a = atoms[i];
      if (a.element == "H" || a.element == "D") continue;
      float r = a.name == "N" ? 1.65f : a.name == "CA" ? 1.87f : a.name == "C" ? 1.76f : a.name == "O" ? 1.4f : 1.8f;
      pts.push_back({ pdbRound(x[a.slot * 3]), pdbRound(x[a.slot * 3 + 1]), pdbRound(x[a.slot * 3 + 2]), r, (int)ri });
    }
  std::vector<float> acc(chain.residues.size(), 0.f);
  // neighbours through a grid of cells as wide as the largest interaction distance
  const float cellW = 2 * (1.87f + kWater);
  auto cell = [&](float v) { return (long)std::floor(v / cellW); };
  std::unordered_map<long long, std::vector<int>> grid;
  auto key = [](long a, long b, long c) { return ((a & 0x1FFFFF) << 42) | ((b & 0x1FFFFF) << 21) | (c & 0x1FFFFF); };
  for (int i = 0; i < (int)pts.size(); ++i) grid[key(cell(pts[i].x), cell(pts[i].y), cell(pts[i].z))].push_back(i);
  std::vector<float> surface(pts.size(), 0.f);
  auto work = [&](int lo, int hi) {
    struct Cand { float x, y, z, r2, d; };
    std::vector<Cand> cands;
    for (int i = lo; i < hi; ++i) {
      const A& a = pts[i];
      cands.clear();
      long cx = cell(a.x), cy = cell(a.y), cz = cell(a.z);
      for (long dx = -1; dx <= 1; ++dx) for (long dy = -1; dy <= 1; ++dy) for (long dz = -1; dz <= 1; ++dz) {
        auto it = grid.find(key(cx + dx, cy + dy, cz + dz));
        if (it == grid.end()) continue;
        for (int j : it->second) {
          const A& b = pts[j];
          float ex = b.x - a.x, ey = b.y - a.y, ez = b.z - a.z, d2 = ex * ex + ey * ey + ez * ez;
          float test = (a.r + kWater) + (b.r + kWater); test *= test;
          if (d2 < test && d2 > 0.0001f) cands.push_back({ ex, ey, ez, (b.r + kWater) * (b.r + kWater), d2 });
        }
      }
      std::sort(cands.begin(), cands.end(), [](const Cand& p, const Cand& q) { return p.d < q.d; });   // nearest first, as DSSP
      float radius = a.r + kWater, s = 0;
      for (size_t k = 0; k < dots.size(); k += 3) {
        float px = dots[k] * radius, py = dots[k + 1] * radius, pz = dots[k + 2] * radius;
        bool free = true;
        for (size_t c = 0; free && c < cands.size(); ++c) {
          float ex = px - cands[c].x, ey = py - cands[c].y, ez = pz - cands[c].z;
          free = cands[c].r2 < ex * ex + ey * ey + ez * ez;
        }
        if (free) s += weight;
      }
      surface[i] = s * radius * radius;
    }
  };
  // (128 atoms a thread: 5CAJ's 2,106 took 5 threads at 512, and each atom is independent - summed in order
  // below, so the count cannot change the answer)
  int nt = std::max(1, std::min((int)std::thread::hardware_concurrency(), (int)(pts.size() / 128) + 1));
  std::vector<std::thread> th;
  for (int t = 0; t < nt; ++t) th.emplace_back(work, (int)(pts.size() * t / nt), (int)(pts.size() * (t + 1) / nt));
  for (auto& t : th) t.join();
  for (size_t i = 0; i < pts.size(); ++i) acc[pts[i].res] += surface[i];
  std::vector<int> out(acc.size());
  for (size_t r = 0; r < acc.size(); ++r) out[r] = (int)std::floor(acc[r] + 0.5f);
  return out;
}

// AF3's windowed rASA of one chain: DSSP accessibility over the residue's maximum (capped at 1),
// then the mean over a 25-residue window, the ends padded by reflection (numpy's 'reflect')
inline std::vector<double> windowedRasa(const std::vector<int>& acc, const ScoreChain& chain) {
  size_t L = acc.size();
  std::vector<double> rasa(L), out(L);
  for (size_t r = 0; r < L; ++r) rasa[r] = std::min(1.0, (double)acc[r] / maxAccessible(chain.residues[r].aa));
  const int window = 25, hw = (window - 1) / 2;
  auto at = [&](long i) -> double {
    if (L == 1) return rasa[0];
    long period = 2 * ((long)L - 1);
    long j = ((i % period) + period) % period;
    return rasa[j < (long)L ? j : period - j];
  };
  for (size_t r = 0; r < L; ++r) {
    double s = 0;
    for (long k = -hw; k <= hw; ++k) s += at((long)r + k);
    out[r] = s / window;
  }
  return out;
}

inline double fractionDisordered(const std::vector<ScoreAtom>& atoms, const std::vector<ScoreChain>& chains, const float* x) {
  std::vector<double> all;
  std::map<std::string, std::vector<double>> bySequence;    // identical chains share it, as in AF3
  for (auto& c : chains) {
    if (!c.protein) continue;
    std::string seq; for (auto& r : c.residues) seq += r.aa;
    auto it = bySequence.find(seq);
    if (it == bySequence.end()) {
      std::vector<int> acc = dsspAccessibility(atoms, c, x);
      if (const char* d = getenv("AF3_ACC_OUT")) {      // the raw accessibilities, for scores_oracle.py --acc
        FILE* f = fopen(d, "a"); fprintf(f, "%s", c.id.c_str());
        for (int v : acc) fprintf(f, " %d", v);
        fprintf(f, "\n"); fclose(f);
      }
      it = bySequence.emplace(seq, windowedRasa(acc, c)).first;
    }
    all.insert(all.end(), it->second.begin(), it->second.end());
  }
  if (all.empty()) return 0.0;
  size_t over = 0; for (double v : all) over += v > 0.581;
  return (double)over / all.size();
}

// the input's atoms and chains, built once a fold rather than once a sample: they are the input's, not the
// coordinates', and reading them re-parsed template.pdb (af3.cu bumps the generation at each fold - a serve
// job's input directory is reused with new contents)
inline int SCORE_ATOMS_GEN = 0;
struct FoldAtoms { std::vector<ScoreAtom> atoms; std::vector<ScoreChain> chains; };
inline const FoldAtoms& foldAtoms() {
  static FoldAtoms cached; static int gen = -1; static std::string dir;
  if (gen != SCORE_ATOMS_GEN || dir != DATA_DIR) {
    cached.atoms = scoreAtoms(DATA_DIR); cached.chains = scoreChains(cached.atoms); gen = SCORE_ATOMS_GEN; dir = DATA_DIR;
  }
  return cached;
}

struct StructureScores { bool clash; double disordered; };
inline StructureScores structureScores(const std::vector<float>& x) {
  const FoldAtoms& fa = foldAtoms();
  return { hasClash(fa.atoms, fa.chains, x.data()), fractionDisordered(fa.atoms, fa.chains, x.data()) };
}
inline double rankingScore(double ptm, double iptm, const StructureScores& s) {
  double base = std::isnan(iptm) ? ptm : 0.8 * iptm + 0.2 * ptm;
  return base + 0.5 * s.disordered - 100.0 * (s.clash ? 1 : 0);
}

// af3 --score-pdb=FILE: the two terms for any PDB (the oracle's input, without a fold)
inline int scorePdbMain(const std::string& path) {
  std::ifstream in(path);
  if (!in) { fprintf(stderr, "cannot read %s\n", path.c_str()); return 1; }
  std::vector<float> x;
  std::vector<ScoreAtom> atoms = pdbAtoms(in, &x);
  std::vector<ScoreChain> chains = scoreChains(atoms);
  printf("%s: fraction_disordered %.6f  has_clash %d\n", path.c_str(), fractionDisordered(atoms, chains, x.data()),
         (int)hasClash(atoms, chains, x.data()));
  return 0;
}

// The structure as mmCIF (AlphaFold 3's own output format), for --out=*.cif: the PDB's records as
// an _atom_site table, the per-atom pLDDT in B_iso_or_equiv. Each chain its own entity; a polymer
// residue carries label_seq_id, a ligand '.'. Returns the slots in record order, as writePdb does.
inline std::vector<size_t> writeCif(const std::string& path, const std::vector<float>& x, const float* bfactors) {
  const std::vector<ScoreAtom>& atoms = foldAtoms().atoms;
  const std::vector<ScoreChain>& chains = foldAtoms().chains;
  // one ENTITY per distinct chain (its residue names in order), as mmCIF means it and AF3's own
  // writer does: copies of one sequence share it (one entity a chain also broke AF3's reader past
  // ten entities - it paired chain A with another entity's sequence)
  std::vector<int> entityOf(chains.size());
  std::vector<size_t> firstOfEntity;
  {
    std::map<std::string, int> byContent;
    for (size_t c = 0; c < chains.size(); ++c) {
      std::string key = chains[c].polymer ? "P" : "L";
      for (auto& r : chains[c].residues) key += " " + atoms[r.atoms[0]].res;
      auto it = byContent.find(key);
      if (it == byContent.end()) { it = byContent.emplace(key, (int)firstOfEntity.size() + 1).first; firstOfEntity.push_back(c); }
      entityOf[c] = it->second;
    }
  }
  std::vector<size_t> order;
  std::string name = path.substr(path.find_last_of('/') + 1);
  name = name.substr(0, name.find_last_of('.'));
  FILE* f = fopen(path.c_str(), "w");
  fprintf(f, "data_%s\n#\n_entry.id %s\n#\nloop_\n_entity.id\n_entity.type\n", name.c_str(), name.c_str());
  for (size_t e = 0; e < firstOfEntity.size(); ++e) fprintf(f, "%zu %s\n", e + 1, chains[firstOfEntity[e]].polymer ? "polymer" : "non-polymer");
  // the polymers' types and sequences, and the chains' entities (what AF3's own reader requires)
  auto dnaRes = [](const std::string& r) { return r == "DA" || r == "DC" || r == "DG" || r == "DT"; };
  auto polyType = [&](const ScoreChain& c) {
    if (c.protein) return "polypeptide(L)";
    for (auto& r : c.residues) if (dnaRes(atoms[r.atoms[0]].res)) return "polydeoxyribonucleotide";
    return "polyribonucleotide";
  };
  fprintf(f, "#\nloop_\n_entity_poly.entity_id\n_entity_poly.type\n");
  for (size_t e = 0; e < firstOfEntity.size(); ++e)
    if (chains[firstOfEntity[e]].polymer) fprintf(f, "%zu %s\n", e + 1, polyType(chains[firstOfEntity[e]]));
  fprintf(f, "#\nloop_\n_entity_poly_seq.entity_id\n_entity_poly_seq.num\n_entity_poly_seq.mon_id\n_entity_poly_seq.hetero\n");
  for (size_t e = 0; e < firstOfEntity.size(); ++e) {
    const ScoreChain& chain = chains[firstOfEntity[e]];
    if (!chain.polymer) continue;
    for (auto& r : chain.residues) {
      const ScoreAtom& a = atoms[r.atoms[0]];
      fprintf(f, "%zu %d %s n\n", e + 1, a.seq, a.res.c_str());
    }
  }
  fprintf(f, "#\nloop_\n_struct_asym.id\n_struct_asym.entity_id\n");
  for (size_t c = 0; c < chains.size(); ++c) fprintf(f, "%s %d\n", chains[c].id.c_str(), entityOf[c]);
  fprintf(f, "#\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n"
             "_atom_site.label_alt_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_entity_id\n"
             "_atom_site.label_seq_id\n_atom_site.pdbx_PDB_ins_code\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n"
             "_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n_atom_site.auth_seq_id\n"
             "_atom_site.auth_asym_id\n_atom_site.pdbx_PDB_model_num\n");
  size_t serial = 1;
  for (size_t c = 0; c < chains.size(); ++c)
    for (auto& r : chains[c].residues)
      for (size_t i : r.atoms) {
        const ScoreAtom& a = atoms[i];
        order.push_back(a.slot);
        std::string el = a.element.empty() ? a.name.substr(0, 1) : a.element;
        std::string atomId = a.name.find('\'') != std::string::npos ? "\"" + a.name + "\"" : a.name;
        std::string seq = chains[c].polymer ? std::to_string(a.seq) : ".";
        fprintf(f, "%s %zu %s %s . %s %s %d %s %s %.3f %.3f %.3f 1.00 %.2f %d %s 1\n", a.het ? "HETATM" : "ATOM", serial++,
                el.c_str(), atomId.c_str(), a.res.c_str(), a.chain.c_str(), entityOf[c], seq.c_str(),
                a.icode == ' ' ? "?" : std::string(1, a.icode).c_str(), x[a.slot * 3], x[a.slot * 3 + 1], x[a.slot * 3 + 2],
                bfactors ? bfactors[a.slot] : 0.0, a.seq, a.chain.c_str());
      }
  fprintf(f, "#\n");
  fclose(f);
  return order;
}
inline bool cifPath(const std::string& p) { return p.size() > 4 && p.substr(p.size() - 4) == ".cif"; }
inline std::vector<size_t> writeStructure(const std::string& path, const std::vector<float>& x, const float* bfactors) {
  return cifPath(path) ? writeCif(path, x, bfactors) : writePdb(path, x, bfactors);
}
