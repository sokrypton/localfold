// Boltz-2's own parameters behind the port's (cuda/grad/make_boltz2_map.py has why): the parameters train-of3 --source
// steps are Boltz-2's 5,035 tensors, by their Boltz names; the port's tensors - what the training graph
// reads - are a function of them, rebuilt after every step, and the port's gradients are carried back through it.
//
// The function, per element of a port tensor: a constant, k * m[a] (a copy of one Boltz element, k 1, -1 or 0.5),
// or k * m[a] * m[b] (a LayerNorm scale folded into a projection). It is the converter's map (bundle tensor ->
// Boltz-2, written by make_boltz2_map.py) composed here with the port's weight walk (port tensor -> bundle tensor,
// cuda/featurise/af3_weights.h: a slice, zeros, literals, or strided views and products of bundle tensors), and stored
// as 2D runs: a block of rows of consecutive port elements whose sources step by constant strides (a transposed
// matrix is one run).
//
// At load: every source element is set from the port's own values through the copies (m[a] = port / k; several port
// elements naming one source - a tied tensor - must agree), the rest from the map's carried values (the folded
// factors, the softmax-invariant LayerNorm offsets, the B-factor head); then the sources are gathered back and must
// reproduce every port tensor (copies and constants to the bit, products to their rounding).
#pragma once
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace grad {

struct SourceRun {                      // rows x cols port elements from pw (+ r * pRow + c), their sources at
  real *pw, *pg;                        // a0 + r * aRow + c * aCol (and b likewise)
  unsigned rows, cols; long long pRow;
  long long a0, aRow, aCol, b0, bRow, bCol;
  real k; int kind;                     // 0 the constant k, 1 k * m[a], 2 k * m[a] * m[b]
};

__global__ void sourceGatherK(const SourceRun* runs, const real* mw) {
  const SourceRun R = runs[blockIdx.x];
  const size_t n = (size_t)R.rows * R.cols;
  for (size_t e = threadIdx.x; e < n; e += blockDim.x) {
    const size_t r = e / R.cols, c = e % R.cols;
    const long long a = R.a0 + (long long)r * R.aRow + (long long)c * R.aCol;
    real v = R.k;
    if (R.kind == 1) v = R.k * mw[a];
    else if (R.kind == 2) v = R.k * (mw[a] * mw[R.b0 + (long long)r * R.bRow + (long long)c * R.bCol]);
    R.pw[(long long)r * R.pRow + c] = v;
  }
}
__global__ void sourceScatterK(const SourceRun* runs, const real* mw, real* mg) {
  const SourceRun R = runs[blockIdx.x];
  if (R.kind == 0) return;
  const size_t n = (size_t)R.rows * R.cols;
  for (size_t e = threadIdx.x; e < n; e += blockDim.x) {
    const size_t r = e / R.cols, c = e % R.cols;
    const real g = R.k * R.pg[(long long)r * R.pRow + c];
    const long long a = R.a0 + (long long)r * R.aRow + (long long)c * R.aCol;
    if (R.kind == 1) { atomicAdd(mg + a, g); continue; }
    const long long b = R.b0 + (long long)r * R.bRow + (long long)c * R.bCol;
    atomicAdd(mg + a, g * mw[b]); atomicAdd(mg + b, g * mw[a]);
  }
}
__global__ void sourceFromPortK(const SourceRun* runs, real* mw) {
  const SourceRun R = runs[blockIdx.x];
  if (R.kind != 1 || R.k == 0) return;
  const size_t n = (size_t)R.rows * R.cols;
  for (size_t e = threadIdx.x; e < n; e += blockDim.x) {
    const size_t r = e / R.cols, c = e % R.cols;
    mw[R.a0 + (long long)r * R.aRow + (long long)c * R.aCol] = R.pw[(long long)r * R.pRow + c] / R.k;
  }
}
// the sources gathered against the port's values: mismatches (copies and constants bitwise, products to 2 ulp)
__global__ void sourceCheckK(const SourceRun* runs, const real* mw, unsigned long long* bad, double* worst) {
  const SourceRun R = runs[blockIdx.x];
  const size_t n = (size_t)R.rows * R.cols;
  for (size_t e = threadIdx.x; e < n; e += blockDim.x) {
    const size_t r = e / R.cols, c = e % R.cols;
    const long long a = R.a0 + (long long)r * R.aRow + (long long)c * R.aCol, b = R.b0 + (long long)r * R.bRow + (long long)c * R.bCol;
    real v = R.kind == 0 ? R.k : R.kind == 1 ? R.k * mw[a] : R.k * (mw[a] * mw[b]);
    const real p = R.pw[(long long)r * R.pRow + c];
    const double d = fabs((double)v - (double)p);
    const bool ok = R.kind == 2 ? d <= 2.5e-7 * fabs((double)p) : (v == p);
    if (!ok || isnan((double)v)) { atomicAdd(bad, 1ull); atomicMax((unsigned long long*)worst, __double_as_longlong(d)); }
  }
}
__global__ void nanCountK(const real* w, size_t n, unsigned long long* bad) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    if (isnan((double)w[i])) atomicAdd(bad, 1ull);
}

struct Boltz2Source {
  std::vector<std::string> names;       // the sources, Boltz-2's keys
  std::vector<Param> params; Params ps; // the optimiser's parameters (w and g in the two slabs below)
  real *w = nullptr, *g = nullptr; size_t total = 0;
  std::vector<size_t> offset;           // each source's offset in the slabs
  SourceRun* runs = nullptr; unsigned nRuns = 0;

  // ports: the training graph's parameters (their names the walk's, their w the bundle's values as loaded);
  // walk: the weight walk's lines for the bundle; dir: make_boltz2_map.py's output
  void init(const Params& ports, const std::vector<std::string>& walk, const std::string& dir) {
    M.load(dir);
    std::vector<std::string> all;
    { std::ifstream f(dir + "/names.txt"); std::string s; while (std::getline(f, s)) if (!s.empty()) all.push_back(s); }
    const int nb = (int)M.meta("boltz2map.bundleTensors"), nm = (int)M.meta("boltz2source.count");
    if ((int)all.size() != nb + nm) { fprintf(stderr, "%s/names.txt: %zu names, %d expected\n", dir.c_str(), all.size(), nb + nm); exit(1); }
    std::map<std::string, int> bundleId;
    for (int i = 0; i < nb; ++i) bundleId[all[i]] = i;
    names.assign(all.begin() + nb, all.end());
    // the slabs: every source 64-element aligned
    offset.resize(nm);
    for (int i = 0; i < nm; ++i) { offset[i] = total; total += ((size_t)M.meta("boltz2source." + std::to_string(i) + ".n") + 63) / 64 * 64; }
    GK(cudaMalloc(&w, total * sizeof(real))); GK(cudaMalloc(&g, total * sizeof(real)));
    GK(cudaMemset(g, 0, total * sizeof(real)));
    { std::vector<real> nan(total, (real)NAN); GK(cudaMemcpy(w, nan.data(), total * sizeof(real), cudaMemcpyHostToDevice)); }
    params.resize(nm);
    for (int i = 0; i < nm; ++i) {
      Param& p = params[i]; p.name = names[i]; p.n = (size_t)M.meta("boltz2source." + std::to_string(i) + ".n");
      p.w = w + offset[i]; p.g = g + offset[i];
    }
    for (auto& p : params) ps.add(p);
    // the converter's runs, by bundle tensor
    const int* RR = M.i("boltz2map.runs"); const int* BS = M.i("boltz2map.bstep"); const float* KK = M.f("boltz2map.k");
    const size_t nr = M.len("boltz2map.runs") / 9;
    std::vector<std::vector<size_t>> byTensor(nb);
    for (size_t r = 0; r < nr; ++r) byTensor[RR[r * 9]].push_back(r);
    struct E { int kind; float k; long long a, b; };
    // (a stacked tensor is read once per block: kept - shared, since a product holds two at once)
    std::map<int, std::shared_ptr<std::vector<E>>> cache;
    auto bundle = [&](const std::string& nmB) -> std::shared_ptr<std::vector<E>> {
      auto it = bundleId.find(nmB);
      if (it == bundleId.end()) { fprintf(stderr, "boltz2 source: the walk reads %s, which the map does not cover\n", nmB.c_str()); exit(1); }
      auto c = cache.find(it->second);
      if (c != cache.end()) return c->second;
      if (cache.size() > 8) cache.erase(cache.begin());
      auto sp = std::make_shared<std::vector<E>>(); cache[it->second] = sp;
      std::vector<E>& v = *sp;
      for (size_t r : byTensor[it->second]) {
        const int* q = RR + r * 9;   // tensor, first, count, kind, a, afirst, astep, b, bfirst
        if ((size_t)q[1] + q[2] > v.size()) v.resize((size_t)q[1] + q[2]);
        for (int j = 0; j < q[2]; ++j) {
          E e{q[3], KK[r], -1, -1};
          if (q[3] >= 1) e.a = (long long)offset[q[4]] + q[5] + (long long)j * q[6];
          if (q[3] == 2) e.b = (long long)offset[q[7]] + q[8] + (long long)j * BS[r];
          v[q[1] + j] = e;
        }
      }
      return sp;
    };
    // the walk's lines by port name
    std::map<std::string, std::vector<std::string>> lines;
    for (const auto& l : walk) {
      std::istringstream in(l); char kind; std::string nmP; in >> kind >> nmP;
      if (kind == 'm') continue;
      lines[nmP].push_back(l);
    }
    std::vector<SourceRun> host, others;  // (others: the walk's tensors the graph does not read - their values only)
    size_t covered = 0;
    std::set<std::string> loaded;
    for (Param* P : ports.all) {
      if (!lines.count(P->name)) { fprintf(stderr, "boltz2 source: %s is not a tensor of the weight walk\n", P->name.c_str()); exit(1); }
      loaded.insert(P->name);
    }
    std::vector<Param> extra;             // (each a temporary: the bundle's values, on the device for the check)
    for (auto& [nmP, ls] : lines) if (!loaded.count(nmP)) {
      Param q; q.name = nmP; q.n = M.len(nmP); q.w = nullptr; q.g = nullptr;
      std::vector<real> h(M.f(nmP), M.f(nmP) + q.n);
      GK(cudaMalloc(&q.w, std::max<size_t>(q.n, 1) * sizeof(real)));
      GK(cudaMemcpy(q.w, h.data(), q.n * sizeof(real), cudaMemcpyHostToDevice));
      extra.push_back(q);
    }
    std::vector<Param*> todo(ports.all.begin(), ports.all.end());
    for (auto& q : extra) todo.push_back(&q);
    for (Param* P : todo) {
      const bool trains = loaded.count(P->name) > 0;
      auto it = lines.find(P->name);
      std::vector<E> d(P->n, E{0, 0.f, -1, -1});
      for (const std::string& l : it->second) {
        std::istringstream in(l); char kind; std::string nmP; in >> kind >> nmP;
        if (kind == 'b') {
          std::string src; size_t first, n; in >> src >> first >> n;
          const auto bp = bundle(src); const auto& b = *bp;
          if (n != P->n || first + n > b.size()) { fprintf(stderr, "boltz2 source: %s: a slice the map cannot hold\n", nmP.c_str()); exit(1); }
          for (size_t i = 0; i < n; ++i) d[i] = b[first + i];
        } else if (kind == 'z') {
        } else if (kind == 'c') {
          size_t n; in >> n; for (size_t i = 0; i < n; ++i) { float v; in >> v; d[i] = E{0, v, -1, -1}; }
        } else if (kind == 'p') {
          size_t n; char op; int rank; in >> n >> op >> rank;
          size_t dims[6]; long long dst, ds[6], off[2] = {0, 0}, ss[2][6] = {};
          for (int k = 0; k < rank; ++k) in >> dims[k];
          in >> dst; for (int k = 0; k < rank; ++k) in >> ds[k];
          const int nsrc = op == 'o' ? 0 : op == 'x' ? 2 : 1;
          std::shared_ptr<std::vector<E>> src[2];
          std::string sn[2];
          for (int q = 0; q < nsrc; ++q) { in >> sn[q] >> off[q]; for (int k = 0; k < rank; ++k) in >> ss[q][k]; }
          for (int q = 0; q < nsrc; ++q) src[q] = bundle(sn[q]);
          if (op == 'i') { fprintf(stderr, "boltz2 source: %s is an integer view, not a parameter\n", nmP.c_str()); exit(1); }
          size_t total = 1; for (int k = 0; k < rank; ++k) total *= dims[k];
          for (size_t o = 0; o < total; ++o) {
            size_t r = o; long long dd = dst, a = off[0], c = off[1];
            for (int k = rank - 1; k >= 0; --k) { long long i = (long long)(r % dims[k]); r /= dims[k]; dd += i * ds[k]; a += i * ss[0][k]; c += i * ss[1][k]; }
            E v{0, 1.f, -1, -1};
            if (op == 'v') v = (*src[0])[a];
            else if (op == 'x') {           // (a constant scales the other side; two copies make a product)
              E x = (*src[0])[a], y = (*src[1])[c];
              if (x.kind == 0 && y.kind == 0) v = E{0, x.k * y.k, -1, -1};
              else if (x.kind == 0 || y.kind == 0) { v = x.kind ? x : y; v.k *= x.kind ? y.k : x.k; }
              else if (x.kind == 1 && y.kind == 1) v = E{2, x.k * y.k, x.a, y.a};
              else { fprintf(stderr, "boltz2 source: %s: a product of a product\n", nmP.c_str()); exit(1); }
            }
            d[dd] = v;
          }
        } else { fprintf(stderr, "boltz2 source: %s: a walk line of kind %c\n", nmP.c_str(), kind); exit(1); }
      }
      // 1D runs, then rows of them
      struct R1 { size_t p0, n; int kind; float k; long long a, as, b, bs; };
      std::vector<R1> r1;
      for (size_t i = 0; i < P->n; ++i) {
        const E& e = d[i];
        if (!r1.empty()) {
          R1& r = r1.back();
          if (r.kind == e.kind && r.k == e.k && r.p0 + r.n == i) {
            const long long as = e.a - (r.a + (long long)(r.n - 1) * r.as), bs = e.b - (r.b + (long long)(r.n - 1) * r.bs);
            if (r.n == 1) { r.as = e.kind ? e.a - r.a : 0; r.bs = e.kind == 2 ? e.b - r.b : 0; ++r.n; continue; }
            if ((e.kind == 0 || as == r.as) && (e.kind != 2 || bs == r.bs)) { ++r.n; continue; }
          }
        }
        r1.push_back(R1{i, 1, e.kind, e.k, e.a, 0, e.b, 0});
      }
      for (size_t q = 0; q < r1.size();) {
        const R1& s = r1[q];
        size_t rows = 1; long long pRow = 0, aRow = 0, bRow = 0;
        while (q + rows < r1.size()) {
          const R1& t = r1[q + rows];
          if (t.n != s.n || t.kind != s.kind || t.k != s.k || t.as != s.as || t.bs != s.bs) break;
          const long long dp = (long long)t.p0 - (long long)r1[q + rows - 1].p0, da = t.a - r1[q + rows - 1].a, db = t.b - r1[q + rows - 1].b;
          if (rows == 1) { pRow = dp; aRow = da; bRow = db; }
          else if (dp != pRow || (s.kind && da != aRow) || (s.kind == 2 && db != bRow)) break;
          ++rows;
          if ((size_t)rows * s.n >= (1u << 16)) break;      // (a block's share of work)
        }
        (trains ? host : others).push_back(SourceRun{P->w + s.p0, P->g ? P->g + s.p0 : nullptr, (unsigned)rows, (unsigned)s.n, pRow, s.a, aRow, s.as, s.b, bRow, s.bs, (real)s.k, s.kind});
        q += rows;
      }
      if (trains) covered += P->n;
    }
    nRuns = (unsigned)host.size();
    const unsigned nOther = (unsigned)others.size();
    SourceRun* otherRuns = nullptr;
    GK(cudaMalloc(&otherRuns, std::max<size_t>(others.size(), 1) * sizeof(SourceRun)));
    GK(cudaMemcpy(otherRuns, others.data(), others.size() * sizeof(SourceRun), cudaMemcpyHostToDevice));
    GK(cudaMalloc(&runs, std::max<size_t>(host.size(), 1) * sizeof(SourceRun)));
    GK(cudaMemcpy(runs, host.data(), host.size() * sizeof(SourceRun), cudaMemcpyHostToDevice));
    // the sources: from the port through the copies, then the carried ones
    if (nOther) sourceFromPortK<<<nOther, 256, 0, STREAM>>>(otherRuns, w);
    sourceFromPortK<<<nRuns, 256, 0, STREAM>>>(runs, w);
    for (int i = 0; i < nm; ++i) {
      const std::string key = "boltz2source." + std::to_string(i);
      if (!M.has(key)) continue;
      if (M.len(key) != params[i].n) { fprintf(stderr, "boltz2 source: %s carries %zu values, %zu expected\n", names[i].c_str(), M.len(key), params[i].n); exit(1); }
      std::vector<real> h(M.f(key), M.f(key) + params[i].n);
      GK(cudaMemcpy(params[i].w, h.data(), h.size() * sizeof(real), cudaMemcpyHostToDevice));
    }
    unsigned long long* bad; double* worst; GK(cudaMalloc(&bad, 8)); GK(cudaMalloc(&worst, 8));
    GK(cudaMemset(bad, 0, 8)); GK(cudaMemset(worst, 0, 8));
    for (auto& p : params) nanCountK<<<64, 256, 0, STREAM>>>(p.w, p.n, bad);
    unsigned long long nNan; GK(cudaMemcpy(&nNan, bad, 8, cudaMemcpyDeviceToHost));
    if (nNan) {
      for (auto& p : params) {
        GK(cudaMemset(bad, 0, 8)); nanCountK<<<64, 256, 0, STREAM>>>(p.w, p.n, bad);
        unsigned long long c; GK(cudaMemcpy(&c, bad, 8, cudaMemcpyDeviceToHost));
        if (c) fprintf(stderr, "  %s: %llu of %zu elements set by nothing\n", p.name.c_str(), c, p.n);
      }
      fprintf(stderr, "boltz2 source: %llu source elements are neither read by the port nor carried by the map\n", nNan); exit(1);
    }
    GK(cudaMemset(bad, 0, 8));
    sourceCheckK<<<nRuns, 256, 0, STREAM>>>(runs, w, bad, worst);
    if (nOther) sourceCheckK<<<nOther, 256, 0, STREAM>>>(otherRuns, w, bad, worst);
    unsigned long long nb2; double wd; GK(cudaMemcpy(&nb2, bad, 8, cudaMemcpyDeviceToHost)); GK(cudaMemcpy(&wd, worst, 8, cudaMemcpyDeviceToHost));
    GK(cudaFree(bad)); GK(cudaFree(worst)); GK(cudaFree(otherRuns));
    for (auto& q : extra) GK(cudaFree(q.w));
    size_t nmEl = 0; for (auto& p : params) nmEl += p.n;
    printf("boltz2 source: %d Boltz-2 tensors (%.1f M parameters) behind %zu port tensors (%.1f M elements) in %u runs; "
           "the port rebuilt from them: %llu elements differ%s (and %zu walk tensors the graph does not read: values only)\n",
           nm, nmEl / 1e6, ports.all.size(), covered / 1e6, nRuns, nb2,
           nb2 ? (" (worst " + std::to_string(wd) + ")").c_str() : "", extra.size());
    for (auto& q : extra) printf("  (not read by the graph: %s)\n", q.name.c_str());
    if (nb2) exit(1);
  }
  // the port's tensors from the sources (after every optimiser step, and after swapping in the EMA)
  void gather() { sourceGatherK<<<nRuns, 256, 0, STREAM>>>(runs, w); }
  // the port's gradients onto the sources (their g zeroed first)
  void scatter() {
    GK(cudaMemsetAsync(g, 0, total * sizeof(real), STREAM));
    sourceScatterK<<<nRuns, 256, 0, STREAM>>>(runs, w, g);
  }
  int index(const std::string& nm) const { for (size_t i = 0; i < names.size(); ++i) if (names[i] == nm) return (int)i; return -1; }
};

}  // namespace grad
