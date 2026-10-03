// Shared infrastructure: the exported model file, device weights, scratch, cuBLAS.
#pragma once
#include <cublas_v2.h>
#include <cublasLt.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <fstream>
#include <functional>
#include <array>
#include <map>
#include <deque>
#include <tuple>
#include <sstream>
#include <thread>
#include <string>
#include <type_traits>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1); } } while (0)

using half = __half;
inline cublasHandle_t H;
inline cudaStream_t STREAM = cudaStreamPerThread;

// ---------------------------------------------------------------- the exported model
struct Entry { char kind; size_t offset, length; double value; int seg; size_t devOffset = 0; };
// model.idx/model.bin pairs, mapped: the input's directory and (af3 --weights=DIR) the weights'
// On the device every tensor starts on 16 bytes (devOffset), wherever the file packed it: cuBLAS's
// vector-load kernels need it (align1 kernels otherwise, and a batched GEMM, which cannot see its
// pointers, faults). runs: the file's byte ranges and where each lands, in file order.
struct Run { size_t src, dst, bytes; };
// data: the file mapped, only once the host reads an entry (f()): a large file goes to the device by
// pread into pinned buffers - a mapping of the 2.9 GB native/ef2 weights cost ~300 ms of page faults to
// read and 275 ms more to tear down when the process exited, after its PDB was written
struct Segment { const float* data; size_t bytes; float* device; std::map<std::string, void*> halfMirrors;
                 int fd = -1;
                 size_t deviceBytes = 0; std::vector<Run> runs; };
struct Model {
  std::map<std::string, Entry> index;
  mutable std::set<std::string> touched;  // every entry whose values were read (see unreadWeights)
  std::deque<Segment> segs;          // (a deque: an upload thread holds a reference while another file loads)
  void load(const std::string& dir) {
    std::ifstream idx(dir + "/model.idx");
    if (!idx) { fprintf(stderr, "no %s/model.idx\n", dir.c_str()); exit(1); }
    int seg = (int)segs.size();
    std::string line;
    while (std::getline(idx, line)) {
      std::istringstream in(line); char kind; std::string name; in >> kind >> name;
      Entry e{kind, 0, 0, 0, seg};
      if (kind == 'm') in >> e.value; else in >> e.offset >> e.length;
      if (index.count(name)) { fprintf(stderr, "%s is in two model directories\n", name.c_str()); exit(1); }
      index[name] = e;
    }
    // neither read nor mapped yet (a 1.4 GB read into a host vector was 1.7 s of every run)
    int fd = open((dir + "/model.bin").c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "no %s/model.bin\n", dir.c_str()); exit(1); }
    struct stat st; fstat(fd, &st);
    size_t bytes = (size_t)st.st_size;
    Segment sg{nullptr, bytes, nullptr};
    sg.fd = fd;
    std::vector<Entry*> ts;
    for (auto& [name, e] : index) if (e.seg == seg && e.kind != 'm') ts.push_back(&e);
    std::sort(ts.begin(), ts.end(), [](const Entry* a, const Entry* b) { return a->offset < b->offset; });
    size_t at = 0, end = 0;          // a tensor sharing (or overlapping) the last one's range keeps its place in it
    for (Entry* e : ts) {
      if (!sg.runs.empty() && e->offset < end) {
        const Run& r = sg.runs.back();
        e->devOffset = (r.dst + (e->offset * 4 - r.src)) / 4;
        if (e->offset + e->length > end) { fprintf(stderr, "%s/model.bin: overlapping tensors\n", dir.c_str()); exit(1); }
        continue;
      }
      at = (at + 3) / 4 * 4;
      e->devOffset = at;
      sg.runs.push_back({e->offset * 4, at * 4, e->length * 4});
      at += e->length; end = e->offset + e->length;
      if (end * 4 > bytes) { fprintf(stderr, "%s/model.bin is shorter than its index\n", dir.c_str()); exit(1); }
    }
    sg.deviceBytes = at * 4;
    segs.push_back(sg);
  }
  // drop a directory's entries, its mapping and its device copy (one input of a batch, done);
  // returns the names it held so the caches keyed on them can be cleared too
  std::vector<std::string> unload(int seg) {
    std::vector<std::string> names;
    for (auto it = index.begin(); it != index.end();) {
      if (it->second.seg == seg) { names.push_back(it->first); touched.erase(it->first); it = index.erase(it); }
      else ++it;
    }
    Segment& s = segs[seg];
    if (s.device) { cudaDeviceSynchronize(); cudaFree(s.device); s.device = nullptr; }
    for (auto& [g, h] : s.halfMirrors) cudaFree(h);
    s.halfMirrors.clear();
    if (s.data) { munmap((void*)s.data, std::max<size_t>(s.bytes, 1)); s.data = nullptr; }
    if (s.fd >= 0) { close(s.fd); s.fd = -1; }
    return names;
  }
  // the whole directory's device copy now (af3 --wait-input does this while the input is exported)
  void upload(int seg) {
    for (auto& [name, e] : index) if (e.seg == seg && e.kind != 'm') { dev(name); touched.erase(name); return; }
  }
  // ...or in the background: the device copy is allocated at once and filled by a thread, so kernels
  // can be launched against it meanwhile - a warm-up, whose answers are garbage until waitUploads()
  // (and whose derived weights must then be forgotten: forgetDerivedWeights())
  std::map<int, std::thread> pending;
  void uploadAsync(int seg) {
    Segment& s = segs[seg];
    if (s.device || pending.count(seg)) return;
    if (cudaMalloc(&s.device, std::max<size_t>(s.deviceBytes, 4)) != cudaSuccess) {
      fprintf(stderr, "cannot put a model.bin (%zu bytes) on the device\n", s.bytes); exit(1);
    }
    pending[seg] = std::thread([&s] {
      if (!copyUp(s)) { fprintf(stderr, "cannot put a model.bin (%zu bytes) on the device\n", s.bytes); exit(1); }
    });
  }
  void waitUploads() { for (auto& [seg, th] : pending) th.join(); pending.clear(); }
  // the device copy rebuilt without the entries `drop` names - read from their f16 mirror from now on
  // (build it first); dev() of a dropped entry is an error. The caller forgets any pointer it cached.
  static constexpr size_t DROPPED = ~(size_t)0;
  size_t compact(int seg, const std::function<bool(const std::string&)>& drop) {
    Segment& s = segs[seg];
    if (!s.device) { fprintf(stderr, "compact: the file is not on the device\n"); exit(1); }
    std::vector<std::pair<Entry*, std::string>> keep;
    for (auto& [name, e] : index) if (e.seg == seg && e.kind != 'm' && e.devOffset != DROPPED) keep.push_back({&e, name});
    std::sort(keep.begin(), keep.end(), [](auto& a, auto& b) { return a.first->devOffset < b.first->devOffset; });
    size_t at = 0; std::vector<std::array<size_t, 3>> moves; std::vector<Entry*> dropped;
    for (auto& [e, name] : keep) {
      if (drop(name)) { dropped.push_back(e); continue; }
      at = (at + 3) / 4 * 4;
      moves.push_back({e->devOffset, at, e->length}); e->devOffset = at; at += e->length;
    }
    float* fresh; CK(cudaMalloc(&fresh, std::max<size_t>(at, 1) * 4));
    for (auto& m : moves) CK(cudaMemcpyAsync(fresh + m[1], s.device + m[0], m[2] * 4, cudaMemcpyDeviceToDevice, STREAM));
    CK(cudaStreamSynchronize(STREAM));
    CK(cudaFree(s.device));
    size_t freed = s.deviceBytes - at * 4;
    s.device = fresh; s.deviceBytes = at * 4;
    for (Entry* e : dropped) e->devOffset = DROPPED;
    return freed;
  }
  // a file's bytes onto the device: from the mapping through small pinned buffers, three threads
  // copying each 8 MB piece while the last one's DMA runs - 77 against 170 ms for the 1.47 GB of
  // weights from pageable memory (the pinning costs 19 of it, a larger piece costs more)
  static const float* mapped(Segment& s) {
    if (!s.data) {
      void* p = mmap(nullptr, std::max<size_t>(s.bytes, 1), PROT_READ, MAP_PRIVATE, s.fd, 0);
      if (p == MAP_FAILED) { fprintf(stderr, "cannot map a model.bin (%zu bytes)\n", s.bytes); exit(1); }
      s.data = (const float*)p;
    }
    return s.data;
  }
  static bool copyUp(Segment& s) {
    if (s.bytes < ((size_t)64 << 20)) {
      mapped(s);
      for (const Run& r : s.runs)
        if (cudaMemcpy((char*)s.device + r.dst, (const char*)s.data + r.src, r.bytes, cudaMemcpyHostToDevice) != cudaSuccess) return false;
      return true;
    }
    const size_t CH = (size_t)8 << 20; const int NB = 3, THREADS = 3;
    char* stage[NB]; cudaEvent_t ev[NB]; cudaStream_t st;
    if (cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess) return false;
    for (int b = 0; b < NB; ++b)
      if (cudaHostAlloc(&stage[b], CH, cudaHostAllocDefault) != cudaSuccess ||
          cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming) != cudaSuccess) return false;
    const char* src = (const char*)s.data; char* dst = (char*)s.device;     // (src: only if already mapped)
    size_t run = 0; bool readOk = true;
    for (size_t off = 0, i = 0; off < s.bytes; off += CH, ++i) {
      int b = (int)(i % NB);
      if (i >= (size_t)NB && cudaEventSynchronize(ev[b]) != cudaSuccess) return false;
      size_t n = std::min(CH, s.bytes - off), per = (n + THREADS - 1) / THREADS;
      std::vector<std::thread> pool;
      for (int t = 0; t < THREADS; ++t) {
        size_t lo = t * per; if (lo >= n) break;
        size_t len = std::min(per, n - lo);
        if (src) pool.emplace_back([=] { memcpy(stage[b] + lo, src + off + lo, len); });
        else pool.emplace_back([=, &readOk] {
          for (size_t got = 0; got < len;) {
            ssize_t r = pread(s.fd, stage[b] + lo + got, len - got, (off_t)(off + lo + got));
            if (r <= 0) { readOk = false; return; }
            got += (size_t)r;
          }
        });
      }
      for (auto& th : pool) th.join();
      if (!readOk) return false;
      // every run's part inside this piece, to where it lands
      while (run < s.runs.size() && s.runs[run].src + s.runs[run].bytes <= off) ++run;
      for (size_t k = run; k < s.runs.size() && s.runs[k].src < off + n; ++k) {
        const Run& r = s.runs[k];
        size_t lo = std::max(r.src, off), hi = std::min(r.src + r.bytes, off + n);
        if (cudaMemcpyAsync(dst + r.dst + (lo - r.src), stage[b] + (lo - off), hi - lo, cudaMemcpyHostToDevice, st) != cudaSuccess)
          return false;
      }
      if (cudaEventRecord(ev[b], st) != cudaSuccess) return false;
    }
    bool ok = cudaStreamSynchronize(st) == cudaSuccess;
    for (int b = 0; b < NB; ++b) { cudaFreeHost(stage[b]); cudaEventDestroy(ev[b]); }
    cudaStreamDestroy(st);
    return ok;
  }
  // the device copy of an entry: one allocation and one copy per file
  const float* dev(const std::string& k) {
    const Entry& e = at(k);
    touched.insert(k);
    Segment& s = segs[e.seg];
    if (!s.device) {
      if (cudaMalloc(&s.device, std::max<size_t>(s.deviceBytes, 4)) != cudaSuccess || !copyUp(s)) {
        fprintf(stderr, "cannot put a model.bin (%zu bytes) on the device\n", s.bytes); exit(1);
      }
    }
    if (e.devOffset == DROPPED) { fprintf(stderr, "%s: its f32 device copy was dropped (read its f16 mirror)\n", k.c_str()); exit(1); }
    return s.device + e.devOffset;
  }
  bool has(const std::string& k) const { return index.count(k) > 0; }
  const Entry& at(const std::string& k) const {
    auto it = index.find(k);
    if (it == index.end()) { fprintf(stderr, "model has no %s\n", k.c_str()); exit(1); }
    return it->second;
  }
  double meta(const std::string& k) const { return at(k).value; }
  double meta(const std::string& k, double fallback) const { return has(k) ? at(k).value : fallback; }
  bool flag(const std::string& k) const { return has(k) && at(k).value != 0; }
  const float* f(const std::string& k) const {
    const Entry& e = at(k); touched.insert(k);
    return mapped(const_cast<Segment&>(segs[e.seg])) + e.offset;
  }
  const int* i(const std::string& k) const { return (const int*)f(k); }
  size_t len(const std::string& k) const { return at(k).length; }
};
inline Model M;
// The weight tensors a fold never read, digits collapsed: a convention the checkpoint carries and
// this port does not apply shows up here instead of as a quietly wrong structure (rf3's atom q/k
// norms were found this way, after the fact).
inline void unreadWeights() {
  std::map<std::string, int> families;
  for (auto& [name, e] : M.index) {
    if (e.kind == 'm' || M.touched.count(name)) continue;
    if (name.rfind("trunk.", 0) && name.rfind("diffusion.", 0) && name.rfind("confidence.", 0) &&
        name.rfind("targetFeat.", 0) && name.rfind("atomReference.", 0) && name.rfind("expander.", 0) &&
        name.rfind("refiner.", 0) && name.rfind("ddeConfidence.", 0)) continue;
    std::string f; bool digit = false;
    for (char c : name) { if (isdigit((unsigned char)c)) { if (!digit) f += 'N'; digit = true; } else { f += c; digit = false; } }
    ++families[f];
  }
  if (families.empty()) return;
  printf("weights never read (%zu families):", families.size());
  for (auto& [f, k] : families) printf(" %s(%d)", f.c_str(), k);
  printf("\n");
}

// ---------------------------------------------------------------- device memory
inline float* dalloc(size_t n) { float* p; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * 4)); return p; }
template <class T> T* dallocT(size_t n) { T* p; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T))); return p; }
template <class T> T* upload(const T* h, size_t n) {
  T* p = dallocT<T>(n); CK(cudaMemcpy(p, h, n * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
// Named scratch, kept for the process and grown when asked for more.
inline std::map<std::string, std::pair<void*, size_t>> SCRATCH;
// Every scratch buffer given back: between the trunk, the denoiser and the confidence head of a
// large input, so each phase has the whole card (every phase asks for its buffers again).
inline void releaseScratch() {
  CK(cudaDeviceSynchronize());
  for (auto& [name, slot] : SCRATCH) { if (slot.first) CK(cudaFree(slot.first)); slot = {nullptr, 0}; }
}
template <class T> T* scratch(const std::string& name, size_t n) {
  auto& [p, have] = SCRATCH[name];
  if (have < n * sizeof(T)) {
    if (p) CK(cudaFree(p));
    if (cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)) != cudaSuccess) {
      size_t held = 0; for (auto& [k, v] : SCRATCH) held += v.second;
      size_t freeB, totalB; cudaMemGetInfo(&freeB, &totalB);
      fprintf(stderr, "out of device memory: scratch %s wants %.2f GB; scratch holds %.2f GB, %.2f of %.2f GB free\n",
              name.c_str(), n * sizeof(T) / 1e9, held / 1e9, freeB / 1e9, totalB / 1e9);
      std::vector<std::pair<size_t, std::string>> big;
      for (auto& [k, v] : SCRATCH) big.push_back({v.second, k});
      std::sort(big.rbegin(), big.rend());
      for (size_t i = 0; i < big.size() && i < 12; ++i) fprintf(stderr, "  %8.2f GB  %s\n", big[i].first / 1e9, big[i].second.c_str());
      exit(1);
    }
    have = n * sizeof(T);
  }
  return (T*)p;
}

// The process's end once its outputs are written: flushed first. (std::exit, not _exit: _exit measured no
// faster - the time is the driver's release of the device, which --detach-output takes off a caller's
// clock - and it skipped the atexit reports, AF2_GEMM_TIMES among them.)
[[noreturn]] inline void finish(int rc) {
  fflush(stdout); fflush(stderr);
  std::exit(rc);
}
// LOCALFOLD_MEM=1: device memory in use at a phase boundary, and the largest scratch buffers
inline void memReport(const char* at) {
  if (!getenv("LOCALFOLD_MEM")) return;
  CK(cudaDeviceSynchronize()); size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
  size_t held = 0; std::vector<std::pair<size_t, std::string>> big;
  for (auto& [k, v] : SCRATCH) { held += v.second; if (v.second) big.push_back({v.second, k}); }
  std::sort(big.rbegin(), big.rend());
  printf("  memory %-22s %6.2f GB in use, scratch %.2f:", at, (tot - fr) / 1e9, held / 1e9);
  for (size_t i = 0; i < big.size() && i < (getenv("LOCALFOLD_MEM_ALL") ? big.size() : 6); ++i) printf(" %s %.2f", big[i].second.c_str(), big[i].first / 1e9);
  printf("\n");
}
__global__ void toHalfK(const float* x, half* y, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = __float2half(x[i]);
}
inline unsigned blocks(size_t n, int t = 256) { return (unsigned)((n + t - 1) / t); }

// Weights: device copies made on first use, f32 and (on request) f16.
inline std::map<std::string, float*> WF;
inline std::map<std::string, half*> WH;
inline std::map<std::string, size_t> WLEN;
// a weight built on the device (the folded conditioning projections), under a name W() serves
inline void deviceWeight(const std::string& k, float* p, size_t n);
inline bool hasW(const std::string& k) { return M.has(k) || WLEN.count(k); }
inline size_t lenW(const std::string& k) {
  auto it = WLEN.find(k);
  return it != WLEN.end() && !M.has(k) ? it->second : M.len(k);
}
inline const float* W(const std::string& k) {
  auto it = WF.find(k);
  if (it != WF.end()) return it->second;
  WLEN[k] = M.len(k);
  return WF[k] = const_cast<float*>(M.dev(k));
}
inline void deviceWeight(const std::string& k, float* p, size_t n) { WF[k] = p; WLEN[k] = n; }
// A weight concatenated along its output columns, on the device from its copies of the parts:
// each part (C, width) row-major (in, out), or stored (width, C) if `transposed`; a part with no
// name is zero columns. Returns `key`, under which W()/Wh() serve the (C, sum of widths) result.
struct Part { std::string name; int width; bool transposed; };
__global__ void transposeIntoK(const float* src, float* dst, int C, int width, size_t ld) {   // dst[c][o] = src[o][c]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)C * width) return;
  int c = (int)(t / width), o = (int)(t % width);
  dst[c * ld + o] = src[(size_t)o * C + c];
}
inline std::string concatColumns(const std::string& key, int C, const std::vector<Part>& parts) {
  if (WF.count(key)) return key;
  size_t total = 0; for (auto& p : parts) total += p.width;
  float* d = dalloc((size_t)C * total);
  size_t off = 0;
  for (auto& p : parts) {
    if (p.name.empty()) {
      CK(cudaMemset2DAsync(d + off, total * 4, 0, p.width * 4, C, STREAM));
    } else {
      if (lenW(p.name) != (size_t)C * p.width) {
        fprintf(stderr, "%s has %zu elements, not %d x %d\n", p.name.c_str(), lenW(p.name), C, p.width); exit(1);
      }
      if (p.transposed) transposeIntoK<<<blocks((size_t)C * p.width), 256, 0, STREAM>>>(W(p.name), d + off, C, p.width, total);
      else CK(cudaMemcpy2DAsync(d + off, total * 4, W(p.name), p.width * 4, p.width * 4, C, cudaMemcpyDeviceToDevice, STREAM));
    }
    off += p.width;
  }
  deviceWeight(key, d, (size_t)C * total);
  return key;
}
inline std::set<std::string> WH_MIRROR;     // the f16 views into a file's mirror (freed with it)
// a file's f16 copy, made once per GROUP - the first f16 read of any of its weights converts every
// float tensor of that group in ONE launch (it was ~800 allocations and conversions, one a weight, in
// the first fold); each tensor starts on 16 bytes, as its own allocation did, for the vector loads.
// A group is a name's first '/'-separated part ("" without one): native/ef2's file holds the folding
// bundle (f/) and ESM-C (c/), which reads only its f32 copy - mirroring it too was 1.2 GB never read
inline std::string halfGroup(const std::string& k) { size_t at = k.find('/'); return at == std::string::npos ? "" : k.substr(0, at); }
inline std::map<std::string, half*> WH_AT;
__global__ void convertTableK(const float* src, half* dst, const size_t* from, const size_t* to, const size_t* len) {
  size_t e = blockIdx.y, n = len[e];
  const float* f = src + from[e]; half* h = dst + to[e];
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) h[i] = __float2half(f[i]);
}
inline const half* segmentHalf(const std::string& k) {
  const Entry& e0 = M.at(k);
  Segment& s = M.segs[e0.seg];
  std::string group = halfGroup(k);
  void*& mirror = s.halfMirrors[group];
  if (!mirror) {
    M.dev(k);
    std::vector<size_t> from, to, len; std::vector<std::string> names; size_t total = 0;
    for (auto& [name, e] : M.index) {
      if (e.seg != e0.seg || e.kind != 't' || halfGroup(name) != group) continue;
      from.push_back(e.devOffset); to.push_back(total); len.push_back(e.length); names.push_back(name);
      total += (e.length + 7) / 8 * 8;
    }
    CK(cudaMalloc(&mirror, std::max<size_t>(total, 1) * 2));
    size_t* table; CK(cudaMalloc(&table, from.size() * 3 * sizeof(size_t)));
    CK(cudaMemcpy(table, from.data(), from.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(table + from.size(), to.data(), to.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(table + 2 * from.size(), len.data(), len.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    for (size_t first = 0; first < from.size(); first += 65535) {
      unsigned count = (unsigned)std::min<size_t>(65535, from.size() - first);
      convertTableK<<<dim3(8, count), 256, 0, STREAM>>>(s.device, (half*)mirror, table + first,
                                                       table + from.size() + first, table + 2 * from.size() + first);
    }
    CK(cudaStreamSynchronize(STREAM)); CK(cudaFree(table));
    for (size_t i = 0; i < names.size(); ++i) WH_AT[names[i]] = (half*)mirror + to[i];
  }
  auto it = WH_AT.find(k);
  if (it == WH_AT.end()) { fprintf(stderr, "%s is not a float tensor of its file\n", k.c_str()); exit(1); }
  return it->second;
}
inline const half* Wh(const std::string& k) {
  auto it = WH.find(k);
  if (it != WH.end()) return it->second;
  if (M.has(k) && (!WF.count(k) || WF[k] == M.dev(k))) {     // a file's entry, not one built here
    WLEN[k] = M.len(k); WH_MIRROR.insert(k);
    return WH[k] = const_cast<half*>(segmentHalf(k));
  }
  const float* f = W(k); size_t n = WLEN[k];
  half* h = dallocT<half>(n);
  toHalfK<<<blocks(n), 256, 0, STREAM>>>(f, h, n);
  return WH[k] = h;
}
// a file's device copy without the entries `drop` names (they must have their f16 mirror already):
// every cached f32 pointer into it forgotten
inline size_t compactWeights(int seg, const std::function<bool(const std::string&)>& drop) {
  size_t freed = M.compact(seg, drop);
  for (auto it = WF.begin(); it != WF.end();) {
    if (M.has(it->first) && M.at(it->first).seg == seg) it = WF.erase(it); else ++it;
  }
  return freed;
}
// every weight derived from the device copies forgotten - f16 mirrors and copies, weights built on the
// device - after a warm-up that ran while a copy was still arriving (M.uploadAsync): they are rebuilt
// from the finished copy on their next use. A port's own caches of derived weights register here.
// (Not for native/af3 as it stands: TCACHE folds its conditioning weights once per process and keeps
// the names, and OpenDDE's structural.cuh registers VIEWS into a file's copy, which this would free.)
inline std::vector<std::function<void()>> FORGET_HOOKS;
inline void forgetDerivedWeights() {
  CK(cudaDeviceSynchronize());
  for (auto& [k, h] : WH) if (!WH_MIRROR.count(k)) CK(cudaFree(h));
  WH.clear(); WH_MIRROR.clear(); WH_AT.clear();
  for (auto& s : M.segs) { for (auto& [g, h] : s.halfMirrors) CK(cudaFree(h)); s.halfMirrors.clear(); }
  for (auto it = WF.begin(); it != WF.end();) {
    if (M.has(it->first) && it->second == M.dev(it->first)) { ++it; continue; }
    CK(cudaFree(it->second)); WLEN.erase(it->first); it = WF.erase(it);
  }
  for (auto& f : FORGET_HOOKS) f();
}
inline std::map<std::string, int*> IDEV;
inline const int* Idev(const std::string& k) {
  auto it = IDEV.find(k);
  if (it != IDEV.end()) return it->second;
  return IDEV[k] = (int*)M.dev(k);
}
// an unloaded input's device views and f16 copies, so the next input's fields are read afresh
inline void forgetEntries(const std::vector<std::string>& names) {
  for (auto& k : names) {
    auto h = WH.find(k);
    if (h != WH.end()) { if (!WH_MIRROR.erase(k)) cudaFree(h->second); WH.erase(h); }
    WF.erase(k); WLEN.erase(k); IDEV.erase(k); WH_AT.erase(k);
  }
}
inline const float* Fdev(const std::string& k) {     // non-weight float inputs (batch fields)
  return W(k);
}

// ---------------------------------------------------------------- precision
// The activations' storage type is T throughout a stage: float for the precise path that
// is checked against AF3, half for the fast one. Residual streams stay f32 either way.
template <class T> constexpr cudaDataType cudaType() {
  return std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_16F;
}
template <class T> __device__ __forceinline__ float toF(T v) {
  if constexpr (std::is_same_v<T, float>) return v; else return __half2float(v);
}
template <class T> __device__ __forceinline__ T fromF(float v) {
  if constexpr (std::is_same_v<T, float>) return v; else return __float2half(v);
}
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }

// The fast path's remaining f32 GEMMs (the conditioning, the atom blocks' aggregation and
// broadcast projections, ...) on the tensor cores in TF32 - a 10-bit mantissa, as f16 has
inline bool F32_TF32 = false;
// Row-major Y[rows x out] = X[rows x in] W + beta Y, W (in,out) or (out,in) if transposed.
// X and W in T (W's f16 copy for half); Y in TY; f32 accumulation always.
template <class T, class TY>
void linear(const T* X, TY* Y, size_t rows, int in, int out, const std::string& w,
            bool transposed = false, float beta = 0.f) {
  const float one = 1.f;
  const void* Wp;
  if constexpr (std::is_same_v<T, float>) Wp = W(w); else Wp = Wh(w);
  if (lenW(w) != (size_t)in * out) {
    fprintf(stderr, "%s has %zu elements, not %d x %d\n", w.c_str(), lenW(w), in, out); exit(1);
  }
  bool tf32 = std::is_same_v<T, float> && F32_TF32;
  // LOCALFOLD_GEMM_TIMES: every call timed by events, summed by weight and shape, printed at exit (an
  // analysis aid; not under graph capture)
  static bool timing = getenv("LOCALFOLD_GEMM_TIMES") != nullptr;
  struct Timed { std::string key; double flop; cudaEvent_t a, b; };
  static std::vector<Timed>* timed = nullptr;
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  if (timing) cudaStreamIsCapturing(STREAM, &cap);
  bool timeThis = timing && cap == cudaStreamCaptureStatusNone;
  cudaEvent_t ea = nullptr, eb = nullptr;
  if (timeThis) {
    if (!timed) {
      timed = new std::vector<Timed>;
      atexit([] {
        std::map<std::string, std::tuple<double, double, int>> sum;
        for (auto& t : *timed) { float ms; cudaEventElapsedTime(&ms, t.a, t.b); auto& v = sum[t.key]; std::get<0>(v) += ms; std::get<1>(v) += t.flop; std::get<2>(v)++; }
        std::vector<std::pair<double, std::string>> order;
        for (auto& [k, v] : sum) order.push_back({std::get<0>(v), k});
        std::sort(order.rbegin(), order.rend());
        for (size_t i = 0; i < order.size() && i < 30; ++i) {
          auto& v = sum[order[i].second];
          printf("  %9.2f ms %6d x  %-60s %6.1f TFLOP/s\n", std::get<0>(v), std::get<2>(v), order[i].second.c_str(),
                 std::get<1>(v) / (std::get<0>(v) * 1e-3) / 1e12);
        }
      });
    }
    cudaEventCreate(&ea); cudaEventCreate(&eb); cudaEventRecord(ea, STREAM);
  }
  CB(cublasGemmEx(H, transposed ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, out, (int)rows, in, &one,
                  Wp, cudaType<T>(), transposed ? in : out, X, cudaType<T>(), in, &beta,
                  Y, cudaType<TY>(), out, tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F,
                  std::is_same_v<T, float> && !tf32 ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  if (timeThis) {
    cudaEventRecord(eb, STREAM);
    std::string k = w; for (char& ch : k) if (isdigit((unsigned char)ch)) ch = 'N';
    char shape[80]; snprintf(shape, sizeof shape, " %zux%dx%d%s", rows, in, out, std::is_same_v<TY, float> ? " f32out" : "");
    timed->push_back({k + shape, 2.0 * rows * in * out, ea, eb});
  }
}

// ---------------------------------------------------------------- checking and timing
inline double relRms(const float* a, const float* b, size_t n) {
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) { double e = (double)a[i] - b[i]; num += e * e; den += (double)b[i] * b[i]; }
  return std::sqrt(num / std::max(den, 1e-300));
}
inline std::vector<float> download(const float* d, size_t n) {
  std::vector<float> h(n);
  CK(cudaStreamSynchronize(STREAM));
  CK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
  return h;
}
// Compare a device tensor against an exported oracle tensor, if that oracle exists.
inline void check(const char* label, const float* d, size_t n, const std::string& oracle) {
  if (!M.has(oracle)) { printf("  %-24s (no oracle %s)\n", label, oracle.c_str()); return; }
  if (M.len(oracle) != n) {
    printf("  %-24s LENGTH %zu against the oracle's %zu\n", label, n, M.len(oracle)); return;
  }
  auto h = download(d, n);
  printf("  %-24s relRMS %.3e\n", label, relRms(h.data(), M.f(oracle), n));
}
inline bool STAGES = false;
inline std::map<std::string, double> STAGE_MS;
inline void stage(const char* name) {
  static auto last = std::chrono::steady_clock::now();
  // a launch that never ran (too much shared memory, a bad grid) reports nothing by itself: ask at
  // every stage boundary, which costs no synchronisation (IntelliFold-2's centre norm was silently
  // skipped this way)
  cudaError_t launch = cudaGetLastError();
  if (launch != cudaSuccess) { fprintf(stderr, "CUDA %s before stage %s\n", cudaGetErrorString(launch), name ? name : "(start)"); exit(1); }
  if (!STAGES) return;
  CK(cudaStreamSynchronize(STREAM));
  auto now = std::chrono::steady_clock::now();
  if (name) STAGE_MS[name] += std::chrono::duration<double, std::milli>(now - last).count();
  last = now;
}
