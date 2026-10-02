// Shared infrastructure: the exported model file, device weights, scratch, cuBLAS.
#pragma once
#include <cublas_v2.h>
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
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
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
struct Entry { char kind; size_t offset, length; double value; int seg; };
// model.idx/model.bin pairs, mapped: the input's directory and (af3 --weights=DIR) the weights'
struct Segment { const float* data; size_t bytes; float* device; };
struct Model {
  std::map<std::string, Entry> index;
  std::vector<Segment> segs;
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
    // mapped, not read: a 1.4 GB read into a host vector was 1.7 s of every run
    int fd = open((dir + "/model.bin").c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "no %s/model.bin\n", dir.c_str()); exit(1); }
    struct stat st; fstat(fd, &st);
    size_t bytes = (size_t)st.st_size;
    void* p = mmap(nullptr, std::max<size_t>(bytes, 1), PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { fprintf(stderr, "cannot map %s/model.bin\n", dir.c_str()); exit(1); }
    close(fd);
    segs.push_back({(const float*)p, bytes, nullptr});
  }
  // the device copy of an entry: one allocation and one copy per file
  const float* dev(const std::string& k) {
    const Entry& e = at(k);
    Segment& s = segs[e.seg];
    if (!s.device) {
      if (cudaMalloc(&s.device, std::max<size_t>(s.bytes, 4)) != cudaSuccess ||
          cudaMemcpy(s.device, s.data, s.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "cannot put a model.bin (%zu bytes) on the device\n", s.bytes); exit(1);
      }
    }
    return s.device + e.offset;
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
  const float* f(const std::string& k) const { const Entry& e = at(k); return segs[e.seg].data + e.offset; }
  const int* i(const std::string& k) const { return (const int*)f(k); }
  size_t len(const std::string& k) const { return at(k).length; }
};
inline Model M;

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
inline const half* Wh(const std::string& k) {
  auto it = WH.find(k);
  if (it != WH.end()) return it->second;
  const float* f = W(k); size_t n = WLEN[k];
  half* h = dallocT<half>(n);
  toHalfK<<<blocks(n), 256, 0, STREAM>>>(f, h, n);
  return WH[k] = h;
}
inline const int* Idev(const std::string& k) {
  static std::map<std::string, int*> cache;
  auto it = cache.find(k);
  if (it != cache.end()) return it->second;
  return cache[k] = (int*)M.dev(k);
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
  CB(cublasGemmEx(H, transposed ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, out, (int)rows, in, &one,
                  Wp, cudaType<T>(), transposed ? in : out, X, cudaType<T>(), in, &beta,
                  Y, cudaType<TY>(), out, tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F,
                  std::is_same_v<T, float> && !tf32 ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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
  if (!STAGES) return;
  CK(cudaStreamSynchronize(STREAM));
  auto now = std::chrono::steady_clock::now();
  if (name) STAGE_MS[name] += std::chrono::duration<double, std::milli>(now - last).count();
  last = now;
}
