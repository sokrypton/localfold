// Shared infrastructure: the exported model file, device weights, scratch, cuBLAS.
#pragma once
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <charconv>
#include <dirent.h>
#include <dlfcn.h>
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
#include <mutex>
#include <condition_variable>
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
// The named scratch comes from one stream-ordered pool that keeps what it is given back (release threshold: never):
// a large input gives each stage's scratch back for the next, and a fresh cudaMalloc of a big buffer is not cheap
// here - 0.9 ms at 1 GB, 37 at 2, 145 at 4 - while the pool hands the same pages to the next stage's buffers, of
// any shape, in 0.01 ms (it reserves the most a stage asked for, not the sum). What it holds idle is free memory to
// every question asked of the card (deviceMemInfo), and a plain allocation refused while it holds some trims it
// and asks again (devMalloc).
inline cudaMemPool_t scratchPool() {
  static cudaMemPool_t pool = [] {
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDevice;
    CK(cudaGetDevice(&props.location.id));
    cudaMemPool_t p; CK(cudaMemPoolCreate(&p, &props));
    uint64_t never = ~0ull; CK(cudaMemPoolSetAttribute(p, cudaMemPoolAttrReleaseThreshold, &never));
    return p;
  }();
  return pool;
}
inline size_t poolIdle() {
  // (a pool may not be asked while a stream captures - a recycle pass asks roomFor as it is captured - so there the
  // last answer, which the eager pass it repeats was given)
  static size_t last = 0;
  cudaStreamCaptureStatus capturing; CK(cudaStreamIsCapturing(STREAM, &capturing));
  if (capturing != cudaStreamCaptureStatusNone) return last;
  uint64_t reserved = 0, used = 0;
  CK(cudaMemPoolGetAttribute(scratchPool(), cudaMemPoolAttrReservedMemCurrent, &reserved));
  CK(cudaMemPoolGetAttribute(scratchPool(), cudaMemPoolAttrUsedMemCurrent, &used));
  return last = reserved > used ? reserved - used : 0;
}
inline void deviceMemInfo(size_t* f, size_t* t) { CK(cudaMemGetInfo(f, t)); *f += poolIdle(); }
inline void trimPool() { CK(cudaDeviceSynchronize()); CK(cudaMemPoolTrimTo(scratchPool(), 0)); }
template <class P> cudaError_t devMalloc(P** p, size_t bytes) {
  cudaError_t e = cudaMalloc((void**)p, bytes);
  if (e == cudaErrorMemoryAllocation && poolIdle()) { cudaGetLastError(); trimPool(); e = cudaMalloc((void**)p, bytes); }
  return e;
}

// ---------------------------------------------------------------- the exported model
struct Entry { char kind; size_t offset, length; double value; int seg; size_t devOffset = 0; int rec = -1; };

// ---------------------------------------------------------------- a published bundle, read as it is
// A page bundle (manifest.json and its shards, int5/int3/int8/float16/float32 as tools/quantize_af3.py
// writes them) loads straight into a Model: its packed codes go to the device and are decoded there into
// the float32 copy, so nothing is decoded on the host and no float32 file is ever written (ESMFold2's was
// 2.9 GB from 0.35 GB of bundles, and a 2-vCPU Colab VM spent minutes making it). The decode is
// shared/weights/dtype.js's, to the bit: code * scale + zero in double, rounded once to float32.
struct BRec { int file; size_t byteOffset, scaleOffset, zeroOffset, elements, dst, first = 0; int kind, bits, block;
               int round16 = 0, accumulate = 0, addRec = -1; size_t rows = 0; int rowBlocks = 0; };
// kind: 0 float32, 1 float16, 2 int8 (symmetric), 3 packed int<bits> (asymmetric), 4 zeros (a map's `z`),
// 5 gathered (a map's `p` lines: parts of bundle tensors decoded into scratch, see GPart),
// 6 int8 as af3-any-model's blobs store it (a float32 scale per CHANNEL of the last axis, `block` of them, and per
//   BLOCK of rows: `rowBlocks` of the `rows`, ceil(rows / rowBlocks) rows each - its params.dequantise_int8),
// 7 bfloat16 (a blob's uint16 bit patterns),
// 8 a map's `c` line: literal values (byteOffset indexes Segment::consts) - stock AlphaFold 3's frozen Fourier
//   embedding, a constant of its source that DeepMind's af3.bin.zst does not carry;
// first: the element of the bundle tensor this entry starts at (a map's slice of a stacked tensor)
// round16 / accumulate / addRec: a DELTA bundle's `addTo` tensor (shared/bundles/delta-tensor-store.js) - the
// base's value rounded to float16, then the delta's decode added in float32 by the record addRec names
// A map's `p` line: one PART of a gathered tensor - a grid of up to 6 axes walked over the destination
// (dst + sum i_k * dstStride_k) and one or two sources (srcOff + sum i_k * srcStride_k, into a bundle
// tensor decoded whole into scratch). op: 'v' the source, 'x' the product of two (one float32 multiply,
// as the page's loader folds a LayerNorm scale into a projection), 'i' the source as an int32 (a
// residue table the bundle stores as float32), 'o' one. What no part covers is zero.
struct GPart { int rec, op, rank, nsrc; size_t dims[6], dst; long long dstStride[6]; int src[2]; size_t srcOff[2]; long long srcStride[2][6]; };
struct GPartD { unsigned long long n, dst, off[2], dims[6]; long long ds[6], ss[2][6]; int rank, op; };
__global__ void bundleGatherK(const float* scratch, const GPartD* parts, float* out) {
  GPartD p = parts[blockIdx.y];
  for (unsigned long long o = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; o < p.n;
       o += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long r = o; long long d = (long long)p.dst, a = (long long)p.off[0], b = (long long)p.off[1];
    for (int k = p.rank - 1; k >= 0; --k) {
      long long i = (long long)(r % p.dims[k]); r /= p.dims[k];
      d += i * p.ds[k]; a += i * p.ss[0][k]; b += i * p.ss[1][k];
    }
    float v;
    if (p.op == 'o') v = 1.f;
    else if (p.op == 'x') v = __fmul_rn(scratch[a], scratch[b]);
    else if (p.op == 'i') v = __int_as_float((int)scratch[a]);
    else v = scratch[a];
    out[d] = v;
  }
}
// A JSON value (enough of JSON for a manifest)
struct Json {
  enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
  double num = 0; std::string str; std::vector<Json> arr; std::vector<std::pair<std::string, Json>> obj;
  const Json* get(const std::string& k) const { for (auto& [n, v] : obj) if (n == k) return &v; return nullptr; }
  static Json parse(const std::string& text) { size_t i = 0; Json v = value(text, i); return v; }
  static void ws(const std::string& s, size_t& i) { while (i < s.size() && isspace((unsigned char)s[i])) ++i; }
  static std::string string_(const std::string& s, size_t& i) {
    std::string out; ++i;
    while (i < s.size() && s[i] != '"') {
      if (s[i] == '\\') {
        ++i; char c = s[i++];
        if (c == 'n') out += '\n'; else if (c == 't') out += '\t'; else if (c == 'u') { out += '?'; i += 4; }
        else out += c;
      } else out += s[i++];
    }
    ++i; return out;
  }
  static Json value(const std::string& s, size_t& i) {
    ws(s, i); Json v;
    if (i >= s.size()) { fprintf(stderr, "manifest: unexpected end\n"); exit(1); }
    char c = s[i];
    if (c == '{') {
      v.t = OBJ; ++i; ws(s, i);
      if (s[i] == '}') { ++i; return v; }
      for (;;) {
        ws(s, i); std::string k = string_(s, i); ws(s, i); ++i;     // ':'
        v.obj.push_back({k, value(s, i)}); ws(s, i);
        if (s[i] == ',') { ++i; continue; }
        ++i; return v;                                            // '}'
      }
    }
    if (c == '[') {
      v.t = ARR; ++i; ws(s, i);
      if (s[i] == ']') { ++i; return v; }
      for (;;) { v.arr.push_back(value(s, i)); ws(s, i); if (s[i] == ',') { ++i; continue; } ++i; return v; }
    }
    if (c == '"') { v.t = STR; v.str = string_(s, i); return v; }
    if (!strncmp(s.c_str() + i, "true", 4)) { v.t = BOOL; v.num = 1; i += 4; return v; }
    if (!strncmp(s.c_str() + i, "false", 5)) { v.t = BOOL; i += 5; return v; }
    if (!strncmp(s.c_str() + i, "null", 4)) { i += 4; return v; }
    char* end; v.t = NUM; v.num = strtod(s.c_str() + i, &end); i = end - s.c_str(); return v;
  }
};
// the decode, one tensor of the shard a blockIdx.y
struct BDecode { unsigned long long src, dst, n, scale, zero, first, rows; int kind, bits, block, flags, rowBlocks; };   // flags: 1 round16, 2 accumulate
__device__ __forceinline__ float bundleHalf(const unsigned char* p) {
  unsigned short h = (unsigned short)(p[0] | (p[1] << 8)); return __half2float(__ushort_as_half(h));
}
__global__ void bundleDecodeK(const unsigned char* raw, const BDecode* table, float* out) {
  BDecode d = table[blockIdx.y];
  for (unsigned long long o = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; o < d.n;
       o += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long i = d.first + o;     // the bundle tensor's element
    float v;
    if (d.kind == 4) { out[d.dst + o] = 0.f; continue; }
    if (d.kind == 0) { unsigned int w; memcpy(&w, raw + d.src + 4 * i, 4); v = __uint_as_float(w); }
    else if (d.kind == 1) v = bundleHalf(raw + d.src + 2 * i);
    else if (d.kind == 2) {
      double scale = (double)bundleHalf(raw + d.scale + 2 * (i / d.block));
      v = (float)__dmul_rn((double)(signed char)raw[d.src + i], scale);
    } else if (d.kind == 6) {          // (float32 times float32, as numpy multiplies them)
      unsigned long long row = i / d.block, col = i % d.block, g = (d.rows + d.rowBlocks - 1) / d.rowBlocks;
      unsigned int w; memcpy(&w, raw + d.scale + 4 * ((row / g) * d.block + col), 4);
      v = __fmul_rn((float)(signed char)raw[d.src + i], __uint_as_float(w));
    } else if (d.kind == 7) {
      unsigned int h = (unsigned int)(raw[d.src + 2 * i] | (raw[d.src + 2 * i + 1] << 8));
      v = __uint_as_float(h << 16);
    } else {
      unsigned long long g = i / d.block, groupBytes = (unsigned long long)d.block * d.bits / 8;
      unsigned long long bit = g * groupBytes * 8 + (i % d.block) * d.bits, byte = bit >> 3;
      unsigned int code = ((raw[d.src + byte] | (raw[d.src + byte + 1] << 8)) >> (bit & 7)) & ((1u << d.bits) - 1);
      double scale = (double)bundleHalf(raw + d.scale + 2 * g), zero = (double)bundleHalf(raw + d.zero + 2 * g);
      v = (float)__dadd_rn(__dmul_rn((double)code, scale), zero);
    }
    if (d.flags & 1) v = __half2float(__float2half_rn(v));
    out[d.dst + o] = (d.flags & 2) ? __fadd_rn(out[d.dst + o], v) : v;
  }
}
inline float hostHalf(const unsigned char* p) { __half_raw r; r.x = (unsigned short)(p[0] | (p[1] << 8)); return __half2float(__half(r)); }
// model.idx/model.bin pairs, mapped: the input's directory and (af3 --weights=DIR) the weights'
// On the device every tensor starts on 16 bytes (devOffset), wherever the file packed it: cuBLAS's
// vector-load kernels need it (align1 kernels otherwise, and a batched GEMM, which cannot see its
// pointers, faults). runs: the file's byte ranges and where each lands, in file order.
struct Run { size_t src, dst, bytes; };
// data: the file mapped, only once the host reads an entry (f()): a large file goes to the device by
// pread into pinned buffers - a mapping of the 2.9 GB cuda/ef2 weights cost ~300 ms of page faults to
// read and 275 ms more to tear down when the process exited, after its PDB was written
struct Segment { const float* data; size_t bytes; float* device; std::map<std::string, void*> halfMirrors;
                 int fd = -1;
                 size_t deviceBytes = 0; std::vector<Run> runs;
                 bool bundle = false; std::string dir; std::vector<std::string> files; std::vector<BRec> recs;
                 std::map<int, std::vector<float>> hostCopies;     // a bundle tensor the host read, decoded
                 std::vector<BRec> srcRecs; size_t scratchElems = 0; std::vector<GPart> parts;   // (gathered tensors)
                 // RESIDENT int8 (loadBundle's residentPrefix): tensors kept on the device as their codes and scales,
                 // never decoded into the float32 copy - a model too large for it (ESM-C 6B: 25 GB as float32, 6.4 GB
                 // as codes; ESM2 3B: 11 and 2.7) - each with where its codes and scales land in `resident`
                 struct Res { BRec b; size_t codes, scales; };
                 std::vector<Res> residentRecs; unsigned char* resident = nullptr; size_t residentBytes = 0;
                 std::vector<std::vector<float>> consts;   // (a map's `c` lines, kind 8)
               };
// A float32 array as a NumPy .npy file (format 1.0: magic, header dict padded to 64 bytes, data)
inline void writeNpy(const std::string& path, const std::vector<float>& data, const std::vector<size_t>& shape) {
  std::string dims;
  for (size_t k = 0; k < shape.size(); ++k) dims += std::to_string(shape[k]) + (shape.size() == 1 || k + 1 < shape.size() ? "," : "");
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + dims + "), }";
  size_t total = 10 + header.size() + 1;
  header += std::string((64 - total % 64) % 64, ' ') + "\n";
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); }
  const char magic[] = "\x93NUMPY\x01\x00";
  fwrite(magic, 1, 8, f);
  uint16_t len = (uint16_t)header.size(); fwrite(&len, 2, 1, f);
  fwrite(header.data(), 1, header.size(), f);
  fwrite(data.data(), 4, data.size(), f);
  fclose(f);
}
// ---------------------------------------------------------------- af3-any-model's published blobs
// sokrypton/af3-any-model on Hugging Face publishes each model as ONE zstd-compressed stream of haiku records
// (alphafold3/model/params.py encode_record: <5i> scope, name, dtype, shape lengths and the payload's bytes,
// then the strings, the shape and the C-order payload). Its tensor names are the ones LocalFold's bundles were
// exported under, so a port's .map reads a blob as it reads a bundle. Decompressed ONCE, to `<blob>.raw` beside
// it, through the system's libzstd (dlopen: no headers, no build dependency), and read as a one-shard bundle.
struct ZstdIn { const void* src; size_t size, pos; };
struct ZstdOut { void* dst; size_t size, pos; };
// decompressed into `<blob>.raw/NNN` shards of at most BLOB_SHARD bytes, split at record boundaries, so the upload
// reads it as it reads a bundle's shards: two pinned buffers of one shard each, not two of a 2.8 GB stream
inline constexpr size_t BLOB_SHARD = 256ull << 20;
inline std::vector<std::string> blobShards(const std::string& blob) {
  std::string dir = blob + ".raw";
  auto listed = [&]() {
    std::vector<std::string> out; std::ifstream done(dir + "/done"); std::string f;
    while (done >> f) out.push_back(dir + "/" + f);
    return out;
  };
  if (auto have = listed(); !have.empty()) return have;
  void* lib = dlopen("libzstd.so.1", RTLD_NOW);
  if (!lib) lib = dlopen("libzstd.so", RTLD_NOW);
  if (!lib) { fprintf(stderr, "reading %s needs libzstd (libzstd.so.1): %s\n", blob.c_str(), dlerror()); exit(1); }
  auto sym = [&](const char* n) { void* f = dlsym(lib, n); if (!f) { fprintf(stderr, "libzstd has no %s\n", n); exit(1); } return f; };
  auto create = (void* (*)())sym("ZSTD_createDStream");
  auto init = (size_t (*)(void*))sym("ZSTD_initDStream");
  auto step = (size_t (*)(void*, ZstdOut*, ZstdIn*))sym("ZSTD_decompressStream");
  auto isError = (unsigned (*)(size_t))sym("ZSTD_isError");
  auto errName = (const char* (*)(size_t))sym("ZSTD_getErrorName");
  auto release = (size_t (*)(void*))sym("ZSTD_freeDStream");
  FILE* in = fopen(blob.c_str(), "rb");
  if (!in) { fprintf(stderr, "cannot read %s\n", blob.c_str()); exit(1); }
  std::string tmp = dir + ".part." + std::to_string(getpid());
  mkdir(tmp.c_str(), 0755);
  // the decompressed stream in order; whole records cut into shards as they complete
  std::vector<char> pending; std::vector<std::string> names; FILE* out = nullptr; size_t inShard = 0;
  auto emit = [&](bool flushAll) {
    size_t at = 0;
    for (;;) {
      if (pending.size() - at < 20) break;
      int32_t h[5]; memcpy(h, pending.data() + at, 20);
      size_t len = 20 + (size_t)h[0] + h[1] + h[2] + 4 * (size_t)h[3] + (size_t)(uint32_t)h[4];
      if (pending.size() - at < len) break;
      // (never before a `__q_scale`: an int8 tensor's scales share its shard)
      std::string nm(pending.data() + at + 20 + h[0], (size_t)h[1]);
      bool scale = nm.size() > 9 && !nm.compare(nm.size() - 9, 9, "__q_scale");
      if (!out || (inShard > 0 && inShard + len > BLOB_SHARD && !scale)) {
        if (out && fclose(out)) { fprintf(stderr, "cannot write %s\n", tmp.c_str()); exit(1); }
        char file[16]; snprintf(file, sizeof file, "%03zu", names.size()); names.push_back(file);
        out = fopen((tmp + "/" + file).c_str(), "wb"); inShard = 0;
        if (!out) { fprintf(stderr, "cannot write %s/%s\n", tmp.c_str(), file); exit(1); }
      }
      fwrite(pending.data() + at, 1, len, out); inShard += len; at += len;
    }
    pending.erase(pending.begin(), pending.begin() + at);
    if (flushAll && !pending.empty()) { fprintf(stderr, "%s: a truncated record\n", blob.c_str()); exit(1); }
  };
  void* ds = create(); init(ds);
  std::vector<char> ib(1 << 20), ob(1 << 22);
  size_t got, last = 0;
  auto take = [&](ZstdIn& zi) {
    ZstdOut zo{ ob.data(), ob.size(), 0 };
    last = step(ds, &zo, &zi);
    if (isError(last)) { fprintf(stderr, "%s: %s\n", blob.c_str(), errName(last)); exit(1); }
    pending.insert(pending.end(), ob.data(), ob.data() + zo.pos);
    if (pending.size() > (64u << 20)) emit(false);
    return zo.pos;
  };
  while ((got = fread(ib.data(), 1, ib.size(), in)) > 0) { ZstdIn zi{ ib.data(), got, 0 }; while (zi.pos < zi.size) take(zi); }
  while (last != 0) { ZstdIn zi{ nullptr, 0, 0 }; if (take(zi) == 0) break; }   // (0: the frame is done)
  release(ds); fclose(in);
  if (last != 0) { fprintf(stderr, "%s: a truncated stream\n", blob.c_str()); exit(1); }
  emit(true);
  if (out && fclose(out)) { fprintf(stderr, "cannot write %s\n", tmp.c_str()); exit(1); }
  { std::ofstream done(tmp + "/done"); for (auto& n : names) done << n << "\n"; }
  if (rename(tmp.c_str(), dir.c_str())) { fprintf(stderr, "cannot write %s\n", dir.c_str()); exit(1); }
  return listed();
}
// the one *.bin.zst a directory holds ("" when it is a bundle: a manifest.json)
inline std::string findBlob(const std::string& dir) {
  struct stat st;
  if (!stat((dir + "/manifest.json").c_str(), &st)) return "";
  std::vector<std::string> found;
  if (DIR* d = opendir(dir.c_str())) {
    while (dirent* e = readdir(d)) { std::string n = e->d_name; if (n.size() > 8 && n.compare(n.size() - 8, 8, ".bin.zst") == 0) found.push_back(n); }
    closedir(d);
  }
  if (found.size() != 1) { fprintf(stderr, "%s holds %s and no manifest.json\n", dir.c_str(), found.empty() ? "no *.bin.zst" : "more than one *.bin.zst"); exit(1); }
  char* real = realpath((dir + "/" + found[0]).c_str(), nullptr);      // (absolute: its shards' paths are)
  std::string path = real ? real : dir + "/" + found[0];
  free(real);
  return path;
}
// every tensor of a decompressed blob as a record of its one shard: name, BRec, shape
inline std::vector<std::tuple<std::string, BRec, std::vector<double>>> blobRecords(const std::vector<std::string>& shards) {
  struct Rec { std::string dtype; std::vector<double> shape; size_t offset, bytes; int file; };
  std::map<std::string, Rec> recs;
  for (size_t fi = 0; fi < shards.size(); ++fi) {
    FILE* f = fopen(shards[fi].c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", shards[fi].c_str()); exit(1); }
    size_t at = 0;
    for (;;) {
      int32_t h[5];
      if (fread(h, 4, 5, f) != 5) break;
      std::string scope(h[0], 0), name(h[1], 0), dtype(h[2], 0);
      std::vector<int32_t> shape(h[3]);
      if (fread(scope.data(), 1, h[0], f) != (size_t)h[0] || fread(name.data(), 1, h[1], f) != (size_t)h[1] ||
          fread(dtype.data(), 1, h[2], f) != (size_t)h[2] || fread(shape.data(), 4, h[3], f) != (size_t)h[3]) {
        fprintf(stderr, "%s: a truncated record\n", shards[fi].c_str()); exit(1);
      }
      at += 20 + h[0] + h[1] + h[2] + 4 * (size_t)h[3];
      recs[scope + "/" + name] = { dtype, std::vector<double>(shape.begin(), shape.end()), at, (size_t)(uint32_t)h[4], (int)fi };
      at += (size_t)(uint32_t)h[4];
      fseek(f, (long)at, SEEK_SET);
    }
    fclose(f);
  }
  std::vector<std::tuple<std::string, BRec, std::vector<double>>> out;
  for (auto& [name, r] : recs) {
    if (!name.compare(0, 9, "__meta__/")) continue;
    if (name.size() > 9 && !name.compare(name.size() - 9, 9, "__q_scale")) continue;
    BRec b{}; b.file = r.file; b.byteOffset = r.offset;
    size_t n = 1; for (double d : r.shape) n *= (size_t)d;
    b.elements = n;
    if (r.dtype == "float32") b.kind = 0;
    else if (r.dtype == "float16") b.kind = 1;
    else if (r.dtype == "uint16") b.kind = 7;
    else if (r.dtype == "int8") {
      auto sc = recs.find(name + "__q_scale");
      if (sc == recs.end() || sc->second.dtype != "float32" || r.shape.empty() || sc->second.file != r.file) {
        fprintf(stderr, "%s: int8 with no float32 __q_scale\n", name.c_str()); exit(1);
      }
      b.kind = 6; b.scaleOffset = sc->second.offset; b.block = (int)r.shape.back(); b.rows = n / b.block;
      const auto& ss = sc->second.shape;
      b.rowBlocks = ss.size() == 1 ? 1 : (int)ss[0];
      if (ss.back() != r.shape.back() || (ss.size() != 1 && ss.size() != 2)) {
        fprintf(stderr, "%s: a __q_scale of an unknown layout\n", name.c_str()); exit(1);
      }
    } else { fprintf(stderr, "%s: unsupported blob dtype %s\n", name.c_str(), r.dtype.c_str()); exit(1); }
    out.emplace_back(name, b, r.shape);
  }
  return out;
}
// a resident int8 tensor on the device: codes and either a float16 scale per `block` consecutive elements (a bundle's,
// kind 2) or - an af3-any-model blob's, kind 6 - float32 scales per channel of the last axis (`block` of them) and
// per block of rows (`rowBlocks` of the `rows`)
struct ResidentInt8 { const signed char* codes; const __half* scales; size_t elements; int block;
                      const float* scales32 = nullptr; size_t rows = 0; int rowBlocks = 0; };
// a resident record's scales, in bytes (its codes are one a element)
inline size_t residentScaleBytes(const BRec& b) {
  return b.kind == 6 ? 4 * (size_t)b.rowBlocks * b.block : 2 * ((b.elements + b.block - 1) / b.block);
}
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
  // a page bundle (its manifest.json and shards) under `prefix/`: every tensor as `prefix/<name>` with its
  // rank and dims (`#r`, `#k`), and the manifest's trunk/languageModel numbers as `meta/<key>` - the
  // entries cuda/ef2/export_weights.mjs wrote, from the same bytes
  // With `lines` (the port's WEIGHT WALK - cuda/featurise/af3_weights.h for the AF3 lineage, af2_weights.h for
  // AlphaFold 2 - handed the bundle's tensor names and shapes), the entries are the walk's instead: each `b` line a
  // slice of a bundle tensor under the port's own name, each `z` zeros, `p` derived (ones, a scale folded into a
  // projection, a strided view), `c` literal values, each `m` metadata as it is. (These lines were once files,
  // cuda/*/maps/*.map, written offline from a float32 export; the walk works them out from the bundle itself.)
  using WeightLines = std::function<std::vector<std::string>(const std::map<std::string, std::vector<long long>>& shapes)>;
  void loadBundle(const std::string& dir, const std::string& prefix, const WeightLines& lines = nullptr,
                  const std::string& delta = "", const std::string& residentPrefix = "") {
    const bool map = (bool)lines;
    if (!delta.empty() && !map) { fprintf(stderr, "a delta bundle is read through a weight walk\n"); exit(1); }
    std::string blob = findBlob(dir);       // (af3-any-model's: a directory holding one *.bin.zst)
    std::string text = "{\"tensors\": {}}";
    if (blob.empty()) {
      std::ifstream in(dir + "/manifest.json");
      text.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }
    Json m = Json::parse(text);
    int seg = (int)segs.size();
    Segment sg{nullptr, 0, nullptr}; sg.bundle = true; sg.dir = dir;
    auto addMeta = [&](const std::string& name, double v) {
      auto it = index.find(name);
      if (it != index.end()) {
        if (it->second.kind != 'm' || it->second.value != v) { fprintf(stderr, "%s is in two bundles, differently\n", name.c_str()); exit(1); }
        return;
      }
      Entry e{'m', 0, 0, v, seg}; index[name] = e;
    };
    for (const char* block : {"trunk", "languageModel"})
      if (const Json* b = m.get(block)) for (auto& [k, v] : b->obj) if (v.t == Json::NUM) addMeta("meta/" + k, v.num);
    const Json* tensors = m.get("tensors");
    if (!tensors) { fprintf(stderr, "%s/manifest.json has no tensors\n", dir.c_str()); exit(1); }
    std::map<std::string, int> fileIndex;
    std::map<std::string, BRec> byName;      // (walk mode) every bundle tensor, by its bundle name
    std::map<std::string, std::vector<long long>> shapeOf;   // ...and its shape, for the walk
    std::map<std::string, BRec> deltaOf;     // (a delta) the record added to an addTo tensor's base
    size_t at = 0;
    // a manifest record as a BRec; `where` prefixes its file (a delta's shards live in another directory)
    auto record = [&](const std::string& name, const Json& r, const std::string& where, std::vector<double>* shapeOut) {
      BRec b{}; const Json* j;
      std::string file = where + r.get("file")->str;
      if (!fileIndex.count(file)) { fileIndex[file] = (int)sg.files.size(); sg.files.push_back(file); }
      b.file = fileIndex[file];
      b.byteOffset = (j = r.get("byteOffset")) ? (size_t)j->num : 0;
      std::string dtype = r.get("dtype")->str;
      size_t n = 1; std::vector<double> shape;
      for (auto& d : r.get("shape")->arr) { n *= (size_t)d.num; shape.push_back(d.num); }
      b.elements = n;
      if (dtype == "float32") b.kind = 0;
      else if (dtype == "float16") b.kind = 1;
      else if (dtype == "int8") b.kind = 2;
      else if (dtype.size() == 4 && dtype.compare(0, 3, "int") == 0 && dtype[3] >= '1' && dtype[3] <= '7') { b.kind = 3; b.bits = dtype[3] - '0'; }
      else { fprintf(stderr, "%s: unsupported dtype %s\n", name.c_str(), dtype.c_str()); exit(1); }
      if (b.kind >= 2) {
        b.block = (int)r.get("block")->num; b.scaleOffset = (size_t)r.get("scaleOffset")->num;
        if (b.kind == 3) {
          if (!(j = r.get("zeroOffset"))) { fprintf(stderr, "%s: %s with no zero offset\n", name.c_str(), dtype.c_str()); exit(1); }
          b.zeroOffset = (size_t)j->num;
          if ((b.block * b.bits) % 8) { fprintf(stderr, "%s: %s at group %d is not whole bytes\n", name.c_str(), dtype.c_str(), b.block); exit(1); }
        }
      }
      if ((b.kind == 0 && b.byteOffset % 4) || (b.kind == 1 && b.byteOffset % 2)) {
        fprintf(stderr, "%s: unaligned in its shard\n", name.c_str()); exit(1);
      }
      if (shapeOut) *shapeOut = shape;
      return b;
    };
    std::vector<std::tuple<std::string, BRec, std::vector<double>>> entries;
    if (!blob.empty()) {
      std::vector<std::string> shards = blobShards(blob);
      for (auto& f : shards) { fileIndex[f] = (int)sg.files.size(); sg.files.push_back(f); }
      entries = blobRecords(shards);
    } else {
      for (auto& [name, r] : tensors->obj) { std::vector<double> shape; BRec b = record(name, r, "", &shape); entries.emplace_back(name, b, shape); }
    }
    for (auto& [name, b, shape] : entries) {
      size_t n = b.elements;
      if (map) { byName[name] = b; shapeOf[name] = std::vector<long long>(shape.begin(), shape.end()); continue; }
      std::string key = prefix + "/" + name;
      if (index.count(key)) { fprintf(stderr, "%s is in two model directories\n", key.c_str()); exit(1); }
      // (int8 under the prefix, a bundle's or a blob's; packed codes - ESM-C 600M's int3 - decode into the float32 copy
      // its tower reads)
      if (!residentPrefix.empty() && (b.kind == 2 || b.kind == 6) && !name.compare(0, residentPrefix.size(), residentPrefix)) {
        auto align = [](size_t v) { return (v + 255) / 256 * 256; };
        Segment::Res r{b, align(sg.residentBytes), 0};
        r.scales = align(r.codes + n); sg.residentBytes = r.scales + residentScaleBytes(b);
        Entry e{'q', 0, n, 0, seg}; e.rec = (int)sg.residentRecs.size();
        index[key] = e; sg.residentRecs.push_back(r);
        addMeta(key + "#r", (double)shape.size());
        for (size_t k = 0; k < shape.size(); ++k) addMeta(key + "#" + std::to_string(k), shape[k]);
        continue;
      }
      at = (at + 3) / 4 * 4;
      Entry e{'t', 0, n, 0, seg}; e.devOffset = at; e.rec = (int)sg.recs.size(); b.dst = at; at += n;
      index[key] = e;
      sg.recs.push_back(b);
      addMeta(key + "#r", (double)shape.size());
      for (size_t k = 0; k < shape.size(); ++k) addMeta(key + "#" + std::to_string(k), shape[k]);
    }
    // 🔴 A DELTA BUNDLE, READ AS THE PAGE READS ONE (shared/bundles/delta-tensor-store.js): AlphaFold 2's
    // models 2-5 are published as int3 differences on model 1, and its header says what each tensor is -
    // `addTo` (the base's value ROUNDED TO FLOAT16, because that is what the delta was taken against, plus
    // the delta's decode), `whole` (the delta's own), `absent` (gone: model_3/4/5 have no template
    // embedder); anything unnamed is the base's. The map names the base's tensors, as for model 1.
    std::string deltaModel;
    if (!delta.empty()) {
      std::ifstream din(delta + "/manifest.json");
      if (!din) { fprintf(stderr, "no %s/manifest.json\n", delta.c_str()); exit(1); }
      Json dm = Json::parse(std::string((std::istreambuf_iterator<char>(din)), std::istreambuf_iterator<char>()));
      const Json* header = dm.get("delta");
      const Json* dt = dm.get("tensors");
      if (!header || !dt) { fprintf(stderr, "%s carries no delta header - it is not a delta\n", delta.c_str()); exit(1); }
      if (const Json* mname = header->get("model")) deltaModel = mname->str;
      std::string where = (delta[0] == '/' ? delta : std::string(realpath(delta.c_str(), nullptr))) + "/";
      auto names = [&](const char* key) {
        std::vector<std::string> out;
        if (const Json* list = header->get(key)) for (auto& v : list->arr) out.push_back(v.str);
        return out;
      };
      for (auto& name : names("absent")) { byName.erase(name); shapeOf.erase(name); }
      for (auto& name : names("whole")) {
        const Json* r = dt->get(name);
        if (!r) { fprintf(stderr, "%s: the delta's header names %s whole and holds no such tensor\n", delta.c_str(), name.c_str()); exit(1); }
        byName[name] = record(name, *r, where, nullptr);
      }
      for (auto& name : names("addTo")) {
        const Json* r = dt->get(name);
        auto it = byName.find(name);
        if (!r || it == byName.end()) { fprintf(stderr, "%s: %s is addTo but missing from the delta or the base\n", delta.c_str(), name.c_str()); exit(1); }
        BRec d = record(name, *r, where, nullptr);
        if (d.elements != it->second.elements) { fprintf(stderr, "%s: %s is %zu in the delta, %zu in the base\n", delta.c_str(), name.c_str(), d.elements, it->second.elements); exit(1); }
        it->second.round16 = 1;
        d.accumulate = 1;
        deltaOf[name] = d;
      }
    }
    // an addTo tensor is two records into one place: the base rounded to f16 (`b`, already flagged), then
    // the delta added; the second goes after the first in `pool`, and decodes after it (its shard does)
    auto withDelta = [&](const std::string& source, BRec& b, std::vector<BRec>& pool) {
      auto it = deltaOf.find(source);
      if (it == deltaOf.end()) return;
      BRec d = it->second; d.first = b.first; d.elements = b.elements; d.dst = b.dst;
      b.addRec = (int)pool.size() + 1;     // (b is pushed first, then d)
      pool.push_back(b); pool.push_back(d);
    };
    if (map) {
      std::vector<std::string> walked;
      try { walked = lines(shapeOf); }
      catch (const std::exception& e) { fprintf(stderr, "%s: %s\n", dir.c_str(), e.what()); exit(1); }
      std::string joined;
      for (auto& l : walked) joined += l + "\n";
      std::istringstream mf(joined);
      const std::string mapName = "the weight walk";
      std::string line; std::map<std::string, int> srcIndex;
      bool mapDelta = false;
      while (std::getline(mf, line)) {
        std::istringstream in(line); char kind; std::string name; in >> kind >> name;
        if (kind == 'm') { double v; in >> v; addMeta(name, v); continue; }
        if (kind == 'D') {          // this map is for a delta model: the one named, on the bundle it was made from
          if (delta.empty() || name != deltaModel) {
            fprintf(stderr, "%s is for %s on a delta bundle; %s\n", mapName.c_str(), name.c_str(),
                    delta.empty() ? "no --delta was given" : ("the delta given is " + deltaModel).c_str());
            exit(1);
          }
          mapDelta = true; continue;
        }
        BRec b{};
        size_t n;
        std::string bSource;
        if (kind == 'b') {
          std::string source; size_t first; in >> source >> first >> n;
          auto it = byName.find(source);
          if (it == byName.end() || first + n > it->second.elements) {
            fprintf(stderr, "%s: %s does not hold [%zu, %zu) of %s (another export of the bundle?)\n", mapName.c_str(),
                    dir.c_str(), first, first + n, source.c_str());
            exit(1);
          }
          b = it->second; b.first = first; b.addRec = -1;
          bSource = source;
        } else if (kind == 'z') { in >> n; b.kind = 4; b.file = -1; }
        else if (kind == 'c') {
          in >> n; std::vector<float> v(n);
          for (size_t k = 0; k < n && in; ++k) in >> v[k];
          if (in.fail()) { fprintf(stderr, "%s: a malformed c line for %s\n", mapName.c_str(), name.c_str()); exit(1); }
          b.kind = 8; b.file = -1; b.byteOffset = sg.consts.size(); sg.consts.push_back(std::move(v));
        }
        else if (kind == 'p') {
          GPart g{}; char op; in >> n >> op >> g.rank;
          g.op = op; g.nsrc = op == 'o' ? 0 : op == 'x' ? 2 : 1;
          bool okLine = (op == 'v' || op == 'x' || op == 'i' || op == 'o') && g.rank >= 1 && g.rank <= 6;
          size_t count = 1;
          for (int k = 0; okLine && k < g.rank; ++k) { in >> g.dims[k]; count *= g.dims[k]; }
          long long lo = 0, hi = 0;
          if (okLine) in >> g.dst;
          for (int k = 0; okLine && k < g.rank; ++k) { in >> g.dstStride[k]; (g.dstStride[k] < 0 ? lo : hi) += g.dstStride[k] * (long long)(g.dims[k] - 1); }
          okLine = okLine && (long long)g.dst + lo >= 0 && (long long)g.dst + hi < (long long)n && count > 0;
          for (int q = 0; okLine && q < g.nsrc; ++q) {
            std::string source; in >> source >> g.srcOff[q];
            auto it = byName.find(source);
            long long slo = 0, shi = 0;
            for (int k = 0; k < g.rank; ++k) { in >> g.srcStride[q][k]; (g.srcStride[q][k] < 0 ? slo : shi) += g.srcStride[q][k] * (long long)(g.dims[k] - 1); }
            if (it == byName.end() || (long long)g.srcOff[q] + slo < 0 || (long long)g.srcOff[q] + shi >= (long long)it->second.elements) {
              fprintf(stderr, "%s: %s does not hold that view of %s (another export of the bundle?)\n", mapName.c_str(), dir.c_str(), source.c_str());
              exit(1);
            }
            if (!srcIndex.count(source)) {
              BRec r = it->second; sg.scratchElems = (sg.scratchElems + 3) / 4 * 4; r.dst = sg.scratchElems; sg.scratchElems += r.elements;
              srcIndex[source] = (int)sg.srcRecs.size();
              if (deltaOf.count(source)) withDelta(source, r, sg.srcRecs);
              else sg.srcRecs.push_back(r);
            }
            g.src[q] = srcIndex[source];
          }
          if (!okLine || in.fail()) { fprintf(stderr, "%s: a malformed p line for %s\n", mapName.c_str(), name.c_str()); exit(1); }
          auto it = index.find(name);
          if (it != index.end()) {
            if (it->second.seg != seg || it->second.rec < 0 || sg.recs[it->second.rec].kind != 5 || it->second.length != n) {
              fprintf(stderr, "%s: %s is a p tensor and something else\n", mapName.c_str(), name.c_str()); exit(1);
            }
            g.rec = it->second.rec; sg.parts.push_back(g); continue;
          }
          b.kind = 5; b.file = -1; g.rec = (int)sg.recs.size(); sg.parts.push_back(g);
        }
        else { fprintf(stderr, "%s: a line of kind %c\n", mapName.c_str(), kind); exit(1); }
        if (index.count(name)) { fprintf(stderr, "%s is in two model directories\n", name.c_str()); exit(1); }
        b.elements = n;
        at = (at + 3) / 4 * 4;
        Entry e{'t', 0, n, 0, seg}; e.devOffset = at; e.rec = (int)sg.recs.size(); b.dst = at; at += n;
        index[name] = e;
        if (!bSource.empty() && deltaOf.count(bSource)) withDelta(bSource, b, sg.recs);
        else sg.recs.push_back(b);
      }
      if (!delta.empty() && !mapDelta) { fprintf(stderr, "%s is not a map for a delta model (no D line)\n", mapName.c_str()); exit(1); }
    }
    sg.deviceBytes = at * 4;
    for (auto& f : sg.files) { struct stat st; if (stat(shardPath(sg, f).c_str(), &st)) { fprintf(stderr, "no %s\n", shardPath(sg, f).c_str()); exit(1); } sg.bytes += (size_t)st.st_size; }
    segs.push_back(sg);
  }
  // a shard's path: relative to the bundle's directory, or absolute (a delta's, in its own directory)
  static std::string shardPath(const Segment& s, const std::string& f) { return f[0] == '/' ? f : s.dir + "/" + f; }
  static std::vector<unsigned char> readFile(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
    struct stat sb; fstat(fd, &sb);
    std::vector<unsigned char> v((size_t)sb.st_size + 2, 0);     // (+2: a packed code's second byte past the last group)
    for (size_t got = 0; got < (size_t)sb.st_size;) {
      ssize_t r = read(fd, v.data() + got, (size_t)sb.st_size - got);
      if (r <= 0) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
      got += (size_t)r;
    }
    close(fd);
    return v;
  }
  // a bundle onto the device: each shard's bytes up as they are (read into pinned memory, two buffers so
  // one shard is read while the last is copied and decoded), decoded there into the float32 copy
  static bool bundleUp(Segment& s) {
    cudaStream_t st; if (cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess) return false;
    size_t most = 0;
    std::vector<size_t> sizes;
    for (auto& f : s.files) { struct stat sb; stat(shardPath(s, f).c_str(), &sb); sizes.push_back((size_t)sb.st_size + 2); most = std::max(most, sizes.back()); }
    // each shard read through a small ring of pinned pieces, kept for the process, into its device buffer:
    // pinning costs ~0.8 ms a MB here, so the two shard-sized pinned buffers this used were 0.4 s to allocate
    // and 0.16 to free for 256 MB shards - more than the whole read (AF3's cold start 1.7 -> 1.25 s, Chai-1's
    // 2.8 -> 1.8) - and 16 MB of pieces cost ~13 ms once
    constexpr size_t PIECE = (size_t)4 << 20; constexpr int NP = 4;
    static unsigned char* ring[NP]; static cudaEvent_t freed[NP];
    static std::mutex ringMu; std::lock_guard<std::mutex> ringHeld(ringMu);   // (uploadAsync: a thread a segment)
    if (!ring[0])
      for (int b = 0; b < NP; ++b)
        if (cudaHostAlloc(&ring[b], PIECE, cudaHostAllocDefault) != cudaSuccess ||
            cudaEventCreateWithFlags(&freed[b], cudaEventDisableTiming) != cudaSuccess) return false;
    unsigned char* raw[2]; BDecode* dt[2]; cudaEvent_t done[2];
    size_t maxRecs = s.recs.size();
    for (int k = 0; k < 2; ++k)
      if (devMalloc(&raw[k], most) != cudaSuccess ||
          devMalloc(&dt[k], maxRecs * sizeof(BDecode)) != cudaSuccess || cudaEventCreateWithFlags(&done[k], cudaEventDisableTiming) != cudaSuccess)
        return false;
    size_t piece = 0;                                        // pieces issued, over every shard
    std::vector<std::vector<BDecode>> tables(2), scratchTables(2);
    if (s.residentBytes && !s.resident && devMalloc(&s.resident, s.residentBytes) != cudaSuccess) return false;
    float* scratch = nullptr;
    if (s.scratchElems && devMalloc(&scratch, s.scratchElems * 4) != cudaSuccess) return false;
    BDecode* sdt[2] = {nullptr, nullptr};
    for (int k = 0; k < 2 && !s.srcRecs.empty(); ++k) if (devMalloc(&sdt[k], s.srcRecs.size() * sizeof(BDecode)) != cudaSuccess) return false;
    for (size_t fi = 0; fi < s.files.size(); ++fi) {
      int k = (int)(fi & 1);
      if (fi >= 2 && cudaEventSynchronize(done[k]) != cudaSuccess) return false;     // buffer k free again
      int fd = open(shardPath(s, s.files[fi]).c_str(), O_RDONLY);
      if (fd < 0) return false;
      size_t n = sizes[fi] - 2;
      for (size_t off = 0; off < n; off += PIECE, ++piece) {
        int b = (int)(piece % NP);
        if (piece >= NP && cudaEventSynchronize(freed[b]) != cudaSuccess) { close(fd); return false; }
        size_t len = std::min(PIECE, n - off), got = 0;
        while (got < len) { ssize_t r = read(fd, ring[b] + got, len - got); if (r <= 0) { close(fd); return false; } got += (size_t)r; }
        if (cudaMemcpyAsync(raw[k] + off, ring[b], len, cudaMemcpyHostToDevice, st) != cudaSuccess ||
            cudaEventRecord(freed[b], st) != cudaSuccess) { close(fd); return false; }
      }
      close(fd);
      if (cudaMemsetAsync(raw[k] + n, 0, 2, st) != cudaSuccess) return false;   // (a packed code's second byte past the last group)
      for (const auto& r : s.residentRecs)    // the resident tensors' codes and scales, as they are
        if (r.b.file == (int)fi &&
            (cudaMemcpyAsync(s.resident + r.codes, raw[k] + r.b.byteOffset, r.b.elements, cudaMemcpyDeviceToDevice, st) != cudaSuccess ||
             cudaMemcpyAsync(s.resident + r.scales, raw[k] + r.b.scaleOffset, residentScaleBytes(r.b),
                             cudaMemcpyDeviceToDevice, st) != cudaSuccess)) return false;
      auto& table = tables[k]; table.clear();
      for (const BRec& b : s.recs)
        if (b.file == (int)fi || (b.kind == 4 && fi == 0))
          table.push_back({b.byteOffset, b.dst, b.elements, b.scaleOffset, b.zeroOffset, b.first, b.rows, b.kind, b.bits, b.block,
                           b.round16 | (b.accumulate << 1), b.rowBlocks});
      if (!table.empty()) {
        if (cudaMemcpyAsync(dt[k], table.data(), table.size() * sizeof(BDecode), cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
        for (size_t t0 = 0; t0 < table.size(); t0 += 65535)
          bundleDecodeK<<<dim3(64, (unsigned)std::min<size_t>(65535, table.size() - t0)), 256, 0, st>>>(raw[k], dt[k] + t0, s.device);
      }
      auto& stable = scratchTables[k]; stable.clear();     // (the sources of gathered tensors, whole, into scratch)
      for (const BRec& b : s.srcRecs)
        if (b.file == (int)fi) stable.push_back({b.byteOffset, b.dst, b.elements, b.scaleOffset, b.zeroOffset, b.first, b.rows, b.kind,
                                                 b.bits, b.block, b.round16 | (b.accumulate << 1), b.rowBlocks});
      if (!stable.empty()) {
        if (cudaMemcpyAsync(sdt[k], stable.data(), stable.size() * sizeof(BDecode), cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
        for (size_t t0 = 0; t0 < stable.size(); t0 += 65535)
          bundleDecodeK<<<dim3(64, (unsigned)std::min<size_t>(65535, stable.size() - t0)), 256, 0, st>>>(raw[k], sdt[k] + t0, scratch);
      }
      if (cudaEventRecord(done[k], st) != cudaSuccess) return false;
    }
    GPartD* pt = nullptr;
    if (!s.parts.empty()) {
      for (const BRec& b : s.recs) if (b.kind == 5 && cudaMemsetAsync(s.device + b.dst, 0, b.elements * 4, st) != cudaSuccess) return false;
      std::vector<GPartD> pd;
      for (const GPart& g : s.parts) {
        GPartD d{}; d.n = 1; d.rank = g.rank; d.op = g.op; d.dst = s.recs[g.rec].dst + g.dst;
        for (int k = 0; k < g.rank; ++k) { d.dims[k] = g.dims[k]; d.n *= g.dims[k]; d.ds[k] = g.dstStride[k]; }
        for (int q = 0; q < g.nsrc; ++q) {
          d.off[q] = s.srcRecs[g.src[q]].dst + g.srcOff[q];
          for (int k = 0; k < g.rank; ++k) d.ss[q][k] = g.srcStride[q][k];
        }
        pd.push_back(d);
      }
      if (devMalloc(&pt, pd.size() * sizeof(GPartD)) != cudaSuccess ||
          cudaMemcpyAsync(pt, pd.data(), pd.size() * sizeof(GPartD), cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
      for (size_t t0 = 0; t0 < pd.size(); t0 += 65535)
        bundleGatherK<<<dim3(64, (unsigned)std::min<size_t>(65535, pd.size() - t0)), 256, 0, st>>>(scratch, pt + t0, s.device);
    }
    for (const BRec& b : s.recs)            // a map's literal tensors, as they are
      if (b.kind == 8 && cudaMemcpyAsync(s.device + b.dst, s.consts[b.byteOffset].data(), b.elements * 4,
                                         cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
    bool ok = cudaStreamSynchronize(st) == cudaSuccess && cudaGetLastError() == cudaSuccess;
    cudaFree(pt); cudaFree(scratch); cudaFree(sdt[0]); cudaFree(sdt[1]);
    for (int k = 0; k < 2; ++k) { cudaFree(raw[k]); cudaFree(dt[k]); cudaEventDestroy(done[k]); }
    cudaStreamDestroy(st);
    return ok;
  }
  // a bundle tensor the host reads (a few small ones: ESM-C's mix weights), decoded as the device does
  const float* hostTensor(const Segment& s, int rec) const {
    Segment& m = const_cast<Segment&>(s);
    auto it = m.hostCopies.find(rec);
    if (it != m.hostCopies.end()) return it->second.data();
    const BRec& b = s.recs[rec];
    if (b.kind == 5) {
      std::vector<float> out(b.elements, 0.f);
      std::map<int, std::vector<float>> src;
      for (const GPart& g : s.parts) {
        if (g.rec != rec) continue;
        for (int q = 0; q < g.nsrc; ++q) if (!src.count(g.src[q])) src[g.src[q]] = decodeComposite(s, s.srcRecs, g.src[q]);
        size_t total = 1; for (int k = 0; k < g.rank; ++k) total *= g.dims[k];
        for (size_t o = 0; o < total; ++o) {
          size_t r = o; long long d = (long long)g.dst, a = (long long)g.srcOff[0], c = (long long)g.srcOff[1];
          for (int k = g.rank - 1; k >= 0; --k) {
            long long i = (long long)(r % g.dims[k]); r /= g.dims[k];
            d += i * g.dstStride[k]; a += i * g.srcStride[0][k]; c += i * g.srcStride[1][k];
          }
          float v;
          if (g.op == 'o') v = 1.f;
          else if (g.op == 'x') v = src[g.src[0]][a] * src[g.src[1]][c];
          else if (g.op == 'i') { int iv = (int)src[g.src[0]][a]; memcpy(&v, &iv, 4); }
          else v = src[g.src[0]][a];
          out[d] = v;
        }
      }
      return (m.hostCopies[rec] = std::move(out)).data();
    }
    return (m.hostCopies[rec] = decodeComposite(s, s.recs, rec)).data();
  }
  // a record as the device decodes it: a delta's addTo is the base rounded to float16 plus the delta's
  static std::vector<float> decodeComposite(const Segment& s, const std::vector<BRec>& pool, int i) {
    const BRec& b = pool[i];
    std::vector<float> out = decodeHost(s, b);
    if (b.round16) for (float& v : out) v = __half2float(__float2half_rn(v));
    if (b.addRec >= 0) {
      std::vector<float> d = decodeHost(s, pool[b.addRec]);
      for (size_t k = 0; k < out.size(); ++k) out[k] = out[k] + d[k];
    }
    return out;
  }
  static std::vector<float> decodeHost(const Segment& s, const BRec& b) {
    std::vector<float> out(b.elements, 0.f);
    if (b.kind == 4) return out;
    if (b.kind == 8) return s.consts[b.byteOffset];
    std::vector<unsigned char> raw = readFile(shardPath(s, s.files[b.file]));
    for (size_t o = 0; o < b.elements; ++o) {
      size_t i = b.first + o;
      if (b.kind == 0) memcpy(&out[o], &raw[b.byteOffset + 4 * i], 4);
      else if (b.kind == 1) out[o] = hostHalf(&raw[b.byteOffset + 2 * i]);
      else if (b.kind == 2) out[o] = (float)((double)(signed char)raw[b.byteOffset + i] * (double)hostHalf(&raw[b.scaleOffset + 2 * (i / b.block)]));
      else if (b.kind == 6) {
        size_t row = i / b.block, col = i % b.block, g = (b.rows + b.rowBlocks - 1) / b.rowBlocks;
        float sc; memcpy(&sc, &raw[b.scaleOffset + 4 * ((row / g) * b.block + col)], 4);
        volatile float p = (float)(signed char)raw[b.byteOffset + i] * sc;
        out[o] = p;
      } else if (b.kind == 7) {
        unsigned int h = (unsigned int)(raw[b.byteOffset + 2 * i] | (raw[b.byteOffset + 2 * i + 1] << 8)) << 16;
        memcpy(&out[o], &h, 4);
      } else {
        size_t g = i / b.block, groupBytes = (size_t)b.block * b.bits / 8, bit = g * groupBytes * 8 + (i % b.block) * b.bits, byte = bit >> 3;
        unsigned code = ((raw[b.byteOffset + byte] | (raw[b.byteOffset + byte + 1] << 8)) >> (bit & 7)) & ((1u << b.bits) - 1);
        volatile double p = (double)code * (double)hostHalf(&raw[b.scaleOffset + 2 * g]);
        out[o] = (float)(p + (double)hostHalf(&raw[b.zeroOffset + 2 * g]));
      }
    }
    return out;
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
    for (auto& [name, e] : index) if (e.seg == seg && e.kind == 't') { dev(name); touched.erase(name); return; }
  }
  // ...or in the background: the device copy is allocated at once and filled by a thread, so kernels
  // can be launched against it meanwhile - a warm-up, whose answers are garbage until waitUploads()
  // (and whose derived weights must then be forgotten: forgetDerivedWeights())
  std::map<int, std::thread> pending;
  void uploadAsync(int seg) {
    Segment& s = segs[seg];
    if (s.device || pending.count(seg)) return;
    if (devMalloc(&s.device, std::max<size_t>(s.deviceBytes, 4)) != cudaSuccess) {
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
    for (auto& [name, e] : index) if (e.seg == seg && e.kind == 't' && e.devOffset != DROPPED) keep.push_back({&e, name});
    std::sort(keep.begin(), keep.end(), [](auto& a, auto& b) { return a.first->devOffset < b.first->devOffset; });
    size_t at = 0; std::vector<std::array<size_t, 3>> moves; std::vector<Entry*> dropped;
    for (auto& [e, name] : keep) {
      if (drop(name)) { dropped.push_back(e); continue; }
      at = (at + 3) / 4 * 4;
      moves.push_back({e->devOffset, at, e->length}); e->devOffset = at; at += e->length;
    }
    float* fresh; CK(devMalloc(&fresh, std::max<size_t>(at, 1) * 4));
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
    if (s.bundle) return bundleUp(s);
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
    { std::lock_guard<std::mutex> g(hostMu); touched.insert(k); }
    Segment& s = segs[e.seg];
    if (!s.device) {
      if (devMalloc(&s.device, std::max<size_t>(s.deviceBytes, 4)) != cudaSuccess || !copyUp(s)) {
        fprintf(stderr, "cannot put a model.bin (%zu bytes) on the device\n", s.bytes); exit(1);
      }
    }
    if (e.devOffset == DROPPED) { fprintf(stderr, "%s: its f32 device copy was dropped (read its f16 mirror)\n", k.c_str()); exit(1); }
    if (e.kind == 'q') { fprintf(stderr, "%s is resident int8: read it with residentInt8\n", k.c_str()); exit(1); }
    return s.device + e.devOffset;
  }
  // The resident codes off the device and back: a fold short of room parks a tower it has finished with (ESM-C 6B's
  // 6.4 GB, idle once the language model has run), and the next fold reads it back from its shards - the bytes
  // bundleUp copied, a tensor at a time through one pinned buffer
  size_t parkResident() {
    size_t freed = 0;
    for (auto& s : segs) if (s.resident) { CK(cudaFree(s.resident)); s.resident = nullptr; freed += s.residentBytes; }
    return freed;
  }
  bool residentParked() const { for (auto& s : segs) if (s.residentBytes && !s.resident) return true; return false; }
  void unparkResident() {
    for (auto& s : segs) {
      if (!s.residentBytes || s.resident) continue;
      CK(devMalloc(&s.resident, s.residentBytes));
      size_t most = 0;
      for (const auto& r : s.residentRecs) most = std::max(most, r.b.elements + residentScaleBytes(r.b));
      unsigned char* pin; CK(cudaHostAlloc(&pin, most, cudaHostAllocDefault));
      int file = -1, fd = -1;
      for (const auto& r : s.residentRecs) {
        if (r.b.file != file) { if (fd >= 0) close(fd); file = r.b.file; fd = open(shardPath(s, s.files[file]).c_str(), O_RDONLY); }
        size_t cbytes = r.b.elements, sbytes = residentScaleBytes(r.b);
        auto readAt = [&](unsigned char* to, size_t n, size_t at) {
          for (size_t got = 0; got < n;) {
            ssize_t k = pread(fd, to + got, n - got, (off_t)(at + got));
            if (k <= 0) { fprintf(stderr, "cannot read %s\n", s.files[file].c_str()); exit(1); }
            got += (size_t)k;
          }
        };
        readAt(pin, cbytes, r.b.byteOffset); readAt(pin + cbytes, sbytes, r.b.scaleOffset);
        CK(cudaMemcpy(s.resident + r.codes, pin, cbytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(s.resident + r.scales, pin + cbytes, sbytes, cudaMemcpyHostToDevice));
      }
      if (fd >= 0) close(fd);
      CK(cudaFreeHost(pin));
    }
  }
  bool isResident(const std::string& k) const { auto it = index.find(k); return it != index.end() && it->second.kind == 'q'; }
  ResidentInt8 residentInt8(const std::string& k) {
    const Entry& e = at(k);
    if (e.kind != 'q') { fprintf(stderr, "%s is not resident int8\n", k.c_str()); exit(1); }
    { std::lock_guard<std::mutex> g(hostMu); touched.insert(k); }
    Segment& s = segs[e.seg];
    if (!s.device && (devMalloc(&s.device, std::max<size_t>(s.deviceBytes, 4)) != cudaSuccess || !copyUp(s))) {
      fprintf(stderr, "cannot put a model.bin (%zu bytes) on the device\n", s.bytes); exit(1);
    }
    const auto& r = s.residentRecs[e.rec];
    if (r.b.kind == 6)
      return { (const signed char*)(s.resident + r.codes), nullptr, r.b.elements, r.b.block,
               (const float*)(s.resident + r.scales), r.b.rows, r.b.rowBlocks };
    return { (const signed char*)(s.resident + r.codes), (const __half*)(s.resident + r.scales), r.b.elements, r.b.block };
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
  // (host reads may come from the fold's output thread too - af3.cu writes each sample's files beside the next
  // sample's confidence - so what they mutate, `touched`, the host copies and a file's mapping, is guarded)
  mutable std::mutex hostMu;
  const float* f(const std::string& k) const {
    std::lock_guard<std::mutex> g(hostMu);
    const Entry& e = at(k); touched.insert(k);
    if (segs[e.seg].bundle) return hostTensor(segs[e.seg], e.rec);
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
inline void scratchReportOOM(const char* what, size_t bytes);
template <class T> T* dallocT(size_t n) {
  T* p;
  if (devMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)) != cudaSuccess) { scratchReportOOM("a direct allocation", n * sizeof(T)); exit(1); }
  return p;
}
inline float* dalloc(size_t n) { return dallocT<float>(n); }
template <class T> T* upload(const T* h, size_t n) {
  T* p = dallocT<T>(n); CK(cudaMemcpy(p, h, n * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
// Named scratch, kept for the process and grown when asked for more.
inline std::map<std::string, std::pair<void*, size_t>> SCRATCH;
// Every scratch buffer given back: between the trunk, the denoiser and the confidence head of a
// large input, so each phase has the whole card (every phase asks for its buffers again).
inline void releaseScratch() {
  CK(cudaDeviceSynchronize());
  for (auto& [name, slot] : SCRATCH) { if (slot.first) CK(cudaFreeAsync(slot.first, STREAM)); slot = {nullptr, 0}; }
}
// A large input's pair is over 128 MB (512 tokens at 128 channels): there each phase gets the card
// to itself, and a stage's own scratch is given back when the stage is done.
// LOCALFOLD_BIG=1: every big-input path at any size (shortPair true, roomFor false) - how they are checked
// against the ordinary ones on an input small enough to fold both ways
inline const bool BIG_FORCED = getenv("LOCALFOLD_BIG") != nullptr;
inline bool tightPair(size_t pairs, int C) { return pairs * C * 4 > ((size_t)128 << 20); }
// ...and short of room. AF3 (SHORT_PAIR_TIMES, set by af3.cu): when SHORT_PAIR_TIMES times the f32 pair does not
// fit the room this process had at its first ask - free memory, held scratch counted, a 20th of the card spare,
// fixed then so a warm fold decides as a cold one does (and a second process holding memory moves it, which a card
// fraction did not). The ordinary paths peak at 16-18x the f32 pair, weights included (AF3 32.7 GB at 2000 tokens,
// a 2.05 GB pair; OpenDDE 16.5 GB at 765, 0.90 GB): 20 puts the line at ~1900 tokens for AF3 on 40 GB, ~1100 for
// OpenDDE. It was a 64th of the card, which sent AF3 down the big-input paths from 1118 tokens and OpenDDE from
// 646 with the card two-thirds empty - 55.2 against 31.8 s a fold at 2000 tokens, OpenDDE's trunk 17.8 against
// 12.8 s at 765. AF2 and ESMFold2 (0) keep the 64th until measured. Under it the trunk gives each stage's scratch
// back between stages, which costs the recycles their graph and a reallocation a pass.
inline double SHORT_PAIR_TIMES = 0;
inline bool shortPair(size_t pairs, int C) {
  if (BIG_FORCED) return true;
  if (SHORT_PAIR_TIMES > 0) {
    static const size_t room = [] {
      size_t f, t; deviceMemInfo(&f, &t);
      for (auto& [name, slot] : SCRATCH) f += slot.second;
      size_t r = f > t / 20 ? f - t / 20 : 0;
      if (getenv("LOCALFOLD_MEM")) fprintf(stderr, "  memory room for the ordinary paths %.2f GB (a pair up to %.2f GB)\n",
                                           r / 1e9, r / 1e9 / SHORT_PAIR_TIMES);
      return r;
    }();
    return (double)pairs * C * 4 * SHORT_PAIR_TIMES > (double)room;
  }
  static const size_t card = [] { size_t f, t; CK(cudaMemGetInfo(&f, &t)); return t; }();
  return pairs * C * 4 > card / 64;
}
// the scratch buffers named, given back (the next use allocates afresh): a name ending in '.' is a
// prefix ("tmpl."), any other is exact ("tri.a", which must not take "tri.abf" with it)
inline void releaseScratch(std::initializer_list<const char*> names) {
  bool synced = false;
  for (auto& [name, slot] : SCRATCH) {
    if (!slot.first) continue;
    bool match = false;
    for (const char* p : names) {
      size_t len = strlen(p);
      if (len && p[len - 1] == '.' ? !name.compare(0, len, p) : name == p) match = true;
    }
    if (!match) continue;
    if (!synced) { CK(cudaDeviceSynchronize()); synced = true; }
    CK(cudaFreeAsync(slot.first, STREAM)); slot = {nullptr, 0};
  }
}
// Whether `bytes` more would fit on the card now with an eighth of it to spare - for the memory levers that
// cost time (blocked contractions, recomputed biases), which only a short card should pay. Scratch held
// under any of `held` already counts toward it, since the whole form would reuse it.
inline bool roomFor(size_t bytes, std::initializer_list<const char*> held = {}) {
  if (BIG_FORCED) return false;
  for (auto& [name, slot] : SCRATCH)
    for (const char* p : held) if (name == p) bytes -= std::min(bytes, slot.second);
  size_t f, t; deviceMemInfo(&f, &t);
  return f > bytes + t / 8;
}
// what holds the device when an allocation is refused: the largest scratch buffers, and the free memory
inline void scratchReportOOM(const char* what, size_t bytes) {
  cudaGetLastError();
  size_t held = 0; for (auto& [k, v] : SCRATCH) held += v.second;
  size_t freeB, totalB; deviceMemInfo(&freeB, &totalB);
  fprintf(stderr, "out of device memory: %s wants %.2f GB; scratch holds %.2f GB, %.2f of %.2f GB free\n",
          what, bytes / 1e9, held / 1e9, freeB / 1e9, totalB / 1e9);
  std::vector<std::pair<size_t, std::string>> big;
  for (auto& [k, v] : SCRATCH) big.push_back({v.second, k});
  std::sort(big.rbegin(), big.rend());
  for (size_t i = 0; i < big.size() && i < 12; ++i) fprintf(stderr, "  %8.2f GB  %s\n", big[i].first / 1e9, big[i].second.c_str());
}
// A pair-sized tensor PARKED in pinned host memory while nothing on the device reads it - the template
// stack's blocks and the diffusion sampler leave the trunk's pair alone - on a card short of room. One
// pinned buffer for the process, grown as needed; parking frees the device copy and unparking makes a new
// one (callers take the new pointer).
inline float* PARK_HOST = nullptr; inline size_t PARK_HAVE = 0;
inline bool parkWorthIt(size_t bytes) { return !roomFor(bytes); }
inline void parkToHost(float*& dev, size_t bytes) {
  if (PARK_HAVE < bytes) {
    if (PARK_HOST) CK(cudaFreeHost(PARK_HOST));
    if (cudaMallocHost(&PARK_HOST, bytes) != cudaSuccess) {
      fprintf(stderr, "cannot pin %.2f GB of host memory to park the pair: this input needs more host RAM\n", bytes / 1e9);
      exit(1);
    }
    PARK_HAVE = bytes;
  }
  CK(cudaMemcpyAsync(PARK_HOST, dev, bytes, cudaMemcpyDeviceToHost, STREAM));
  CK(cudaStreamSynchronize(STREAM));
  CK(cudaFree(dev)); dev = nullptr;
}
inline void unparkFromHost(float*& dev, size_t bytes) {
  dev = dallocT<float>(bytes / 4);
  CK(cudaMemcpyAsync(dev, PARK_HOST, bytes, cudaMemcpyHostToDevice, STREAM));
}
template <class T> T* scratch(const std::string& name, size_t n) {
  auto& [p, have] = SCRATCH[name];
  if (have < n * sizeof(T)) {
    // (from the pool, stream-ordered - and freed only after the device is idle, as cudaFree was: the frame tap's
    // copies read scratch on a stream of their own. Never inside a capture, where a pool allocation would become
    // the graph's own; a recycle pass is captured once every buffer is sized)
    size_t bytes = std::max<size_t>(n, 1) * sizeof(T);
    cudaStreamCaptureStatus capturing; CK(cudaStreamIsCapturing(STREAM, &capturing));
    if (p) { CK(cudaDeviceSynchronize()); CK(cudaFreeAsync(p, STREAM)); p = nullptr; }
    cudaError_t e = capturing != cudaStreamCaptureStatusNone ? cudaMalloc(&p, bytes)
                                                             : cudaMallocFromPoolAsync(&p, bytes, scratchPool(), STREAM);
    if (e != cudaSuccess && capturing == cudaStreamCaptureStatusNone) {     // (fragmented idle pages: trimmed, asked again)
      cudaGetLastError(); trimPool(); e = cudaMallocFromPoolAsync(&p, bytes, scratchPool(), STREAM);
    }
    if (e != cudaSuccess) {
      size_t held = 0; for (auto& [k, v] : SCRATCH) held += v.second;
      size_t freeB, totalB; deviceMemInfo(&freeB, &totalB);
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
// --serve=DIR: the process stays up, its weights resident, folding each job dropped in DIR - a cold fold is
// mostly start-up (the CUDA context, the weight upload, the kernels' first launches) and this pays it once.
// A job is DIR/<id>.job, renamed into place: its first line the input's directory, then one flag a line;
// its output goes to <id>.log and its exit status to <id>.done (renamed into place). Jobs run in name
// order; one whose first line reads "quit" stops the server. `fold(input, flags)` returns the exit status.
inline void serveJobs(const std::string& dir, const char* name,
                      const std::function<int(const std::string&, const std::vector<std::string>&)>& fold) {
  printf("%s: serving %s\n", name, dir.c_str()); fflush(stdout);
  for (;;) {
    std::string id;
    if (DIR* d = opendir(dir.c_str())) {
      std::vector<std::string> jobs;
      while (dirent* e = readdir(d)) {
        std::string file = e->d_name;
        if (file.size() > 4 && file.substr(file.size() - 4) == ".job") jobs.push_back(file.substr(0, file.size() - 4));
      }
      closedir(d);
      if (!jobs.empty()) { std::sort(jobs.begin(), jobs.end()); id = jobs[0]; }
    }
    if (id.empty()) { usleep(2000); continue; }
    std::string base = dir + "/" + id;
    std::ifstream job(base + ".job");
    std::string input, line; std::getline(job, input);
    std::vector<std::string> flags; while (std::getline(job, line)) if (!line.empty()) flags.push_back(line);
    job.close(); unlink((base + ".job").c_str());
    if (input == "quit") { printf("%s: stopped\n", name); return; }
    fflush(stdout); fflush(stderr);
    int saved = dup(1), savedErr = dup(2), log = open((base + ".log").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    dup2(log, 1); dup2(log, 2); close(log);
    int code = fold(input, flags);
    fflush(stdout); fflush(stderr); dup2(saved, 1); dup2(savedErr, 2); close(saved); close(savedErr);
    CK(cudaDeviceSynchronize());
    FILE* df = fopen((base + ".done.tmp").c_str(), "w"); fprintf(df, "%d\n", code); fclose(df);
    rename((base + ".done.tmp").c_str(), (base + ".done").c_str());
  }
}
// LOCALFOLD_MEM=1: device memory in use at a phase boundary, and the largest scratch buffers
inline void memReport(const char* at) {
  if (!getenv("LOCALFOLD_MEM")) return;
  cudaStreamCaptureStatus capturing;                 // (a recycle pass being captured: no sync there)
  CK(cudaStreamIsCapturing(STREAM, &capturing));
  if (capturing != cudaStreamCaptureStatusNone) return;
  CK(cudaDeviceSynchronize()); size_t fr, tot; deviceMemInfo(&fr, &tot);     // (the pool's idle pages not "in use")
  size_t held = 0; std::vector<std::pair<size_t, std::string>> big;
  for (auto& [k, v] : SCRATCH) { held += v.second; if (v.second) big.push_back({v.second, k}); }
  std::sort(big.rbegin(), big.rend());
  printf("  memory %-22s %6.2f GB in use, scratch %.2f:", at, (tot - fr) / 1e9, held / 1e9);
  for (size_t i = 0; i < big.size() && i < (getenv("LOCALFOLD_MEM_ALL") ? big.size() : 6); ++i) printf(" %s %.2f", big[i].second.c_str(), big[i].first / 1e9);
  printf("\n");
}
// Shared memory a block may take: the device's opt-in limit (227 KB on an A100, 64 KB on a T4),
// lowered by LOCALFOLD_SMEM_LIMIT to rehearse a smaller device here. A fused path that would need more
// asks fitsSmem() and takes its unfused path; smemAttr() refuses, naming the kernel, rather than
// letting a launch fail with "invalid argument".
inline size_t SMEM_LIMIT = 0;
inline size_t smemLimit() {
  if (!SMEM_LIMIT) {
    int dev = 0, v = 0; CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    SMEM_LIMIT = (size_t)v;
    if (const char* e = getenv("LOCALFOLD_SMEM_LIMIT")) SMEM_LIMIT = std::min(SMEM_LIMIT, (size_t)atol(e));
  }
  return SMEM_LIMIT;
}
inline bool fitsSmem(size_t bytes) { return bytes <= smemLimit(); }
template <class K> void smemAttrNamed(K kernel, size_t bytes, const char* name) {
  if (!fitsSmem(bytes)) {
    fprintf(stderr, "%s wants %zu bytes of shared memory a block; this device allows %zu (a fused path missing its fitsSmem check)\n",
            name, bytes, smemLimit());
    fflush(stderr); _exit(1);    // (_exit: a background upload thread may still be running)
  }
  CK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
}
#define smemAttr(kernel, bytes) smemAttrNamed(kernel, (size_t)(bytes), #kernel)
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
// A weight concatenated along its output columns (concatColumns): each part (C, width) row-major
// (in, out), or stored (width, C) if `transposed`; a part with no name is zero columns.
struct Part { std::string name; int width; bool transposed; };
// ...and how each was built, so its f32 copy can be given back once its f16 one exists (Wh) and built
// again, exactly, by a later W() - the fast path reads only the f16 one
inline std::map<std::string, std::pair<int, std::vector<Part>>> CONCAT;
inline std::vector<float*> CONCAT_FREE;
// the f32 copies Wh() queued, given back (one drain for all of them; at a phase boundary)
inline void releaseConcatCopies() {
  if (CONCAT_FREE.empty()) return;
  CK(cudaDeviceSynchronize());
  for (float* p : CONCAT_FREE) CK(cudaFree(p));
  CONCAT_FREE.clear();
}
inline void buildConcat(const std::string& key);
inline const float* W(const std::string& k) {
  auto it = WF.find(k);
  if (it != WF.end()) return it->second;
  if (CONCAT.count(k)) { buildConcat(k); return WF[k]; }
  WLEN[k] = M.len(k);
  return WF[k] = const_cast<float*>(M.dev(k));
}
inline void deviceWeight(const std::string& k, float* p, size_t n) { WF[k] = p; WLEN[k] = n; }
// The concatenation on the device from its copies of the parts. concatColumns returns `key`, under
// which W()/Wh() serve the (C, sum of widths) result.
__global__ void transposeIntoK(const float* src, float* dst, int C, int width, size_t ld) {   // dst[c][o] = src[o][c]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)C * width) return;
  int c = (int)(t / width), o = (int)(t % width);
  dst[c * ld + o] = src[(size_t)o * C + c];
}
inline std::string concatColumns(const std::string& key, int C, const std::vector<Part>& parts) {
  CONCAT.emplace(key, std::make_pair(C, parts));
  if (!WF.count(key) && !WH.count(key)) buildConcat(key);
  return key;
}
inline void buildConcat(const std::string& key) {
  const auto& [C, parts] = CONCAT.at(key);
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
}
inline std::set<std::string> WH_MIRROR;     // the f16 views into a file's mirror (freed with it)
// a file's f16 copy, made once per GROUP - the first f16 read of any of its weights converts every
// float tensor of that group in ONE launch (it was ~800 allocations and conversions, one a weight, in
// the first fold); each tensor starts on 16 bytes, as its own allocation did, for the vector loads.
// A group is a name's first '/'-separated part ("" without one): cuda/ef2's file holds the folding
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
    CK(devMalloc(&mirror, std::max<size_t>(total, 1) * 2));
    size_t* table; CK(devMalloc(&table, from.size() * 3 * sizeof(size_t)));
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
__global__ void toBf16WK(const float* in, __nv_bfloat16* out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16(in[i]);
}
// a weight's bf16 copy (a GEMM whose operands and output are bf16: a product accumulated straight into a bf16 pair)
inline const __nv_bfloat16* Wbf(const std::string& k) {
  static std::map<std::string, __nv_bfloat16*> cache;
  auto it = cache.find(k);
  if (it != cache.end()) return it->second;
  const float* f = W(k); size_t n = lenW(k);
  __nv_bfloat16* b = dallocT<__nv_bfloat16>(n);
  toBf16WK<<<blocks(n), 256, 0, STREAM>>>(f, b, n);
  return cache[k] = b;
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
  // a concatenation's f32 copy given back at the next phase boundary (buildConcat makes it again if
  // asked) - AF3's peak 7.06 -> 6.57 GB at 262 tokens. Queued, not freed here: a cudaFree drains the
  // device, and one per weight was 27 ms of a cold 464 ms trunk
  if (CONCAT.count(k)) { CONCAT_FREE.push_back(const_cast<float*>(f)); WF.erase(k); }
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
// (Not for cuda/af3 as it stands: TCACHE folds its conditioning weights once per process and keeps
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
  return std::is_same_v<T, float> ? CUDA_R_32F : std::is_same_v<T, __nv_bfloat16> ? CUDA_R_16BF : CUDA_R_16F;
}
template <class T> __device__ __forceinline__ float toF(T v) {
  if constexpr (std::is_same_v<T, float>) return v;
  else if constexpr (std::is_same_v<T, __nv_bfloat16>) return __bfloat162float(v);
  else return __half2float(v);
}
template <class T> __device__ __forceinline__ T fromF(float v) {
  if constexpr (std::is_same_v<T, float>) return v;
  else if constexpr (std::is_same_v<T, __nv_bfloat16>) return __float2bfloat16(v);
  else return __float2half(v);
}
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
// the sigmoid on one MUFU op (tanh.approx: |error| of the sigmoid <= ~2.5e-4) where exp and a reciprocal
// were two - only where that is below the noise already there: a result rounded to f16 straight after
// (its own step ~4.9e-4 relative), or the triangle output's gate, whose update comes off a bf16 product
// (~4e-3 relative). The transition, triangle-input and triangle-output kernels' sigmoids (ncu: 11-18% of
// their stall samples); a 5CAJ fold moves 0.006 A rms
__device__ __forceinline__ float sigmH(float x) {
  float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x)); return fmaf(0.5f, t, 0.5f);
}

// The fast path's remaining f32 GEMMs (the conditioning, the atom blocks' aggregation and
// broadcast projections, ...) on the tensor cores in TF32 - a 10-bit mantissa, as f16 has
inline bool F32_TF32 = false;
// Row-major Y[rows x out] = X[rows x in] W + beta Y, W (in,out) or (out,in) if transposed.
// X and W in T (W's f16 copy for half); Y in TY; f32 accumulation always.
template <class T, class TY>
void linear(const T* X, TY* Y, size_t rows, int in, int out, const std::string& w,
            bool transposed = false, float beta = 0.f, float alpha = 1.f) {
  const float one = alpha;
  const void* Wp;
  if constexpr (std::is_same_v<T, float>) Wp = W(w);
  else if constexpr (std::is_same_v<T, __nv_bfloat16>) Wp = Wbf(w);
  else Wp = Wh(w);
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

// linear() over `batch` blocks of `rows` rows, X and Y each at their own row stride and block stride (in
// elements): a transposed operand read or written in place - block b's row j at X + b * sx + j * ldx
template <class T, class TY>
void linearStrided(const T* X, size_t ldx, size_t sx, TY* Y, size_t ldy, size_t sy, int rows, int batch, int in,
                   int out, const std::string& w, float beta = 0.f) {
  static_assert(!std::is_same_v<T, float>, "the f16 path's");
  const float one = 1.f;
  if (lenW(w) != (size_t)in * out) {
    fprintf(stderr, "%s has %zu elements, not %d x %d\n", w.c_str(), lenW(w), in, out); exit(1);
  }
  CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, out, rows, in, &one, Wh(w), cudaType<T>(), out, 0,
                                X, cudaType<T>(), (long long)ldx, (long long)sx, &beta, Y, cudaType<TY>(), (long long)ldy,
                                (long long)sy, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ---------------------------------------------------------------- checking and timing
inline double relRms(const float* a, const float* b, size_t n) {
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) { double e = (double)a[i] - b[i]; num += e * e; den += (double)b[i] * b[i]; }
  return std::sqrt(num / std::max(den, 1e-300));
}
// The rotation taking `moving` (centred) onto `fixed` (centred) in the least-squares sense - Horn's
// quaternion method: the largest eigenvector of a symmetric 4x4 built from their covariance (Jacobi).
inline void bestRotation(const std::vector<double>& moving, const std::vector<double>& fixed, double R[9]) {
  double S[3][3] = {};
  for (size_t i = 0; i + 2 < moving.size(); i += 3)
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) S[a][b] += moving[i + a] * fixed[i + b];
  double N[4][4] = {
    {S[0][0] + S[1][1] + S[2][2], S[1][2] - S[2][1], S[2][0] - S[0][2], S[0][1] - S[1][0]},
    {S[1][2] - S[2][1], S[0][0] - S[1][1] - S[2][2], S[0][1] + S[1][0], S[2][0] + S[0][2]},
    {S[2][0] - S[0][2], S[0][1] + S[1][0], -S[0][0] + S[1][1] - S[2][2], S[1][2] + S[2][1]},
    {S[0][1] - S[1][0], S[2][0] + S[0][2], S[1][2] + S[2][1], -S[0][0] - S[1][1] + S[2][2]}};
  double V[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  for (int sweep = 0; sweep < 50; ++sweep) {
    double off = 0; for (int p = 0; p < 4; ++p) for (int q = p + 1; q < 4; ++q) off += N[p][q] * N[p][q];
    if (off < 1e-22) break;
    for (int p = 0; p < 4; ++p) for (int q = p + 1; q < 4; ++q) {
      if (std::fabs(N[p][q]) < 1e-300) continue;
      double theta = (N[q][q] - N[p][p]) / (2 * N[p][q]);
      double t = (theta >= 0 ? 1 : -1) / (std::fabs(theta) + std::sqrt(theta * theta + 1)), c = 1 / std::sqrt(t * t + 1), sn = t * c;
      for (int k = 0; k < 4; ++k) { double a = N[k][p], b = N[k][q]; N[k][p] = c * a - sn * b; N[k][q] = sn * a + c * b; }
      for (int k = 0; k < 4; ++k) { double a = N[p][k], b = N[q][k]; N[p][k] = c * a - sn * b; N[q][k] = sn * a + c * b; }
      for (int k = 0; k < 4; ++k) { double a = V[k][p], b = V[k][q]; V[k][p] = c * a - sn * b; V[k][q] = sn * a + c * b; }
    }
  }
  int best = 0; for (int k = 1; k < 4; ++k) if (N[k][k] > N[best][best]) best = k;
  double w = V[0][best], x = V[1][best], y = V[2][best], z = V[3][best];
  double R0[9] = {w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z};
  for (int k = 0; k < 9; ++k) R[k] = R0[k];
}

// ---------------------------------------------------------------- the asynchronous tap
// Device buffers handed to a host thread WITHOUT the fold's stream waiting for them: what streams a fold's
// intermediate results (sampler frames, each trunk pass's contact map, AF2's per-pass confidences) to the
// page. offer() snapshots the parts into a free slot on the fold's own stream (device-to-device, a few
// hundred KB at most, in stream order so the next step cannot overwrite them first) and records an event;
// a second stream - the copy engine - waits on it and copies the slot to pinned host memory; one host
// thread waits for THAT and calls the job's `done` with the host bytes, in the order offered.
// 🔴 THE SLOTS ARE RESERVED BEFORE A FOLD (reserve), because cudaMalloc synchronises the device and one
// allocated mid-fold would stall it - and kept between folds. With no free slot an offer is DROPPED: an
// intermediate result is a picture, and the fold never waits for one.
struct AsyncTap {
  struct Slot { char* dev = nullptr; char* host = nullptr; size_t cap = 0; cudaEvent_t ready{}, copied{}; bool busy = false; };
  struct Job { int slot; std::vector<size_t> offsets; std::function<void(const char*, const std::vector<size_t>&)> done; };
  std::vector<Slot> slots; cudaStream_t copy = nullptr;
  std::mutex mu; std::condition_variable cv; std::deque<Job> queue; bool stopping = false; int pending = 0;
  std::thread writer; int dropped = 0;
  // at least `count` slots of at least `bytes` each (allocating, so: before a fold)
  void reserve(int count, size_t bytes) {
    bytes = (bytes + 255) / 256 * 256;         // (an offer rounds each part so)
    if (!copy) {
      CK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
      writer = std::thread([this] { loop(); });
    }
    drain();
    for (auto& sl : slots) if (sl.cap < bytes) {
      CK(cudaFree(sl.dev)); CK(cudaFreeHost(sl.host));
      CK(devMalloc(&sl.dev, bytes)); CK(cudaMallocHost(&sl.host, bytes)); sl.cap = bytes;
    }
    while ((int)slots.size() < count) {
      Slot sl; CK(devMalloc(&sl.dev, bytes)); CK(cudaMallocHost(&sl.host, bytes)); sl.cap = bytes;
      CK(cudaEventCreateWithFlags(&sl.ready, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&sl.copied, cudaEventDisableTiming));
      slots.push_back(sl);
    }
  }
  bool offer(const std::vector<std::pair<const void*, size_t>>& parts,
             std::function<void(const char*, const std::vector<size_t>&)> done) {
    size_t total = 0; std::vector<size_t> offsets;
    for (auto& p : parts) { offsets.push_back(total); total += (p.second + 255) / 256 * 256; }
    int k = -1;
    { std::lock_guard<std::mutex> lock(mu);
      for (size_t j = 0; j < slots.size(); ++j) if (!slots[j].busy && slots[j].cap >= total) { k = (int)j; slots[j].busy = true; ++pending; break; } }
    if (k < 0) { ++dropped; return false; }
    Slot& sl = slots[k];
    for (size_t q = 0; q < parts.size(); ++q)
      CK(cudaMemcpyAsync(sl.dev + offsets[q], parts[q].first, parts[q].second, cudaMemcpyDeviceToDevice, STREAM));
    CK(cudaEventRecord(sl.ready, STREAM));
    CK(cudaStreamWaitEvent(copy, sl.ready, 0));
    CK(cudaMemcpyAsync(sl.host, sl.dev, total, cudaMemcpyDeviceToHost, copy));
    CK(cudaEventRecord(sl.copied, copy));
    { std::lock_guard<std::mutex> lock(mu); queue.push_back({k, offsets, std::move(done)}); }
    cv.notify_one();
    return true;
  }
  // every offered job handed over (the end of a fold, before its outputs are declared written)
  void drain() {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [this] { return pending == 0; });
  }
  void loop() {
    for (;;) {
      Job job;
      { std::unique_lock<std::mutex> lock(mu); cv.wait(lock, [this] { return stopping || !queue.empty(); });
        if (queue.empty()) return; job = std::move(queue.front()); queue.pop_front(); }
      Slot& sl = slots[job.slot];
      CK(cudaEventSynchronize(sl.copied));                // (this thread waits; the fold does not)
      job.done(sl.host, job.offsets);
      { std::lock_guard<std::mutex> lock(mu); sl.busy = false; --pending; }
      cv.notify_all();
    }
  }
};
inline AsyncTap& TAP() { static AsyncTap* tap = new AsyncTap(); return *tap; }   // (never destroyed: its thread lives with the process)
// a probability or a distance as one byte a pair, on the device, before it is tapped: x / scale, clamped
__global__ void quantiseK(const float* in, unsigned char* out, size_t n, float scale) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (unsigned char)fminf(255.f, fmaxf(0.f, rintf(in[i] / scale)));
}
// a file written whole: a temporary renamed into place, so a reader never sees half of one
inline void writeWhole(const std::string& path, const void* data, size_t bytes) {
  FILE* f = fopen((path + ".tmp").c_str(), "wb"); fwrite(data, 1, bytes, f); fclose(f);
  rename((path + ".tmp").c_str(), path.c_str());
}
// a square matrix as the confidence files write it - rows "[a, b, ...]" joined by ",\n  ", two decimals - into j
// (std::to_chars's fixed precision rounds as "%.2f" does; a fprintf a number was 27 ms of a 261 x 261 matrix)
inline void appendMatrix2(std::string& j, const float* m, int L) {
  j.reserve(j.size() + (size_t)L * L * 7 + 16);
  j += "[";
  for (int i = 0; i < L; ++i) {
    j += i ? ",\n  [" : "[";
    for (int c = 0; c < L; ++c) {
      if (c) j += ", ";
      char b[64]; auto r = std::to_chars(b, b + sizeof b, m[(size_t)i * L + c], std::chars_format::fixed, 2); j.append(b, r.ptr);
    }
    j += "]";
  }
  j += "]";
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
#include "elementwise.cuh"    // the element-wise kernels every port shares
