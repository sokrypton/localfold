#pragma once
// One fold across several GPUs: one PROCESS a GPU (a rank), each with the port's single-device state as it is -
// weights, scratch, streams, cuBLAS plans - and the pair split by rows between them. Ranks share buffers through CUDA
// IPC (cudaIpcGetMemHandle; a peer's buffer is copied with cudaMemcpyAsync, which takes NVLink or PCIe peer-to-peer
// where the box has it and stages through the host where it does not) and meet at a host barrier in a shared mapping
// under /dev/shm. A collective is: the stream drained, a barrier, the copies, the stream drained, a barrier - so a
// rank never overwrites a buffer a peer is still reading.
//
// mg::launch(world) forks the ranks before any CUDA call; LOCALFOLD_GPU_MAP=0,0 puts two ranks on one GPU (what this
// port's single-GPU development box checks with), otherwise rank r takes device r. Rank 0 is the one that reads the
// input and writes the outputs; the others fold silently. A rank that dies takes the others down (the shared abort
// flag), rather than leaving them at a barrier.
#include "common.cuh"
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace mg {
constexpr int MAX_RANKS = 16, MAX_SLOTS = 256;
inline int RANK = 0, WORLD = 1;
struct Slot { char name[48]; size_t bytes; cudaIpcMemHandle_t handle[MAX_RANKS]; };
struct Shm {
  std::atomic<int> arrived, generation, abort, slots;
  Slot slot[MAX_SLOTS];
};
inline Shm* SHM = nullptr;
inline bool FINISHED = false;
inline std::vector<pid_t> CHILDREN;

inline void abortAll() { if (SHM) SHM->abort.store(1); }
inline void onExit() { if (WORLD > 1 && !FINISHED) abortAll(); }
// every rank calls this at the end of a fold that succeeded; any other exit tells the others to stop
inline void finish() {
  FINISHED = true;
  if (RANK == 0) for (pid_t p : CHILDREN) { int st = 0; waitpid(p, &st, 0); }
}

inline void barrier() {
  if (WORLD == 1) return;
  const int gen = SHM->generation.load();
  if (SHM->arrived.fetch_add(1) == WORLD - 1) { SHM->arrived.store(0); SHM->generation.fetch_add(1); return; }
  while (SHM->generation.load() == gen) {
    if (SHM->abort.load()) { fprintf(stderr, "rank %d: a peer failed\n", RANK); fflush(stderr); _exit(1); }
    usleep(20);
  }
}

// before any CUDA call: fork world - 1 ranks, choose each one's device
inline void launch(int world) {
  WORLD = world;
  if (world <= 1) return;
  if (world > MAX_RANKS) { fprintf(stderr, "at most %d GPUs\n", MAX_RANKS); exit(1); }
  std::string path = "/dev/shm/localfold-mg-" + std::to_string(getpid());
  int fd = open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0600);
  if (fd < 0 || ftruncate(fd, sizeof(Shm)) != 0) { perror("mg shared mapping"); exit(1); }
  SHM = (Shm*)mmap(nullptr, sizeof(Shm), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd); unlink(path.c_str());               // (the mapping outlives the name)
  new (SHM) Shm();
  for (int r = 1; r < world; ++r) {
    pid_t p = fork();
    if (p == 0) { RANK = r; CHILDREN.clear(); break; }
    CHILDREN.push_back(p);
  }
  atexit(onExit);
  int dev = RANK;
  if (const char* m = getenv("LOCALFOLD_GPU_MAP")) {       // "0,0" or "0,1,2,3"
    std::vector<int> map; for (const char* s = m; *s;) { map.push_back(atoi(s)); while (*s && *s != ',') ++s; if (*s) ++s; }
    if ((int)map.size() < world) { fprintf(stderr, "LOCALFOLD_GPU_MAP names %zu devices for %d ranks\n", map.size(), world); exit(1); }
    dev = map[RANK];
  }
  CK(cudaSetDevice(dev));
  if (RANK != 0) { fflush(stdout); if (!freopen("/dev/null", "w", stdout)) {} }   // (rank 0 reports)
}

// rows [begin, end) of n that rank r owns: the first n % world ranks take one row more
inline int rowBegin(int n, int r) { int q = n / WORLD, m = n % WORLD; return r * q + std::min(r, m); }
inline int rowEnd(int n, int r) { return rowBegin(n, r + 1); }

// A buffer every rank holds `bytes` of, named so each rank finds its peers' copies: local is this rank's, peer[r]
// rank r's (peer[RANK] == local). Collective: every rank asks for the same name, in the same order, at the same size.
struct Shared {
  void* local = nullptr; void* peer[MAX_RANKS] = {}; size_t bytes = 0;
  template <class T> T* at(int r) const { return (T*)peer[r]; }
};
inline std::map<std::string, Shared>& sharedBuffers() { static std::map<std::string, Shared> m; return m; }
inline Shared& shared(const std::string& name, size_t bytes) {
  auto& m = sharedBuffers();
  auto it = m.find(name);
  if (it != m.end()) {
    if (it->second.bytes < bytes) {
      fprintf(stderr, "mg: shared buffer %s grows %zu -> %zu bytes (size it for the largest use first)\n", name.c_str(),
              it->second.bytes, bytes);
      exit(1);
    }
    return it->second;
  }
  Shared s; s.bytes = bytes;
  CK(cudaMalloc(&s.local, std::max<size_t>(bytes, 256)));
  int idx;
  if (RANK == 0) {
    idx = SHM ? SHM->slots.fetch_add(1) : 0;
    if (idx >= MAX_SLOTS) { fprintf(stderr, "mg: more than %d shared buffers\n", MAX_SLOTS); exit(1); }
    if (SHM) { snprintf(SHM->slot[idx].name, sizeof(SHM->slot[idx].name), "%s", name.c_str()); SHM->slot[idx].bytes = bytes; }
  }
  barrier();
  if (WORLD > 1) {
    idx = -1;
    for (int i = 0, k = SHM->slots.load(); i < k; ++i) if (name == SHM->slot[i].name) idx = i;
    if (idx < 0 || SHM->slot[idx].bytes != bytes) { fprintf(stderr, "rank %d: shared buffer %s disagrees\n", RANK, name.c_str()); exit(1); }
    CK(cudaIpcGetMemHandle(&SHM->slot[idx].handle[RANK], s.local));
  }
  barrier();
  for (int r = 0; r < WORLD; ++r) {
    if (r == RANK) { s.peer[r] = s.local; continue; }
    CK(cudaIpcOpenMemHandle(&s.peer[r], SHM->slot[idx].handle[r], cudaIpcMemLazyEnablePeerAccess));
  }
  barrier();
  return m.emplace(name, s).first->second;
}

// the stream drained and every rank at the same point: what each wrote to its shared buffers is visible to the others
inline void fence() {
  if (WORLD == 1) return;
  CK(cudaStreamSynchronize(STREAM));
  barrier();
}

// Every rank's `bytesOf(r)` bytes at offset `srcOff(r)` of its shared buffer `src`, gathered into this rank's `dst`
// at `dstOff(r)` (dst a plain device pointer). The fence before and after: the senders' writes are finished, and no
// sender reuses `src` while a receiver is copying it.
template <class F1, class F2, class F3>
inline void gather(const Shared& src, void* dst, F1 bytesOf, F2 srcOff, F3 dstOff) {
  fence();
  for (int r = 0; r < WORLD; ++r) {
    size_t b = bytesOf(r);
    if (!b) continue;
    CK(cudaMemcpyAsync((char*)dst + dstOff(r), (const char*)src.peer[r] + srcOff(r), b, cudaMemcpyDefault, STREAM));
  }
  fence();
}

// ---------------------------------------------------------------- phase 1: the trunk pair replicated, its work split
// Every rank holds the whole trunk pair (a Shared buffer, SPLIT); each of its updates computes this rank's share of
// its output blocks - rows (the triangle outgoing, row attention, the transition) or columns (the triangle incoming,
// column attention) - and the other ranks' shares are then copied in. Only the trunk's own pair splits: the template
// stack's and the confidence head's tensors are rank-local and run whole on every rank.
inline Shared* SPLIT = nullptr;
inline bool splitting(const void* pair) { return WORLD > 1 && SPLIT && pair == SPLIT->local; }
// rank r's share of [0, total), total a multiple of align, in whole units of align
inline void shareOf(int total, int align, int r, int& lo, int& hi) {
  const int units = total / align, q = units / WORLD, m = units % WORLD;
  lo = (r * q + std::min(r, m)) * align; hi = ((r + 1) * q + std::min(r + 1, m)) * align;
}
// [0, total) when this pair does not split, else this rank's share
inline void myShare(const void* pair, int total, int align, int& lo, int& hi) {
  if (!splitting(pair)) { lo = 0; hi = total; return; }
  shareOf(total, align, RANK, lo, hi);
}
// after each rank wrote its share of rows (cols false) or columns (cols true) of the n x n x C pair (elem bytes an
// element), every other rank's share copied in: the share of [0, total) in units of align, clipped to n
inline void exchange(bool cols, int n, int C, size_t elem, int total, int align) {
  fence();
  const size_t row = (size_t)n * C * elem;
  for (int r = 0; r < WORLD; ++r) {
    if (r == RANK) continue;
    int lo, hi; shareOf(total, align, r, lo, hi); hi = std::min(hi, n);
    if (hi <= lo) continue;
    char* dst = (char*)SPLIT->local; const char* src = (const char*)SPLIT->peer[r];
    if (!cols) CK(cudaMemcpyAsync(dst + lo * row, src + lo * row, (hi - lo) * row, cudaMemcpyDefault, STREAM));
    else CK(cudaMemcpy2DAsync(dst + lo * C * elem, row, src + lo * C * elem, row, (size_t)(hi - lo) * C * elem, n,
                              cudaMemcpyDefault, STREAM));
  }
  fence();
}
}  // namespace mg
