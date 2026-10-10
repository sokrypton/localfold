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
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
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
  int pid[MAX_RANKS];                 // every rank's process, so a barrier notices one that died without exiting
  Slot slot[MAX_SLOTS];
};
inline Shm* SHM = nullptr;
// LOCALFOLD_MG_SIMULATE=1 with --gpus=N: no fork - this one process is rank 0 of N, every peer's buffer its own (a
// peer copy becomes a copy on this card, a barrier nothing). The results are wrong; the TIMES are one rank's compute at
// the real shapes and its real memory, which is what a box of N GPUs is short of when it is not N times faster - so
// the per-rank work can be measured and tuned on one GPU. (The interconnect is the part it cannot see.)
inline bool SIM = false;
inline bool FINISHED = false;
inline std::vector<pid_t> CHILDREN;

inline void abortAll() { if (SHM) SHM->abort.store(1); }
// (rank 0 reaps the ranks that left only as it exits itself: their CUDA teardown runs beside the rest of its fold)
inline void onExit() {
  if (WORLD > 1 && !FINISHED) abortAll();
  if (FINISHED) for (pid_t p : CHILDREN) { int st = 0; waitpid(p, &st, 0); }
}
// every rank calls this at the end of a fold that succeeded; any other exit tells the others to stop
inline void finish() { FINISHED = true; }
// a rank other than 0 at the end of its part of a fold that succeeded: gone at once, without waiting on its CUDA
// teardown (the driver frees its memory as the process ends; rank 0 waited for that, in its fold's time - 0.4 s at
// 741 tokens on 4 ranks sharing one A100)
[[noreturn]] inline void leave() { finish(); fflush(stdout); fflush(stderr); _exit(0); }

// a peer gone without passing through exit (a signal, the OOM killer, an error): rank 0's watchdog reaps its children
// (one that left with status 0 has finished its part); another rank sees rank 0 gone through its parent pid (and
// PR_SET_PDEATHSIG kills it when rank 0 dies anyway)
inline std::atomic<bool> CHILD_FAILED{false};
inline bool peerDied() { return RANK == 0 ? CHILD_FAILED.load() : getppid() != SHM->pid[0]; }
// Every rank's watchdog thread: the fences are on the DEVICE (fence() below), so a rank waiting for a peer that died
// sits in a kernel, not in a host loop that could notice - this thread notices instead and ends the process (a
// peer's failure, or the abort flag another rank set) rather than leave it waiting for ever.
inline void watchdog() {
  std::thread([] {
    std::vector<pid_t> live = CHILDREN;
    for (;;) {
      usleep(20000);
      if (RANK == 0)
        for (size_t k = 0; k < live.size();) {
          int st = 0;
          if (waitpid(live[k], &st, WNOHANG) == live[k]) {
            if (!(WIFEXITED(st) && WEXITSTATUS(st) == 0)) CHILD_FAILED.store(true);
            live.erase(live.begin() + k);
          } else ++k;
        }
      if ((SHM->abort.load() || peerDied()) && !FINISHED) {
        SHM->abort.store(1);
        fprintf(stderr, "rank %d: a peer failed\n", RANK); fflush(stderr); _exit(1);
      }
    }
  }).detach();
}
inline void barrier() {
  if (WORLD == 1 || SIM) return;
  const int gen = SHM->generation.load();
  if (SHM->arrived.fetch_add(1) == WORLD - 1) { SHM->arrived.store(0); SHM->generation.fetch_add(1); return; }
  for (long spin = 0; SHM->generation.load() == gen; ++spin) {
    if (SHM->abort.load() || (spin % 4096 == 4095 && peerDied())) {
      SHM->abort.store(1);
      fprintf(stderr, "rank %d: a peer failed\n", RANK); fflush(stderr); _exit(1);
    }
    usleep(20);
  }
}

// the GPUs this process may use, counted without the CUDA runtime (which must not start before the fork):
// CUDA_VISIBLE_DEVICES's entries where it is set, else the driver's /proc entries
inline int visibleGpus() {
  if (const char* v = getenv("CUDA_VISIBLE_DEVICES")) {
    int k = 0; for (const char* s = v; *s; ++s) if (*s != ',' && (s == v || s[-1] == ',')) ++k;
    return k;
  }
  int k = 0;
  if (DIR* d = opendir("/proc/driver/nvidia/gpus")) {
    while (dirent* e = readdir(d)) if (e->d_name[0] != '.') ++k;
    closedir(d);
  }
  return k;
}
// --gpus=N | --gpus=all (or LOCALFOLD_GPUS) from the command line, the flag taken out of it
inline int gpusArg(int& argc, char** argv) {
  int world = getenv("LOCALFOLD_GPUS") ? atoi(getenv("LOCALFOLD_GPUS")) : 1;
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--gpus=", 7)) continue;
    world = !strcmp(argv[i] + 7, "all") ? visibleGpus() : atoi(argv[i] + 7);
    for (int j = i; j + 1 < argc; ++j) argv[j] = argv[j + 1];
    --argc; --i;
  }
  return std::max(world, 1);
}
// before any CUDA call: fork world - 1 ranks, choose each one's device
inline void launch(int world) {
  WORLD = world;
  if (world <= 1) return;
  if (world > MAX_RANKS) { fprintf(stderr, "at most %d GPUs\n", MAX_RANKS); exit(1); }
  if (getenv("LOCALFOLD_MG_SIMULATE")) {
    SIM = true;
    fprintf(stderr, "mg: SIMULATING rank 0 of %d on one GPU (LOCALFOLD_MG_SIMULATE): times only, the fold is wrong\n", world);
    setenv("LOCALFOLD_RANK", "0", 1);
    CK(cudaSetDevice(getenv("LOCALFOLD_GPU_MAP") ? atoi(getenv("LOCALFOLD_GPU_MAP")) : 0));
    return;
  }
  std::string path = "/dev/shm/localfold-mg-" + std::to_string(getpid());
  int fd = open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0600);
  if (fd < 0 || ftruncate(fd, sizeof(Shm)) != 0) { perror("mg shared mapping"); exit(1); }
  SHM = (Shm*)mmap(nullptr, sizeof(Shm), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd); unlink(path.c_str());               // (the mapping outlives the name)
  new (SHM) Shm();
  SHM->pid[0] = getpid();
  const std::string runId = std::to_string(getpid());        // (the run's id, taken before any rank exists)
  for (int r = 1; r < world; ++r) {
    pid_t p = fork();
    if (p == 0) { RANK = r; CHILDREN.clear(); prctl(PR_SET_PDEATHSIG, SIGKILL); if (getppid() != SHM->pid[0]) _exit(1); break; }
    CHILDREN.push_back(p);
  }
  SHM->pid[RANK] = getpid();
  atexit(onExit);
  watchdog();
  // (the standalone front end reads these: rank 0 alone featurises and fetches, cuda/featurise/standalone.h)
  setenv("LOCALFOLD_RANK", std::to_string(RANK).c_str(), 1);
  setenv("LOCALFOLD_MG_ID", runId.c_str(), 1);
  int dev = RANK;
  if (const char* m = getenv("LOCALFOLD_GPU_MAP")) {       // "0,0" or "0,1,2,3"
    std::vector<int> map; for (const char* s = m; *s;) { map.push_back(atoi(s)); while (*s && *s != ',') ++s; if (*s) ++s; }
    if ((int)map.size() < world) { fprintf(stderr, "LOCALFOLD_GPU_MAP names %zu devices for %d ranks\n", map.size(), world); exit(1); }
    dev = map[RANK];
  }
  CK(cudaSetDevice(dev));
  // CUDA IPC maps another GPU's memory only where the two can reach each other (NVLink, or PCIe peer-to-peer)
  std::vector<int> devs; for (int r = 0; r < world; ++r) devs.push_back(r);
  if (const char* mp = getenv("LOCALFOLD_GPU_MAP")) { devs.clear(); for (const char* s = mp; *s;) { devs.push_back(atoi(s)); while (*s && *s != ',') ++s; if (*s) ++s; } }
  for (int r = 0; r < world; ++r) {
    int ok = 1;
    if (devs[r] != dev) CK(cudaDeviceCanAccessPeer(&ok, dev, devs[r]));
    if (!ok) { fprintf(stderr, "GPU %d cannot reach GPU %d's memory (no peer-to-peer): one fold cannot span them\n", dev, devs[r]); exit(1); }
  }
  if (RANK != 0) { fflush(stdout); if (!freopen("/dev/null", "w", stdout)) {} }   // (rank 0 reports)
}

// rows [begin, end) of n that rank r owns: the first n % world ranks take one row more
inline int rowBegin(int n, int r) { int q = n / WORLD, m = n % WORLD; return r * q + std::min(r, m); }
inline int rowEnd(int n, int r) { return rowBegin(n, r + 1); }

// A buffer every rank holds `bytes` of, named so each rank finds its peers' copies: local is this rank's, peer[r]
// rank r's (peer[RANK] == local). Collective: every rank asks for the same name, in the same order, at the same size.
struct Shared {
  void* local = nullptr; void* peer[MAX_RANKS] = {}; size_t bytes = 0; int world = 1;
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
  Shared s; s.bytes = bytes; s.world = WORLD;
  CK(cudaMalloc(&s.local, std::max<size_t>(bytes, 256)));
  if (SIM) { for (int r = 0; r < WORLD; ++r) s.peer[r] = s.local; return m.emplace(name, s).first->second; }
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

// Shared buffers whose names start with one of `prefixes` given back: every rank's imports of its peers' closed, then
// (after a barrier, so no peer still maps it) its own freed. Collective while the ranks run; after the others have
// left (rank 0 alone, WORLD 1) it closes and frees without meeting anyone. The name is not asked for again.
inline void release(const std::vector<std::string>& prefixes) {
  auto& m = sharedBuffers();
  auto match = [&](const std::string& k) { for (auto& p : prefixes) if (!k.compare(0, p.size(), p)) return true; return false; };
  bool any = false; for (auto& [k, s] : m) any = any || match(k);
  if (WORLD > 1) { CK(cudaStreamSynchronize(STREAM)); barrier(); }
  else if (any) CK(cudaStreamSynchronize(STREAM));
  for (auto& [k, s] : m)
    if (match(k)) for (int r = 0; r < s.world; ++r) if (r != RANK && s.peer[r]) { if (!SIM) CK(cudaIpcCloseMemHandle(s.peer[r])); s.peer[r] = nullptr; }
  barrier();
  for (auto it = m.begin(); it != m.end();) {
    if (match(it->first)) { CK(cudaFree(it->second.local)); it = m.erase(it); } else ++it;
  }
}
// The fences, on the DEVICE: each rank's stream writes its arrival (a generation) into every rank's flags and waits
// until every rank's has arrived in its own - so what each rank's stream did before its fence is done and visible to
// the others when any of them moves past it. A host fence (the stream drained, then a host barrier) leaves every GPU
// idle while the slowest rank and the host catch up: ~5,000 such gaps in a 2,964-token trunk pass, 12% of one rank's
// time. The flag buffer is a shared buffer like any other; the abort flag (/dev/shm) is mapped in, so a rank that
// failed releases the others' waits (the watchdog ends them).
struct DevFlags { int* peer[MAX_RANKS]; int world, rank; const volatile int* abort; };
__global__ void arriveAndWaitK(DevFlags f, int gen) {
  const int t = threadIdx.x;
  __threadfence_system();
  if (t < f.world) ((volatile int*)f.peer[t])[f.rank] = gen;
  __threadfence_system();
  if (t < f.world) {
    const volatile int* mine = f.peer[f.rank];
    while (mine[t] < gen && !*f.abort) __nanosleep(64);
  }
  __syncthreads();
  __threadfence_system();
}
inline DevFlags devFlags() {
  static DevFlags f{};
  static bool made = false;
  if (!made) {
    Shared& s = shared("mg.flags", MAX_RANKS * sizeof(int));
    CK(cudaMemset(s.local, 0, MAX_RANKS * sizeof(int)));
    CK(cudaDeviceSynchronize());
    void* dshm = nullptr;                // (the whole mapping, page-aligned as it is)
    CK(cudaHostRegister((void*)SHM, sizeof(Shm), cudaHostRegisterMapped | cudaHostRegisterPortable));
    CK(cudaHostGetDevicePointer(&dshm, (void*)SHM, 0));
    void* ab = (char*)dshm + ((char*)&SHM->abort - (char*)SHM);
    for (int r = 0; r < WORLD; ++r) f.peer[r] = (int*)s.peer[r];
    f.world = WORLD; f.rank = RANK; f.abort = (const volatile int*)ab;
    barrier();                           // (every rank's flags zeroed before any rank arrives)
    made = true;
  }
  return f;
}
inline void fence() {
  if (WORLD == 1 || SIM) return;
  static int gen = 0;
  arriveAndWaitK<<<1, 32, 0, STREAM>>>(devFlags(), ++gen);
  CK(cudaGetLastError());
}
// ...and on the host, where the HOST then reads what the others wrote (a synchronous copy off a peer's buffer)
inline void hostFence() {
  if (WORLD == 1) return;
  CK(cudaStreamSynchronize(STREAM));
  barrier();
}

// LOCALFOLD_MG_TIMES=1: every rank drained and met at named points, rank 0 printing the time since the last one (what
// a phase costs across the ranks; the meeting itself adds a sync, so off by default)
inline void tick(const char* what) {
  static const bool on = getenv("LOCALFOLD_MG_TIMES") != nullptr;
  if (!on) return;
  static auto last = std::chrono::steady_clock::now();
  CK(cudaStreamSynchronize(STREAM));
  if (WORLD > 1) barrier();
  auto now = std::chrono::steady_clock::now();
  if (RANK == 0) fprintf(stderr, "  mg %-30s %9.1f ms\n", what, std::chrono::duration<double, std::milli>(now - last).count());
  last = now;
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
