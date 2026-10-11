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
  int vote[MAX_RANKS];                // (release: whether each rank has the room to keep what it gives back)
  size_t value[MAX_RANKS];            // (minOverRanks)
  Slot slot[MAX_SLOTS];
};
inline Shm* SHM = nullptr;
// LOCALFOLD_MG_SIMULATE=1 with --gpus=N: no fork - this one process is rank 0 of N, every peer's buffer its own (a
// peer copy becomes a copy on this card, a barrier nothing). The results are wrong; the TIMES are one rank's compute at
// the real shapes and its real memory, which is what a box of N GPUs is short of when it is not N times faster - so
// the per-rank work can be measured and tuned on one GPU. (The interconnect is the part it cannot see.)
inline bool SIM = false;
// ranks SHARING a GPU (LOCALFOLD_GPU_MAP=0,0 - how a one-GPU box checks a fold): their fences on the host, since a fence
// waiting in a kernel holds the card while the peer it waits for needs it (two contexts on one card take turns by time
// slice: a 3-rank trunk 14.4 s against 0.4); LOCALFOLD_MG_DEVICE_FENCES=1 keeps them on the device there
inline bool SHARED_DEVICE = false;
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
    for (int a = 0; a < world; ++a) for (int b = a + 1; b < world; ++b) SHARED_DEVICE = SHARED_DEVICE || map[a] == map[b];
    if (getenv("LOCALFOLD_MG_DEVICE_FENCES")) SHARED_DEVICE = false;
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

struct Shared {
  void* local = nullptr; void* peer[MAX_RANKS] = {}; size_t bytes = 0; int world = 1;
  int seg = -1;                        // (the segment it lies in, below)
  template <class T> T* at(int r) const { return (T*)peer[r]; }
};
inline std::map<std::string, Shared>& sharedBuffers() { static std::map<std::string, Shared> m; return m; }
// Shared buffers live in SEGMENTS - one device allocation a rank, every peer's opened once - several buffers to one where
// they are small enough, and a buffer given back is kept, mapped, for the next of about its size (the pool): opening a
// peer's allocation is ~3 ms of driver time with 8 ranks at once (1 with 2) whatever its size, closing one ~1.5, so a
// buffer an allocation of its own, made and given back, cost every rank ~35 ms of a GPU left idle - ~1.6 s a fold at
// 2,964 tokens on 8 x A100, a third of one at 988.
struct Segment { void* local = nullptr; void* peer[MAX_RANKS] = {}; size_t size = 0, used = 0; int live = 0; bool dead = false; };
inline std::vector<Segment>& segments() { static std::vector<Segment> v; return v; }
inline std::vector<Shared>& sharedPool() { static std::vector<Shared> p; return p; }
// every segment holding nothing named given back - its kept buffers dropped from the pool, every rank's imports of its
// peers' closed, then (after a barrier, so no peer still maps it) its own freed (collective)
inline void freeIdle() {
  auto& sg = segments(); auto& pl = sharedPool();
  for (size_t i = 0; i < pl.size();) { if (!sg[pl[i].seg].live) pl.erase(pl.begin() + i); else ++i; }
  for (auto& g : sg)
    if (!g.dead && !g.live)
      for (int r = 0; r < WORLD; ++r) if (r != RANK && g.peer[r]) { if (!SIM) CK(cudaIpcCloseMemHandle(g.peer[r])); g.peer[r] = nullptr; }
  if (WORLD > 1) barrier();
  for (auto& g : sg) if (!g.dead && !g.live) { CK(cudaFree(g.local)); g.dead = true; }
}
// (LOCALFOLD_MEM: the shared segments this rank holds - named, kept in the pool, freed)
inline void segmentReport(const char* at) {
  if (!getenv("LOCALFOLD_MEM") || RANK != 0) return;
  size_t live = 0, idle = 0; int nl = 0, ni = 0;
  for (auto& g : segments()) if (!g.dead) { if (g.live) { live += g.size; ++nl; } else { idle += g.size; ++ni; } }
  size_t pooled = 0; for (auto& s : sharedPool()) pooled += s.bytes;
  size_t f = 0, t = 0; CK(cudaMemGetInfo(&f, &t));
  size_t held = 0; for (auto& [k, v] : SCRATCH) held += v.second;
  fprintf(stderr, "  segments %-24s %d holding named buffers %.2f GB, %d idle %.2f GB; the pool %.2f GB | device in use %.2f, scratch %.2f\n",
          at, nl, live / 1e9, ni, idle / 1e9, pooled / 1e9, (t - f) / 1e9, held / 1e9);
}
// the least of every rank's `v` (collective)
inline size_t minOverRanks(size_t v) {
  if (WORLD == 1 || SIM) return v;
  SHM->value[RANK] = v;
  barrier();
  for (int r = 0; r < WORLD; ++r) v = std::min(v, SHM->value[r]);
  barrier();
  return v;
}
// whether every rank has `bytes` free and a margin (collective: each rank's answer in the shared mapping)
inline bool everyRankHas(size_t bytes) {
  size_t f = 0, t = 0; CK(cudaMemGetInfo(&f, &t));
  const bool mine = f >= bytes;
  if (WORLD == 1 || SIM) return mine;
  SHM->vote[RANK] = mine;
  barrier();
  bool all = true; for (int r = 0; r < WORLD; ++r) all = all && SHM->vote[r];
  barrier();                             // (every vote read before the next writes them)
  return all;
}
constexpr size_t SEGMENT = (size_t)1 << 30, SEGMENT_ALIGN = 4096;
// a new segment of `bytes` on every rank, each peer's opened (collective)
inline int openSegment(size_t bytes, const std::string& name) {
  Segment g; g.size = bytes;
  // (devMalloc: where it is refused, the scratch pool's idle pages given back first - the sampler's, held by the pool
  // past its releaseScratch, refused a 12,103-token rank its 4.7 GB contacts slab with 4.1 GB said free)
  if (devMalloc(&g.local, std::max<size_t>(bytes, 256)) != cudaSuccess) {
    size_t f = 0, t = 0; cudaGetLastError(); cudaMemGetInfo(&f, &t);
    fprintf(stderr, "rank %d: out of device memory for the shared buffer %s (%.2f GB; %.2f of %.2f GB free)\n", RANK,
            name.c_str(), bytes / 1e9, f / 1e9, t / 1e9);
    exit(1);
  }
  if (SIM) { for (int r = 0; r < WORLD; ++r) g.peer[r] = g.local; segments().push_back(g); return (int)segments().size() - 1; }
  int idx;
  if (RANK == 0) {
    idx = SHM ? SHM->slots.fetch_add(1) : 0;
    if (idx >= MAX_SLOTS) { fprintf(stderr, "mg: more than %d shared segments\n", MAX_SLOTS); exit(1); }
    if (SHM) { snprintf(SHM->slot[idx].name, sizeof(SHM->slot[idx].name), "%s", name.c_str()); SHM->slot[idx].bytes = bytes; }
  }
  barrier();
  if (WORLD > 1) {
    idx = -1;
    for (int i = 0, k = SHM->slots.load(); i < k; ++i) if (name == SHM->slot[i].name) idx = i;
    if (idx < 0 || SHM->slot[idx].bytes != bytes) { fprintf(stderr, "rank %d: shared buffer %s disagrees\n", RANK, name.c_str()); exit(1); }
    CK(cudaIpcGetMemHandle(&SHM->slot[idx].handle[RANK], g.local));
  }
  barrier();
  for (int r = 0; r < WORLD; ++r) {
    if (r == RANK) { g.peer[r] = g.local; continue; }
    CK(cudaIpcOpenMemHandle(&g.peer[r], SHM->slot[idx].handle[r], cudaIpcMemLazyEnablePeerAccess));
  }
  barrier();
  segments().push_back(g);
  return (int)segments().size() - 1;
}
// A buffer every rank holds `bytes` of, named so each rank finds its peers' copies: local is this rank's, peer[r]
// rank r's (peer[RANK] == local). Collective: every rank asks for the same name, in the same order, at the same size -
// so every rank's pool, segments and choices below are alike.
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
  auto& sg = segments();
  // one given back and kept: the smallest that holds `bytes` and is not much larger
  auto& pl = sharedPool();
  int best = -1;
  for (int i = 0; i < (int)pl.size(); ++i)
    if (pl[i].bytes >= bytes && pl[i].bytes <= bytes + bytes / 4 + (16u << 20) && (best < 0 || pl[i].bytes < pl[best].bytes)) best = i;
  Shared s;
  if (best >= 0) { s = pl[best]; pl.erase(pl.begin() + best); }
  else {
    // (a large buffer to 64 MiB: the template stack's 64-channel slabs and the triangle's 128-channel planes, 4,709.5
    // and 4,713.0 MB at 12,103 tokens on 8, then fall in one size and each takes the other's from the pool)
    const size_t align = bytes > SEGMENT / 2 ? (size_t)64 << 20 : SEGMENT_ALIGN;
    const size_t want = (std::max<size_t>(bytes, 1) + align - 1) / align * align;
    int k = -1;
    if (want <= SEGMENT / 2)          // (the tail of a shared segment; a large buffer a segment of its own)
      for (int i = 0; i < (int)sg.size() && k < 0; ++i) if (!sg[i].dead && sg[i].size == SEGMENT && sg[i].size - sg[i].used >= want) k = i;
    if (k < 0) {
      const size_t size = want <= SEGMENT / 2 ? SEGMENT : want;
      // (what the pool keeps is given back first where any rank is short of room for the new segment: a 10,127-token
      // fold's confidence slabs did not fit beside the diffusion's 6.6 GB encoder pair, kept)
      bool idle = false; for (auto& g : sg) idle = idle || (!g.dead && !g.live);
      if (idle && !everyRankHas(size + ((size_t)1 << 30))) freeIdle();
      k = openSegment(size, "seg" + std::to_string(sg.size()));
    }
    Segment& g = sg[k];
    s.seg = k; s.bytes = want; s.world = WORLD;
    s.local = (char*)g.local + g.used;
    for (int r = 0; r < WORLD; ++r) s.peer[r] = (char*)g.peer[r] + g.used;
    g.used += want;
  }
  static const bool diag = getenv("LOCALFOLD_MG_FENCE_TIMES") != nullptr;
  if (diag && RANK == 0) fprintf(stderr, "  shared %-16s %9.1f MB %s seg %d\n", name.c_str(), bytes / 1e6, best >= 0 ? "(pool)" : "NEW", s.seg);
  ++sg[s.seg].live;
  return m.emplace(name, s).first->second;
}

// Shared buffers whose names start with one of `prefixes` given back (collective while the ranks run; after the others
// have left - rank 0 alone, WORLD 1 - it meets no one). KEEP: kept in the pool for the next shared() of about their size,
// where every rank has a quarter of its card free with them held - else, and with FREE, every segment left holding
// nothing named given back (every rank's imports of its peers' closed, then - after a barrier, so no peer still maps it
// - its own freed), the pool's buffers in those with it. FINAL: the ranks part at once - none closes a peer's segment
// (the others exit; the driver frees theirs), rank 0 frees its own. The name is not asked for again.
enum ReleaseMode { KEEP, FREE, FINAL };
inline void release(const std::vector<std::string>& prefixes, ReleaseMode mode = KEEP) {
  auto& m = sharedBuffers();
  auto& pl = sharedPool();
  auto& sg = segments();
  auto match = [&](const std::string& k) { for (auto& p : prefixes) if (!k.compare(0, p.size(), p)) return true; return false; };
  bool any = false; for (auto& [k, s] : m) any = any || match(k);
  if (WORLD > 1) { CK(cudaStreamSynchronize(STREAM)); barrier(); }
  else if (any) CK(cudaStreamSynchronize(STREAM));
  for (auto it = m.begin(); it != m.end();) {
    if (match(it->first)) { --sg[it->second.seg].live; pl.push_back(it->second); it = m.erase(it); } else ++it;
  }
  if (mode == FINAL) {
    if (RANK == 0) for (auto& g : sg) if (!g.dead && (m.empty() || !g.live)) { CK(cudaFree(g.local)); g.dead = true; }
    pl.clear();
    return;
  }
  if (mode == KEEP && WORLD > 1) {
    size_t f = 0, t = 0; CK(cudaMemGetInfo(&f, &t));
    if (everyRankHas(t / 4)) return;
  }
  freeIdle();
}
// The fences, on the DEVICE: each rank's stream writes its arrival (a generation) into every rank's flags and waits
// until every rank's has arrived in its own - so what each rank's stream did before its fence is done and visible to
// the others when any of them moves past it. A host fence (the stream drained, then a host barrier) leaves every GPU
// idle while the slowest rank and the host catch up: ~5,000 such gaps in a 2,964-token trunk pass, 12% of one rank's
// time. The flag buffer is a shared buffer like any other; the abort flag (/dev/shm) is mapped in, so a rank that
// failed releases the others' waits (the watchdog ends them).
// The generation is counted ON THE DEVICE (gen, this rank's own), so a fence captured in a CUDA graph is a new fence at
// every replay - every rank replays the same sequence, so their counts stay together.
struct DevFlags { int* peer[MAX_RANKS]; int* gen; int world, rank; const volatile int* abort; };
__global__ void arriveAndWaitK(DevFlags f) {
  const int t = threadIdx.x;
  __shared__ int gs;
  if (t == 0) gs = *f.gen + 1;
  __syncthreads();
  const int gen = gs;
  __threadfence_system();
  if (t < f.world) ((volatile int*)f.peer[t])[f.rank] = gen;
  __threadfence_system();
  if (t < f.world) {
    const volatile int* mine = f.peer[f.rank];
    while (mine[t] < gen && !*f.abort) __nanosleep(64);
  }
  __syncthreads();
  if (t == 0) *f.gen = gen;
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
    CK(cudaMalloc(&f.gen, sizeof(int))); CK(cudaMemset(f.gen, 0, sizeof(int))); CK(cudaDeviceSynchronize());
    f.world = WORLD; f.rank = RANK; f.abort = (const volatile int*)ab;
    barrier();                           // (every rank's flags zeroed before any rank arrives)
    made = true;
  }
  return f;
}
inline void hostFence();
// (DIAG LOCALFOLD_MG_FENCE_TIMES=1: every rank's time inside its fences, reported at each tick - not under a capture)
inline std::vector<std::pair<cudaEvent_t, cudaEvent_t>> FENCE_EV;
inline void fenceByStage(const char* name);
inline void fence() {
  if (WORLD == 1 || SIM) return;
  if (SHARED_DEVICE) { hostFence(); return; }
  static const bool ft = getenv("LOCALFOLD_MG_FENCE_TIMES") != nullptr;
  if (ft && STAGES) STAGE_HOOK = fenceByStage;
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  if (ft) CK(cudaStreamIsCapturing(STREAM, &cap));
  cudaEvent_t a = nullptr, b = nullptr;
  if (ft && cap == cudaStreamCaptureStatusNone) { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); CK(cudaEventRecord(a, STREAM)); }
  arriveAndWaitK<<<1, 32, 0, STREAM>>>(devFlags());
  CK(cudaGetLastError());
  if (a) { CK(cudaEventRecord(b, STREAM)); FENCE_EV.push_back({ a, b }); }
}
inline std::map<std::string, std::pair<double, int>> FENCE_BY_STAGE;
inline void fenceByStage(const char* name) {
  double tot = 0;
  for (auto& [a, b] : FENCE_EV) { float m = 0; CK(cudaEventElapsedTime(&m, a, b)); tot += m; CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b)); }
  auto& e = FENCE_BY_STAGE[name]; e.first += tot; e.second += (int)FENCE_EV.size();
  FENCE_EV.clear();
}
inline void fenceTimes(const char* what) {
  for (auto& [k, v] : FENCE_BY_STAGE)
    fprintf(stderr, "  rank %d stage %-22s %5d fences %8.1f ms waiting of %8.1f -> working %8.1f\n", RANK, k.c_str(), v.second, v.first,
            STAGE_MS[k], STAGE_MS[k] - v.first);
  FENCE_BY_STAGE.clear();
  if (FENCE_EV.empty()) return;
  CK(cudaDeviceSynchronize());
  double tot = 0, mx = 0;
  for (auto& [a, b] : FENCE_EV) { float m = 0; CK(cudaEventElapsedTime(&m, a, b)); tot += m; mx = std::max(mx, (double)m); CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b)); }
  fprintf(stderr, "  rank %d %-24s %5zu fences, %8.1f ms waiting (longest %.2f)\n", RANK, what, FENCE_EV.size(), tot, mx);
  FENCE_EV.clear();
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
  segmentReport(what);
  fenceTimes(what);
  last = now;
}

// Every rank's own part [off, off + bytes) of its shared buffer `s` written into every other rank's (collective; each
// rank its own part). PUSHED by a kernel: reading another GPU's memory through a PCIe host bridge - a copy engine's
// peer copy too - ran at a thirtieth of writing it (sharded.cuh's triangle). The fence before and after: every part
// written, and no rank rewrites its part while another still reads it.
struct PushPeers { uint4* p[MAX_RANKS]; int world, rank; };
__global__ void pushPartK(PushPeers pp, size_t off16, size_t n16) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n16 * (pp.world - 1)) return;
  const size_t at = off16 + t % n16;
  pp.p[(pp.rank + 1 + (int)(t / n16)) % pp.world][at] = pp.p[pp.rank][at];
}
inline void allGather(const Shared& s, size_t off, size_t bytes) {
  if (off % 16 || bytes % 16) { fprintf(stderr, "mg::allGather: a part of %zu bytes at %zu, not on 16 bytes\n", bytes, off); exit(1); }
  fence();
  PushPeers pp{}; pp.world = WORLD; pp.rank = RANK;
  for (int r = 0; r < WORLD; ++r) pp.p[r] = (uint4*)s.peer[r];
  const size_t n16 = bytes / 16;
  if (n16 && WORLD > 1) pushPartK<<<(unsigned)((n16 * (WORLD - 1) + 255) / 256), 256, 0, STREAM>>>(pp, off / 16, n16);
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
