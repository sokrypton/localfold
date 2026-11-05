// Data-parallel training for train-of3 (OpenFold3's DDP with per-sample clipping, core/utils/grad_manager.py): each
// rank clips and accumulates its own examples' gradients; at a step the ranks' accumulators AND their participation
// counts are summed (_sync_and_average_grads), so every rank divides by the same global count and takes the same Adam
// step from the same weights - the weights stay identical without being exchanged.
//
//   NcclComm   one GPU a rank (device rank % devices), NCCL's all-reduce; the unique id passed through --rendezvous
//   FileComm   any placement, ranks sharing a GPU included: each rank's buffer through files in --rendezvous (a tmpfs
//              such as /dev/shm), summed in rank order by every rank so every rank's sum is the same bits. For testing
//              the protocol on one GPU (NCCL refuses two ranks on one device), not for speed.
#pragma once
#include <nccl.h>
#include <chrono>
#include <fstream>
#include <thread>
#include <unistd.h>
#include "core.cuh"

namespace grad {

#define NK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { fprintf(stderr, "NCCL %s at %s:%d\n", ncclGetErrorString(r_), __FILE__, __LINE__); exit(1); } } while (0)

struct Comm {
  int rank = 0, world = 1;
  virtual ~Comm() {}
  // every rank's `n` reals summed into each rank's `dev`, in place
  virtual void sumReals(real* dev, size_t n) { (void)dev; (void)n; }
  // every rank's integers summed, in place
  virtual void sumInts(std::vector<int>& h) { (void)h; }
};

static inline bool waitFile(const std::string& path, double seconds = 3600) {
  const auto t0 = std::chrono::steady_clock::now();
  while (access(path.c_str(), F_OK) != 0) {
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > seconds) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return true;
}

struct NcclComm : Comm {
  ncclComm_t comm{};
  int* ints = nullptr; size_t intCap = 0;
  NcclComm(int r, int w, const std::string& dir) {
    rank = r; world = w;
    ncclUniqueId id;
    const std::string path = dir + "/nccl.id";
    if (rank == 0) {
      NK(ncclGetUniqueId(&id));
      { std::ofstream f(path + ".tmp", std::ios::binary); f.write((const char*)&id, sizeof(id)); }
      std::rename((path + ".tmp").c_str(), path.c_str());
    } else {
      if (!waitFile(path, 600)) { fprintf(stderr, "rank %d: no %s from rank 0\n", rank, path.c_str()); exit(1); }
      std::ifstream f(path, std::ios::binary); f.read((char*)&id, sizeof(id));
    }
    NK(ncclCommInitRank(&comm, world, id, rank));
  }
  void sumReals(real* dev, size_t n) override {
    NK(ncclAllReduce(dev, dev, n, sizeof(real) == 8 ? ncclDouble : ncclFloat, ncclSum, comm, STREAM));
  }
  void sumInts(std::vector<int>& h) override {
    if (h.size() > intCap) { if (ints) GK(cudaFree(ints)); intCap = h.size(); GK(cudaMalloc(&ints, intCap * sizeof(int))); }
    GK(cudaMemcpyAsync(ints, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice, STREAM));
    NK(ncclAllReduce(ints, ints, h.size(), ncclInt32, ncclSum, comm, STREAM));
    GK(cudaMemcpyAsync(h.data(), ints, h.size() * sizeof(int), cudaMemcpyDeviceToHost, STREAM));
    GK(cudaStreamSynchronize(STREAM));
  }
};

struct FileComm : Comm {
  std::string dir;
  long seq = 0;
  FileComm(int r, int w, const std::string& d) : dir(d) { rank = r; world = w; }
  std::string name(long k, int r, const char* what) const { return dir + "/" + std::to_string(k) + "." + std::to_string(r) + "." + what; }
  // every rank's bytes, in rank order (the exchange itself: write, wait for all, read all, then retire)
  template <class T> std::vector<std::vector<T>> exchange(const std::vector<T>& mine) {
    const long k = seq++;
    {
      const std::string p = name(k, rank, "bin");
      std::ofstream f(p + ".tmp", std::ios::binary); f.write((const char*)mine.data(), mine.size() * sizeof(T)); f.close();
      std::rename((p + ".tmp").c_str(), p.c_str());
    }
    std::vector<std::vector<T>> all(world);
    for (int r = 0; r < world; ++r) {
      const std::string p = name(k, r, "bin");
      if (!waitFile(p)) { fprintf(stderr, "rank %d: rank %d never wrote %s\n", rank, r, p.c_str()); exit(1); }
      all[r].resize(mine.size());
      std::ifstream f(p, std::ios::binary); f.read((char*)all[r].data(), mine.size() * sizeof(T));
      if (!f) { fprintf(stderr, "rank %d: %s is short\n", rank, p.c_str()); exit(1); }
    }
    // (every rank read every file: then each retires its own)
    { std::ofstream f(name(k, rank, "done")); }
    for (int r = 0; r < world; ++r) waitFile(name(k, r, "done"));
    std::remove(name(k, rank, "bin").c_str());
    if (k > 0) std::remove(name(k - 1, rank, "done").c_str());
    return all;
  }
  void sumReals(real* dev, size_t n) override {
    std::vector<real> h(n);
    GK(cudaMemcpyAsync(h.data(), dev, n * sizeof(real), cudaMemcpyDeviceToHost, STREAM));
    GK(cudaStreamSynchronize(STREAM));
    auto all = exchange(h);
    for (size_t i = 0; i < n; ++i) { real s = 0; for (int r = 0; r < world; ++r) s += all[r][i]; h[i] = s; }
    GK(cudaMemcpy(dev, h.data(), n * sizeof(real), cudaMemcpyHostToDevice));
  }
  void sumInts(std::vector<int>& h) override {
    auto all = exchange(h);
    for (size_t i = 0; i < h.size(); ++i) { int s = 0; for (int r = 0; r < world; ++r) s += all[r][i]; h[i] = s; }
  }
};

}  // namespace grad
