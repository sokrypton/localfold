// The published weights, fetched natively - what cuda/fetch_bundles.py did, so a fold needs no Python either.
//
//   lf::fetch::bundle(root, "af3")      a registry bundle (shared/bundles/manifests/index.js, generated into
//                                       bundles.inc): its manifest.json and every shard it names, into root/<directory>
//   lf::fetch::blob(root, "boltz2")     af3-any-model's own int8 blob (huggingface.co/sokrypton/af3-any-model, pinned
//                                       below) into root/af3am-<name>/ - the weights cuda/af3 folds the AF3 lineage with
//
// Each prints "  <name>: <file> (k/n)" as a piece lands and "<name> -> <dir>" when whole (cuda/worker.py turns the
// first into the page's download status). 🔴 THE MANIFEST LAST: a bundle is "here" when its manifest.json is, so
// it is written only once every shard it names is, and every file lands through a `.part` renamed when whole - an
// interrupted download leaves nothing that reads as finished. One fetch a bundle at a time (flock), since the
// notebook prefetches the default model while a fold may already be asking for it.
//
// AlphaFold 3's blob is Google DeepMind's, for academic non-commercial use: it is fetched only once its terms are
// accepted - LOCALFOLD_ACCEPT_MODEL_TERMS naming alphafold3, or a yes at a terminal's prompt.
//
// 🔴 NO LIBRARY, AS search.h: curl runs as a child, eight at a time (one connection to Hugging Face is ~21 MB/s
// from here, eight 112 - AF3's 277 MB in 2.5 s against 13.5).
#pragma once
#include <fcntl.h>
#include <strings.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ccd.h"
#include "json.h"
#include "search.h"

namespace lf::fetch {

struct BundleEntry { std::string key, directory, remote; };
#include "bundles.inc"

// af3-any-model's int8 blobs, at the commit that added AlphaFold 3's (after chai1's structure token-pair weights)
inline const std::string AF3_ANY_MODEL = "https://huggingface.co/sokrypton/af3-any-model/resolve/98f787eacd6e49cdba88e044622773a8702f03fe/";
inline const std::vector<std::pair<std::string, std::string>>& blobTable() {
  static const std::vector<std::pair<std::string, std::string>> t = {
    {"af3", "alphafold3/af3.int8.bin.zst"},     // (Google DeepMind's: academic, non-commercial use, terms first)
    {"chai1", "chai1/chai1.int8.bin.zst"},
    {"boltz2", "boltz2/boltz2.int8.bin.zst"},
    {"protenix2", "protenix/protenix2.int8.bin.zst"},
    {"intellifold2", "intellifold2/intellifold2.int8.bin.zst"},
    {"rosettafold3", "rosettafold3/rosettafold3.int8.bin.zst"},
    {"openbind0", "openfold3/openbind0.int8.bin.zst"},
    {"opendde", "opendde/opendde.int8.bin.zst"},
    {"esm2", "lm/esm2.bin.zst"},                // (chai-1's ESM2 3B: int8 as published, its only form)
  };
  return t;
}

inline const char* AF3_TERMS =
    "AlphaFold 3's parameters are Google DeepMind's, for academic, non-commercial use only, under the\n"
    "AlphaFold 3 Model Parameters Terms of Use and Prohibited Use Policy:\n"
    "  https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md\n"
    "  https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_PROHIBITED_USE_POLICY.md\n";

inline bool exists(const std::string& path) { struct stat st; return stat(path.c_str(), &st) == 0; }
inline long long fileSize(const std::string& path) { struct stat st; return stat(path.c_str(), &st) == 0 ? (long long)st.st_size : -1; }
inline void makeDirs(const std::string& dir) {
  if (system(("mkdir -p " + search::shellQuote(dir)).c_str()) != 0) throw std::runtime_error("cannot create " + dir);
}

// Where the weights live: LOCALFOLD_HOME when set; else the checkout this binary was built in (it sits at
// <checkout>/cuda/<port>/<port> or <checkout>/cuda/featurise/<tool>), where every wrapper and the worker keep them;
// else ~/.cache/localfold for a binary copied out of its checkout
inline std::string home() {
  if (const char* h = std::getenv("LOCALFOLD_HOME"); h && *h) return h;
  char self[4096];
  ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
  if (n > 0) {
    std::string exe(self, (size_t)n);
    std::string dir = exe.substr(0, exe.rfind('/'));          // <checkout>/cuda/<port>
    std::string cuda = dir.substr(0, dir.rfind('/'));         // <checkout>/cuda
    if (cuda.size() > 5 && cuda.substr(cuda.size() - 5) == "/cuda" && exists(cuda + "/featurise"))
      return cuda.substr(0, cuda.size() - 5);
  }
  const char* user = std::getenv("HOME");
  if (!user || !*user) throw std::runtime_error("no LOCALFOLD_HOME and no HOME to keep the weights under");
  return std::string(user) + "/.cache/localfold";
}

struct Lock {          // one fetch of a directory at a time
  int fd;
  explicit Lock(const std::string& dir) {
    fd = open((dir + "/.fetch.lock").c_str(), O_WRONLY | O_CREAT, 0644);
    if (fd < 0 || flock(fd, LOCK_EX) != 0) throw std::runtime_error("cannot lock " + dir);
  }
  ~Lock() { if (fd >= 0) close(fd); }
};

// one URL to one file through a `.part` (a byte range of it when `range` is given)
inline void download(const std::string& url, const std::string& path, const std::string& range = "") {
  std::string part = path + ".part";
  std::string command = "curl -s -f -L --retry 3 --max-time 3600" + (range.empty() ? "" : " -r " + range) +
                        " -o " + search::shellQuote(part) + " " + search::shellQuote(url);
  if (system(command.c_str()) != 0) { std::remove(part.c_str()); throw std::runtime_error("could not download " + url); }
  if (rename(part.c_str(), path.c_str()) != 0) throw std::runtime_error("cannot rename " + part);
}

// a file's size from its headers, after any redirect (the last Content-Length curl saw)
inline long long remoteSize(const std::string& url) {
  int status = 0;
  std::string headers = search::runCapture("curl -s -f -L -I --retry 3 " + search::shellQuote(url), status);
  if (status != 0) throw std::runtime_error("could not reach " + url);
  long long size = -1;
  size_t at = 0;
  while ((at = headers.find('\n', at)) != std::string::npos) {
    ++at;
    if (headers.size() - at > 15 && strncasecmp(headers.c_str() + at, "content-length:", 15) == 0) size = atoll(headers.c_str() + at + 15);
  }
  if (size < 0) throw std::runtime_error(url + " gave no Content-Length");
  return size;
}

// `jobs` run on up to eight threads, each said as it finishes
inline void inParallel(const std::vector<std::function<void()>>& jobs, const std::function<void(size_t done)>& said) {
  std::atomic<size_t> next{0}, done{0};
  std::mutex m;
  std::string failure;
  std::vector<std::thread> pool;
  for (int w = 0; w < 8; ++w)
    pool.emplace_back([&] {
      for (size_t k; (k = next++) < jobs.size();) {
        try { jobs[k](); } catch (const std::exception& e) { std::lock_guard<std::mutex> g(m); if (failure.empty()) failure = e.what(); continue; }
        std::lock_guard<std::mutex> g(m);
        said(++done);
      }
    });
  for (auto& t : pool) t.join();
  if (!failure.empty()) throw std::runtime_error(failure);
}

inline bool acceptedAf3Terms() {
  if (const char* named = std::getenv("LOCALFOLD_ACCEPT_MODEL_TERMS")) {
    std::string list = std::string(",") + named + ",", padded;
    for (char c : list) if (c != ' ' && c != '\t') padded += c;
    if (padded.find(",alphafold3,") != std::string::npos) return true;
  }
  std::cerr << AF3_TERMS;
  if (!isatty(0)) return false;
  std::cerr << "Do you accept these terms? [y/N] " << std::flush;
  std::string answer;
  std::getline(std::cin, answer);
  answer = trimWs(answer);
  return answer == "y" || answer == "Y" || answer == "yes" || answer == "Yes";
}

inline std::string blob(const std::string& root, const std::string& name) {
  std::string file;
  for (auto& [k, v] : blobTable()) if (k == name) file = v;
  if (file.empty()) throw std::runtime_error("af3-any-model has no blob named " + name);
  std::string dest = root + "/af3am-" + name, path = dest + "/" + file.substr(file.rfind('/') + 1);
  if (exists(path)) return dest;
  if (name == "af3" && !acceptedAf3Terms())
    throw std::runtime_error("AlphaFold 3's parameters need its terms accepted first: set LOCALFOLD_ACCEPT_MODEL_TERMS=alphafold3");
  makeDirs(dest);
  Lock lock(dest);
  if (exists(path)) return dest;
  std::string url = AF3_ANY_MODEL + file;
  long long size = remoteSize(url);
  const int parts = 8;
  long long step = (size + parts - 1) / parts;
  std::vector<std::function<void()>> jobs;
  int count = 0;
  for (int k = 0; k < parts && k * step < size; ++k, ++count) {
    long long lo = k * step, hi = std::min(size, (k + 1) * step) - 1;
    std::string piece = path + ".part" + std::to_string(k);
    jobs.push_back([=] {
      download(url, piece, std::to_string(lo) + "-" + std::to_string(hi));
      if (fileSize(piece) != hi - lo + 1) throw std::runtime_error(url + ": range " + std::to_string(k) + " came back short");
    });
  }
  std::string base = path.substr(path.rfind('/') + 1);
  inParallel(jobs, [&](size_t done) { printf("  %s: %s (%zu/%d)\n", name.c_str(), base.c_str(), done, count); fflush(stdout); });
  {
    std::string joined = path + ".part";
    FILE* out = fopen(joined.c_str(), "wb");
    if (!out) throw std::runtime_error("cannot write " + joined);
    std::vector<char> buf(1 << 24);
    for (int k = 0; k < count; ++k) {
      std::string piece = path + ".part" + std::to_string(k);
      FILE* in = fopen(piece.c_str(), "rb");
      if (!in) throw std::runtime_error("cannot read " + piece);
      for (size_t n; (n = fread(buf.data(), 1, buf.size(), in)) > 0;) fwrite(buf.data(), 1, n, out);
      fclose(in);
      std::remove(piece.c_str());
    }
    fclose(out);
    if (fileSize(joined) != size) throw std::runtime_error(url + ": " + std::to_string(fileSize(joined)) + " bytes, not " + std::to_string(size));
    if (rename(joined.c_str(), path.c_str()) != 0) throw std::runtime_error("cannot rename " + joined);
  }
  printf("%s -> %s\n", name.c_str(), dest.c_str()); fflush(stdout);
  return dest;
}

inline std::string bundle(const std::string& root, const std::string& key) {
  const BundleEntry* entry = nullptr;
  for (auto& b : bundleTable()) if (b.key == key) entry = &b;
  if (!entry) throw std::runtime_error("no bundle " + key + " in shared/bundles/manifests/index.js");
  std::string dest = root + "/" + entry->directory, manifest = dest + "/manifest.json";
  if (exists(manifest)) return dest;
  if (entry->remote.empty()) throw std::runtime_error(key + " has no published remote");
  makeDirs(dest);
  Lock lock(dest);
  if (exists(manifest)) return dest;
  download(entry->remote + "manifest.json", manifest + ".fetching");
  Json m = parseJson(readFile(manifest + ".fetching"));
  std::set<std::string> files;
  if (const Json* tensors = m.get("tensors"))
    for (auto& [name, record] : tensors->o) if (const Json* f = record.get("file")) files.insert(f->s);
  std::vector<std::function<void()>> jobs;
  size_t present = 0;
  for (auto& f : files) {
    if (exists(dest + "/" + f)) { ++present; continue; }
    jobs.push_back([=] { download(entry->remote + f, dest + "/" + f); });
  }
  inParallel(jobs, [&](size_t done) { printf("  %s: shard (%zu/%zu)\n", key.c_str(), present + done, files.size()); fflush(stdout); });
  if (rename((manifest + ".fetching").c_str(), manifest.c_str()) != 0) throw std::runtime_error("cannot rename " + manifest);
  printf("%s -> %s\n", key.c_str(), dest.c_str()); fflush(stdout);
  return dest;
}

}  // namespace lf::fetch
