// --profile: CUPTI's activity API records every kernel's start and end on the device; a table
// of GPU time by kernel name (and the share of the wall the device was busy) for the warm fold.
// CUPTI is LOADED when --profile asks for it (dlopen), never linked: a fold needs no CUPTI, and a prebuilt binary
// (the wheel) must not depend on a library only the profiler reads.
#pragma once
#include <cupti.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <memory>
#include <string>

namespace prof {
inline CUptiResult (*getNextRecord)(uint8_t*, size_t, CUpti_Activity**) = nullptr;
inline CUptiResult (*registerCallbacks)(CUpti_BuffersCallbackRequestFunc, CUpti_BuffersCallbackCompleteFunc) = nullptr;
inline CUptiResult (*enable)(CUpti_ActivityKind) = nullptr;
inline CUptiResult (*flushAll)(uint32_t) = nullptr;
inline std::map<std::string, std::pair<double, int>> byName;   // name -> (ns, calls)
inline uint64_t first = UINT64_MAX, last = 0; inline double busy = 0;
inline bool on = false;
inline void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxRecords) {
  *size = 8 << 20; *buffer = (uint8_t*)aligned_alloc(8, *size); *maxRecords = 0;
}
inline std::string shortName(const char* mangled) {
  int status = 0;
  std::unique_ptr<char, void (*)(void*)> d(abi::__cxa_demangle(mangled, nullptr, nullptr, &status), std::free);
  std::string n = status == 0 ? d.get() : mangled;
  size_t p = n.find('(');            // drop the argument list; keep template arguments
  return p == std::string::npos ? n : n.substr(0, p);
}
inline void CUPTIAPI bufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t validSize) {
  CUpti_Activity* record = nullptr;
  while (getNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
    if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      auto* k = (CUpti_ActivityKernel5*)record;
      if (!on) continue;
      auto& e = byName[shortName(k->name)];
      e.first += (double)(k->end - k->start); e.second += 1;
      busy += (double)(k->end - k->start);
      first = std::min<uint64_t>(first, k->start); last = std::max<uint64_t>(last, k->end);
    }
  }
  free(buffer);
}
inline void init() {
  // the CUPTI of the toolkit this was compiled with (12 here, 13 on Colab): its activity records are read through that
  // toolkit's own struct layouts, which another major version need not keep
  const std::string name = "libcupti.so." + std::to_string(CUDART_VERSION / 1000);
  void* lib = dlopen(name.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!lib) lib = dlopen("libcupti.so", RTLD_NOW | RTLD_GLOBAL);
  if (!lib) { fprintf(stderr, "--profile needs CUPTI (%s, the CUDA toolkit's extras/CUPTI/lib64): %s\n", name.c_str(), dlerror()); exit(1); }
  getNextRecord = (decltype(getNextRecord))dlsym(lib, "cuptiActivityGetNextRecord");
  registerCallbacks = (decltype(registerCallbacks))dlsym(lib, "cuptiActivityRegisterCallbacks");
  enable = (decltype(enable))dlsym(lib, "cuptiActivityEnable");
  flushAll = (decltype(flushAll))dlsym(lib, "cuptiActivityFlushAll");
  if (!getNextRecord || !registerCallbacks || !enable || !flushAll) { fprintf(stderr, "--profile: this CUPTI lacks the activity API\n"); exit(1); }
  registerCallbacks(bufferRequested, bufferCompleted);
  enable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
}
inline void start() { flushAll(0); byName.clear(); busy = 0; first = UINT64_MAX; last = 0; on = true; }
inline void stop(int top = 25) {
  CK(cudaDeviceSynchronize());
  flushAll(0);
  on = false;
  std::vector<std::pair<std::string, std::pair<double, int>>> rows(byName.begin(), byName.end());
  std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.second.first > b.second.first; });
  double span = last > first ? (double)(last - first) : 1;
  printf("GPU kernel time %.1f ms over a %.1f ms span (%.0f%% busy):\n", busy / 1e6, span / 1e6, 100 * busy / span);
  for (int i = 0; i < (int)rows.size() && i < top; ++i)
    printf("  %8.2f ms %5.1f%% %7d  %s\n", rows[i].second.first / 1e6, 100 * rows[i].second.first / busy,
           rows[i].second.second, rows[i].first.substr(0, 90).c_str());
}
}  // namespace prof
