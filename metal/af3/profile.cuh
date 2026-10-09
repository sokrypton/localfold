// The Metal build's profiler: cuda/af3/src/profile.cuh's interface (prof::init, start, stop) over the Metal
// runtime's per-kernel GPU times (lfcuda.mm, which times each dispatch when profiling is on), in place of CUPTI.
#pragma once
#include <string>
namespace lf { void profileStart(); void profileStop(int top); }
namespace prof {
inline bool on = false;
inline void init() {}
inline void start() { on = true; lf::profileStart(); }
inline void stop(int top = 25) { on = false; lf::profileStop(top); }
}  // namespace prof
