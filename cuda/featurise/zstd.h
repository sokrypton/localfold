// Zstandard decompression, in memory, through the system's libzstd.
//
// 🔴 `dlopen`, NOT A LINK-TIME DEPENDENCY. `cuda/af3/src/common.cuh` already reads af3-any-model's
// published blobs this way and for the same reason: libzstd is on every machine that matters and
// on none of their build images, so linking it would make the featuriser unbuildable where it is
// merely unused. A caller that never meets a compressed file never loads the library.
//
// 🔴 AND THE FOUR MAGIC BYTES ARE THE TEST, NOT THE FILE EXTENSION. The page's examples are
// `.cif.zst`, but a reader that trusted the name would refuse a decompressed copy under its
// original name and accept a plain file somebody renamed.
#pragma once
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace lf {
namespace zstd {

struct In { const void* src; size_t size, pos; };
struct Out { void* dst; size_t size, pos; };

inline bool isCompressed(const std::string& bytes) {
  return bytes.size() > 3 && (unsigned char)bytes[0] == 0x28 && (unsigned char)bytes[1] == 0xb5 &&
         (unsigned char)bytes[2] == 0x2f && (unsigned char)bytes[3] == 0xfd;
}

/**
 * The whole stream, decompressed into a string.
 *
 * 🔴 STREAMING RATHER THAN `ZSTD_getFrameContentSize`, because a frame need not declare its
 * size - a stream written without one reports UNKNOWN and a reader that trusted the field would
 * allocate nothing and silently return an empty structure.
 */
inline std::string decompress(const std::string& in, const char* what = "input") {
  // 🔴 THE PROCESS ITSELF FIRST, THEN THE SYSTEM, AND BOTH PLATFORMS' NAMES. macOS ships no libzstd at all and
  // has no `.so` even where one is installed, so `libzstd.so.1` alone made every compressed structure
  // unreadable on a Mac - which is a refusal in the middle of a job rather than at its build. A port that
  // statically links the decompressor (metal/build.sh does, for exactly that reason) exports these symbols into
  // its own image, and `RTLD_DEFAULT` finds them there; a Mac with Homebrew's zstd has `libzstd.1.dylib`; Linux
  // keeps `libzstd.so.1`. The comment above still holds for the Linux case: nothing is linked at build time.
  void* lib = dlsym(RTLD_DEFAULT, "ZSTD_createDStream") ? RTLD_DEFAULT : nullptr;
  for (const char* name : {"libzstd.so.1", "libzstd.so", "libzstd.1.dylib", "libzstd.dylib"}) {
    if (lib) break;
    lib = dlopen(name, RTLD_NOW);
  }
  if (!lib) {
    fprintf(stderr, "%s is zstd and this needs libzstd (libzstd.so.1 / libzstd.1.dylib), or a port that links "
                    "the decompressor: %s\n", what, dlerror());
    exit(1);
  }
  auto sym = [&](const char* n) {
    void* f = dlsym(lib, n);
    if (!f) { fprintf(stderr, "libzstd has no %s\n", n); exit(1); }
    return f;
  };
  auto create = (void* (*)())sym("ZSTD_createDStream");
  auto init = (size_t(*)(void*))sym("ZSTD_initDStream");
  auto step = (size_t(*)(void*, Out*, In*))sym("ZSTD_decompressStream");
  auto isError = (unsigned (*)(size_t))sym("ZSTD_isError");
  auto errName = (const char* (*)(size_t))sym("ZSTD_getErrorName");
  auto release = (size_t(*)(void*))sym("ZSTD_freeDStream");

  void* ds = create();
  init(ds);
  std::string out;
  std::vector<char> buf(1 << 20);
  In zi{in.data(), in.size(), 0};
  while (zi.pos < zi.size) {
    Out zo{buf.data(), buf.size(), 0};
    const size_t rc = step(ds, &zo, &zi);
    if (isError(rc)) {
      fprintf(stderr, "%s: zstd: %s\n", what, errName(rc));
      exit(1);
    }
    out.append(buf.data(), zo.pos);
    // A frame that consumed nothing and produced nothing cannot make progress; without this a
    // truncated stream spins rather than saying so.
    if (zo.pos == 0 && rc == 0) break;
  }
  release(ds);
  return out;
}

/** The file's text, decompressed if it is a zstd frame and returned as it is if it is not. */
inline std::string readTextOrCompressed(const std::string& path, bool* ok = nullptr) {
  std::string text;
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    if (ok) { *ok = false; return text; }
    fprintf(stderr, "no %s\n", path.c_str());
    exit(1);
  }
  char buf[65536];
  size_t got;
  while ((got = fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, got);
  fclose(f);
  if (ok) *ok = true;
  return isCompressed(text) ? decompress(text, path.c_str()) : text;
}

}  // namespace zstd
}  // namespace lf
