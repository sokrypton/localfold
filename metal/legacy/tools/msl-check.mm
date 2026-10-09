// Compiles a Metal source with the runtime compiler and prints its diagnostics: the translator's test loop.
//   msl-check <file.metal> [include-dir] [instantiation-lines-file]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
static std::string slurp(const std::string& p) { std::ifstream f(p); std::stringstream s; s << f.rdbuf(); return s.str(); }
int main(int argc, char** argv) { @autoreleasepool {
  std::string src = slurp(argv[1]);
  if (argc > 2) {          // (the runtime compiler has no include path: inline the prelude)
    std::string pre = slurp(std::string(argv[2]) + "/prelude.metal");
    size_t at = src.find("#include \"prelude.metal\"");
    if (at != std::string::npos) src.replace(at, 24, pre);
  }
  if (argc > 3) src += slurp(argv[3]);
  { std::ofstream dump(std::string(argv[1]) + ".combined"); dump << src; }
  id<MTLDevice> dev = MTLCopyAllDevices()[0];
  MTLCompileOptions* o = [MTLCompileOptions new]; o.languageVersion = MTLLanguageVersion3_0; o.fastMathEnabled = NO;
  NSError* e = nil;
  auto t0 = [NSDate date];
  id<MTLLibrary> lib = [dev newLibraryWithSource:@(src.c_str()) options:o error:&e];
  double s = -[t0 timeIntervalSinceNow];
  if (e) printf("%s\n", e.localizedDescription.UTF8String);
  printf("%s in %.2f s, %lu functions\n", lib ? "compiled" : "FAILED", s, (unsigned long)lib.functionNames.count);
  return lib ? 0 : 1;
}}
