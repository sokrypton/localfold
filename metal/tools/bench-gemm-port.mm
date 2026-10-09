#include "../runtime/lfcuda.h"
namespace lf {
extern const KernelInfo PORT_KERNELS[] = {{"none", "none", 0, 2}};
extern const int PORT_KERNEL_COUNT = 1;
extern const char* PORT_NAME = "bench";
extern const char* PORT_SOURCE = "#include <metal_stdlib>\n";
}
