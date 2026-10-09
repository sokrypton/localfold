// The CUDA runtime, cuBLAS and cuBLASLt, as much of them as the CUDA ports (cuda/af3, af2, ef2) use, on Metal.
// metal/tools/cu2metal.py compiles the ports' HOST code against this: a device pointer is a Metal GPU address,
// every allocation one shared MTLBuffer (the host never dereferences a device pointer, in CUDA or here, so the
// GPU's address space is the one the host code passes around), and every stream is the one ordered queue.
// Implemented in lfcuda.mm.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <charconv>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <vector>

// ---------------------------------------------------------------- qualifiers (device code is translated elsewhere)
#define __global__
#define __device__
#define __host__
#define __constant__
#define __shared__
#define __forceinline__ inline
#define __noinline__
#define __launch_bounds__(...)
#define __align__(n) alignas(n)
#ifndef __CUDACC__
#define __CUDACC__ 1
#endif
#define __CUDA_METAL__ 1

// ---------------------------------------------------------------- vector types (CUDA's layouts: float3 is 12 bytes)
#define LF_VEC2(T, N) struct alignas(2 * sizeof(T)) N##2 { T x, y; };
#define LF_VEC3(T, N) struct N##3 { T x, y, z; };
#define LF_VEC4(T, N) struct alignas(4 * sizeof(T) > 16 ? 16 : 4 * sizeof(T)) N##4 { T x, y, z, w; };
LF_VEC2(float, float) LF_VEC3(float, float) LF_VEC4(float, float)
LF_VEC2(int, int) LF_VEC3(int, int) LF_VEC4(int, int)
LF_VEC2(unsigned, uint) LF_VEC3(unsigned, uint) LF_VEC4(unsigned, uint)
LF_VEC2(short, short) LF_VEC4(short, short)
LF_VEC2(unsigned short, ushort) LF_VEC4(unsigned short, ushort)
LF_VEC2(unsigned char, uchar) LF_VEC4(unsigned char, uchar)
LF_VEC2(char, char) LF_VEC4(char, char)
struct alignas(16) double2 { double x, y; };
struct longlong2 { long long x, y; };
inline float2 make_float2(float x, float y) { return {x, y}; }
inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
inline int2 make_int2(int x, int y) { return {x, y}; }
inline int3 make_int3(int x, int y, int z) { return {x, y, z}; }
inline int4 make_int4(int x, int y, int z, int w) { return {x, y, z, w}; }
inline uint2 make_uint2(unsigned x, unsigned y) { return {x, y}; }
inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) { return {x, y, z, w}; }

struct dim3 {
  unsigned x, y, z;
  constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

// ---------------------------------------------------------------- half and bfloat16 on the host
struct __half_raw { unsigned short x; };
struct __half {
  _Float16 v;
  __half() = default;
  constexpr __half(float f) : v((_Float16)f) {}
  __half(double f) : v((_Float16)f) {}
  __half(int i) : v((_Float16)i) {}
  __half(__half_raw r) { std::memcpy(&v, &r.x, 2); }
  operator float() const { return (float)v; }
  operator __half_raw() const { __half_raw r; std::memcpy(&r.x, &v, 2); return r; }
  __half& operator=(float f) { v = (_Float16)f; return *this; }
};
using half = __half;
struct alignas(4) __half2 { __half x, y; };
using half2 = __half2;
inline __half __float2half(float f) { return __half(f); }
inline __half __float2half_rn(float f) { return __half(f); }
inline float __half2float(__half h) { return (float)h.v; }
inline __half2 __floats2half2_rn(float a, float b) { return {__half(a), __half(b)}; }
inline __half2 __float2half2_rn(float a) { return {__half(a), __half(a)}; }
inline float __low2float(__half2 h) { return (float)h.x.v; }
inline float __high2float(__half2 h) { return (float)h.y.v; }
inline unsigned short __half_as_ushort(__half h) { unsigned short u; std::memcpy(&u, &h.v, 2); return u; }
inline __half __ushort_as_half(unsigned short u) { __half h; std::memcpy(&h.v, &u, 2); return h; }

struct __nv_bfloat16_raw { unsigned short x; };
struct __nv_bfloat16 {
  unsigned short x;
  __nv_bfloat16() = default;
  __nv_bfloat16(float f) { x = fromFloat(f); }
  __nv_bfloat16(__nv_bfloat16_raw r) : x(r.x) {}
  static unsigned short fromFloat(float f) {
    uint32_t b; std::memcpy(&b, &f, 4);
    if ((b & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((b >> 16) | 0x40);
    return (unsigned short)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
  }
  operator float() const { uint32_t b = (uint32_t)x << 16; float f; std::memcpy(&f, &b, 4); return f; }
};
struct __nv_bfloat162 { __nv_bfloat16 x, y; };
inline __nv_bfloat16 __float2bfloat16(float f) { return __nv_bfloat16(f); }
inline __nv_bfloat16 __float2bfloat16_rn(float f) { return __nv_bfloat16(f); }
inline float __bfloat162float(__nv_bfloat16 b) { return (float)b; }

// ---------------------------------------------------------------- the runtime API
typedef int cudaError_t;
enum : int { cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorMemoryAllocation = 2, cudaErrorNotReady = 600 };
typedef cudaError_t cudaError;
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2,
                      cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4 };
typedef struct LfStream_* cudaStream_t;
#define cudaStreamPerThread ((cudaStream_t)0x2)
#define cudaStreamLegacy ((cudaStream_t)0x1)
enum { cudaStreamDefault = 0, cudaStreamNonBlocking = 1 };
typedef struct LfEvent_* cudaEvent_t;
enum { cudaEventDefault = 0, cudaEventBlockingSync = 1, cudaEventDisableTiming = 2 };
typedef struct LfGraph_* cudaGraph_t;
typedef struct LfGraph_* cudaGraphExec_t;
typedef struct LfGraphNode_* cudaGraphNode_t;
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone = 0, cudaStreamCaptureStatusActive = 1, cudaStreamCaptureStatusInvalidated = 2 };
enum cudaStreamCaptureMode { cudaStreamCaptureModeGlobal = 0, cudaStreamCaptureModeThreadLocal = 1, cudaStreamCaptureModeRelaxed = 2 };
enum cudaDeviceAttr { cudaDevAttrMaxThreadsPerBlock = 1, cudaDevAttrMultiProcessorCount = 16,
                      cudaDevAttrComputeCapabilityMajor = 75, cudaDevAttrComputeCapabilityMinor = 76,
                      cudaDevAttrMaxSharedMemoryPerBlockOptin = 97, cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
                      cudaDevAttrMaxSharedMemoryPerBlock = 8, cudaDevAttrL2CacheSize = 38 };
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8, cudaFuncAttributePreferredSharedMemoryCarveout = 9 };
typedef struct LfMemPool_* cudaMemPool_t;
enum cudaMemPoolAttr { cudaMemPoolReuseFollowEventDependencies = 1, cudaMemPoolReuseAllowOpportunistic, cudaMemPoolReuseAllowInternalDependencies,
                       cudaMemPoolAttrReleaseThreshold, cudaMemPoolAttrReservedMemCurrent, cudaMemPoolAttrReservedMemHigh,
                       cudaMemPoolAttrUsedMemCurrent, cudaMemPoolAttrUsedMemHigh };
enum cudaMemAllocationType { cudaMemAllocationTypeInvalid = 0, cudaMemAllocationTypePinned = 1 };
enum cudaMemLocationType { cudaMemLocationTypeInvalid = 0, cudaMemLocationTypeDevice = 1 };
enum cudaMemAllocationHandleType { cudaMemHandleTypeNone = 0 };
struct cudaMemLocation { cudaMemLocationType type; int id; };
struct cudaMemPoolProps { cudaMemAllocationType allocType; cudaMemAllocationHandleType handleTypes; cudaMemLocation location;
                          void* win32SecurityAttributes; unsigned char reserved[64]; };
enum { cudaHostAllocDefault = 0, cudaHostAllocPortable = 1, cudaHostAllocMapped = 2, cudaHostAllocWriteCombined = 4 };
enum { cudaMemAttachGlobal = 1 };

cudaError_t cudaMalloc(void** p, size_t bytes);
template <class T> inline cudaError_t cudaMalloc(T** p, size_t bytes) { return cudaMalloc((void**)p, bytes); }
cudaError_t cudaFree(void* p);
cudaError_t cudaMallocAsync(void** p, size_t bytes, cudaStream_t s);
template <class T> inline cudaError_t cudaMallocAsync(T** p, size_t bytes, cudaStream_t s) { return cudaMallocAsync((void**)p, bytes, s); }
cudaError_t cudaMallocFromPoolAsync(void** p, size_t bytes, cudaMemPool_t pool, cudaStream_t s);
template <class T> inline cudaError_t cudaMallocFromPoolAsync(T** p, size_t bytes, cudaMemPool_t pool, cudaStream_t s) {
  return cudaMallocFromPoolAsync((void**)p, bytes, pool, s);
}
cudaError_t cudaFreeAsync(void* p, cudaStream_t s);
cudaError_t cudaMallocHost(void** p, size_t bytes);
template <class T> inline cudaError_t cudaMallocHost(T** p, size_t bytes) { return cudaMallocHost((void**)p, bytes); }
cudaError_t cudaHostAlloc(void** p, size_t bytes, unsigned flags);
template <class T> inline cudaError_t cudaHostAlloc(T** p, size_t bytes, unsigned flags) { return cudaHostAlloc((void**)p, bytes, flags); }
cudaError_t cudaFreeHost(void* p);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t s = nullptr);
cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                              cudaMemcpyKind kind, cudaStream_t s = nullptr);
cudaError_t cudaMemset(void* p, int value, size_t bytes);
cudaError_t cudaMemsetAsync(void* p, int value, size_t bytes, cudaStream_t s = nullptr);
cudaError_t cudaMemset2DAsync(void* p, size_t pitch, int value, size_t width, size_t height, cudaStream_t s = nullptr);
cudaError_t cudaMemGetInfo(size_t* free, size_t* total);
cudaError_t cudaGetDevice(int* d);
cudaError_t cudaSetDevice(int d);
cudaError_t cudaGetDeviceCount(int* n);
cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaDeviceReset();
cudaError_t cudaStreamSynchronize(cudaStream_t s);
cudaError_t cudaStreamCreate(cudaStream_t* s);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned flags);
cudaError_t cudaStreamDestroy(cudaStream_t s);
cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned flags = 0);
cudaError_t cudaStreamQuery(cudaStream_t s);
cudaError_t cudaEventCreate(cudaEvent_t* e);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned flags);
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s = nullptr);
cudaError_t cudaEventSynchronize(cudaEvent_t e);
cudaError_t cudaEventQuery(cudaEvent_t e);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b);
cudaError_t cudaEventDestroy(cudaEvent_t e);
cudaError_t cudaGetLastError();
cudaError_t cudaPeekAtLastError();
const char* cudaGetErrorString(cudaError_t e);
const char* cudaGetErrorName(cudaError_t e);
template <class F> inline cudaError_t cudaFuncSetAttribute(F, cudaFuncAttribute, int) { return cudaSuccess; }
// (an estimate: Apple GPUs hold up to 1024 threads and 64 KB of threadgroup memory a core)
template <class F> inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, F, int threads, size_t smem) {
  int byThreads = threads > 0 ? 1024 / threads : 1, bySmem = smem > 0 ? (int)(65536 / smem) : 64;
  *n = byThreads < bySmem ? byThreads : bySmem; if (*n < 1) *n = 1; return cudaSuccess;
}
cudaError_t cudaStreamBeginCapture(cudaStream_t s, cudaStreamCaptureMode mode);
cudaError_t cudaStreamEndCapture(cudaStream_t s, cudaGraph_t* g);
cudaError_t cudaStreamIsCapturing(cudaStream_t s, cudaStreamCaptureStatus* status);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t g, unsigned long long flags = 0);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t g, cudaGraphNode_t* errNode, char* log, size_t logSize);
cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t s);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec);
cudaError_t cudaGraphDestroy(cudaGraph_t g);
cudaError_t cudaMemPoolCreate(cudaMemPool_t* pool, const cudaMemPoolProps* props);
cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t pool, cudaMemPoolAttr attr, void* value);
cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t pool, cudaMemPoolAttr attr, void* value);
cudaError_t cudaMemPoolTrimTo(cudaMemPool_t pool, size_t keep);
cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* pool, int device);
cudaError_t lfSymbolAddress(void** p, const void* host, size_t bytes);
template <class T> inline cudaError_t cudaGetSymbolAddress(void** p, const T& sym) { return lfSymbolAddress(p, &sym, sizeof(T)); }

// ---------------------------------------------------------------- cuBLAS
typedef struct LfBlas_* cublasHandle_t;
typedef int cublasStatus_t;
enum : int { CUBLAS_STATUS_SUCCESS = 0, CUBLAS_STATUS_NOT_SUPPORTED = 15 };
enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 };
enum cudaDataType { CUDA_R_32F = 0, CUDA_R_64F = 1, CUDA_R_16F = 2, CUDA_R_8I = 3, CUDA_R_32I = 10, CUDA_R_16BF = 14 };
typedef cudaDataType cudaDataType_t;
enum cublasComputeType_t { CUBLAS_COMPUTE_16F = 64, CUBLAS_COMPUTE_32F = 68, CUBLAS_COMPUTE_32F_FAST_16F = 74,
                           CUBLAS_COMPUTE_32F_FAST_16BF = 75, CUBLAS_COMPUTE_32F_FAST_TF32 = 77, CUBLAS_COMPUTE_32F_PEDANTIC = 69 };
enum cublasGemmAlgo_t { CUBLAS_GEMM_DEFAULT = -1, CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99 };
enum cublasMath_t { CUBLAS_DEFAULT_MATH = 0, CUBLAS_TENSOR_OP_MATH = 1, CUBLAS_PEDANTIC_MATH = 2, CUBLAS_TF32_TENSOR_OP_MATH = 3 };
cublasStatus_t cublasCreate(cublasHandle_t* h);
cublasStatus_t cublasDestroy(cublasHandle_t h);
cublasStatus_t cublasSetStream(cublasHandle_t h, cudaStream_t s);
cublasStatus_t cublasSetWorkspace(cublasHandle_t h, void* ws, size_t bytes);
cublasStatus_t cublasSetMathMode(cublasHandle_t h, cublasMath_t mode);
cublasStatus_t cublasSgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                           const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                         const float* alpha, const float* A, int lda, long long sa, const float* B, int ldb,
                                         long long sb, const float* beta, float* C, int ldc, long long sc, int batch);
cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                            const void* alpha, const void* A, cudaDataType ta_, int lda, const void* B, cudaDataType tb_, int ldb,
                            const void* beta, void* C, cudaDataType tc_, int ldc, cublasComputeType_t compute, cublasGemmAlgo_t algo);
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                          const void* alpha, const void* A, cudaDataType ta_, int lda, long long sa,
                                          const void* B, cudaDataType tb_, int ldb, long long sb, const void* beta,
                                          void* C, cudaDataType tc_, int ldc, long long sc, int batch,
                                          cublasComputeType_t compute, cublasGemmAlgo_t algo);
cublasStatus_t cublasGemmBatchedEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                   const void* alpha, const void* const A[], cudaDataType ta_, int lda,
                                   const void* const B[], cudaDataType tb_, int ldb, const void* beta,
                                   void* const C[], cudaDataType tc_, int ldc, int batch,
                                   cublasComputeType_t compute, cublasGemmAlgo_t algo);
cublasStatus_t cublasSscal(cublasHandle_t h, int n, const float* alpha, float* x, int incx);

// ---------------------------------------------------------------- cuBLASLt (the matmuls the ports configure)
typedef struct LfLt_* cublasLtHandle_t;
struct LfLtDesc { int compute = 0, scale = 0, ta = 0, tb = 0, epilogue = 1; const void* bias = nullptr; int biasType = 0; };
struct LfLtLayout { int type = 0; unsigned long long rows = 0, cols = 0; long long ld = 0; int batch = 1; long long stride = 0; int order = 0; };
typedef LfLtDesc* cublasLtMatmulDesc_t;
typedef LfLtLayout* cublasLtMatrixLayout_t;
typedef struct LfLtPref_* cublasLtMatmulPreference_t;
struct cublasLtMatmulAlgo_t { uint64_t data[8]; };
struct cublasLtMatmulHeuristicResult_t { cublasLtMatmulAlgo_t algo; size_t workspaceSize; cublasStatus_t state; float wavesCount; int reserved[4]; };
enum cublasLtMatmulDescAttributes_t { CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = 0, CUBLASLT_MATMUL_DESC_SCALE_TYPE = 1,
                                      CUBLASLT_MATMUL_DESC_POINTER_MODE = 2, CUBLASLT_MATMUL_DESC_TRANSA = 3,
                                      CUBLASLT_MATMUL_DESC_TRANSB = 4, CUBLASLT_MATMUL_DESC_TRANSC = 5,
                                      CUBLASLT_MATMUL_DESC_EPILOGUE = 7, CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8,
                                      CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 26 };
enum cublasLtMatrixLayoutAttribute_t { CUBLASLT_MATRIX_LAYOUT_TYPE = 0, CUBLASLT_MATRIX_LAYOUT_ORDER = 1,
                                       CUBLASLT_MATRIX_LAYOUT_ROWS = 2, CUBLASLT_MATRIX_LAYOUT_COLS = 3,
                                       CUBLASLT_MATRIX_LAYOUT_LD = 4, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5,
                                       CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6 };
enum cublasLtMatmulPreferenceAttributes_t { CUBLASLT_MATMUL_PREF_SEARCH_MODE = 0, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
                                            CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK = 3 };
enum cublasLtMatmulAlgoConfigAttributes_t { CUBLASLT_ALGO_CONFIG_ID = 0, CUBLASLT_ALGO_CONFIG_TILE_ID = 1,
                                            CUBLASLT_ALGO_CONFIG_SPLITK_NUM = 2, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME = 3 };
enum cublasLtEpilogue_t { CUBLASLT_EPILOGUE_DEFAULT = 1, CUBLASLT_EPILOGUE_RELU = 2, CUBLASLT_EPILOGUE_BIAS = 4,
                          CUBLASLT_EPILOGUE_RELU_BIAS = 6, CUBLASLT_EPILOGUE_GELU = 32, CUBLASLT_EPILOGUE_GELU_BIAS = 36 };
enum { CUBLASLT_REDUCTION_SCHEME_NONE = 0, CUBLASLT_MATMUL_TILE_128 = 0, CUBLASLT_MATMUL_TILE_128x128 = 20,
       CUBLASLT_MATMUL_TILE_UNDEFINED = 0, CUBLASLT_MATMUL_TILE_64x64 = 15, CUBLASLT_MATMUL_TILE_128x64 = 19, CUBLASLT_ORDER_COL = 0, CUBLASLT_ORDER_ROW = 1 };
cublasStatus_t cublasLtCreate(cublasLtHandle_t* h);
cublasStatus_t cublasLtDestroy(cublasLtHandle_t h);
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* d, cublasComputeType_t compute, cudaDataType scale);
cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t d);
cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t d, cublasLtMatmulDescAttributes_t a, const void* v, size_t n);
cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* l, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld);
cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t l);
cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t l, cublasLtMatrixLayoutAttribute_t a, const void* v, size_t n);
cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* p);
cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t p);
cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t p, cublasLtMatmulPreferenceAttributes_t a, const void* v, size_t n);
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t h, cublasLtMatmulDesc_t d, cublasLtMatrixLayout_t a,
                                              cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t dl,
                                              cublasLtMatmulPreference_t p, int requested, cublasLtMatmulHeuristicResult_t* out, int* got);
cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t* algo, cublasLtMatmulAlgoConfigAttributes_t a,
                                                    void* v, size_t n, size_t* written);
cublasStatus_t cublasLtMatmul(cublasLtHandle_t h, cublasLtMatmulDesc_t d, const void* alpha, const void* A, cublasLtMatrixLayout_t la,
                              const void* B, cublasLtMatrixLayout_t lb, const void* beta, const void* C, cublasLtMatrixLayout_t lc,
                              void* D, cublasLtMatrixLayout_t ld, const cublasLtMatmulAlgo_t* algo, void* ws, size_t wsBytes,
                              cudaStream_t s);

// ---------------------------------------------------------------- kernel launches (the generated stubs call these)
namespace lf {
struct KernelInfo { const char* name; const char* structName; int tparams; int status; };   // status 0 translated, 1 overridden, 2 not translated
struct Launch { dim3 grid, block; size_t smem = 0; };
void setLaunchRaw(dim3 grid, dim3 block, size_t smem);
inline dim3 asDim(dim3 d) { return d; }
template <class I, typename std::enable_if<std::is_integral<I>::value, int>::type = 0> inline dim3 asDim(I v) { return dim3((unsigned)v); }
template <class G, class B> inline int setLaunch(G grid, B block, size_t smem = 0, cudaStream_t = nullptr) {
  setLaunchRaw(asDim(grid), asDim(block), smem); return 0;
}
// a kernel's arguments, packed as C lays out a struct of them (the Metal side's argument struct, member for member)
struct KArgs {
  std::vector<unsigned char> bytes;
  size_t align = 1;
  template <class T> void add(const T& v) {
    size_t a = alignof(T);
    if (a > align) align = a;
    size_t at = (bytes.size() + a - 1) / a * a;
    bytes.resize(at + sizeof(T));
    std::memcpy(bytes.data() + at, &v, sizeof(T));
  }
  void finish() { bytes.resize((bytes.size() + align - 1) / align * align); }
};
void launch(int kernel, std::initializer_list<std::string> targs, KArgs& args);
// a template argument as the Metal source spells it
template <class T> std::string tname();
template <> inline std::string tname<float>() { return "float"; }
template <> inline std::string tname<double>() { return "float"; }
template <> inline std::string tname<__half>() { return "half"; }
template <> inline std::string tname<__half2>() { return "half2"; }
template <> inline std::string tname<__nv_bfloat16>() { return "lf_bf16"; }
template <> inline std::string tname<int>() { return "int"; }
template <> inline std::string tname<unsigned>() { return "uint"; }
template <> inline std::string tname<short>() { return "short"; }
template <> inline std::string tname<unsigned short>() { return "ushort"; }
template <> inline std::string tname<char>() { return "char"; }
template <> inline std::string tname<signed char>() { return "char"; }
template <> inline std::string tname<unsigned char>() { return "uchar"; }
template <> inline std::string tname<bool>() { return "bool"; }
template <> inline std::string tname<long long>() { return "long"; }
template <> inline std::string tname<unsigned long long>() { return "ulong"; }
template <> inline std::string tname<long>() { return "long"; }
template <> inline std::string tname<unsigned long>() { return "ulong"; }
template <> inline std::string tname<float2>() { return "float2"; }
template <> inline std::string tname<float4>() { return "float4"; }
template <class T, class V> std::string tval(V v, const char* type) {
  if constexpr (std::is_same<T, bool>::value) return v ? "true" : "false";
  else if constexpr (std::is_enum<T>::value) return std::string("(") + type + ")" + std::to_string((long long)v);
  else if constexpr (std::is_unsigned<T>::value) return std::to_string((unsigned long long)v) + "u";
  else return std::to_string((long long)v);
}
// std::to_chars, where a float's form needs macOS 13.3: fixed with a precision prints as "%.*f" does
inline std::to_chars_result to_chars(char* first, char* last, double v, std::chars_format fmt, int precision) {
  int n = fmt == std::chars_format::fixed ? snprintf(first, last - first, "%.*f", precision, v)
        : fmt == std::chars_format::scientific ? snprintf(first, last - first, "%.*e", precision, v)
        : snprintf(first, last - first, "%.*g", precision, v);
  if (n < 0 || n >= last - first) return {last, std::errc::value_too_large};
  return {first + n, std::errc()};
}
inline std::to_chars_result to_chars(char* first, char* last, double v) {
  char buf[64];
  for (int p = 1; p <= 17; ++p) { snprintf(buf, sizeof buf, "%.*g", p, v); if (strtod(buf, nullptr) == v) break; }
  size_t n = strlen(buf);
  if ((ptrdiff_t)n > last - first) return {last, std::errc::value_too_large};
  memcpy(first, buf, n); return {first + n, std::errc()};
}
template <class I, typename std::enable_if<std::is_integral<I>::value, int>::type = 0>
inline std::to_chars_result to_chars(char* first, char* last, I v) { return std::to_chars(first, last, v); }
// where a Linux build would use /dev/shm: $TMPDIR (macOS's per-user temporary directory), else /tmp
inline const char* tmpDir() {
  const char* t = getenv("TMPDIR");
  static std::string dir = (t && *t) ? std::string(t) : std::string("/tmp");
  while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
  return dir.c_str();
}
// the host memory behind a device pointer (unified memory: the same bytes), for tools and tests
void* hostView(const void* device);
}  // namespace lf
