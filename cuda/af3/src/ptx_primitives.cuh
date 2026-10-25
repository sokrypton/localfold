// Inline PTX primitives for SM80+ (Ampere/Ada/Hopper): loads/stores, warp shuffles,
// MUFU math, FP16/BF16/FP32 conversions, and tensor-core helpers.
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ---------------------------------------------------------------- shared memory addressing
__device__ __forceinline__ uint32_t ptx_smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------- global memory loads (.nc / .ca / .cg)
__device__ __forceinline__ float ptx_ldg_f32(const float* p) {
  float v;
  asm("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t ptx_ldg_u32(const void* p) {
  uint32_t v;
  asm("ld.global.nc.b32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ int32_t ptx_ldg_s32(const int* p) {
  int32_t v;
  asm("ld.global.nc.s32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ unsigned short ptx_ldg_u16(const void* p) {
  unsigned short v;
  asm("ld.global.nc.b16 %0, [%1];" : "=h"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ float2 ptx_ldg_v2_f32(const void* p) {
  float2 v;
  asm("ld.global.nc.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ float4 ptx_ldg_v4_f32(const void* p) {
  float4 v;
  asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint2 ptx_ldg_v2_u32(const void* p) {
  uint2 v;
  asm("ld.global.nc.v2.b32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 ptx_ldg_v4_u32(const void* p) {
  uint4 v;
  asm("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ float ptx_ld_global_f32(const float* p) {
  float v;
  asm("ld.global.f32 %0, [%1];" : "=f"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t ptx_ld_global_u32(const void* p) {
  uint32_t v;
  asm("ld.global.b32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ unsigned short ptx_ld_global_u16(const void* p) {
  unsigned short v;
  asm("ld.global.b16 %0, [%1];" : "=h"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ float2 ptx_ld_global_v2_f32(const void* p) {
  float2 v;
  asm("ld.global.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ float4 ptx_ld_global_v4_f32(const void* p) {
  float4 v;
  asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint2 ptx_ld_global_v2_u32(const void* p) {
  uint2 v;
  asm("ld.global.v2.b32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 ptx_ld_global_v4_u32(const void* p) {
  uint4 v;
  asm("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// ---------------------------------------------------------------- global memory stores
__device__ __forceinline__ void ptx_st_global_f32(void* p, float v) {
  asm volatile("st.global.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory");
}
__device__ __forceinline__ void ptx_st_global_u32(void* p, uint32_t v) {
  asm volatile("st.global.b32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void ptx_st_global_u16(void* p, unsigned short v) {
  asm volatile("st.global.b16 [%0], %1;" :: "l"(p), "h"(v) : "memory");
}
__device__ __forceinline__ void ptx_st_global_b16(void* p, unsigned short v) { ptx_st_global_u16(p, v); }
__device__ __forceinline__ void ptx_st_global_v2_f32(void* p, float2 v) {
  asm volatile("st.global.v2.f32 [%0], {%1, %2};" :: "l"(p), "f"(v.x), "f"(v.y) : "memory");
}
__device__ __forceinline__ void ptx_st_global_v4_f32(void* p, float4 v) {
  asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ void ptx_st_global_v2_u32(void* p, uint2 v) {
  asm volatile("st.global.v2.b32 [%0], {%1, %2};" :: "l"(p), "r"(v.x), "r"(v.y) : "memory");
}
__device__ __forceinline__ void ptx_st_global_v4_u32(void* p, uint4 v) {
  asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

// ---------------------------------------------------------------- shared memory loads & stores
__device__ __forceinline__ float ptx_ld_shared_f32(const void* p) {
  float v; uint32_t a = ptx_smem_addr(p);
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ptx_ld_shared_u32(const void* p) {
  uint32_t v; uint32_t a = ptx_smem_addr(p);
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ unsigned short ptx_ld_shared_u16(const void* p) {
  unsigned short v; uint32_t a = ptx_smem_addr(p);
  asm volatile("ld.shared.b16 %0, [%1];" : "=h"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ float2 ptx_ld_shared_v2_f32(const void* p) {
  float2 v; uint32_t a = ptx_smem_addr(p);
  asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ float4 ptx_ld_shared_v4_f32(const void* p) {
  float4 v; uint32_t a = ptx_smem_addr(p);
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint2 ptx_ld_shared_v2_u32(const void* p) {
  uint2 v; uint32_t a = ptx_smem_addr(p);
  asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ptx_ld_shared_v4_u32(const void* p) {
  uint4 v; uint32_t a = ptx_smem_addr(p);
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}

__device__ __forceinline__ void ptx_st_shared_f32(void* p, float v) {
  uint32_t a = ptx_smem_addr(p);
  asm volatile("st.shared.f32 [%0], %1;" :: "r"(a), "f"(v) : "memory");
}
__device__ __forceinline__ void ptx_st_shared_u32(void* p, uint32_t v) {
  uint32_t a = ptx_smem_addr(p);
  asm volatile("st.shared.b32 [%0], %1;" :: "r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void ptx_st_shared_u16(void* p, unsigned short v) {
  uint32_t a = ptx_smem_addr(p);
  asm volatile("st.shared.b16 [%0], %1;" :: "r"(a), "h"(v) : "memory");
}
__device__ __forceinline__ void ptx_st_shared_b16(void* p, unsigned short v) { ptx_st_shared_u16(p, v); }
__device__ __forceinline__ void ptx_st_shared_v2_f32(void* p, float2 v) {
  uint32_t a = ptx_smem_addr(p);
  asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(a), "f"(v.x), "f"(v.y) : "memory");
}
__device__ __forceinline__ void ptx_st_shared_v2_f32(void* p, float x, float y) {
  uint32_t a = ptx_smem_addr(p);
  asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(a), "f"(x), "f"(y) : "memory");
}
__device__ __forceinline__ void ptx_st_shared_v4_f32(void* p, float4 v) {
  uint32_t a = ptx_smem_addr(p);
  asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ void ptx_st_shared_v2_u32(void* p, uint2 v) {
  uint32_t a = ptx_smem_addr(p);
  asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(a), "r"(v.x), "r"(v.y) : "memory");
}
__device__ __forceinline__ void ptx_st_shared_v4_u32(void* p, uint4 v) {
  uint32_t a = ptx_smem_addr(p);
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

// ---------------------------------------------------------------- PTX type conversions (FP16 / BF16 / FP32)
// Pack (lo, hi) -> f16x2 in a 32-bit register (lo in [15:0], hi in [31:16])
__device__ __forceinline__ uint32_t ptx_cvt_f16x2_f32(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(d) : "f"(lo), "f"(hi));
  return d;
}
// Pack (lo, hi) -> bf16x2 in a 32-bit register (lo in [15:0], hi in [31:16])
__device__ __forceinline__ uint32_t ptx_cvt_bf16x2_f32(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(d) : "f"(lo), "f"(hi));
  return d;
}
// Unpack f16x2 -> (lo, hi) float2
__device__ __forceinline__ void ptx_f16x2_to_f32x2(uint32_t u, float& lo, float& hi) {
  asm("{\n\t"
      ".reg .b16 h0, h1;\n\t"
      "mov.b32 {h0, h1}, %2;\n\t"
      "cvt.f32.f16 %0, h0;\n\t"
      "cvt.f32.f16 %1, h1;\n\t"
      "}" : "=f"(lo), "=f"(hi) : "r"(u));
}
__device__ __forceinline__ float2 ptx_f16x2_to_f32x2(uint32_t u) {
  float lo, hi;
  ptx_f16x2_to_f32x2(u, lo, hi);
  return make_float2(lo, hi);
}
// Unpack bf16x2 -> (lo, hi) float2
__device__ __forceinline__ void ptx_bf16x2_to_f32x2(uint32_t u, float& lo, float& hi) {
  asm("{\n\t"
      ".reg .b16 h0, h1;\n\t"
      "mov.b32 {h0, h1}, %2;\n\t"
      "mov.b32 %0, {0, h0};\n\t"
      "mov.b32 %1, {0, h1};\n\t"
      "}" : "=f"(lo), "=f"(hi) : "r"(u));
}
__device__ __forceinline__ float2 ptx_bf16x2_to_f32x2(uint32_t u) {
  float lo, hi;
  ptx_bf16x2_to_f32x2(u, lo, hi);
  return make_float2(lo, hi);
}
__device__ __forceinline__ unsigned short ptx_f32_to_f16_raw(float x) {
  unsigned short d;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(d) : "f"(x));
  return d;
}
__device__ __forceinline__ float ptx_f16_raw_to_f32(unsigned short x) {
  float d;
  asm("cvt.f32.f16 %0, %1;" : "=f"(d) : "h"(x));
  return d;
}
__device__ __forceinline__ unsigned short ptx_f32_to_bf16_raw(float x) {
  unsigned short d;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(d) : "f"(x));
  return d;
}
__device__ __forceinline__ float ptx_bf16_raw_to_f32(unsigned short x) {
  float d;
  asm("{\n\t"
      ".reg .b32 r;\n\t"
      "mov.b32 r, {0, %1};\n\t"
      "mov.b32 %0, r;\n\t"
      "}" : "=f"(d) : "h"(x));
  return d;
}
__device__ __forceinline__ uint32_t ptx_splat_f16x2_f32(float x) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %1;" : "=r"(d) : "f"(x));
  return d;
}
__device__ __forceinline__ float ptx_round_f16_f32(float x) {
  float d;
  asm("{\n\t"
      ".reg .b16 h;\n\t"
      "cvt.rn.f16.f32 h, %1;\n\t"
      "cvt.f32.f16 %0, h;\n\t"
      "}" : "=f"(d) : "f"(x));
  return d;
}
__device__ __forceinline__ uint32_t ptx_max_f16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ uint32_t ptx_sub_f16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("sub.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ uint32_t ptx_sub_ex2_f16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("{\n\t"
      ".reg .b32 t;\n\t"
      "sub.f16x2 t, %1, %2;\n\t"
      "ex2.approx.f16x2 %0, t;\n\t"
      "}" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ float ptx_hmax2_to_f32(uint32_t u) {
  float d;
  asm("{\n\t"
      ".reg .b16 h0, h1, hm;\n\t"
      "mov.b32 {h0, h1}, %1;\n\t"
      "max.f16 hm, h0, h1;\n\t"
      "cvt.f32.f16 %0, hm;\n\t"
      "}" : "=f"(d) : "r"(u));
  return d;
}

// ---------------------------------------------------------------- PTX warp shuffles & butterfly reductions
__device__ __forceinline__ float ptx_shfl_xor_f32(float v, int mask) {
  float d;
  asm("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(d) : "f"(v), "r"(mask));
  return d;
}
__device__ __forceinline__ uint32_t ptx_shfl_xor_u32(uint32_t v, int mask) {
  uint32_t d;
  asm("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(d) : "r"(v), "r"(mask));
  return d;
}
__device__ __forceinline__ float ptx_shfl_idx_f32(float v, int srcLane) {
  float d;
  asm("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(d) : "f"(v), "r"(srcLane));
  return d;
}
__device__ __forceinline__ float ptx_warp_sum_f32(float v) {
  asm("{\n\t"
      ".reg .f32 t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 16, 0x1f, 0xffffffff;\n\t"
      "add.ftz.f32 %0, %0, t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 8, 0x1f, 0xffffffff;\n\t"
      "add.ftz.f32 %0, %0, t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 4, 0x1f, 0xffffffff;\n\t"
      "add.ftz.f32 %0, %0, t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 2, 0x1f, 0xffffffff;\n\t"
      "add.ftz.f32 %0, %0, t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 1, 0x1f, 0xffffffff;\n\t"
      "add.ftz.f32 %0, %0, t;\n\t"
      "}" : "+f"(v));
  return v;
}
__device__ __forceinline__ float ptx_warp_max_f32(float v) {
  asm("{\n\t"
      ".reg .f32 t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 16, 0x1f, 0xffffffff;\n\t"
      "max.ftz.f32 %0, %0, t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 8, 0x1f, 0xffffffff;\n\t"
      "max.ftz.f32 %0, %0, t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 4, 0x1f, 0xffffffff;\n\t"
      "max.ftz.f32 %0, %0, t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 2, 0x1f, 0xffffffff;\n\t"
      "max.ftz.f32 %0, %0, t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 1, 0x1f, 0xffffffff;\n\t"
      "max.ftz.f32 %0, %0, t;\n\t"
      "}" : "+f"(v));
  return v;
}
__device__ __forceinline__ float ptx_halfwarp_sum_f32(float v) {
  asm("{\n\t"
      ".reg .f32 t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 8, 0x1f, 0xffffffff;\n\t"
      "add.ftz.f32 %0, %0, t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 4, 0x1f, 0xffffffff;\n\t"
      "add.ftz.f32 %0, %0, t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 2, 0x1f, 0xffffffff;\n\t"
      "add.ftz.f32 %0, %0, t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 1, 0x1f, 0xffffffff;\n\t"
      "add.ftz.f32 %0, %0, t;\n\t"
      "}" : "+f"(v));
  return v;
}
__device__ __forceinline__ float ptx_quad_sum_f32(float v) {
  asm("{\n\t"
      ".reg .f32 t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 2, 0x1f, 0xffffffff;\n\t"
      "add.ftz.f32 %0, %0, t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 1, 0x1f, 0xffffffff;\n\t"
      "add.ftz.f32 %0, %0, t;\n\t"
      "}" : "+f"(v));
  return v;
}
__device__ __forceinline__ float ptx_quad_max_f32(float v) {
  asm("{\n\t"
      ".reg .f32 t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 2, 0x1f, 0xffffffff;\n\t"
      "max.ftz.f32 %0, %0, t;\n\t"
      "shfl.sync.bfly.b32 t, %0, 1, 0x1f, 0xffffffff;\n\t"
      "max.ftz.f32 %0, %0, t;\n\t"
      "}" : "+f"(v));
  return v;
}

// ---------------------------------------------------------------- PTX MUFU & ALU primitives
__device__ __forceinline__ float ptx_rsqrt_f32(float x) {
  float d;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x));
  return d;
}
__device__ __forceinline__ float ptx_rcp_f32(float x) {
  float d;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x));
  return d;
}
__device__ __forceinline__ float ptx_div_f32(float a, float b) {
  float d;
  asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float ptx_sqrt_f32(float x) {
  float d;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x));
  return d;
}
__device__ __forceinline__ float ptx_ex2_f32(float x) {
  float d;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x));
  return d;
}
__device__ __forceinline__ float ptx_exp_f32(float x) {
  float d;
  asm("{\n\t"
      ".reg .f32 t;\n\t"
      "mul.ftz.f32 t, %1, 0f3FB8AA3B;\n\t"
      "ex2.approx.ftz.f32 %0, t;\n\t"
      "}" : "=f"(d) : "f"(x));
  return d;
}
__device__ __forceinline__ float ptx_sigm_f32(float x) {
  float d;
  asm("{\n\t"
      ".reg .f32 t;\n\t"
      "mul.ftz.f32 t, %1, 0fBFB8AA3B;\n\t"
      "ex2.approx.ftz.f32 t, t;\n\t"
      "add.ftz.f32 t, t, 0f3F800000;\n\t"
      "rcp.approx.ftz.f32 %0, t;\n\t"
      "}" : "=f"(d) : "f"(x));
  return d;
}
__device__ __forceinline__ float ptx_sigm_fast_f32(float x) {
  float d;
  asm("{\n\t"
      ".reg .f32 t;\n\t"
      "mul.ftz.f32 t, %1, 0f3F000000;\n\t"
      "tanh.approx.f32 t, t;\n\t"
      "fma.rn.ftz.f32 %0, t, 0f3F000000, 0f3F000000;\n\t"
      "}" : "=f"(d) : "f"(x));
  return d;
}
__device__ __forceinline__ float ptx_relu_f32(float x) {
  float d;
  asm("max.ftz.f32 %0, %1, 0f00000000;" : "=f"(d) : "f"(x));
  return d;
}
__device__ __forceinline__ float ptx_add_f32(float a, float b) {
  float d;
  asm("add.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float ptx_sub_f32(float a, float b) {
  float d;
  asm("sub.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float ptx_mul_f32(float a, float b) {
  float d;
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float ptx_fma_f32(float a, float b, float c) {
  float d;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
__device__ __forceinline__ float ptx_max_f32(float a, float b) {
  float d;
  asm("max.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float ptx_min_f32(float a, float b) {
  float d;
  asm("min.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
