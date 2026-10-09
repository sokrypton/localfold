// cuda/af2/src/fast.cuh's triGateTK with a pair's (row, column) divided out once, in 32 bits: the size_t division a
// channel is emulated on Apple's GPUs (3 ms a call at 261 residues). The same arithmetic.
{
  __shared__ float A[32][33], B[32][33];
  const unsigned r0 = blockIdx.x * 32u; const int c0 = blockIdx.y * 32;
  const int tx = threadIdx.x, ty = threadIdx.y;
  for (int ry = ty; ry < 32; ry += 8) {
    unsigned r = r0 + ry; int c = c0 + tx;
    float va = 0, vb = 0;
    if (r < pairs) {
      const half* p = pg + (size_t)r * 5 * C; float m = mask[p0 + r];
      va = __half2float(p[c]) * m / (1.f + __expf(-__half2float(p[2 * C + c])));
      vb = __half2float(p[C + c]) * m / (1.f + __expf(-__half2float(p[3 * C + c])));
    }
    A[ry][tx] = va; B[ry][tx] = vb;
  }
  __syncthreads();
  const unsigned r = r0 + tx;
  if (r >= pairs) return;
  const unsigned pp = (unsigned)(p0 + r), i = pp / (unsigned)L, j = pp - i * (unsigned)L;
  const size_t q = (size_t)i * Lp + j, plane = (size_t)Lp * Lp;
  for (int cy = ty; cy < 32; cy += 8) {
    const size_t at = (size_t)(c0 + cy) * plane + q;
    a[at] = __float2half(A[tx][cy]); b[at] = __float2half(B[tx][cy]);
  }
}
