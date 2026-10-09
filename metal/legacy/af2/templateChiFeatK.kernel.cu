// cuda/af2/src/templates.cuh's templateChiFeatK, its four atom pointers declared in device memory (Metal names the
// address space of an array of pointers)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  int aaRaw = tAatype[i], a = min(max(aaRaw, 0), 20);
  const float* p = tPos + (size_t)i * 37 * 3; const float* m = tMask + (size_t)i * 37;
  float* f = feat + (size_t)i * 34;
  for (int k = 0; k < 22; ++k) f[k] = (k == min(max(aaRaw, 0), 21)) ? 1.f : 0.f;
  for (int c = 0; c < 4; ++c) {
    device const float* x[4]; float mm = chiMaskT[a * 4 + c];
    for (int k = 0; k < 4; ++k) { int at = chiIdx[(a * 4 + c) * 4 + k]; x[k] = p + at * 3; mm *= m[at]; }
    float v1[3], v2[3], v3[3], c1[3], c2[3], c3[3];
    sub3(x[0], x[1], v1); sub3(x[1], x[2], v2); sub3(x[3], x[2], v3);
    cross3(v1, v2, c1); cross3(v3, v2, c2); cross3(c2, c1, c3);
    float v2m = sqrtf(fmaxf(dot3(v2, v2), 1e-12f));
    float ang = atan2f(dot3(c3, v2), v2m * dot3(c1, c2));
    f[22 + c] = sinf(ang) * mm; f[26 + c] = cosf(ang) * mm; f[30 + c] = mm;
    if (c == 0) rowMask[i] = mm;
  }
}
