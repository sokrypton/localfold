// cuda/af2/src/templates.cuh's templateTorsionFeatK with the 28 atoms it reads COPIED into a local array: the CUDA
// kernel points at either the template (device memory) or a local zero row for residue 0's missing predecessor,
// and Metal has no pointer that can be both. (Its double arithmetic is float here; residue 0's pre-omega, the one
// value that cancellation made fragile, is defined as zero by both.)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  int aaRaw = tAatype[i], a = min(aaRaw, 20);
  const float* p = tPos + (size_t)i * 37 * 3; const float* m = tMask + (size_t)i * 37;
  float prev[37 * 3], pm[37];
  for (int k = 0; k < 37 * 3; ++k) prev[k] = i > 0 ? tPos[(size_t)(i - 1) * 37 * 3 + k] : 0.f;
  for (int k = 0; k < 37; ++k) pm[k] = i > 0 ? tMask[(size_t)(i - 1) * 37 + k] : 0.f;
  float at[7][4][3]; float tm[7];
  for (int k = 0; k < 3; ++k) {
    at[0][0][k] = prev[3 + k]; at[0][1][k] = prev[6 + k]; at[0][2][k] = p[k]; at[0][3][k] = p[3 + k];     // pre-omega
    at[1][0][k] = prev[6 + k]; at[1][1][k] = p[k]; at[1][2][k] = p[3 + k]; at[1][3][k] = p[6 + k];        // phi
    at[2][0][k] = p[k]; at[2][1][k] = p[3 + k]; at[2][2][k] = p[6 + k]; at[2][3][k] = p[12 + k];          // psi
  }
  tm[0] = pm[1] * pm[2] * m[0] * m[1];
  tm[1] = pm[2] * m[0] * m[1] * m[2];
  tm[2] = m[0] * m[1] * m[2] * m[4];
  for (int c = 0; c < 4; ++c) {
    float mm = chiMaskT[a * 4 + c];
    for (int q = 0; q < 4; ++q) {
      int idx = chiIdx[(a * 4 + c) * 4 + q];
      for (int k = 0; k < 3; ++k) at[3 + c][q][k] = p[idx * 3 + k];
      mm *= m[idx];
    }
    tm[3 + c] = mm;
  }
  float* f = feat + (size_t)i * 57;
  for (int k = 0; k < 22; ++k) f[k] = (k == min(max(aaRaw, 0), 21)) ? 1.f : 0.f;
  for (int t = 0; t < 7; ++t) {
    float e0[3], e1[3], e2[3], d[3];
    for (int k = 0; k < 3; ++k) { e0[k] = at[t][2][k] - at[t][1][k]; e1[k] = at[t][0][k] - at[t][2][k]; d[k] = at[t][3][k] - at[t][2][k]; }
    float n0 = sqrtf(e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2] + 1e-8f); for (int k = 0; k < 3; ++k) e0[k] /= n0;
    float c = e1[0] * e0[0] + e1[1] * e0[1] + e1[2] * e0[2]; for (int k = 0; k < 3; ++k) e1[k] -= c * e0[k];
    float n1 = sqrtf(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2] + 1e-8f); for (int k = 0; k < 3; ++k) e1[k] /= n1;
    e2[0] = e0[1] * e1[2] - e0[2] * e1[1]; e2[1] = e0[2] * e1[0] - e0[0] * e1[2]; e2[2] = e0[0] * e1[1] - e0[1] * e1[0];
    float y = e1[0] * d[0] + e1[1] * d[1] + e1[2] * d[2], z = e2[0] * d[0] + e2[1] * d[1] + e2[2] * d[2];
    float nn = sqrtf(z * z + y * y + 1e-8f);
    float sn = z / nn, cs = y / nn;
    if (t == 0 && i == 0) sn = cs = 0.f;
    if (t == 2) { sn = -sn; cs = -cs; }
    float alt = t >= 3 ? 1.f - 2.f * chiPi[a * 4 + t - 3] : 1.f;
    f[22 + t * 2] = sn; f[22 + t * 2 + 1] = cs;
    f[36 + t * 2] = sn * alt; f[36 + t * 2 + 1] = cs * alt;
    f[50 + t] = tm[t];
  }
  rowMask[i] = tm[2];
}
