// The structure module's backward, kernel side. The forward is metal/af2/structure.mm and the tail of
// metal/af2/af2.metal (af3-any-model's alphafold3/af2/model/folding.py: InvariantPointAttention, FoldIteration,
// MultiRigidSidechain). See kernels.h for the rigid's layout and why these are a thread a residue.

// a rigid is 12 floats: the rotation row-major, then the translation. The forward's own helpers, repeated here
// because metal/af2's are in another port's source.
inline void lf_rapply(thread const float* r, thread const float* p, thread float* o) {
  for (int k = 0; k < 3; ++k) o[k] = r[k * 3] * p[0] + r[k * 3 + 1] * p[1] + r[k * 3 + 2] * p[2] + r[9 + k];
}

// ---------------------------------------------------------------- the backbone update
// The forward: q = (1, x, y, z) / |(1, x, y, z)|, U the rotation of q with translation (u3, u4, u5), and
// rigNew = rigOld o U, i.e. R' = A B and t' = A b + a where A/a is rigOld and B/b is U.
//
// So, with G = dR' and g = dt':
//   dA[k][j] += sum_m G[k][m] B[j][m] + g[k] b[j]      da[k] += g[k]
//   dB[j][m]  = sum_k G[k][m] A[k][j]                  db[j] = sum_k g[k] A[k][j]
// and dB goes through the quaternion and then through the normalisation. The norm is never clamped: w is 1, so the
// squared length is at least 1 and the forward's max(1e-6, ...) cannot bind - which is why there is no gate here.
kernel void lf_rigid_upd_bwd(LF_ARGS(RigidUpdBwdArgs)) {
  uint i = (uint)LF_INDEX;
  if (i >= a.L) return;
  device const float* A = a.rigOld + i * 12;
  device const float* u = a.upd + i * 6;
  device const float* dC = a.dRigNew + i * 12;

  // the forward's U, rebuilt
  float w = 1, x = u[0], y = u[1], z = u[2];
  float inv = rsqrt(w * w + x * x + y * y + z * z);
  w *= inv; x *= inv; y *= inv; z *= inv;
  float B[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)};
  float b[3] = {u[3], u[4], u[5]};

  // dRigOld
  device float* dA = a.dRigOld + i * 12;
  for (int k = 0; k < 3; ++k) {
    for (int j = 0; j < 3; ++j) {
      float s = dC[9 + k] * b[j];
      for (int m = 0; m < 3; ++m) s += dC[k * 3 + m] * B[j * 3 + m];
      if (a.accum) dA[k * 3 + j] += s; else dA[k * 3 + j] = s;
    }
    if (a.accum) dA[9 + k] += dC[9 + k]; else dA[9 + k] = dC[9 + k];
  }

  // dU: the rotation, then the translation
  float G[9];
  for (int j = 0; j < 3; ++j)
    for (int m = 0; m < 3; ++m) {
      float s = 0;
      for (int k = 0; k < 3; ++k) s += dC[k * 3 + m] * A[k * 3 + j];
      G[j * 3 + m] = s;
    }
  // the quaternion's four, read straight off U's entries
  float dw = 2 * (-G[1] * z + G[2] * y + G[3] * z - G[5] * x - G[6] * y + G[7] * x);
  float dx = 2 * (G[1] * y + G[2] * z + G[3] * y - 2 * x * G[4] - G[5] * w + G[6] * z + G[7] * w - 2 * x * G[8]);
  float dy = 2 * (-2 * y * G[0] + G[1] * x + G[2] * w + G[3] * x + G[5] * z - G[6] * w + G[7] * z - 2 * y * G[8]);
  float dz = 2 * (-2 * z * G[0] - G[1] * w + G[2] * x + G[3] * w - 2 * z * G[4] + G[5] * y + G[6] * x + G[7] * y);
  // through q = r / |r|, whose only free components are the last three (w's raw value is the constant 1)
  float dot = dw * w + dx * x + dy * y + dz * z;
  device float* du = a.dUpd + i * 6;
  du[0] = (dx - x * dot) * inv;
  du[1] = (dy - y * dot) * inv;
  du[2] = (dz - z * dot) * inv;
  for (int j = 0; j < 3; ++j) {
    float s = 0;
    for (int k = 0; k < 3; ++k) s += dC[9 + k] * A[k * 3 + j];
    du[3 + j] = s;
  }
}

// ---------------------------------------------------------------- points into the global frame
// g = R loc + t, so dloc = R^T dg, dR[k][j] += dg[k] loc[j], dt[k] += dg[k].
kernel void lf_points_global_bwd(LF_ARGS(PointsGlobalBwdArgs)) {
  uint i = (uint)LF_INDEX;
  if (i >= a.L) return;
  float R[9], acc[12];
  for (int k = 0; k < 9; ++k) R[k] = a.rig[i * 12 + k];
  for (int k = 0; k < 12; ++k) acc[k] = 0;
  for (uint h = 0; h < a.H; ++h) {
    device const float* s = a.proj + ((ulong)i * a.H + h) * 3 * a.P;
    device float* ds = a.dProj + ((ulong)i * a.H + h) * 3 * a.P;
    for (uint p = 0; p < a.P; ++p) {
      ulong t = (((ulong)i * a.H + h) * a.P + p) * 3;
      float dg[3] = {a.dOut[t], a.dOut[t + 1], a.dOut[t + 2]};
      float loc[3] = {s[p], s[a.P + p], s[2 * a.P + p]};
      for (int j = 0; j < 3; ++j) {
        float v = R[j] * dg[0] + R[3 + j] * dg[1] + R[6 + j] * dg[2];
        if (a.accumProj) ds[j * a.P + p] += v; else ds[j * a.P + p] = v;
      }
      for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < 3; ++j) acc[k * 3 + j] += dg[k] * loc[j];
        acc[9 + k] += dg[k];
      }
    }
  }
  device float* dr = a.dRig + i * 12;
  for (int k = 0; k < 12; ++k) { if (a.accumRig) dr[k] += acc[k]; else dr[k] = acc[k]; }
}
