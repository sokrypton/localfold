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

// ---------------------------------------------------------------- InvariantPointAttention: the attention weights
// dLogit[q][h][k] = attn (dAttn - sum_j attn dAttn) * sqrt(1/3), the softmax's own backward with the forward's
// scaling carried through. The mask penalty is a constant and contributes nothing.
kernel void lf_ipa_dlogit(LF_ARGS(IpaDlogitArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.L * a.H) return;
  device const float* at = a.attn + (ulong)t * a.L;
  device const float* da = a.dAttn + (ulong)t * a.L;
  device float* dl = a.dLogit + (ulong)t * a.L;
  float corr = 0;
  for (uint k = 0; k < a.L; ++k) corr += at[k] * da[k];
  const float sc = sqrt(1.f / 3.f);
  for (uint k = 0; k < a.L; ++k) dl[k] = at[k] * (da[k] - corr) * sc;
}

// per (query, head): the gradients that accumulate over KEYS
kernel void lf_ipa_dq(LF_ARGS(IpaDqArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.L * a.H) return;
  uint h = t % a.H, q = t / a.H;
  device const float* dl = a.dLogit + (ulong)t * a.L;
  const float w = a.pw[h];
  device float* dqs = a.dqs + (ulong)t * a.Cs;
  for (uint c = 0; c < a.Cs; ++c) {
    float s = 0;
    for (uint k = 0; k < a.L; ++k) s += dl[k] * a.ks[((ulong)k * a.H + h) * a.Cs + c];
    dqs[c] = s;
  }
  float part = 0;
  for (uint p = 0; p < a.Pq; ++p) {
    float acc[3] = {0, 0, 0};
    for (uint k = 0; k < a.L; ++k) {
      float g = dl[k];
      for (int x = 0; x < 3; ++x) {
        float d = a.qp[((ulong)t * a.Pq + p) * 3 + x] - a.kp[(((ulong)k * a.H + h) * a.Pq + p) * 3 + x];
        acc[x] += g * d;
        part += g * (-0.5f) * d * d;
      }
    }
    for (int x = 0; x < 3; ++x) a.dqp[((ulong)t * a.Pq + p) * 3 + x] = -w * acc[x];
  }
  a.dpwPart[t] = part;
}

// per (key, head): the gradients that accumulate over QUERIES
kernel void lf_ipa_dk(LF_ARGS(IpaDkArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.L * a.H) return;
  uint h = t % a.H, k = t / a.H;
  const float w = a.pw[h];
  device float* dks = a.dks + (ulong)t * a.Cs;
  for (uint c = 0; c < a.Cs; ++c) {
    float s = 0;
    for (uint q = 0; q < a.L; ++q)
      s += a.dLogit[((ulong)q * a.H + h) * a.L + k] * a.qs[((ulong)q * a.H + h) * a.Cs + c];
    dks[c] = s;
  }
  for (uint p = 0; p < a.Pq; ++p) {
    float acc[3] = {0, 0, 0};
    for (uint q = 0; q < a.L; ++q) {
      float g = a.dLogit[((ulong)q * a.H + h) * a.L + k];
      for (int x = 0; x < 3; ++x)
        acc[x] += g * (a.qp[(((ulong)q * a.H + h) * a.Pq + p) * 3 + x] - a.kp[((ulong)t * a.Pq + p) * 3 + x]);
    }
    for (int x = 0; x < 3; ++x) a.dkp[((ulong)t * a.Pq + p) * 3 + x] = w * acc[x];
  }
}

kernel void lf_ipa_db2d(LF_ARGS(IpaDb2dArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.L * a.L * a.H) return;
  uint h = (uint)(t % a.H), k = (uint)((t / a.H) % a.L), q = (uint)(t / ((ulong)a.H * a.L));
  float v = a.dLogit[((ulong)q * a.H + h) * a.L + k];
  if (a.accum) a.db2d[t] += v; else a.db2d[t] = v;
}

kernel void lf_ipa_dpw(LF_ARGS(IpaDpwArgs)) {
  uint h = (uint)LF_INDEX;
  if (h >= a.H) return;
  float s = 0;
  for (uint q = 0; q < a.L; ++q) s += a.dpwPart[(ulong)q * a.H + h];
  float r = a.raw[h];
  float v = s * a.base * (1.f / (1.f + exp(-r)));
  if (a.accum) a.dRaw[h] += v; else a.dRaw[h] = v;
}

// ---------------------------------------------------------------- InvariantPointAttention: the outputs
// per (query, head): dAttn from all three blocks, the point cotangent dg the dv pass needs, and dRig's partial
kernel void lf_ipa_out_dq(LF_ARGS(IpaOutDqArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.L * a.H) return;
  uint h = t % a.H, q = t / a.H;
  uint Fw = a.H * a.Cs + 4 * a.H * a.Pv + a.H * a.C2, base = a.H * a.Cs;
  device const float* at = a.attn + (ulong)t * a.L;
  device const float* df = a.df + (ulong)q * Fw;
  device float* da = a.dAttn + (ulong)t * a.L;

  float R[12];
  for (int kk = 0; kk < 12; ++kk) R[kk] = a.rig[q * 12 + kk];
  float dR[12];
  for (int kk = 0; kk < 12; ++kk) dR[kk] = 0;

  // the point block: g = sum_k attn vp, loc = R^T (g - t), and the forward's norm over loc
  for (uint p = 0; p < a.Pv; ++p) {
    float g[3] = {0, 0, 0};
    for (uint k = 0; k < a.L; ++k)
      for (int x = 0; x < 3; ++x) g[x] += at[k] * a.vp[(((ulong)k * a.H + h) * a.Pv + p) * 3 + x];
    float d[3] = {g[0] - R[9], g[1] - R[10], g[2] - R[11]}, loc[3];
    for (int j = 0; j < 3; ++j) loc[j] = R[j] * d[0] + R[3 + j] * d[1] + R[6 + j] * d[2];
    uint idx = h * a.Pv + p;
    float n2 = loc[0] * loc[0] + loc[1] * loc[1] + loc[2] * loc[2];
    float nrm = sqrt(max(n2, 1e-16f));
    float dn = df[base + 3 * a.H * a.Pv + idx];
    float dloc[3];
    for (int x = 0; x < 3; ++x)
      dloc[x] = df[base + x * a.H * a.Pv + idx] + (n2 > 1e-16f ? dn * loc[x] / nrm : 0.f);
    // loc[j] = sum_m R[m][j] d[m], so dd[m] = sum_j R[m][j] dloc[j] and dR[m][j] += dloc[j] d[m]
    float dd[3];
    for (int m = 0; m < 3; ++m) {
      dd[m] = R[m * 3] * dloc[0] + R[m * 3 + 1] * dloc[1] + R[m * 3 + 2] * dloc[2];
      for (int j = 0; j < 3; ++j) dR[m * 3 + j] += dloc[j] * d[m];
      dR[9 + m] -= dd[m];
    }
    for (int x = 0; x < 3; ++x) a.dg[((ulong)t * a.Pv + p) * 3 + x] = dd[x];
  }
  for (int kk = 0; kk < 12; ++kk) a.dRigPart[(ulong)t * 12 + kk] = dR[kk];

  // dAttn: the scalar block, the point block through dg, and the pair block
  for (uint k = 0; k < a.L; ++k) {
    float s = 0;
    for (uint c = 0; c < a.Cs; ++c) s += df[h * a.Cs + c] * a.vs[((ulong)k * a.H + h) * a.Cs + c];
    for (uint p = 0; p < a.Pv; ++p)
      for (int x = 0; x < 3; ++x)
        s += a.dg[((ulong)t * a.Pv + p) * 3 + x] * a.vp[(((ulong)k * a.H + h) * a.Pv + p) * 3 + x];
    for (uint c = 0; c < a.C2; ++c)
      s += df[base + 4 * a.H * a.Pv + h * a.C2 + c] * a.act2d[((ulong)q * a.L + k) * a.C2 + c];
    da[k] = s;
  }
}

kernel void lf_ipa_out_drig(LF_ARGS(IpaOutDrigArgs)) {
  uint q = (uint)LF_INDEX;
  if (q >= a.L) return;
  device float* dr = a.dRig + q * 12;
  for (int kk = 0; kk < 12; ++kk) {
    float s = 0;
    for (uint h = 0; h < a.H; ++h) s += a.dRigPart[(((ulong)q * a.H + h) * 12) + kk];
    if (a.accum) dr[kk] += s; else dr[kk] = s;
  }
}

kernel void lf_ipa_out_dv(LF_ARGS(IpaOutDvArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.L * a.H) return;
  uint h = t % a.H, k = t / a.H;
  uint Fw = a.H * a.Cs + 4 * a.H * a.Pv + a.H * a.C2;
  device float* dvs = a.dvs + (ulong)t * a.Cs;
  for (uint c = 0; c < a.Cs; ++c) {
    float s = 0;
    for (uint q = 0; q < a.L; ++q)
      s += a.attn[((ulong)q * a.H + h) * a.L + k] * a.df[(ulong)q * Fw + h * a.Cs + c];
    dvs[c] = s;
  }
  for (uint p = 0; p < a.Pv; ++p)
    for (int x = 0; x < 3; ++x) {
      float s = 0;
      for (uint q = 0; q < a.L; ++q)
        s += a.attn[((ulong)q * a.H + h) * a.L + k] * a.dg[((((ulong)q * a.H + h) * a.Pv) + p) * 3 + x];
      a.dvp[((ulong)t * a.Pv + p) * 3 + x] = s;
    }
}

kernel void lf_ipa_out_d2d(LF_ARGS(IpaOutD2dArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.L * a.L) return;
  uint k = (uint)(t % a.L), q = (uint)(t / a.L);
  uint Fw = a.H * a.Cs + 4 * a.H * a.Pv + a.H * a.C2, off = a.H * a.Cs + 4 * a.H * a.Pv;
  device float* d = a.dact2d + t * a.C2;
  for (uint c = 0; c < a.C2; ++c) {
    float s = 0;
    for (uint h = 0; h < a.H; ++h)
      s += a.attn[((ulong)q * a.H + h) * a.L + k] * a.df[(ulong)q * Fw + off + h * a.C2 + c];
    if (a.accum) d[c] += s; else d[c] = s;
  }
}
