// tools/oracle/dump_bioemu2.py's dumps, in a browser: every array is {shape, dtype, b64}, little-endian float32 or
// int32 (tools/bioemu2-dump.js is the node twin).
export function decodeArray(entry) {
  const text = atob(entry.b64);
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i += 1) bytes[i] = text.charCodeAt(i);
  const data = entry.dtype === "i32" ? new Int32Array(bytes.buffer) : new Float32Array(bytes.buffer);
  return Object.assign(data, { shape: entry.shape });
}

export async function fetchBioemu2Dump(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const dump = await response.json();
  const stages = {};
  for (const [key, value] of Object.entries(dump.stages)) {
    stages[key] = value && typeof value === "object" && "b64" in value ? decodeArray(value) : value;
  }
  return { mode: dump.mode, stages };
}

/** A dump's feature dict, as webgpu/bioemu2 reads it: the MSA as an index, scalars as numbers. */
export function featuresOf(stages) {
  const f = {};
  for (const [key, value] of Object.entries(stages)) {
    if (key.startsWith("feats.")) f[key.slice(6)] = value;
  }
  const index = new Int32Array(f.msa.length / 33);
  for (let i = 0; i < index.length; i += 1) for (let k = 0; k < 33; k += 1) if (f.msa[i * 33 + k]) index[i] = k;
  f.msa = index;
  f.dt = f.dt[0];
  f.is_dynamics_conditioned = f.is_dynamics_conditioned[0];
  return f;
}

/** relative RMS of a against b over `count` values, refusing to compare nothing. */
export function relRms(a, b, count = b?.length) {
  if (a?.length === undefined || b?.length === undefined || a.length < count || b.length < count || !count) {
    throw new Error(`relRms over ${a?.length} and ${b?.length}`);
  }
  let num = 0;
  let den = 0;
  for (let i = 0; i < count; i += 1) {
    const d = a[i] - b[i];
    if (!Number.isFinite(d)) return Infinity;
    num += d * d;
    den += b[i] * b[i];
  }
  return Math.sqrt(num / Math.max(den, 1e-30));
}
