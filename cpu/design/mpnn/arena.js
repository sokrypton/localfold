// A tiny named-slot allocator.
//
// The layers are called in a strict sequence and never hold a buffer across
// calls, so scratch space can be shared by role instead of per layer. Without
// this, three encoder layers at L = 1000, K = 48 would each want ~100 MB of
// their own intermediates.

export class Arena {
  constructor() {
    /** @type {Map<string, Float32Array>} */
    this.slots = new Map();
  }

  /**
   * A Float32Array of exactly `n` elements for the given role. The backing
   * store grows monotonically and is reused across calls.
   */
  f32(name, n) {
    let buf = this.slots.get(name);
    if (buf === undefined || buf.length < n) {
      buf = new Float32Array(n);
      this.slots.set(name, buf);
    }
    return buf.length === n ? buf : buf.subarray(0, n);
  }

  /** Same, but zeroed. */
  zeros(name, n) {
    const buf = this.f32(name, n);
    buf.fill(0);
    return buf;
  }

  /** Total bytes currently held, for reporting. */
  bytes() {
    let total = 0;
    for (const buf of this.slots.values()) total += buf.buffer.byteLength;
    return total;
  }

  release() {
    this.slots.clear();
  }
}
