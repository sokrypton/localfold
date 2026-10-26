/**
 * Python's `random.Random(seed)`, as far as atomworks draws from it: MT19937 seeded through
 * `init_by_array`, `getrandbits(k)` for k <= 32, `_randbelow` by rejection, and `shuffle`.
 *
 * 🔴 BIT FOR BIT, BECAUSE THE DRAWS DECIDE WHICH ATOMS EXIST. Upstream picks a raw structure's
 * alternate conformers with `random.Random(0)` (`select_altlocs_clash_aware`), so a generator
 * that is merely a good one picks other conformers and loads a different structure. Every call
 * here is CPython's (`Modules/_randommodule.c`, `Lib/random.py`) in the same order.
 */

const N = 624;
const M = 397;

export class PyRandom {
  constructor(seed) {
    this.mt = new Uint32Array(N);
    this.mti = N + 1;
    // `random_seed`: an int seed is its absolute value as 32-bit words, least significant first;
    // zero is the one-word key [0].
    let n = BigInt(Math.abs(Math.trunc(seed)));
    const key = [];
    do {
      key.push(Number(n & 0xffffffffn));
      n >>= 32n;
    } while (n > 0n);
    this.initByArray(key);
  }

  initGenrand(s) {
    const mt = this.mt;
    mt[0] = s >>> 0;
    for (let i = 1; i < N; i += 1) {
      const prev = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = (Math.imul(1812433253, prev) + i) >>> 0;
    }
    this.mti = N;
  }

  initByArray(key) {
    const mt = this.mt;
    this.initGenrand(19650218);
    let i = 1;
    let j = 0;
    for (let k = Math.max(N, key.length); k > 0; k -= 1) {
      const prev = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = ((mt[i] ^ Math.imul(prev, 1664525)) + key[j] + j) >>> 0;
      i += 1;
      j += 1;
      if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
      if (j >= key.length) j = 0;
    }
    for (let k = N - 1; k > 0; k -= 1) {
      const prev = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = ((mt[i] ^ Math.imul(prev, 1566083941)) - i) >>> 0;
      i += 1;
      if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
    }
    mt[0] = 0x80000000;
  }

  genrandUint32() {
    const mt = this.mt;
    if (this.mti >= N) {
      let kk = 0;
      const mag = (y) => ((y & 1) ? 0x9908b0df : 0);
      for (; kk < N - M; kk += 1) {
        const y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff);
        mt[kk] = (mt[kk + M] ^ (y >>> 1) ^ mag(y)) >>> 0;
      }
      for (; kk < N - 1; kk += 1) {
        const y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff);
        mt[kk] = (mt[kk + (M - N)] ^ (y >>> 1) ^ mag(y)) >>> 0;
      }
      const y = (mt[N - 1] & 0x80000000) | (mt[0] & 0x7fffffff);
      mt[N - 1] = (mt[M - 1] ^ (y >>> 1) ^ mag(y)) >>> 0;
      this.mti = 0;
    }
    let y = mt[this.mti];
    this.mti += 1;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  /** `getrandbits(k)` for 1 <= k <= 32. */
  getrandbits(k) {
    return this.genrandUint32() >>> (32 - k);
  }

  /** `_randbelow_with_getrandbits(n)`: k = n.bit_length(), redraw until below n. */
  randbelow(n) {
    const k = 32 - Math.clz32(n);
    let r = this.getrandbits(k);
    while (r >= n) r = this.getrandbits(k);
    return r;
  }

  /** `shuffle(x)`, in place: from the top down, j = randbelow(i + 1). */
  shuffle(x) {
    for (let i = x.length - 1; i > 0; i -= 1) {
      const j = this.randbelow(i + 1);
      [x[i], x[j]] = [x[j], x[i]];
    }
    return x;
  }
}
