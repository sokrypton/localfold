/**
 * Float32Arrays a kernel has finished with, for the next one that asks for the same length.
 *
 * A block of the CPU's pair stack asks for pair-sized arrays dozens of times, none of them outlives the block, and
 * each is zeroed by the allocator and left for a collector that runs when it gets round to it: a fold at 600 residues
 * reached 14 GiB of resident memory where it needed about 6. So a fold keeps them, for as long as it runs.
 *
 * On only while a fold asks: everywhere else `floats` is `new Float32Array` and `recycle` does nothing.
 */
let free;

export function startRecycling() { free = new Map(); }
export function stopRecycling() { free = undefined; }

/** An array of this length, of whatever a kernel last left in it: the caller writes every element it goes on to read. */
export function floats(length) {
  return free?.get(length)?.pop() ?? new Float32Array(length);
}

/** Hand arrays back. The caller holds no reference and no view of them afterwards. */
export function recycle(...arrays) {
  if (free === undefined) return;
  for (const array of arrays) {
    if (array === undefined || array.buffer.byteLength !== array.byteLength) continue;
    const pool = free.get(array.length);
    if (pool === undefined) free.set(array.length, [array]);
    else pool.push(array);
  }
}
