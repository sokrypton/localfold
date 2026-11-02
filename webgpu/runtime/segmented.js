/**
 * A tensor longer than one buffer holds or one binding can name, kept as a run of buffers of whole rows.
 *
 * D3D12 allows 2 GiB a buffer and every backend about 2 GiB a binding, which is a pair tensor at 1448 tokens whatever the
 * card. Almost nothing binds a pair tensor whole: every stage walks it in chunks of rows a small fraction of a buffer
 * long. So past that size it is several buffers, and a chunk is bound from whichever one holds it. A segment's rows are a
 * power of two, which every chunk size in this port divides, so no chunk crosses a boundary; one that did would throw
 * rather than read the wrong rows. The few kernels that read a pair across rows take every segment: `segmentedRead`.
 */

/** 1 GiB of ESMFold2's 256-channel pair: four of them reach the longest fold two channel halves can bind. */
export const SEGMENT_ROWS = 2 ** 20;

export class SegmentedAllocation {
  constructor(allocator, label, rows, rowBytes, usage, segmentRows) {
    if (!Number.isInteger(Math.log2(segmentRows))) {
      throw new RangeError(`a segment of ${segmentRows} rows: a kernel finds its segment by shifting`);
    }
    this.rows = rows;
    this.rowBytes = rowBytes;
    this.segmentRows = segmentRows;
    this.segments = [];
    try {
      for (let first = 0; first < rows; first += segmentRows) {
        this.segments.push(allocator.allocate(`${label}.${this.segments.length}`,
                                               Math.min(segmentRows, rows - first) * rowBytes, usage));
      }
    } catch (cause) {
      this.release();
      throw cause;
    }
  }

  /** The only buffer, for the few callers that want one. More than one is a caller that has not been converted. */
  get buffer() {
    if (this.segments.length !== 1) {
      throw new Error("this tensor is several buffers: bind it a chunk at a time, through at()");
    }
    return this.segments[0].buffer;
  }

  /** The binding for bytes [byteOffset, byteOffset + byteSize), which have to lie in one segment. */
  at(byteOffset, byteSize) {
    const segmentBytes = this.segmentRows * this.rowBytes;
    const index = Math.floor(byteOffset / segmentBytes);
    const within = byteOffset - index * segmentBytes;
    if (within + byteSize > segmentBytes || index >= this.segments.length) {
      throw new RangeError(`bytes ${byteOffset}+${byteSize} cross a segment of ${segmentBytes}`);
    }
    return { buffer: this.segments[index].buffer, byteOffset: within, byteSize };
  }

  clear(encoder) {
    for (const segment of this.segments) encoder.clearBuffer(segment.buffer);
  }

  release() {
    for (let at = this.segments.length - 1; at >= 0; at -= 1) this.segments[at].release();
    this.segments = [];
  }
}

/** A range of an allocation of either kind, as a pass's buffer list takes it. */
export function rangeOf(allocation, byteOffset, byteSize) {
  return allocation.at?.(byteOffset, byteSize)
    ?? { buffer: allocation.buffer, byteOffset: (allocation.byteOffset ?? 0) + byteOffset, byteSize };
}

/** What a tensor of this length should be cut to on this device: `rows`, which is no cut, wherever one buffer will do. */
export function segmentRowsFor(device, rows, rowBytes) {
  const ceiling = Math.min(device.limits?.maxStorageBufferBindingSize ?? Infinity,
                           device.limits?.maxBufferSize ?? Infinity);
  if (rows * rowBytes <= ceiling) return rows;
  let segmentRows = SEGMENT_ROWS;
  while (segmentRows * rowBytes > ceiling) segmentRows /= 2;
  return segmentRows;
}

export function allocatePairSized(allocator, label, rows, rowBytes, usage, segmentRows = rows) {
  return segmentRows >= rows ? allocator.allocate(label, rows * rowBytes, usage)
    : new SegmentedAllocation(allocator, label, rows, rowBytes, usage, segmentRows);
}

/**
 * WGSL for a kernel that reads a segmented tensor across rows: `${name}_at(element)`, over one binding a segment.
 * The first binding is the one the kernel has always used for this tensor, and the rest go after its others.
 */
export function segmentedRead(name, { firstBinding, moreBindingsFrom, segments, elementsPerSegment }) {
  const shift = Math.log2(elementsPerSegment);
  if (!Number.isInteger(shift)) throw new RangeError(`${elementsPerSegment} elements a segment is not a power of two`);
  const bindings = Array.from({ length: segments }, (_, at) => (at === 0 ? firstBinding : moreBindingsFrom + at - 1));
  return `${bindings.map((binding, at) => `@group(0) @binding(${binding}) var<storage, read> ${name}${at}: array<f32>;`).join("\n")}
fn ${name}_at(element: u32) -> f32 {
  let within = element & ${elementsPerSegment - 1}u;
  switch (element >> ${shift}u) {
${bindings.slice(0, -1).map((_, at) => `    case ${at}u: { return ${name}${at}[within]; }`).join("\n")}
    default: { return ${name}${segments - 1}[within]; }
  }
}`;
}

/** The `{buffer}` of each segment, for a pass that binds them all in the order above: the first, its others, the rest. */
export const segmentBuffers = (allocation) => allocation.segments.map((segment) => ({ buffer: segment.buffer }));
