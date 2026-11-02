/**
 * A device tensor held in host memory while something else needs the card.
 *
 * A stage that only reads a pair-sized tensor once a loop, or not until the end of the fold, does not need it on the
 * device in between: reading it back once and writing it back when it is wanted costs seconds of a fold that takes
 * minutes at the sizes where this runs, and gives back a whole pair-sized tensor.
 *
 * Held in pieces, because a browser refuses one ArrayBuffer past about a gigabyte. A piece is read through a mapped
 * buffer of its own size, so the spill itself costs the card one piece and not one tensor.
 */
export const SPILL_PIECE_BYTES = 64 * 1024 * 1024;

export class HostSpill {
  constructor(byteLength, pieceBytes, pieces) {
    this.byteLength = byteLength;
    this.pieceBytes = pieceBytes;
    this.pieces = pieces;
  }

  /** Queue bytes [byteOffset, byteOffset + byteSize) of the tensor into `buffer` at `bufferOffset`. */
  writeTo(device, buffer, bufferOffset, byteOffset, byteSize) {
    if (byteOffset < 0 || byteOffset + byteSize > this.byteLength) {
      throw new RangeError(`bytes ${byteOffset}+${byteSize} of a spill of ${this.byteLength}`);
    }
    let written = 0;
    while (written < byteSize) {
      const at = byteOffset + written;
      const index = Math.floor(at / this.pieceBytes);
      const within = at - index * this.pieceBytes;
      const count = Math.min(byteSize - written, this.pieceBytes - within);
      // The piece's ArrayBuffer and not the typed array, whose offset and size writeBuffer takes in elements.
      device.queue.writeBuffer(buffer, bufferOffset + written, this.pieces[index].buffer, within, count);
      written += count;
    }
  }

  /** Back into a whole buffer, a piece at a time with a drain between, so the staging memory never holds the tensor. */
  async restore(device, buffer) {
    for (let index = 0; index < this.pieces.length; index += 1) {
      device.queue.writeBuffer(buffer, index * this.pieceBytes, this.pieces[index]);
      // eslint-disable-next-line no-await-in-loop
      if (index % 4 === 3) await device.queue.onSubmittedWorkDone();
    }
    await device.queue.onSubmittedWorkDone();
  }
}

/**
 * Read `byteLength` bytes of `source` into host pieces.
 *
 * @param {GPUBuffer} source needs COPY_SRC
 */
export async function spillToHost(device, allocator, source, byteLength, label, pieceBytes = SPILL_PIECE_BYTES) {
  if (byteLength % 4 !== 0 || pieceBytes % 4 !== 0) {
    throw new RangeError(`a spill of ${byteLength} bytes in pieces of ${pieceBytes}: neither may split a float`);
  }
  const size = Math.min(pieceBytes, byteLength);
  const staging = allocator.allocate(`${label}-spill`, size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const pieces = [];
  try {
    for (let offset = 0; offset < byteLength; offset += size) {
      const count = Math.min(size, byteLength - offset);
      const encoder = device.createCommandEncoder({ label: `${label}-spill` });
      encoder.copyBufferToBuffer(source, offset, staging.buffer, 0, count);
      device.queue.submit([encoder.finish()]);
      // eslint-disable-next-line no-await-in-loop
      await staging.buffer.mapAsync(GPUMapMode.READ, 0, count);
      pieces.push(new Float32Array(staging.buffer.getMappedRange(0, count).slice(0)));
      staging.buffer.unmap();
    }
  } finally {
    staging.release();
  }
  return new HostSpill(byteLength, size, pieces);
}
