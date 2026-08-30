import { describe, expect, it, vi } from "./harness.js";
import { extractMmseqs2A3m, generateMmseqs2Msa, readTarFiles } from "../src/input/mmseqs2-api.js";

function tar(files) {
  const chunks= [];
  for (const [name, value] of Object.entries(files)) {
    const header = new Uint8Array(512); const encodedName = new TextEncoder().encode(name);
    header.set(encodedName, 0);
    header.set(new TextEncoder().encode(value.length.toString(8).padStart(11, "0") + "\0"), 124);
    header[156] = 48;
    const data = new TextEncoder().encode(value); const padded = new Uint8Array(Math.ceil(data.length / 512) * 512); padded.set(data);
    chunks.push(header, padded);
  }
  chunks.push(new Uint8Array(1024));
  const output = new Uint8Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
  let offset = 0; for (const chunk of chunks) { output.set(chunk, offset); offset += chunk.length; }
  return output;
}

const resultTar = tar({
  "uniref.a3m": ">101\nACDE\n>hit1\nAC-E\n\0",
  "bfd.mgnify30.metaeuk30.smag30.a3m": ">101\nACDE\n>env1\nAcCDE\n\0",
});

describe("MMseqs2 API", () => {
  it("reads files and combines the ColabFold UniRef and environmental A3Ms", () => {
    expect([...readTarFiles(resultTar).keys()]).toEqual(["uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"]);
    expect(extractMmseqs2A3m(resultTar)).toBe(">101\nACDE\n>hit1\nAC-E\n>101\nACDE\n>env1\nAcCDE\n");
  });

  it("submits, polls, downloads, and validates a generated MSA", async () => {
    const requests= [];
    const responses = [
      new Response(JSON.stringify({ status: "PENDING", id: "ticket-1" })),
      new Response(JSON.stringify({ status: "RUNNING" })),
      new Response(JSON.stringify({ status: "COMPLETE" })),
      new Response(new Uint8Array([1, 2, 3])),
    ];
    const fetchImplementation = vi.fn(async (input, init) => {
      requests.push({ url: String(input), ...(init === undefined ? {} : { init }) });
      return responses.shift() ;
    });
    const phases= [];
    const result = await generateMmseqs2Msa("ACDE", {
      fetchImplementation, wait: async () => {}, decompress: async () => resultTar,
      onProgress: (progress) => phases.push(progress.phase),
    });
    expect(result.ticket).toBe("ticket-1"); expect(result.depth).toBe(4);
    expect(requests.map((entry) => entry.url)).toEqual([
      "https://api.colabfold.com/ticket/msa", "https://api.colabfold.com/ticket/ticket-1",
      "https://api.colabfold.com/ticket/ticket-1", "https://api.colabfold.com/result/download/ticket-1",
    ]);
    expect(String(requests[0] .init?.body)).toContain("mode=env");
    expect(phases).toEqual(["submitting", "queued", "running", "downloading", "complete"]);
  });
});
