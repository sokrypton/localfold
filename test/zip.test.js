import { describe, expect, it } from "./harness.js";
import { looksLikeZip, readZip, writeZip } from "../web/zip.js";

// A member big enough that deflate has something to do, and repetitive in the
// way an a3m is - long runs of gaps - so the compressed arm is exercised rather
// than falling through to stored because deflate made it bigger.
const alignment = `>query\n${"ACDEFGHIKL".repeat(12)}\n`
  + Array.from({ length: 60 }, (_, row) =>
    `>hit_${row} OS=Something OX=${row}\n${"-".repeat(row)}${"ACDEFGHIKL".repeat(10)}\n`).join("");

describe("zip", () => {
  const members = [
    ["job.json", '{"name":"a fold"}'],
    ["msas/paired_chains_a.a3m", alignment],
    ["msas/unpaired_chains_a.a3m", alignment.toUpperCase()],
  ];

  it("reads back exactly what it wrote", async () => {
    const files = await readZip(await writeZip(members));
    expect([...files.keys()]).toEqual(members.map(([name]) => name));
    for (const [name, content] of members) expect(files.get(name)).toBe(content);
  });

  // 🔴 THE STORED PATH IS NOT HYPOTHETICAL. `CompressionStream` is missing in
  // older browsers, so this is the arm a reader on one of them gets - and an
  // arm that only runs where the feature is absent is an arm that is never
  // tested. Forcing it is the only way it is covered here.
  it("reads back a stored archive too", async () => {
    const archive = await writeZip(members, { store: true });
    const files = await readZip(archive);
    for (const [name, content] of members) expect(files.get(name)).toBe(content);
  });

  it("compresses, so an archive of alignments is not the sum of its members", async () => {
    const deflated = await writeZip(members);
    const stored = await writeZip(members, { store: true });
    expect(deflated.length < stored.length / 2).toBe(true);
  });

  // ...a fold archive carries a3m text, which is ASCII, but a description line
  // out of UniProt is not always: a species name can carry an accent, and a
  // reader that counts characters rather than bytes truncates the member.
  it("keeps non-ASCII text intact", async () => {
    const text = ">sp|P00000|Ostrinia nubilalis (Européenne) · 40°\nACDEF\n";
    const files = await readZip(await writeZip([["msas/a.a3m", text]]));
    expect(files.get("msas/a.a3m")).toBe(text);
  });

  it("survives an empty member and an empty archive", async () => {
    const files = await readZip(await writeZip([["empty.txt", ""]]));
    expect(files.get("empty.txt")).toBe("");
    expect((await readZip(await writeZip([]))).size).toBe(0);
  });

  it("takes a Map as readily as pairs", async () => {
    const files = await readZip(await writeZip(new Map(members)));
    expect([...files.keys()]).toEqual(members.map(([name]) => name));
  });

  it("knows an archive from an alignment", async () => {
    expect(looksLikeZip(await writeZip(members))).toBe(true);
    expect(looksLikeZip(new TextEncoder().encode(alignment))).toBe(false);
    expect(looksLikeZip(new Uint8Array([0x50]))).toBe(false);
  });

  it("says so when handed something that is not an archive", async () => {
    let thrown = null;
    try {
      await readZip(new TextEncoder().encode(alignment));
    } catch (error) {
      thrown = error;
    }
    expect(thrown?.message).toContain("not a zip archive");
  });
});
