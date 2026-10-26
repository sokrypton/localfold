import { readFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { describe, expect, it } from "./harness.js";
import { gunzip } from "../shared/rfd4p/pipeline/inflate.js";
import { parseCifToAtomArray } from "../shared/rfd4p/pipeline/cif.js";
import { decompress } from "../shared/rfd4p/vendor/fzstd.js";

/**
 * The structure formats upstream takes besides plain mmCIF: gzip and BinaryCIF (and Zstandard,
 * which the shipped examples already are). tools/check-rfd4p-load.py holds each to atomworks;
 * this holds them to each other without upstream's venv - a gzipped file and RCSB's own BinaryCIF
 * must read to the SAME atoms as the mmCIF they encode.
 */
const fixture = (name) => decompress(new Uint8Array(readFileSync(new URL(`../tools/fixtures/rcsb/${name}`, import.meta.url))));

describe("the structure formats", () => {
  it("inflates what zlib deflates, at every level", () => {
    const cif = fixture("1brs.cif.zst");
    for (const level of [0, 1, 6, 9]) {
      const back = gunzip(new Uint8Array(gzipSync(cif, { level })));
      expect(back.length).toBe(cif.length);
      expect(back.every((v, i) => v === cif[i])).toBe(true);
    }
  });

  it("reads a gzipped mmCIF and RCSB's BinaryCIF as the mmCIF itself", () => {
    for (const id of ["6mrr", "7m5w"]) {
      const cif = fixture(`${id}.cif.zst`);
      const want = parseCifToAtomArray(cif);
      for (const [label, bytes] of [["gzip", new Uint8Array(gzipSync(cif))], ["bcif", fixture(`${id}.bcif.zst`)]]) {
        const got = parseCifToAtomArray(bytes);
        expect(`${id} ${label} ${got.nAtoms}`).toBe(`${id} ${label} ${want.nAtoms}`);
        for (const key of ["chain_id", "res_id", "res_name", "atom_name", "element", "charge", "occupancy", "b_factor"]) {
          const bad = want[key].findIndex((v, i) => !Object.is(v, got[key][i]) && !(v === got[key][i]));
          expect(`${id} ${label} ${key} ${bad}`).toBe(`${id} ${label} ${key} -1`);
        }
        const coordBad = want.coord.findIndex((c, i) => c.some((x, k) => x !== got.coord[i][k]));
        expect(`${id} ${label} coord ${coordBad}`).toBe(`${id} ${label} coord -1`);
      }
    }
  });
});
