// An exporter kept loaded between requests:
//
//   node --js-float16array native/export_server.mjs <exporter.mjs> <dir>
//
// Loading an exporter's modules is most of a run (~180 ms of AF3's 220 ms export), so this imports the
// exporter once a request with that request's arguments - `${exporter}?request=N`, a new instance of the
// exporter's own module while every module it imports stays cached. A request is <dir>/<id>.req (renamed
// into place), a JSON list of the exporter's arguments; what it printed goes to <id>.log, and it ends with
// <id>.ok, or <id>.err holding the error's message and stack. A request of ["quit"] stops the server.
// tools/native_worker.py and native/af3/fold use it.
import { readFileSync, readdirSync, unlinkSync, writeFileSync, renameSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [exporter, dir] = process.argv.slice(2);
if (!dir) { console.error("usage: export_server.mjs <exporter.mjs> <dir>"); process.exit(1); }
const url = pathToFileURL(resolve(exporter)).href;
const argv0 = [process.argv[0], resolve(exporter)];
const write = { out: process.stdout.write.bind(process.stdout), err: process.stderr.write.bind(process.stderr) };
console.log(`export: serving ${dir}`);
for (let n = 0; ; ) {
  const reqs = readdirSync(dir).filter((f) => f.endsWith(".req")).sort();
  if (reqs.length === 0) { await new Promise((r) => setTimeout(r, 2)); continue; }
  const id = reqs[0].slice(0, -4);
  const request = JSON.parse(readFileSync(`${dir}/${id}.req`, "utf8"));
  unlinkSync(`${dir}/${id}.req`);
  if (request[0] === "quit") process.exit(0);
  process.argv = [...argv0, ...request];
  // the request's own output, as a run of the exporter would have printed it
  let said = "";
  process.stdout.write = process.stderr.write = (chunk, ...rest) => {
    said += typeof chunk === "string" ? chunk : Buffer.from(chunk).toString();
    const done = rest.find((r) => typeof r === "function");
    if (done) done();
    return true;
  };
  let failed;
  try {
    await import(`${url}?request=${n++}`);
  } catch (error) {
    failed = error;
  }
  process.stdout.write = write.out; process.stderr.write = write.err;
  writeFileSync(`${dir}/${id}.log`, said);
  const end = failed === undefined ? "ok" : "err";
  writeFileSync(`${dir}/${id}.${end}.tmp`,
    failed === undefined ? "" : `${failed?.constructor?.name ?? "Error"}: ${failed?.message ?? failed}\n${failed?.stack ?? ""}\n`);
  renameSync(`${dir}/${id}.${end}.tmp`, `${dir}/${id}.${end}`);
}
