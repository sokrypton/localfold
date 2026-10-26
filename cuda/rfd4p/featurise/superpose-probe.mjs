// The same points through `output.js`'s `superimposeCoords3D`.
import { superimposeCoords3D } from "../../../shared/rfd4p/output.js";

const [seed, nStr, noiseStr, reflectStr] = process.argv.slice(2);
let s = Number(seed) >>> 0;
const next = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return (s >>> 8) / 16777216; };
const n = Number(nStr);
const noise = Number(noiseStr);
// 🔴 A REFLECTED SET IS THE ONLY WAY det(H) GOES NEGATIVE, which is where a Kabsch that
// can return an improper rotation does (test/rfd4p-superpose.test.js holds the page's to
// Horn's quaternion method there; this probe holds the native one to the page's).
const flip = reflectStr === '1' ? -1 : 1;
const fixed = new Float64Array(n * 3);
const mobile = new Float64Array(n * 3);
for (let i = 0; i < n * 3; i += 1) fixed[i] = next() * 40 - 20;
const c = 0.8253356149;
const sn = 0.5646424734;
for (let i = 0; i < n; i += 1) {
  const x = fixed[i * 3];
  const y = fixed[i * 3 + 1];
  const z = fixed[i * 3 + 2] * flip;
  mobile[i * 3] = c * x - sn * y + 3.5 + (next() - 0.5) * noise;
  mobile[i * 3 + 1] = sn * x + c * y - 7.25 + (next() - 0.5) * noise;
  mobile[i * 3 + 2] = z + 11.0 + (next() - 0.5) * noise;
}
const { rotation, fixedCenter, mobileCenter } = superimposeCoords3D(fixed, mobile);
console.log(`${[...rotation, ...fixedCenter, ...mobileCenter].map((v) => v.toFixed(12)).join(" ")} `);
