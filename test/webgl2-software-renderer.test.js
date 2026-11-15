/**
 * A WebGL2 drawn on the CPU is recognised by name - the page refuses to fold on it (web/model.js `foldBackend`),
 * because a fold there is hundreds of times slower and reads as a hang (docs/FIREFOX.md, "Linux").
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { SOFTWARE_RENDERER } from "../webgl/af3/model/fold-batch.js";

describe("software WebGL2 renderers", () => {
  it("names the CPU renderers as browsers report them", () => {
    for (const name of [
      "llvmpipe, or similar",                                          // Firefox, sanitised
      "llvmpipe (LLVM 15.0.7, 256 bits)",
      "ANGLE (Mesa, llvmpipe (LLVM 15.0.7 256 bits), OpenGL 4.5)",     // Chrome on Mesa
      "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)",
      "Microsoft Basic Render Driver",
    ]) assert.ok(SOFTWARE_RENDERER.test(name), name);
  });
  it("leaves every GPU alone", () => {
    for (const name of [
      "ANGLE (NVIDIA Corporation, NVIDIA A100-SXM4-40GB/PCIe/SSE2, OpenGL ES 3.2)",
      "ANGLE (NVIDIA, Vulkan 1.4.312 (NVIDIA NVIDIA A100-SXM4-40GB (0x000020B0)), NVIDIA)",
      "ANGLE (Intel, Mesa Intel(R) UHD Graphics 620 (KBL GT2), OpenGL 4.6)",
      "ANGLE (AMD, AMD Radeon RX 6700 XT (radeonsi, navi22, LLVM 15.0.7, DRM 3.49), OpenGL 4.6)",
      "Apple M2", "ANGLE (Apple, ANGLE Metal Renderer: Apple M2, Unspecified Version)",
      "Mali-G78", "Adreno (TM) 740",
    ]) assert.ok(!SOFTWARE_RENDERER.test(name), name);
  });
});
