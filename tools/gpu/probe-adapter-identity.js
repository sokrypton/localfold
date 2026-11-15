// What a browser says about its GPU, by every route a page has: adapter.info, the adapter's features, the platform
// strings and WebGL's renderer. Firefox leaves adapter.info blank, so a device prior has to be chosen from the rest.
//   node tools/gpu-chrome.mjs tools/gpu/probe-adapter-identity.js
export async function main() {
  const adapter = await navigator.gpu.requestAdapter();
  const info = adapter.info ?? {};
  const out = {
    info: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description },
    features: [...adapter.features].sort(),
    platform: navigator.platform, userAgent: navigator.userAgent,
    uaDataPlatform: navigator.userAgentData?.platform ?? null,
    preferredFormat: navigator.gpu.getPreferredCanvasFormat(),
    limits: { binding: adapter.limits.maxStorageBufferBindingSize, buffer: adapter.limits.maxBufferSize,
      workgroupStorage: adapter.limits.maxComputeWorkgroupStorageSize },
  };
  const canvas = typeof OffscreenCanvas === "function" ? new OffscreenCanvas(1, 1) : document.createElement("canvas");
  const gl = canvas.getContext("webgl2");
  if (gl) {
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    out.webgl = { renderer: gl.getParameter(gl.RENDERER), vendor: gl.getParameter(gl.VENDOR),
      unmasked: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : null };
  }
  return out;
}
