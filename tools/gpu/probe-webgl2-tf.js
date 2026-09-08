/** The smallest transform feedback that works, and where the error comes from. */
export async function main() {
  const gl = new OffscreenCanvas(4, 4).getContext("webgl2");
  const E = (where) => { const e = gl.getError(); return e === 0 ? null : `${where}:${e}`; };
  const errors = [];
  const vs = `#version 300 es
out vec4 v0;
void main() { v0 = vec4(float(gl_VertexID) + 1.0, 2.0, 3.0, 4.0);
  gl_Position = vec4(0.0, 0.0, 0.0, 1.0); gl_PointSize = 1.0; }`;
  const fs = `#version 300 es
precision highp float; out vec4 c; void main() { c = vec4(0.0); }`;
  const mk = (t, src) => { const s = gl.createShader(t); gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
  gl.transformFeedbackVaryings(p, ["v0"], gl.INTERLEAVED_ATTRIBS);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  gl.useProgram(p);
  errors.push(E("afterLink"));

  const n = 4;
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, n * 4 * 4, gl.STATIC_COPY);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  errors.push(E("afterAlloc"));

  const tf = gl.createTransformFeedback();
  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, tf);
  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, buf);
  gl.enable(gl.RASTERIZER_DISCARD);
  gl.beginTransformFeedback(gl.POINTS);
  gl.drawArrays(gl.POINTS, 0, n);
  gl.endTransformFeedback();
  gl.disable(gl.RASTERIZER_DISCARD);
  errors.push(E("afterDraw"));

  // Release EVERY binding this buffer has before reading it.
  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
  errors.push(E("afterUnbind"));

  const out = new Float32Array(n * 4);
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.getBufferSubData(gl.ARRAY_BUFFER, 0, out);
  errors.push(E("afterRead"));
  return { values: [...out], errors: errors.filter((e) => e !== null),
    expected: "1,2,3,4, 2,2,3,4, 3,2,3,4, 4,2,3,4" };
}
