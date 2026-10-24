import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ENGINE_PATH = path.resolve(__dirname, '../webgpu/rfd4p/engine.js');

test('Every WGSL compute shader in webgpu/rfd4p/engine.js folds wid.x + wid.y * 65535u and guards bounds', () => {
  const src = fs.readFileSync(ENGINE_PATH, 'utf8');
  const computeCount = [...src.matchAll(/@compute\s+@workgroup_size/g)].length;
  const foldCount = [...src.matchAll(/wid\.x\s*\+\s*wid\.y\s*\*\s*65535u/g)].length;

  assert.ok(computeCount >= 6, `Expected at least 6 compute shader templates in engine.js, found ${computeCount}`);
  assert.ok(
    foldCount >= computeCount,
    `Every @compute shader must fold wid.x + wid.y * 65535u (${foldCount} vs ${computeCount})`
  );

  // Verify each shader section contains an explicit bounds guard (`inBatch`, `live`, or `i >= p.count`)
  const shaders = src.split('@compute').slice(1);
  for (let i = 0; i < shaders.length; i += 1) {
    const body = shaders[i];
    assert.ok(
      body.includes('wid.x + wid.y * 65535u'),
      `Shader #${i + 1} must compute folded workgroup index 'wid.x + wid.y * 65535u'`
    );
    assert.ok(
      body.includes('if (!inBatch) { return; }') ||
        body.includes('if (!live) { return; }') ||
        body.includes('if (live &&') ||
        body.includes('if (i >= p.count) { return; }'),
      `Shader #${i + 1} must guard folded workgroup index before writing storage buffers`
    );
  }
});
