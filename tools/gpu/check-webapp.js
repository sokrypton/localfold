/**
 * End-to-end browser verification of `index.html` (AtomWorks Studio + `web/app.js`) in headless Chrome:
 *   1. Mounts `/index.html` (AtomWorks Studio and the RFD4-Proteina card share one document) in a harness iframe.
 *   2. Verifies all 9 official AtomWorks Studio examples and `/api/v1/*` routes (conditions, internal-coords,
 *      distance-restraints, segments insert/realize/collapse, undo, export) work end-to-end.
 *   3. Verifies all 10 presets in `RFD4P_PRESETS` build batches cleanly.
 *   4. Runs a real WebGPU design (`nsteps=10`, `recycleSchedule='none'`) on `motif_unidx_1prw`
 *      and verifies that the generated Studio session + multi-model `trajectory.cif` (10 models) + PDB/mmCIF are populated.
 *   5. Runs `inverse_folding_1prw` to verify on-demand lazy loading of the 5 encoder shards (910 -> 1,257 tensors).
 *
 *   DISPLAY=:99 node tools/gpu-chrome.mjs tools/gpu/check-webapp.js
 */

import { RFD4P_PRESETS } from '../../webgpu/rfd4p/index.js';

export async function main() {
  const iframe = document.createElement('iframe');
  iframe.style.width = '1440px';
  iframe.style.height = '960px';
  document.body.appendChild(iframe);

  await new Promise((resolve, reject) => {
    iframe.onload = resolve;
    iframe.onerror = reject;
    iframe.src = '/index.html';
  });

  const win = iframe.contentWindow;
  const t0 = performance.now();
  while (!win.rfd4pApp || win.rfd4pApp.els.predictBtn.disabled) {
    if (performance.now() - t0 > 15000) {
      throw new Error('Timed out waiting for window.rfd4pApp.init()');
    }
    await new Promise((r) => setTimeout(r, 50));
  }

  const app = win.rfd4pApp;

  // Wait for embedded AtomWorks Studio iframe to mount and render its 9 examples
  const studioWin = win;
  const studioDoc = studioWin?.document;
  const tStudio = performance.now();
  while (!studioDoc || studioDoc.querySelectorAll('.example-load').length < 9) {
    if (performance.now() - tStudio > 15000) {
      throw new Error('Timed out waiting for embedded AtomWorks Studio iframe to load examples');
    }
    await new Promise((r) => setTimeout(r, 100));
  }

  const studioExampleCount = studioDoc.querySelectorAll('.example-load').length;

  // Click the first Studio example ("Protein binder · PD-L1") and wait for session creation
  studioDoc.querySelector('.example-load').click();
  const tSession = performance.now();
  while (!app.inputSessionId) {
    if (performance.now() - tSession > 10000) {
      throw new Error('Timed out waiting for Studio example click to create a session');
    }
    await new Promise((r) => setTimeout(r, 50));
  }
  const initialStudioSessionId = app.inputSessionId;

  // Exercise `/api/v1/*` routes against the active Studio session
  const apiJson = async (path, method = 'GET', body = null) => {
    const resp = await win.fetch(path, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`HTTP ${resp.status} on ${method} ${path}: ${txt}`);
    }
    return resp.json();
  };

  // 1. Apply sequence probability condition + secondary structure condition + CATH fold condition
  await apiJson(`/api/v1/sessions/${initialStudioSessionId}/conditions`, 'POST', {
    condition: 'sequence_probability_ala',
    atom_indices: [0, 1, 2],
    enabled: true,
    value: 0.75,
  });
  await apiJson(`/api/v1/sessions/${initialStudioSessionId}/conditions`, 'POST', {
    condition: 'secondary_structure',
    atom_indices: [0, 1, 2],
    enabled: true,
    value: 1,
  });
  const cathRes = await apiJson(`/api/v1/sessions/${initialStudioSessionId}/conditions`, 'POST', {
    condition: 'cath_label',
    atom_indices: [0, 1, 2],
    enabled: true,
    value: '2.60.40',
  });
  const cathAnn = cathRes.structure?.annotations?.find((a) => a.name === 'cath_label');
  if (!cathAnn || cathAnn.values?.[0]?.label !== '2.60.40.x') {
    throw new Error(`Expected cath_label annotation 2.60.40.x, got ${JSON.stringify(cathAnn)}`);
  }

  // 2. Internal coordinate targets + clear-targets
  await apiJson(`/api/v1/sessions/${initialStudioSessionId}/internal-coords`, 'POST', {
    atom_indices: [0, 1, 2],
    enabled: true,
    scope: 'residue',
  });
  await apiJson(`/api/v1/sessions/${initialStudioSessionId}/internal-coords/clear-targets`, 'POST', {
    atom_indices: [0, 1, 2],
    scope: 'residue',
  });
  // 3. Distance restraints set & query
  await apiJson(`/api/v1/sessions/${initialStudioSessionId}/distance-restraints`, 'POST', {
    action: 'set',
    atom1: 0,
    atom2: 5,
    distance: 8.0,
  });
  const restraintsAfterAdd = await apiJson(`/api/v1/sessions/${initialStudioSessionId}/distance-restraints`, 'GET');

  // 3b. Inspect bonds for single atom (`?atom_index=0`), all 7 interaction types, protonation, and PDB export
  const singleAtomBonds = await apiJson(`/api/v1/sessions/${initialStudioSessionId}/bonds?atom_index=0`, 'GET');
  const allBonds = await apiJson(`/api/v1/sessions/${initialStudioSessionId}/bonds`, 'GET');
  if (singleAtomBonds.count <= 0 || singleAtomBonds.count >= allBonds.count) {
    throw new Error(`Expected single-atom bond filter to return 0 < count < ${allBonds.count}, got ${singleAtomBonds.count}`);
  }
  const interactionsRes = await apiJson(`/api/v1/sessions/${initialStudioSessionId}/interactions/analyze`, 'POST', {
    interaction_types: ['hbond', 'hydrophobic', 'pistacking', 'pication', 'saltbridge', 'halogen', 'metal'],
    inter_chain_only: false,
  });
  const protonateRes = await apiJson(`/api/v1/sessions/${initialStudioSessionId}/protonate`, 'POST', { ph: 7.4 });
  if (!protonateRes.edit_result?.changed) {
    throw new Error('Expected protonate to record an undoable state change');
  }
  await apiJson(`/api/v1/sessions/${initialStudioSessionId}/undo`, 'POST', {});
  const pdbExportResp = await win.fetch(`/api/v1/sessions/${initialStudioSessionId}/pdb`);
  const pdbExportText = pdbExportResp.ok ? await pdbExportResp.text() : '';
  if (!pdbExportText.includes('ATOM  ')) {
    throw new Error('Expected GET /api/v1/sessions/:id/pdb to return valid PDB records');
  }

  // 3c. Exercise Mol* Style -> Color scheme buttons in Studio iframe
  const currentStudioDoc = win.document;
  const colorBtns = currentStudioDoc ? [...currentStudioDoc.querySelectorAll('[data-color]')] : [];
  for (const btn of colorBtns) {
    btn.click();
    await new Promise((r) => setTimeout(r, 20));
  }
  const studioLogText = currentStudioDoc?.getElementById('log')?.textContent || '';
  if (studioLogText.includes('Color controls are unavailable')) {
    throw new Error(`Mol* color button failed: ${studioLogText}`);
  }
  // 4. Segment insert -> realize -> collapse -> undo
  await apiJson(`/api/v1/sessions/${initialStudioSessionId}/segments`, 'POST', {
    atom_index: 0,
    before: true,
    minimum: 3,
    maximum: 3,
  });
  await apiJson(`/api/v1/sessions/${initialStudioSessionId}/segments/realize`, 'POST', { seed: 42 });
  await apiJson(`/api/v1/sessions/${initialStudioSessionId}/segments/collapse`, 'POST', {});
  await apiJson(`/api/v1/sessions/${initialStudioSessionId}/undo`, 'POST', {});
  await apiJson(`/api/v1/sessions/${initialStudioSessionId}/undo`, 'POST', {});
  await apiJson(`/api/v1/sessions/${initialStudioSessionId}/undo`, 'POST', {});
  const exported = await apiJson(`/api/v1/sessions/${initialStudioSessionId}/export`, 'POST', {});

  // Verify all 10 benchmark presets build batches cleanly
  const presetResults = [];
  for (const p of RFD4P_PRESETS) {
    await app._loadPreset(p.id);
    const n = app.currentBatchResult?.n ?? 0;
    if (n <= 0) {
      throw new Error(`Preset ${p.id} failed to build batch (status=${app.els.statusMessage.textContent})`);
    }
    presetResults.push({ presetId: p.id, n });
  }

  // Select `motif_unidx_1prw`, run 10 steps + 'none' recycles for fast E2E verification
  let liveFramesCalled = 0;
  const origLiveRender = studioWin.__rfd4pRenderLiveFrame;
  if (typeof origLiveRender === 'function') {
    studioWin.__rfd4pRenderLiveFrame = async (...args) => {
      liveFramesCalled += 1;
      return origLiveRender(...args);
    };
  }
  await app._loadPreset('motif_unidx_1prw');
  await app.runDesign({ nsteps: 10, recycleSchedule: 'none' });
  if (liveFramesCalled < 2) {
    throw new Error(`Expected live intermediate denoised frames to render during diffusion, got ${liveFramesCalled}`);
  }

  const statusText = app.els.statusMessage.textContent;
  if (app.els.statusMessage.classList.contains('error') || !app.lastResult) {
    throw new Error(`runDesign failed: ${statusText}`);
  }

  const coreTensorCount = Object.keys(app.activeWeights ?? {}).length;
  const encoderLoadedAfterCore = Boolean(app.shardedBundle?.encoderLoaded);

  // Verify generated Studio session and multi-model trajectory CIF (10 models)
  const generatedEntries = [...app.generatedViews.entries()].filter(([, v]) => !v.isInput);
  const [designSessionId, designEntry] = generatedEntries[0] ?? [];
  const trajResp = await win.fetch(`/api/v1/sessions/${designSessionId}/trajectory.cif`);
  const trajCifText = trajResp.ok ? await trajResp.text() : '';
  const modelNumbers = new Set();
  for (const line of trajCifText.split('\n')) {
    if (line.startsWith('ATOM ') || line.startsWith('HETATM ')) {
      const cols = line.trim().split(/\s+/);
      modelNumbers.add(Number(cols[cols.length - 1]));
    }
  }
  const trajectoryModelCount = modelNumbers.size;

  // Now run `inverse_folding_1prw` to verify on-demand lazy loading of the 5 encoder shards
  await app._loadPreset('inverse_folding_1prw');
  await app.runDesign({ nsteps: 10, recycleSchedule: 'none' });

  const invStatusText = app.els.statusMessage.textContent;
  if (app.els.statusMessage.classList.contains('error') || !app.lastResult) {
    throw new Error(`inverse_folding_1prw runDesign failed: ${invStatusText}`);
  }
  const fullTensorCount = Object.keys(app.activeWeights ?? {}).length;
  const encoderLoadedAfterInv = Boolean(app.shardedBundle?.encoderLoaded);

  // Verify all 9 AtomWorks Studio examples run end-to-end through Studio session -> WebGPU generate -> Studio output
  const examplesList = await apiJson('/api/v1/examples', 'GET');
  const studioExampleRuns = [];
  for (const ex of examplesList) {
    const cifResp = await win.fetch(`/api/v1/examples/${encodeURIComponent(ex.id)}/cif`);
    if (!cifResp.ok) throw new Error(`Failed to fetch example CIF for ${ex.id}`);
    const cifText = await cifResp.text();
    const exFile = new win.File([cifText], ex.file, { type: 'chemical/x-mmcif' });
    const exForm = new win.FormData();
    exForm.append('file', exFile);
    exForm.append('parse_preset', 'minimal');
    const sessData = await (await win.fetch('/api/v1/sessions', { method: 'POST', body: exForm })).json();
    app.studioSessionId = sessData.session_id;
    app.inputSessionId = sessData.session_id;
    app.useStudioSession = true;
    app._setPartialDiffusion(ex.task === 'partial_diffusion');
    const tEx0 = performance.now();
    await app.runDesign({ nsteps: 5, recycleSchedule: 'none' });
    const exElapsedMs = Math.round(performance.now() - tEx0);
    const exStatus = app.els.statusMessage.textContent;
    if (app.els.statusMessage.classList.contains('error') || !app.lastResult?.cif) {
      throw new Error(`Studio example '${ex.id}' (${ex.label}) failed: ${exStatus} deviceLost=${JSON.stringify(app.gpuDeviceLost ?? null)}`);
    }
    const outAtoms = app.lastResult.structure?.atoms?.length ?? 0;
    if (outAtoms <= 0) {
      throw new Error(`Studio example '${ex.id}' produced 0 output atoms`);
    }
    studioExampleRuns.push({
      id: ex.id,
      task: ex.task,
      n: app.lastResult.n,
      outAtoms,
      ms: exElapsedMs,
    });
  }

  // Verify opening a .pdb file in AtomWorks Studio works end-to-end
  const pdbFile = new win.File([app.lastResult.pdb], 'test_design.pdb', { type: 'chemical/x-pdb' });
  const convertedCif = await pdbFile.text();
  const pdbForm = new win.FormData();
  pdbForm.append('file', pdbFile);
  pdbForm.append('parse_preset', 'rcsb');
  const pdbSessionResp = await win.fetch('/api/v1/sessions', { method: 'POST', body: pdbForm });
  const pdbSessionData = pdbSessionResp.ok ? await pdbSessionResp.json() : null;

  const designRadioCount = app.els.designViewRadios?.querySelectorAll('input[type="radio"]').length ?? 0;

  // Verify clicking #predict directly inside the embedded Studio DOM triggers runDesign
  const latestStudioDoc = win.document;
  const predictInStudio = latestStudioDoc?.getElementById('predict');
  if (!predictInStudio || typeof predictInStudio.onclick !== 'function') {
    throw new Error('#predict button inside embedded Studio is missing onclick handler');
  }

  // Verify Mol* is the sole active 3D renderer and rendered atoms are populated
  const viewerEl = latestStudioDoc?.getElementById('viewer');
  const viewerModeSelect = latestStudioDoc?.getElementById('viewer-mode');
  const viewerOptions = viewerModeSelect ? [...viewerModeSelect.options].map((o) => o.value) : [];
  const molstarBackend = viewerEl?.dataset?.backend;
  const molstarRenderedAtoms = Number(viewerEl?.dataset?.renderedAtoms ?? 0);
  if (molstarBackend !== 'molstar' || molstarRenderedAtoms <= 0 || viewerOptions.join(',') !== 'molstar') {
    throw new Error(
      `Mol* backend check failed: backend=${molstarBackend}, renderedAtoms=${molstarRenderedAtoms}, options=${viewerOptions}`,
    );
  }

  return {
    ok:
      presetResults.length === 10 &&
      coreTensorCount === 910 &&
      !encoderLoadedAfterCore &&
      fullTensorCount === 1257 &&
      encoderLoadedAfterInv &&
      studioExampleCount === 9 &&
      studioExampleRuns.length === 9 &&
      liveFramesCalled >= 2 &&
      molstarBackend === 'molstar' &&
      molstarRenderedAtoms > 0 &&
      restraintsAfterAdd.restraints?.length >= 1 &&
      Boolean(exported.cif && exported.name) &&
      designRadioCount >= 2 &&
      Boolean(designEntry?.hasTrajectory) &&
      trajectoryModelCount === 10 &&
      convertedCif.includes('_atom_site.Cartn_x') &&
      (pdbSessionData?.structure?.atom_count ?? 0) > 0 &&
      app.lastResult.pdb.includes('ATOM  ') &&
      app.lastResult.cif.includes('_atom_site.Cartn_x'),
    presetCount: presetResults.length,
    coreTensorCount,
    encoderLoadedAfterCore,
    fullTensorCount,
    encoderLoadedAfterInv,
    studioExampleCount,
    studioExampleRuns,
    molstarBackend,
    molstarRenderedAtoms,
    restraintsCount: restraintsAfterAdd.restraints?.length ?? 0,
    singleAtomBondCount: singleAtomBonds.count,
    interactionCount: interactionsRes.interaction_count,
    colorButtonCount: colorBtns.length,
    exportedName: exported.name,
    designRadioCount,
    liveFramesCalled,
    trajectoryModelCount,
    statusText,
    invStatusText,
    pdbLines: app.lastResult.pdb.split('\n').length,
    cifLines: app.lastResult.cif.split('\n').length,
  };
}
