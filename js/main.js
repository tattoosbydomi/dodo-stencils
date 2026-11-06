import {
  countTones, grayToRGBA, linesToRGBA, tintStencilOverReference,
} from './pipeline.js';
import { encodePng } from './png.js';
import {
  clampView, FIT_VIEW, fitRect, imageRect, pinchView,
} from './viewport.js';
import { blockPageZoom, enhanceSlider } from './touch.js';

// The size the AI model looks at, for the preview and every download alike. The model's
// output isn't just "softer at lower res", it's a genuinely different result, so there is
// one analysis per image and everything else is drawn from it. 1200 was picked from
// measured wasm inference time; analysis reliably failed above ~1400px on the older engine
// (out of memory), and the limit is likely lower on phones, so don't raise this without
// re-measuring on the weakest device in use.
const MAX_PREVIEW_DIM = 1200;
// Each slider move redraws the preview, so it's kept to about this size however large the
// download is. Downloads are drawn separately at full size, from the same analysis.
const PREVIEW_MAX_PIXELS = 2_700_000;
// Larger downloads would run a phone out of memory; better to say so first.
const MAX_OUTPUT_PIXELS = 36_000_000;
const DEFAULT_WIDTH_CM = 17.2;
const DEFAULT_DPI = 300;
const DEBOUNCE_MS = 130;
const PREBUILD_IDLE_MS = 1200;

const el = (id) => document.getElementById(id);

const dom = {
  fileInput: el('file-input'),
  dropzone: el('dropzone'),
  resetBtn: el('reset-btn'),
  uploadView: el('upload-view'),
  editorView: el('editor-view'),
  viewport: el('canvas-viewport'),
  canvas: el('preview-canvas'),
  overlay: el('processing-overlay'),
  overlayLabel: el('processing-label'),
  errorBanner: el('error-banner'),
  modeButtons: Array.from(document.querySelectorAll('.mode-btn')),
  exportButtons: Array.from(document.querySelectorAll('[data-mode-btn]')),
  exportStatus: el('export-status'),
  stencilSliders: el('stencil-sliders'),
  settingsGroup: el('settings-group'),
  rowRefPosterize: el('row-ref-posterize'),
  rowRefLevels: el('row-ref-levels'),
  rowRefOpacity: el('row-ref-opacity'),
  rowColour: el('row-colour'),
  colourSwatches: Array.from(document.querySelectorAll('.swatch-btn')),
  blackSwatch: el('swatch-black'),
  refOpacity: el('in-ref-opacity'),
  refOpacityOut: el('out-ref-opacity'),
  outputSizeHint: el('output-size-hint'),
  printWidth: el('in-print-width'),
  printDpi: el('in-print-dpi'),
};

// --- State -------------------------------------------------------------
let sourceBitmap = null;      // full-resolution upload, kept for downloads
let previewImageData = null;  // the upload at analysis size
let sourceCanvas = null;      // the same, shown while the first analysis runs
let previewW = 0;
let previewH = 0;
let previewLayers = null;     // last computed { lines, reference, width, height }
let imageId = 0;              // changes with every upload; anything older is discarded
let previewAnalyzed = false;
let currentMode = 'stencil';
let debounceTimer = null;
let latestFinalizeToken = 0;
// Each tab keeps its own colour. Black suits the Stencil tab, which is what gets printed;
// a bright colour suits the combined tab, where the lines must stand out from the photo.
const lineColours = { stencil: '#000000', colour: '#FF007F' };
// Until Width or DPI is edited, a download has exactly the upload's pixels, so it lands
// exactly on top of the design when overlaid on the same canvas.
let matchUpload = true;

function showError(message) {
  dom.errorBanner.textContent = message;
  dom.errorBanner.hidden = !message;
  if (message) dom.errorBanner.scrollIntoView({ block: 'nearest' });
}

// A bare numeric message is the wasm engine running out of memory rather than a normal error.
function readableError(err) {
  const message = err?.message || String(err);
  if (/SIMD/i.test(message)) return 'this device’s software is too old for the image engine. On iPhone it needs iOS 16.4 or newer';
  return /^[a-z]/i.test(message) ? message : 'the image is too large or detailed to process on this device';
}

// --- Sliders ------------------------------------------------------------
const sliderIds = ['line-detail', 'thickness', 'ref-levels'];
// Grey Steps is a literal count of greys in the file, so it keeps its own raw scale; the
// others show a normalised 0-100 reading instead of their real underlying range.
const RAW_SCALE_IDS = new Set(['ref-levels']);
const sliders = {};

function percentFromValue(value, min, max) {
  if (max === min) return 0;
  return Math.round(((value - min) / (max - min)) * 100);
}

for (const id of sliderIds) {
  sliders[id] = { input: el(`in-${id}`), output: el(`out-${id}`) };
  const updateOutput = () => {
    const { input, output } = sliders[id];
    output.textContent = RAW_SCALE_IDS.has(id)
      ? input.value
      : percentFromValue(Number(input.value), Number(input.min), Number(input.max));
  };
  updateOutput();
  sliders[id].input.addEventListener('input', () => {
    updateOutput();
    schedulePreviewFinalize();
  });
}
document.querySelectorAll('input[type="range"]').forEach(enhanceSlider);

for (const id of ['invert', 'ref-posterize']) {
  el(`in-${id}`).addEventListener('change', schedulePreviewFinalize);
}
el('in-ref-posterize').addEventListener('change', showGreySteps);

// Grey Steps only does anything once Simplify is ticked, so it's hidden until then.
function showGreySteps() {
  const wanted = SETTINGS_ROWS_BY_MODE[currentMode].includes('rowRefLevels');
  dom.rowRefLevels.hidden = !wanted || !el('in-ref-posterize').checked;
}

// The slider can't offer more steps than the design has greys, so that its number is always
// what the file contains.
const GREY_STEPS_MIN = 4;
const GREY_STEPS_MAX = 16;
function fitGreyStepsToImage(tones) {
  const { input } = sliders['ref-levels'];
  input.min = Math.min(GREY_STEPS_MIN, tones);
  input.max = Math.min(GREY_STEPS_MAX, tones);
  input.disabled = input.min === input.max;
  input.value = Math.min(Number(input.value), Number(input.max));
  input.dispatchEvent(new Event('input', { bubbles: true })); // updates the number and the fill
}

// Was the Keep Detail slider, removed because it barely changed anything: see DEV-NOTES.md
// in the dev folder. This is its old default, so default stencils are unchanged.
const FIXED_BLACK_POINT = 15;

function readParams() {
  const v = (id) => Number(sliders[id].input.value);
  return {
    blackPoint: FIXED_BLACK_POINT,
    whitePoint: v('line-detail'),
    lineThickness: v('thickness'),
    invertLines: el('in-invert').checked,
    referencePosterize: el('in-ref-posterize').checked,
    referenceLevels: v('ref-levels'),
  };
}

// --- Worker plumbing -----------------------------------------------------
const worker = new Worker(new URL('./worker.js', import.meta.url));
let nextRequestId = 1;
const pending = new Map();

worker.onmessage = (e) => {
  const { requestId, error } = e.data;
  const resolver = pending.get(requestId);
  if (!resolver) return;
  pending.delete(requestId);
  if (error) {
    resolver.reject(new Error(error));
    return;
  }
  if (e.data.type === 'previewAnalyzeDone') {
    resolver.resolve({ analyzed: true });
    return;
  }
  const { width, height, lineBuf, referenceBuf } = e.data;
  resolver.resolve({
    width,
    height,
    lines: lineBuf ? new Float32Array(lineBuf) : null,
    reference: referenceBuf ? new Float32Array(referenceBuf) : null,
  });
};

// If the worker itself dies, every waiting request would otherwise wait for ever.
worker.onerror = (e) => {
  const message = e.message || 'the image processor stopped unexpectedly';
  for (const resolver of pending.values()) resolver.reject(new Error(message));
  pending.clear();
};

function send(message, transferList = []) {
  const requestId = nextRequestId++;
  return new Promise((resolve, reject) => {
    pending.set(requestId, { resolve, reject });
    worker.postMessage({ requestId, ...message }, transferList);
  });
}

// --- Preview drawing -----------------------------------------------------
// One canvas the size of the box, showing only what's visible. A zoomed-in preview used to
// mean enlarging four canvases, which could pass the size limit on iPhones and go blank.
const layerCache = new Map(); // mode -> offscreen canvas at the layer's own pixel size
let view = { ...FIT_VIEW };
let boxW = 0;
let boxH = 0;
let drawPending = false;

function canvasFromRGBA(rgba, w, h) {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  canvas.getContext('2d').putImageData(new ImageData(rgba, w, h), 0, 0);
  return canvas;
}

function layerFor(mode) {
  if (!previewLayers) return sourceCanvas;
  if (!layerCache.has(mode)) {
    const { lines, reference, width: w, height: h } = previewLayers;
    let rgba;
    if (mode === 'stencil') rgba = linesToRGBA(lines, w, h, lineColours.stencil);
    else if (mode === 'reference') rgba = grayToRGBA(reference, w, h);
    else rgba = tintStencilOverReference(lines, reference, w, h, lineColours.colour, Number(dom.refOpacity.value) / 100);
    layerCache.set(mode, canvasFromRGBA(rgba, w, h));
  }
  return layerCache.get(mode);
}

function currentFit() {
  return fitRect(previewW || 1, previewH || 1, boxW, boxH);
}

function drawNow() {
  drawPending = false;
  const ctx = dom.canvas.getContext('2d');
  const dpr = dom.canvas.width / (boxW || 1);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, boxW, boxH);
  const layer = layerFor(currentMode);
  if (!layer) return;
  const rect = imageRect(view, currentFit());
  if (currentMode === 'stencil' && previewLayers) {
    // The download is transparent; on screen the lines sit on white paper.
    ctx.fillStyle = '#fff';
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  }
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(layer, rect.x, rect.y, rect.w, rect.h);
}

function scheduleDraw() {
  if (drawPending) return;
  drawPending = true;
  requestAnimationFrame(drawNow);
}

function sizeCanvas() {
  const rect = dom.viewport.getBoundingClientRect();
  if (!rect.width) return;
  boxW = dom.viewport.clientWidth;
  boxH = dom.viewport.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  dom.canvas.width = Math.max(1, Math.round(boxW * dpr));
  dom.canvas.height = Math.max(1, Math.round(boxH * dpr));
  view = clampView(view, currentFit(), boxW, boxH);
  scheduleDraw();
}
new ResizeObserver(sizeCanvas).observe(dom.viewport);

// --- Pinch to zoom -------------------------------------------------------
// Two fingers zoom where they are and move the image together. One finger is left alone,
// so it only ever scrolls the page. Pinching back out is the way back to the full view.
let pinch = null;

function touchInfo(touches) {
  const box = dom.viewport.getBoundingClientRect();
  const [a, b] = [touches[0], touches[1]];
  return {
    mid: {
      x: (a.clientX + b.clientX) / 2 - box.left - dom.viewport.clientLeft,
      y: (a.clientY + b.clientY) / 2 - box.top - dom.viewport.clientTop,
    },
    distance: Math.max(1, Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY)),
  };
}

dom.viewport.addEventListener('touchstart', (e) => {
  if (e.touches.length !== 2) return;
  if (e.cancelable) e.preventDefault();
  pinch = { view: { ...view }, ...touchInfo(e.touches) };
}, { passive: false });

dom.viewport.addEventListener('touchmove', (e) => {
  if (!pinch || e.touches.length !== 2) return;
  if (e.cancelable) e.preventDefault();
  view = pinchView(pinch, touchInfo(e.touches), currentFit(), boxW, boxH);
  scheduleDraw();
}, { passive: false });

const endPinch = (e) => {
  if (e.touches.length < 2) pinch = null;
};
dom.viewport.addEventListener('touchend', endPinch);
dom.viewport.addEventListener('touchcancel', endPinch);

// --- Image loading -------------------------------------------------------
dom.fileInput.addEventListener('change', () => {
  const file = dom.fileInput.files[0];
  if (file) loadFile(file);
});
dom.dropzone.addEventListener('dragover', (e) => e.preventDefault());
dom.dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});

function clearImage() {
  imageId += 1;
  sourceBitmap?.close?.();
  sourceBitmap = null;
  previewImageData = null;
  sourceCanvas = null;
  previewLayers = null;
  previewAnalyzed = false;
  layerCache.clear();
  readyDownload = null;
  clearTimeout(prebuildTimer);
  view = { ...FIT_VIEW };
}

dom.resetBtn.addEventListener('click', () => {
  clearImage();
  showError('');
  setOverlay(false);
  dom.fileInput.value = '';
  dom.editorView.hidden = true;
  dom.uploadView.hidden = false;
  dom.resetBtn.hidden = true;
});

/** Draws the upload at a given size on white, since a transparent background would otherwise read as black. */
function drawOnWhite(bitmap, w, h) {
  const scratch = document.createElement('canvas');
  scratch.width = w;
  scratch.height = h;
  const ctx = scratch.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(bitmap, 0, 0, w, h);
  return { canvas: scratch, imageData: ctx.getImageData(0, 0, w, h) };
}

async function loadFile(file) {
  showError('');
  let bitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    showError('Couldn’t open that image. Please choose a PNG or JPEG.');
    return;
  }
  clearImage();
  sourceBitmap = bitmap;
  matchUpload = true;
  dom.printDpi.value = DEFAULT_DPI;
  dom.printWidth.value = ((bitmap.width / DEFAULT_DPI) * 2.54).toFixed(2);

  const scale = Math.min(1, MAX_PREVIEW_DIM / Math.max(bitmap.width, bitmap.height));
  previewW = Math.max(1, Math.round(bitmap.width * scale));
  previewH = Math.max(1, Math.round(bitmap.height * scale));
  const drawn = drawOnWhite(bitmap, previewW, previewH);
  previewImageData = drawn.imageData;
  sourceCanvas = drawn.canvas;
  // Counted on the full upload: the smaller copy has extra greys from being shrunk.
  fitGreyStepsToImage(countTones(drawOnWhite(bitmap, bitmap.width, bitmap.height).imageData.data));

  dom.uploadView.hidden = true;
  dom.editorView.hidden = false;
  dom.resetBtn.hidden = false;

  sizeCanvas();
  updateOutputSizeHint();
  updateExportButtons();
  setMode(currentMode);
  await runPreviewAnalysis();
}

// --- Preview pipeline (heavy analysis once, cheap redraw per slider move) ---
async function runPreviewAnalysis() {
  if (!previewImageData) return;
  const forImage = imageId;
  previewAnalyzed = false;
  readyDownload = null;
  updateExportButtons();
  setOverlay(true, 'Analyzing artwork…');
  try {
    const { buffer } = previewImageData.data.slice();
    await send({ type: 'previewAnalyze', buffer, width: previewW, height: previewH }, [buffer]);
    if (forImage !== imageId) return; // replaced while it was running
    previewAnalyzed = true;
    await runPreviewFinalize();
    updateExportButtons();
  } catch (err) {
    console.error(err);
    if (forImage === imageId) showError(`Couldn’t analyze the image: ${readableError(err)}.`);
  } finally {
    if (forImage === imageId) setOverlay(false);
  }
}

function schedulePreviewFinalize() {
  if (!previewAnalyzed) return;
  readyDownload = null;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(runPreviewFinalize, DEBOUNCE_MS);
}

// The size the preview's layers are drawn at: the download's own scale where that's
// reasonable, so what you see closely matches what you download, but capped so that
// slider moves stay quick on a phone however large the download is.
function previewRenderSize() {
  const { widthPx } = computeOutputPixelSize();
  const cap = Math.sqrt(PREVIEW_MAX_PIXELS / (previewW * previewH));
  const scale = Math.max(0.25, Math.min(cap, widthPx / previewW));
  return {
    outWidth: Math.max(1, Math.round(previewW * scale)),
    outHeight: Math.max(1, Math.round(previewH * scale)),
  };
}

async function runPreviewFinalize() {
  const token = ++latestFinalizeToken;
  const forImage = imageId;
  setOverlay(true, 'Updating…');
  try {
    const { outWidth, outHeight } = previewRenderSize();
    const result = await send({ type: 'previewFinalize', params: readParams(), outWidth, outHeight });
    if (token !== latestFinalizeToken || forImage !== imageId) return; // superseded
    previewLayers = result;
    layerCache.clear();
    scheduleDraw();
    schedulePrebuild();
  } catch (err) {
    console.error(err);
    if (token === latestFinalizeToken) showError(`Couldn’t update the preview: ${readableError(err)}.`);
  } finally {
    if (token === latestFinalizeToken && forImage === imageId) setOverlay(false);
  }
}

function setOverlay(visible, label) {
  dom.overlay.hidden = !visible;
  if (label) dom.overlayLabel.textContent = label;
}

// --- Colours -------------------------------------------------------------
function showActiveSwatch() {
  const colour = lineColours[currentMode];
  for (const btn of dom.colourSwatches) {
    const active = btn.dataset.colour === colour;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', String(active));
  }
}

for (const btn of dom.colourSwatches) {
  btn.addEventListener('click', () => {
    if (!(currentMode in lineColours)) return;
    lineColours[currentMode] = btn.dataset.colour;
    layerCache.delete(currentMode);
    readyDownload = null;
    showActiveSwatch();
    scheduleDraw();
    schedulePrebuild();
  });
}

dom.refOpacityOut.textContent = dom.refOpacity.value;
dom.refOpacity.addEventListener('input', () => {
  dom.refOpacityOut.textContent = dom.refOpacity.value;
  layerCache.delete('colour');
  readyDownload = null;
  scheduleDraw();
  schedulePrebuild();
});

// --- Tabs ----------------------------------------------------------------
const SETTINGS_ROWS_BY_MODE = {
  stencil: ['rowColour'],
  reference: ['rowRefPosterize', 'rowRefLevels'],
  colour: ['rowRefPosterize', 'rowRefLevels', 'rowRefOpacity', 'rowColour'],
};
const ALL_SETTINGS_ROWS = ['rowRefPosterize', 'rowRefLevels', 'rowRefOpacity', 'rowColour'];

function setMode(mode) {
  currentMode = mode;
  for (const btn of dom.modeButtons) {
    const active = btn.dataset.mode === mode;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', String(active));
  }
  for (const btn of dom.exportButtons) btn.hidden = btn.dataset.modeBtn !== mode;
  dom.stencilSliders.hidden = mode !== 'stencil';
  const visibleRows = new Set(SETTINGS_ROWS_BY_MODE[mode]);
  for (const key of ALL_SETTINGS_ROWS) dom[key].hidden = !visibleRows.has(key);
  showGreySteps();
  dom.blackSwatch.hidden = mode !== 'stencil'; // black lines don't stand out on a photo
  showActiveSwatch();
  dom.exportStatus.textContent = '';
  scheduleDraw();
  schedulePrebuild();
}

for (const btn of dom.modeButtons) {
  btn.addEventListener('click', () => setMode(btn.dataset.mode));
}

// --- Output size ---------------------------------------------------------
function computeOutputPixelSize() {
  const dpi = Number(dom.printDpi.value) || DEFAULT_DPI;
  if (matchUpload && sourceBitmap) {
    return { widthPx: sourceBitmap.width, heightPx: sourceBitmap.height, dpi };
  }
  const widthCm = Number(dom.printWidth.value) || DEFAULT_WIDTH_CM;
  const widthPx = Math.max(1, Math.round((widthCm / 2.54) * dpi));
  const aspect = sourceBitmap ? sourceBitmap.height / sourceBitmap.width : 2870 / 2031;
  return { widthPx, heightPx: Math.max(1, Math.round(widthPx * aspect)), dpi };
}

function outputTooLarge() {
  const { widthPx, heightPx } = computeOutputPixelSize();
  return widthPx * heightPx > MAX_OUTPUT_PIXELS;
}

function updateOutputSizeHint() {
  const { widthPx, heightPx, dpi } = computeOutputPixelSize();
  const note = outputTooLarge() ? ' — too large for a phone, please reduce the width or DPI' : '';
  dom.outputSizeHint.textContent = `Output size: ${widthPx} × ${heightPx} px at ${dpi} DPI${note}`;
}

function updateExportButtons() {
  const blocked = !previewAnalyzed || outputTooLarge();
  for (const btn of dom.exportButtons) btn.disabled = blocked;
}

[dom.printWidth, dom.printDpi].forEach((input) => input.addEventListener('input', () => {
  matchUpload = false;
  readyDownload = null;
  updateOutputSizeHint();
  updateExportButtons();
  schedulePreviewFinalize();
}));

// --- Downloads -----------------------------------------------------------
const OUTPUTS = {
  stencil: 'stencil.png',
  reference: 'reference.png',
  colour: 'stencil-with-reference.png',
};

function downloadKey(mode) {
  return JSON.stringify([
    imageId, mode, readParams(), computeOutputPixelSize(), lineColours[mode], dom.refOpacity.value,
  ]);
}

async function buildDownload(mode) {
  const { widthPx, heightPx, dpi } = computeOutputPixelSize();
  if (widthPx * heightPx > MAX_OUTPUT_PIXELS) throw new Error('that print size is too large for a phone');

  // The reference is made from the full upload, not the smaller analysis copy.
  const transfer = [];
  let referenceBuffer = null;
  if (mode !== 'stencil') {
    const { imageData } = drawOnWhite(sourceBitmap, widthPx, heightPx);
    referenceBuffer = imageData.data.buffer;
    transfer.push(referenceBuffer);
  }

  const result = await send({
    type: 'export',
    params: readParams(),
    outWidth: widthPx,
    outHeight: heightPx,
    referenceBuffer,
    wantLines: mode !== 'reference',
  }, transfer);

  const { width, height, lines, reference } = result;
  if (mode === 'stencil') {
    const rgba = linesToRGBA(lines, width, height, lineColours.stencil);
    return encodePng({ rgba, hasAlpha: true, width, height, dpi });
  }
  if (mode === 'reference') return encodePng({ gray: reference, width, height, dpi });
  const refOpacity = Number(dom.refOpacity.value) / 100;
  const rgba = tintStencilOverReference(lines, reference, width, height, lineColours.colour, refOpacity);
  return encodePng({ rgba, width, height, dpi });
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

// An iPhone download goes to Files, and a website can't save straight to Photos. The share
// sheet is the closest route: its Save Image action goes to Photos.
function usesShareSheet() {
  const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent)
    || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  if (!isIOS || !navigator.canShare) return false;
  return navigator.canShare({ files: [new File(['x'], 'x.png', { type: 'image/png' })] });
}

// Safari only opens the share sheet straight after a tap, and drawing a full-size download
// takes too long for that. So on iPhone the file for the current tab is made ahead of time,
// once the sliders have settled, and the tap can hand it over immediately.
let readyDownload = null; // { key, blob }
let prebuildTimer = 0;

function schedulePrebuild() {
  clearTimeout(prebuildTimer);
  if (!previewAnalyzed || !usesShareSheet()) return;
  prebuildTimer = setTimeout(async () => {
    const mode = currentMode;
    const key = downloadKey(mode);
    if (readyDownload?.key === key || outputTooLarge()) return;
    try {
      const blob = await buildDownload(mode);
      if (key === downloadKey(mode)) readyDownload = { key, blob };
    } catch (err) {
      console.warn('Could not prepare the download ahead of time', err); // a real tap will report it
    }
  }, PREBUILD_IDLE_MS);
}

async function shareFile(blob, filename) {
  try {
    await navigator.share({ files: [new File([blob], filename, { type: 'image/png' })] });
    dom.exportStatus.textContent = '';
    return true;
  } catch (err) {
    if (err.name === 'AbortError') return true; // the sheet was closed on purpose
    if (err.name === 'NotAllowedError') return false; // needs a fresh tap
    throw err;
  }
}

function handleDownloadTap(button, mode) {
  showError('');
  const filename = OUTPUTS[mode];

  // Ready already: go straight to the share sheet, with nothing in between to lose the tap.
  if (usesShareSheet() && readyDownload?.key === downloadKey(mode)) {
    shareFile(readyDownload.blob, filename).catch((err) => showError(`Couldn’t share the file: ${readableError(err)}.`));
    return;
  }

  const original = button.textContent;
  button.disabled = true;
  button.textContent = 'Rendering…';
  const key = downloadKey(mode);
  buildDownload(mode)
    .then(async (blob) => {
      if (!usesShareSheet()) {
        triggerDownload(blob, filename);
        return;
      }
      readyDownload = { key, blob };
      if (!(await shareFile(blob, filename))) {
        dom.exportStatus.textContent = 'Ready. Tap the button again to save it to Photos.';
      }
    })
    .catch((err) => {
      console.error(err);
      showError(`Couldn’t render the download: ${readableError(err)}.`);
    })
    .finally(() => {
      button.textContent = original;
      updateExportButtons();
    });
}

for (const btn of dom.exportButtons) {
  btn.addEventListener('click', () => {
    if (sourceBitmap && previewAnalyzed) handleDownloadTap(btn, btn.dataset.modeBtn);
  });
}

// --- Offline support -----------------------------------------------------
function setUpServiceWorker() {
  if (!('serviceWorker' in navigator)) return;
  // Reload only after Update is tapped. The first install also fires controllerchange,
  // and a reload then would throw away an image being worked on.
  let updating = false;
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (updating) window.location.reload();
  });
  navigator.serviceWorker.register('sw.js').then((registration) => {
    if (!registration) return;
    const offer = (next) => {
      el('update-bar').hidden = false;
      el('update-now').onclick = () => {
        updating = true;
        next.postMessage('skipWaiting');
      };
    };
    if (registration.waiting && navigator.serviceWorker.controller) offer(registration.waiting);
    registration.addEventListener('updatefound', () => {
      const next = registration.installing;
      next?.addEventListener('statechange', () => {
        if (next.state === 'installed' && navigator.serviceWorker.controller) offer(next);
      });
    });
  }).catch((err) => console.warn('Offline support unavailable', err));
}

// --- Start ---------------------------------------------------------------
blockPageZoom();
setMode('stencil');
updateOutputSizeHint();
setUpServiceWorker();
