// Classic (non-module) worker: the AI engine's distributed build is loaded via
// importScripts, while our own pipeline.js is pulled in via dynamic import().
// The engine lives in this site, not on a CDN, so the app works with no signal.
importScripts('../lib/ort/ort.wasm.min.js');

const ENGINE_URL = new URL('../lib/ort/', self.location.href).href;
ort.env.wasm.wasmPaths = ENGINE_URL;
// GitHub Pages can't send the headers multi-threading needs, so be explicit about it.
ort.env.wasm.numThreads = 1;

const MODEL_URL = new URL('../models/informative-drawings.onnx', self.location.href).href;

const pipelinePromise = import('./pipeline.js');
let sessionPromise = null;
function getSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
    sessionPromise.catch(() => { sessionPromise = null; }); // so a later attempt can try again
  }
  return sessionPromise;
}

// The most recently analyzed image. Slider moves and downloads all draw from this, so the
// network runs only for a new upload or the "sharpen" option, and a download has exactly
// the same line data as the preview.
let previewCache = null; // { gray, rawLineMap, width, height }

// The network internally downsamples/upsamples by some fixed stride, so an input whose
// dimensions aren't a multiple of that stride comes back a pixel or two larger than requested.
// Padding up to a safe multiple (with edge-replication, so we're not inventing new content at
// the border) and cropping back down afterward sidesteps the issue for any input size.
const NETWORK_STRIDE = 32;
function padToMultiple(value, multiple) {
  return Math.ceil(value / multiple) * multiple;
}

function rgbaToPlanarTensor(rgba, width, height, padW, padH) {
  const plane = padW * padH;
  const data = new Float32Array(3 * plane);
  for (let y = 0; y < padH; y++) {
    const sy = Math.min(y, height - 1);
    for (let x = 0; x < padW; x++) {
      const sx = Math.min(x, width - 1);
      const srcIdx = (sy * width + sx) * 4;
      const dstIdx = y * padW + x;
      data[dstIdx] = rgba[srcIdx] / 255;
      data[plane + dstIdx] = rgba[srcIdx + 1] / 255;
      data[2 * plane + dstIdx] = rgba[srcIdx + 2] / 255;
    }
  }
  return new ort.Tensor('float32', data, [1, 3, padH, padW]);
}

async function runNetwork(rgba, width, height) {
  const session = await getSession();
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const padW = padToMultiple(width, NETWORK_STRIDE);
  const padH = padToMultiple(height, NETWORK_STRIDE);

  const results = await session.run({ [inputName]: rgbaToPlanarTensor(rgba, width, height, padW, padH) });
  const outputTensor = results[outputName];
  const [, , outH, outW] = outputTensor.dims;
  if (outW !== padW || outH !== padH) {
    throw new Error(`Model output size (${outW}x${outH}) does not match padded input (${padW}x${padH})`);
  }
  const raw = outputTensor.data;

  // Crop back down to the originally requested size (the padding was edge-replicated, not real content).
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = raw[y * padW + x] * 255;
      out[y * width + x] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
  outputTensor.dispose?.();
  return out;
}

function respond(requestId, width, height, lines, reference) {
  const lineBuf = lines ? Float32Array.from(lines).buffer : null;
  const referenceBuf = reference ? Float32Array.from(reference).buffer : null;
  self.postMessage(
    { requestId, width, height, lineBuf, referenceBuf },
    [lineBuf, referenceBuf].filter(Boolean),
  );
}

async function handle(data) {
  const { type, requestId } = data;
  const pipeline = await pipelinePromise;

  if (type === 'previewAnalyze') {
    const { buffer, width, height } = data;
    const rgba = new Uint8ClampedArray(buffer);
    const rawLineMap = await runNetwork(rgba, width, height);
    previewCache = { rawLineMap, gray: pipeline.toGrayscale(rgba, width, height), width, height };
    self.postMessage({ requestId, type: 'previewAnalyzeDone' });
    return;
  }

  if (!previewCache) throw new Error('No image analyzed yet');
  const { params, outWidth, outHeight } = data;
  const { rawLineMap, gray, width, height } = previewCache;

  if (type === 'previewFinalize') {
    const lines = pipeline.finalizeLines(rawLineMap, width, height, params, outWidth, outHeight);
    const reference = pipeline.resizeGray(pipeline.buildReferenceLayer(gray, params), width, height, outWidth, outHeight);
    respond(requestId, outWidth, outHeight, lines, reference);
    return;
  }

  if (type === 'export') {
    const lines = data.wantLines
      ? pipeline.finalizeLines(rawLineMap, width, height, params, outWidth, outHeight)
      : null;
    let reference = null;
    if (data.referenceBuffer) {
      const fullRgba = new Uint8ClampedArray(data.referenceBuffer);
      reference = pipeline.buildReferenceLayer(pipeline.toGrayscale(fullRgba, outWidth, outHeight), params);
    }
    respond(requestId, outWidth, outHeight, lines, reference);
    return;
  }

  throw new Error(`Unknown message type: ${type}`);
}

// One job at a time: the engine can't run two analyses at once, and this keeps results
// in the order they were asked for.
let queue = Promise.resolve();
self.onmessage = (e) => {
  const { requestId } = e.data;
  queue = queue.then(() => handle(e.data)).catch((err) => {
    // Not every failure here is a normal Error with a .message — the wasm backend can throw a
    // bare trap (e.g. out of memory) whose .message is undefined, and posting that straight
    // through would read as "no error" to the caller.
    const message = (err && err.message) ? err.message : String(err);
    self.postMessage({ requestId, error: message });
  });
};
