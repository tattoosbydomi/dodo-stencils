// Pure image-processing primitives. No DOM access — safe to run in a Web Worker.
// All "channel" values below are single-plane Float32Array grayscale buffers, 0-255.
//
// Line art itself comes from a neural network (see worker.js) rather than a
// classical edge filter — a plain filter can only ever find "every edge in the
// pixels," never "the lines an artist chose to keep." Everything in this file
// operates on top of that network's output (or on the original photo, for the
// reference layer).

export function toGrayscale(rgba, w, h) {
  const out = new Float32Array(w * h);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    // Perceptual luminance weighting — holds up better on skin tones than a flat average.
    out[i] = 0.2126 * rgba[p] + 0.7152 * rgba[p + 1] + 0.0722 * rgba[p + 2];
  }
  return out;
}

function clamp255(v) { return v < 0 ? 0 : v > 255 ? 255 : v; }

// Otsu's method: picks the cutoff that best separates a grayscale histogram into two classes
// (line vs. paper) by maximizing between-class variance, so the line thresholds adapt per image.
export function otsuThreshold(src) {
  const hist = new Array(256).fill(0);
  for (let i = 0; i < src.length; i++) {
    const v = Math.max(0, Math.min(255, Math.round(src[i])));
    hist[v]++;
  }
  const total = src.length;
  let sum = 0;
  for (let t = 0; t < 256; t++) sum += t * hist[t];

  let sumB = 0, weightB = 0, maxVariance = 0, threshold = 128;
  for (let t = 0; t < 256; t++) {
    weightB += hist[t];
    if (weightB === 0) continue;
    const weightF = total - weightB;
    if (weightF === 0) break;
    sumB += t * hist[t];
    const meanB = sumB / weightB;
    const meanF = (sum - sumB) / weightF;
    const variance = weightB * weightF * (meanB - meanF) * (meanB - meanF);
    if (variance > maxVariance) { maxVariance = variance; threshold = t; }
  }
  return threshold;
}

function toneBin(v) {
  const t = Math.round(v);
  return t < 0 ? 0 : t > 255 ? 255 : t;
}

/** How many different greys an RGBA image contains, measured as toGrayscale would. */
export function countTones(rgba) {
  const seen = new Uint8Array(256);
  let count = 0;
  for (let p = 0; p < rgba.length; p += 4) {
    const t = toneBin(0.2126 * rgba[p] + 0.7152 * rgba[p + 1] + 0.0722 * rgba[p + 2]);
    if (!seen[t]) {
      seen[t] = 1;
      count += 1;
    }
  }
  return count;
}

// Reduces an image to exactly `levels` greys, or to every grey it has if that's fewer. The
// steps go where this image's tones actually are: evenly spaced steps from black to white
// would leave empty ones whenever a photo doesn't use the full range, so the slider's number
// wouldn't match the file. This finds the grouping of the image's tones into that many
// neighbouring bands that changes the image least (Fisher-Jenks, solved exactly by dynamic
// programming over the at most 256 tones), and paints each band its own average tone.
export function posterize(src, levels) {
  const hist = new Float64Array(256);
  for (let i = 0; i < src.length; i++) hist[toneBin(src[i])]++;
  const tones = [];
  for (let t = 0; t < 256; t++) if (hist[t]) tones.push(t);
  const n = tones.length;
  const k = Math.max(1, Math.min(Math.round(levels), n));

  // Prefix sums, so any band's cost (its squared error around its mean) is instant to work out.
  const sumW = new Float64Array(n + 1);
  const sumT = new Float64Array(n + 1);
  const sumTT = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    const t = tones[i];
    const w = hist[t];
    sumW[i + 1] = sumW[i] + w;
    sumT[i + 1] = sumT[i] + w * t;
    sumTT[i + 1] = sumTT[i] + w * t * t;
  }
  const cost = (from, to) => {
    const w = sumW[to] - sumW[from];
    const t = sumT[to] - sumT[from];
    return (sumTT[to] - sumTT[from]) - (t * t) / w;
  };

  // best[i] = least cost of splitting the first i tones into the current number of bands.
  let best = new Float64Array(n + 1);
  for (let i = 1; i <= n; i++) best[i] = cost(0, i);
  const cuts = [];
  for (let c = 2; c <= k; c++) {
    const next = new Float64Array(n + 1).fill(Infinity);
    const cut = new Int32Array(n + 1);
    for (let i = c; i <= n; i++) {
      for (let j = c - 1; j < i; j++) {
        const total = best[j] + cost(j, i);
        if (total < next[i]) {
          next[i] = total;
          cut[i] = j;
        }
      }
    }
    best = next;
    cuts.push(cut);
  }

  // Walk the cuts back to get the bands, and paint each with its average tone. Two bands can't
  // round to the same grey, because each average rounds to within that band's own tones.
  const paint = new Float32Array(256);
  let end = n;
  for (let c = k; c >= 1; c--) {
    const start = c === 1 ? 0 : cuts[c - 2][end];
    const mean = Math.round((sumT[end] - sumT[start]) / (sumW[end] - sumW[start]));
    for (let i = start; i < end; i++) paint[tones[i]] = mean;
    end = start;
  }
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = paint[toneBin(src[i])];
  return out;
}

export function grayToRGBA(gray, w, h) {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    const v = gray[i];
    out[p] = v; out[p + 1] = v; out[p + 2] = v; out[p + 3] = 255;
  }
  return out;
}

// The stencil on nothing: each pixel is the line colour, as opaque as that pixel is dark.
// Paper is fully transparent, so the lines sit directly on whatever layer is below.
export function linesToRGBA(lines, w, h, colourHex) {
  const { r, g, b } = hexToRgb(colourHex);
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0, p = 0; i < lines.length; i++, p += 4) {
    out[p] = r; out[p + 1] = g; out[p + 2] = b;
    out[p + 3] = 255 - lines[i];
  }
  return out;
}

function hexToRgb(hex) {
  const num = parseInt(hex.replace('#', ''), 16);
  return { r: (num >> 16) & 255, g: (num >> 8) & 255, b: num & 255 };
}

// Layers the stencil (tinted with colourHex) on top of the grayscale reference
// photo: darker stencil pixels (lines) become more opaque colour, while
// paper-white stencil pixels are fully transparent and let the reference show
// through underneath — so the two outputs read as one combined image.
// refOpacity (0-1) fades the reference photo toward white before that blend, so
// the tinted lines pop out more without changing their own colour.
export function tintStencilOverReference(stencilGray, referenceGray, w, h, colourHex, refOpacity = 1) {
  const { r, g, b } = hexToRgb(colourHex);
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0, p = 0; i < stencilGray.length; i++, p += 4) {
    const alpha = (255 - stencilGray[i]) / 255;
    const ref = referenceGray[i] * refOpacity + 255 * (1 - refOpacity);
    out[p] = r * alpha + ref * (1 - alpha);
    out[p + 1] = g * alpha + ref * (1 - alpha);
    out[p + 2] = b * alpha + ref * (1 - alpha);
    out[p + 3] = 255;
  }
  return out;
}

// --- Smooth stencil lines ------------------------------------------------
// The helpers below back the 'stencil' line style. They work on "darkness"
// planes (0 = paper, 1 = full ink) rather than the 0-255 paper-white scale above.

// Catmull-Rom bicubic resample of a single-channel plane (separable, pixel-centre
// aligned). Used to enlarge the network's SOFT output to the final pixel size
// before anything gets thresholded — see stencilLines for why that order matters.
export function resampleBicubic(src, w, h, W, H) {
  if (W === w && H === h) return Float32Array.from(src);
  // Shrinking (a small print size): pre-blur so thin lines don't alias away.
  if (W < w) src = gaussianBlur(src, w, h, 0.5 * (w / W));
  const taps = (n, N) => {
    const idx = new Int32Array(N * 4), wt = new Float32Array(N * 4);
    for (let x = 0; x < N; x++) {
      const s = (x + 0.5) * n / N - 0.5;
      const i0 = Math.floor(s), t = s - i0, t2 = t * t, t3 = t2 * t;
      const c = [(-t3 + 2 * t2 - t) / 2, (3 * t3 - 5 * t2 + 2) / 2, (-3 * t3 + 4 * t2 + t) / 2, (t3 - t2) / 2];
      for (let k = 0; k < 4; k++) {
        idx[x * 4 + k] = Math.min(n - 1, Math.max(0, i0 - 1 + k));
        wt[x * 4 + k] = c[k];
      }
    }
    return { idx, wt };
  };
  const tx = taps(w, W), ty = taps(h, H);
  const tmp = new Float32Array(W * h);
  for (let y = 0; y < h; y++) {
    const row = y * w, orow = y * W;
    for (let x = 0; x < W; x++) {
      const b = x * 4;
      tmp[orow + x] = src[row + tx.idx[b]] * tx.wt[b] + src[row + tx.idx[b + 1]] * tx.wt[b + 1]
        + src[row + tx.idx[b + 2]] * tx.wt[b + 2] + src[row + tx.idx[b + 3]] * tx.wt[b + 3];
    }
  }
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    const b = y * 4;
    const r0 = ty.idx[b] * W, r1 = ty.idx[b + 1] * W, r2 = ty.idx[b + 2] * W, r3 = ty.idx[b + 3] * W;
    const w0 = ty.wt[b], w1 = ty.wt[b + 1], w2 = ty.wt[b + 2], w3 = ty.wt[b + 3];
    const orow = y * W;
    for (let x = 0; x < W; x++) {
      out[orow + x] = tmp[r0 + x] * w0 + tmp[r1 + x] * w1 + tmp[r2 + x] * w2 + tmp[r3 + x] * w3;
    }
  }
  return out;
}

export function gaussianBlur(src, w, h, sigma) {
  if (sigma <= 0.05) return Float32Array.from(src);
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma)); sum += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const tmp = new Float32Array(src.length), out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        const xx = x + i < 0 ? 0 : x + i >= w ? w - 1 : x + i;
        acc += src[row + xx] * k[i + r];
      }
      tmp[row + x] = acc;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        const yy = y + i < 0 ? 0 : y + i >= h ? h - 1 : y + i;
        acc += tmp[yy * w + x] * k[i + r];
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

// Square-window local maximum (separable).
function maxFilter(src, w, h, r) {
  const tmp = new Float32Array(src.length), out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let m = 0;
      const x1 = Math.min(w - 1, x + r);
      for (let xx = Math.max(0, x - r); xx <= x1; xx++) if (src[row + xx] > m) m = src[row + xx];
      tmp[row + x] = m;
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let m = 0;
      const y1 = Math.min(h - 1, y + r);
      for (let yy = Math.max(0, y - r); yy <= y1; yy++) if (tmp[yy * w + x] > m) m = tmp[yy * w + x];
      out[y * w + x] = m;
    }
  }
  return out;
}

// Labels connected regions of mask === value (8- or 4-connected). Returns a
// per-pixel label (-1 for pixels not equal to value) and each region's size.
function labelComponents(mask, w, h, value, eightConnected) {
  const labels = new Int32Array(w * h).fill(-1);
  const sizes = [];
  const stack = new Int32Array(w * h);
  for (let start = 0; start < mask.length; start++) {
    if (mask[start] !== value || labels[start] !== -1) continue;
    const id = sizes.length;
    let sp = 0, size = 0;
    stack[sp++] = start; labels[start] = id;
    while (sp) {
      const i = stack[--sp]; size++;
      const x = i % w, y = (i - x) / w;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy; if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if ((!dx && !dy) || (!eightConnected && dx && dy)) continue;
          const xx = x + dx; if (xx < 0 || xx >= w) continue;
          const j = yy * w + xx;
          if (mask[j] === value && labels[j] === -1) { labels[j] = id; stack[sp++] = j; }
        }
      }
    }
    sizes.push(size);
  }
  return { labels, sizes };
}

// Flips every region of mask === value smaller than minArea to the other value
// (removes ink specks when value = 1, fills pinholes inside strokes when value = 0).
function removeSmallRegions(mask, w, h, value, minArea, eightConnected) {
  const { labels, sizes } = labelComponents(mask, w, h, value, eightConnected);
  for (let i = 0; i < mask.length; i++) {
    if (labels[i] >= 0 && sizes[labels[i]] < minArea) mask[i] = 1 - value;
  }
}

// Default levels the adaptive thresholds are measured against, so the thresholds stay fixed
// per image and moving Line Detail actually changes the result.
const REFERENCE_BLACK = 15, REFERENCE_WHITE = 225;

// The stencil's lines: pure black/white output with smooth edges and
// continuous strokes, rendered directly at the final pixel size (outW × outH).
//
// Why edges stay crisp: thresholding at the network's analysis resolution (at most 1200px) and
// only then enlarging would make every edge a pixel staircase that the enlargement smears into
// grey. Here the SOFT darkness map is enlarged first, lightly blurred, and only then cut to
// black/white, so each edge lands where the smooth gradient crosses the cutoff, at the output's
// own pixel precision.
//
// Why strokes don't break up: the network draws faint strokes much lighter than
// strong ones, so one global cutoff either drops the faint ones or bloats the
// strong ones. Two things fix that:
//   1. Stroke-relative cut: each pixel is compared to the darkest value near it,
//      so every stroke is cut at the same fraction of its own peak and faint
//      strokes come out as solid, continuous lines instead of dotted fragments.
//   2. Hysteresis (as in Canny edge detection): faint pixels are only kept when
//      they connect to a clearly-dark one, so faint continuations of real lines
//      survive while isolated background haze doesn't.
// Leftover specks are then removed and pinholes inside strokes filled.
function stencilLines(rawLineMap, w, h, outW, outH, params) {
  const toDark = (bp, wp) => {
    const range = Math.max(1, wp - bp);
    const out = new Float32Array(w * h);
    for (let i = 0; i < out.length; i++) {
      const v = (wp - rawLineMap[i]) / range;
      out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
    return out;
  };

  // Per-image reference level: Otsu split of the darkness histogram at default levels.
  const refDark = toDark(REFERENCE_BLACK, REFERENCE_WHITE);
  const refPaper = new Float32Array(refDark.length); // otsuThreshold expects the 0-255 paper-white scale
  for (let i = 0; i < refDark.length; i++) refPaper[i] = 255 - refDark[i] * 255;
  const ref = Math.max(0.05, (255 - otsuThreshold(refPaper)) / 255);
  // The black point is now fixed at REFERENCE_BLACK, so this is 1. It's kept so the old
  // Keep Detail slider can be restored (see DEV-NOTES.md in the dev folder).
  const detail = Math.max(0.2, (REFERENCE_WHITE - params.blackPoint) / (REFERENCE_WHITE - REFERENCE_BLACK));
  const strongLevel = 0.8 * ref * detail * detail, weakLevel = strongLevel * 0.5;

  const scale = outW / w; // output px per analysis px — keeps the look identical at any DPI
  const dark = toDark(params.blackPoint, params.whitePoint);
  const D = gaussianBlur(resampleBicubic(dark, w, h, outW, outH), outW, outH, 0.5 * scale);

  const normRadius = 1.5 * scale;
  const localPeak = gaussianBlur(maxFilter(D, outW, outH, Math.max(1, Math.round(normRadius))), outW, outH, normRadius * 0.5);

  // Line Thickness: where within each stroke's own profile the edge falls
  // (lower = further out along the stroke's soft falloff = thicker).
  // Thinning is capped at 0.75: cutting any higher up the profile starts breaking
  // strokes apart instead of just slimming them.
  const t = params.lineThickness || 0;
  const cut = t >= 0 ? Math.max(0.35, 0.6 - t * 0.01) : Math.min(0.75, 0.6 - t * 0.006);

  const n = outW * outH;
  const weak = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    weak[i] = (D[i] >= weakLevel && D[i] >= cut * Math.max(localPeak[i], weakLevel)) ? 1 : 0;
  }
  const { labels, sizes } = labelComponents(weak, outW, outH, 1, true);
  const keep = new Uint8Array(sizes.length);
  for (let i = 0; i < n; i++) if (labels[i] >= 0 && D[i] >= strongLevel) keep[labels[i]] = 1;
  const ink = new Uint8Array(n);
  for (let i = 0; i < n; i++) if (labels[i] >= 0 && keep[labels[i]]) ink[i] = 1;

  const area = scale * scale;
  removeSmallRegions(ink, outW, outH, 1, 4 * area, true);
  removeSmallRegions(ink, outW, outH, 0, 3 * area, false);

  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = ink[i] ? 0 : 255;
  return out;
}

// Turns the neural network's raw line-art output (0-255 grayscale) into the final "lines"
// layer at outW x outH. The app once offered four other line styles: see DEV-NOTES.md.
export function finalizeLines(rawLineMap, width, height, params, outW = width, outH = height) {
  const lines = stencilLines(rawLineMap, width, height, outW, outH, params);
  if (params.invertLines) {
    for (let i = 0; i < lines.length; i++) lines[i] = 255 - lines[i];
  }
  return lines;
}

// Smooth enlargement of a 0-255 plane (clamped, since bicubic can overshoot).
export function resizeGray(src, w, h, W, H) {
  if (W === w && H === h) return src;
  const out = resampleBicubic(src, w, h, W, H);
  for (let i = 0; i < out.length; i++) out[i] = clamp255(out[i]);
  return out;
}

// Grayscale reference photo, derived from the original artwork rather than
// the line-art network.
export function buildReferenceLayer(originalGray, params) {
  if (params.referencePosterize) return posterize(originalGray, params.referenceLevels);
  return originalGray;
}
