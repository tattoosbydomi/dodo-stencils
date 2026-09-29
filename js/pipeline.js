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

// Photoshop-style levels remap: push everything at/below blackPoint to 0 (line),
// everything at/above whitePoint to 255 (paper), linear ramp between. This is how
// the "Keep Detail" / "Background Cleanup" sliders clean up the network's raw
// (slightly soft/grayish) output into confident line art.
export function applyLevels(src, blackPoint, whitePoint) {
  const range = Math.max(1, whitePoint - blackPoint);
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    out[i] = clamp255(((src[i] - blackPoint) / range) * 255);
  }
  return out;
}

export function binarize(src, threshold = 128) {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = src[i] >= threshold ? 255 : 0;
  return out;
}

// Otsu's method: picks the cutoff that best separates a grayscale histogram into
// two classes (line vs. paper) by maximizing between-class variance. Used instead
// of a fixed threshold (128) so "Crisp" lines stay clean across photos where the
// network's soft output happens to sit at a different brightness overall — a
// fixed cutoff either eats thin lines or lets background haze through depending
// on the image, while this adapts per photo.
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

// Zhang-Suen thinning: iteratively erodes a binary foreground (ink) mask down to
// a 1-pixel-wide skeleton while preserving connectivity and branching (Y/T
// junctions survive, unlike a naive contour trace which can only represent
// non-branching paths). This is what turns a thick, unevenly-blurry threshold
// result into a single clean centerline per stroke.
// maxIterations bounds worst-case runtime: each outer iteration only erodes one
// layer off every blob's boundary, so thin stencil lines (the normal case, a
// handful of iterations) finish long before this cap is ever reached — it only
// kicks in for a pathological input (e.g. a large solid dark region the network/
// threshold misclassified as "line"), where full erosion could otherwise take
// tens of seconds. Any such remainder is still usable, just not fully thinned.
export function skeletonize(binaryGray, w, h, maxIterations = 40) {
  const fg = new Uint8Array(w * h);
  for (let i = 0; i < fg.length; i++) fg[i] = binaryGray[i] < 128 ? 1 : 0;

  const at = (x, y) => (x < 0 || x >= w || y < 0 || y >= h) ? 0 : fg[y * w + x];

  let changed = true;
  let iterations = 0;
  while (changed && iterations++ < maxIterations) {
    changed = false;
    for (let subIter = 0; subIter < 2; subIter++) {
      const toRemove = [];
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          if (!fg[y * w + x]) continue;
          const p2 = at(x, y - 1), p3 = at(x + 1, y - 1), p4 = at(x + 1, y);
          const p5 = at(x + 1, y + 1), p6 = at(x, y + 1), p7 = at(x - 1, y + 1);
          const p8 = at(x - 1, y), p9 = at(x - 1, y - 1);
          const ring = [p2, p3, p4, p5, p6, p7, p8, p9];
          const blackNeighbours = ring.reduce((a, b) => a + b, 0);
          if (blackNeighbours < 2 || blackNeighbours > 6) continue;
          let transitions = 0;
          for (let k = 0; k < 8; k++) {
            if (ring[k] === 0 && ring[(k + 1) % 8] === 1) transitions++;
          }
          if (transitions !== 1) continue;
          if (subIter === 0) {
            if (p2 * p4 * p6 !== 0) continue;
            if (p4 * p6 * p8 !== 0) continue;
          } else {
            if (p2 * p4 * p8 !== 0) continue;
            if (p2 * p6 * p8 !== 0) continue;
          }
          toRemove.push(y * w + x);
        }
      }
      if (toRemove.length) {
        changed = true;
        for (const i of toRemove) fg[i] = 0;
      }
    }
  }

  const out = new Float32Array(w * h).fill(255);
  for (let i = 0; i < out.length; i++) if (fg[i]) out[i] = 0;
  return out;
}

// Re-thickens a (typically skeletonized) binary line to a constant radius, so
// every stroke reads at the same weight regardless of how thick/thin the network's
// soft output happened to make it before thresholding.
export function dilate(binaryGray, w, h, radius) {
  if (radius <= 0) return binaryGray;
  const fg = new Uint8Array(w * h);
  for (let i = 0; i < fg.length; i++) fg[i] = binaryGray[i] < 128 ? 1 : 0;
  const out = new Uint8Array(w * h);
  const r2 = radius * radius;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!fg[y * w + x]) continue;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          if (dx * dx + dy * dy > r2) continue;
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          out[yy * w + xx] = 1;
        }
      }
    }
  }
  const result = new Float32Array(w * h).fill(255);
  for (let i = 0; i < result.length; i++) if (out[i]) result[i] = 0;
  return result;
}

function boxBlur3(src, w, h) {
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - 1), x1 = Math.min(w - 1, x + 1);
      tmp[row + x] = (src[row + x0] + src[row + x] + src[row + x1]) / 3;
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      const y0 = Math.max(0, y - 1), y1 = Math.min(h - 1, y + 1);
      out[y * w + x] = (tmp[y0 * w + x] + tmp[y * w + x] + tmp[y1 * w + x]) / 3;
    }
  }
  return out;
}

// Cheap unsharp mask on the source photo (before it ever reaches the line-art
// network): blur each channel, then push the original away from that blur.
// Helps when the *photo* itself is soft/low-contrast, which is a case no amount
// of post-processing the network's output can fix — the network never had a
// strong gradient to lock onto in the first place.
export function unsharpMaskRGBA(rgba, w, h, amount = 0.8) {
  const plane = w * h;
  const out = new Uint8ClampedArray(rgba.length);
  for (let c = 0; c < 3; c++) {
    const channel = new Float32Array(plane);
    for (let i = 0, p = c; i < plane; i++, p += 4) channel[i] = rgba[p];
    const blurred = boxBlur3(channel, w, h);
    for (let i = 0, p = c; i < plane; i++, p += 4) {
      out[p] = channel[i] + amount * (channel[i] - blurred[i]);
    }
  }
  for (let i = 0, p = 3; i < plane; i++, p += 4) out[p] = 255;
  return out;
}

export function posterize(src, levels) {
  const step = 255 / (levels - 1);
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = Math.round(Math.round(src[i] / step) * step);
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

// Default levels the adaptive thresholds are measured against — deliberately the
// slider DEFAULTS, not the live slider values, so the thresholds stay fixed per
// image and moving Keep Detail / Background Cleanup actually changes the result
// (re-measuring after the sliders move would just cancel the slider back out).
const REFERENCE_BLACK = 15, REFERENCE_WHITE = 225;

// The 'stencil' line style: pure black/white output with smooth edges and
// continuous strokes, rendered directly at the final pixel size (outW × outH).
//
// Why the other crisp styles look soft/jagged: they threshold at the network's
// analysis resolution (≤1200px) and only then get enlarged, so every edge is a
// pixel staircase that the enlargement smears into grey. Here the SOFT darkness
// map is enlarged first, lightly blurred, and only then cut to black/white — so
// each edge lands where the smooth gradient crosses the cutoff, at the output's
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
  // Keep Detail (the black point) also lowers both thresholds, so it recovers
  // faint strokes directly rather than only darkening the ones already kept.
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

// Post-processes the neural network's raw line-art output (0-255 grayscale,
// network already produced) into the final "lines" layer, at outW × outH.
//
// "Line Thickness" is implemented as a shift of the black/white points
// together (same knobs "Keep Detail"/"Background Cleanup" already expose),
// applied before those levels run — not as pixel-radius morphology. Morphology
// reassigns each pixel to the darkest/lightest value within a radius, which for
// soft anti-aliased line art is a flood, not a nudge: even radius 1 could wipe
// thin strokes out entirely (thinner) or blow them into solid blobs (thicker),
// with nothing gradual in between. Shifting the levels window instead moves
// where *within the network's existing soft gradient* the line/paper cutoff
// falls, so it stays continuous and reuses the exact remap that already
// produces clean results for the other two sliders.
//
// `lineStyle` picks how (or whether) that soft gradient then gets crisped up —
// see index.html's "Line Style" dropdown for the four methodologies being compared:
//   'soft'           — the network's gradient as-is, no thresholding.
//   'threshold'      — flat cutoff at 128 (the original "Crisp lines" behaviour).
//   'auto-threshold' — same cutoff, but picked per-image via Otsu instead of fixed.
//   'clean'          — auto-threshold, then skeletonize + re-thicken to a constant
//                      width, so every stroke reads the same regardless of how
//                      thick/thin/blurry the raw network output made it.
//   'stencil'        — smooth pure black/white, rendered natively at outW × outH
//                      (see stencilLines). The others are computed at the analysis
//                      size and then enlarged smoothly, as they always were.
export function finalizeLines(rawLineMap, width, height, params, outW = width, outH = height) {
  const style = params.lineStyle || 'soft';
  if (style === 'stencil') {
    const lines = stencilLines(rawLineMap, width, height, outW, outH, params);
    if (params.invertLines) {
      for (let i = 0; i < lines.length; i++) lines[i] = 255 - lines[i];
    }
    return lines;
  }

  const shift = params.lineThickness || 0;
  const blackPoint = clamp255(params.blackPoint + shift);
  const whitePoint = clamp255(params.whitePoint + shift);
  let lines = applyLevels(rawLineMap, blackPoint, whitePoint);

  if (style === 'threshold') {
    lines = binarize(lines, 128);
  } else if (style === 'auto-threshold') {
    lines = binarize(lines, otsuThreshold(lines));
  } else if (style === 'clean') {
    lines = binarize(lines, otsuThreshold(lines));
    lines = skeletonize(lines, width, height);
    // radius 0 = no dilation at all, i.e. the raw 1px skeleton — the thinnest a
    // line can be. Line Thickness only ever ADDS width on top of that from here;
    // it used to have a floor of 1 (a forced "+"-shaped halo on every skeleton
    // pixel), so "thinnest" was never actually thin — that was the bug.
    const radius = Math.max(0, Math.min(4, Math.round(shift / 10)));
    lines = dilate(lines, width, height, radius);
  }

  if (params.invertLines) {
    for (let i = 0; i < lines.length; i++) lines[i] = 255 - lines[i];
  }
  return resizeGray(lines, width, height, outW, outH);
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
