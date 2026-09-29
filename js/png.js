// Minimal PNG encoder, used instead of canvas.toBlob because the canvas can only
// write 8-bit RGBA with no resolution info. This one writes:
//   - a true 1-bit black/white PNG when the image is pure black/white (the
//     'stencil' line style), so nothing downstream can mistake it for greyscale;
//   - a pHYs chunk with the chosen DPI, so printer software opens the file at
//     the intended physical size instead of rescaling it.
// Compression uses the browser's built-in CompressionStream ('deflate' = zlib,
// which is what PNG's IDAT expects); without it, it falls back to the canvas.

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

async function zlibDeflate(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// Every row starts with a filter-type byte (0 = none).
function buildRows(width, height, { gray, rgba }) {
  if (rgba) {
    const stride = 1 + width * 3;
    const raw = new Uint8Array(stride * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const s = (y * width + x) * 4, d = y * stride + 1 + x * 3;
        raw[d] = rgba[s]; raw[d + 1] = rgba[s + 1]; raw[d + 2] = rgba[s + 2];
      }
    }
    return { raw, bitDepth: 8, colorType: 2 };
  }

  let binary = true;
  for (let i = 0; i < gray.length; i++) {
    if (gray[i] !== 0 && gray[i] !== 255) { binary = false; break; }
  }
  if (binary) {
    const stride = 1 + Math.ceil(width / 8);
    const raw = new Uint8Array(stride * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (gray[y * width + x]) raw[y * stride + 1 + (x >> 3)] |= 0x80 >> (x & 7); // 1 = white
      }
    }
    return { raw, bitDepth: 1, colorType: 0 };
  }
  const stride = 1 + width;
  const raw = new Uint8Array(stride * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = gray[y * width + x];
      raw[y * stride + 1 + x] = v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
    }
  }
  return { raw, bitDepth: 8, colorType: 0 };
}

function canvasFallback(width, height, { gray, rgba }) {
  let pixels = rgba;
  if (!pixels) {
    pixels = new Uint8ClampedArray(width * height * 4);
    for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
      pixels[p] = pixels[p + 1] = pixels[p + 2] = gray[i]; pixels[p + 3] = 255;
    }
  }
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  canvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(pixels), width, height), 0, 0);
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
}

// image: { width, height, dpi, gray (0-255 plane) | rgba (Uint8ClampedArray) }
export async function encodePng(image) {
  const { width, height, dpi } = image;
  if (typeof CompressionStream === 'undefined') return canvasFallback(width, height, image);

  const { raw, bitDepth, colorType } = buildRows(width, height, image);

  const ihdr = new Uint8Array(13);
  const ihdrView = new DataView(ihdr.buffer);
  ihdrView.setUint32(0, width);
  ihdrView.setUint32(4, height);
  ihdr[8] = bitDepth; ihdr[9] = colorType; // compression, filter, interlace all 0

  const phys = new Uint8Array(9);
  const physView = new DataView(phys.buffer);
  const pixelsPerMetre = Math.round(dpi / 0.0254);
  physView.setUint32(0, pixelsPerMetre);
  physView.setUint32(4, pixelsPerMetre);
  phys[8] = 1; // unit: metre

  const signature = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  return new Blob([
    signature,
    chunk('IHDR', ihdr),
    chunk('pHYs', phys),
    chunk('IDAT', await zlibDeflate(raw)),
    chunk('IEND', new Uint8Array(0)),
  ], { type: 'image/png' });
}
