// Offline support. Your images never pass through here: they stay in the page's memory.
//
// Cache first, so the app opens instantly and works with no signal. When you change any
// file, bump VERSION. That's the only version number in the app: the browser spots the
// changed sw.js, downloads the new set in the background, and the app offers an Update button.
const VERSION = 'v5';
const CACHE = `dodo-stencils-${VERSION}`;

const ASSETS = [
  './',
  './manifest.json',
  './css/styles.css',
  './js/main.js',
  './js/pipeline.js',
  './js/png.js',
  './js/touch.js',
  './js/viewport.js',
  './js/worker.js',
  './lib/ort/ort.wasm.min.js',
  './lib/ort/ort-wasm-simd-threaded.mjs',
  './lib/ort/ort-wasm-simd-threaded.wasm',
  './models/informative-drawings.onnx',
  './fonts/fraunces-latin-ext.woff2',
  './fonts/fraunces-variable.woff2',
  './fonts/newsreader-italic-latin-ext.woff2',
  './fonts/newsreader-italic-variable.woff2',
  './fonts/newsreader-latin-ext.woff2',
  './fonts/newsreader-variable.woff2',
  './apple-touch-icon.png',
  './favicon-16.png',
  './favicon-32.png',
  './favicon.ico',
  './icon-192.png',
  './icon-512.png',
];

self.addEventListener('install', (event) => {
  // GitHub Pages caches files for up to 10 minutes, on its servers as well as in the
  // browser. cache: 'reload' gets past the browser and the version in the URL gets past
  // GitHub, so a new version can't install with old files. Stored without the query.
  event.waitUntil(caches.open(CACHE).then((cache) => Promise.all(ASSETS.map(async (url) => {
    const res = await fetch(`${url}?v=${VERSION}`, { cache: 'reload' });
    if (!res.ok) throw new Error(`${url}: ${res.status}`);
    await cache.put(url, res);
  }))));
});

self.addEventListener('activate', (event) => {
  // Only this app's caches: other Pages sites on this account share the same origin.
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys
        .filter((k) => k.startsWith('dodo-stencils-') && k !== CACHE)
        .map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('message', (event) => {
  if (event.data === 'skipWaiting') self.skipWaiting();
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) return;
  if (req.mode === 'navigate') {
    event.respondWith(caches.open(CACHE).then((c) => c.match('./')).then((hit) => hit || fetch(req)));
    return;
  }
  event.respondWith(caches.open(CACHE).then((c) => c.match(req, { ignoreSearch: true })).then((hit) => hit || fetch(req)));
});
