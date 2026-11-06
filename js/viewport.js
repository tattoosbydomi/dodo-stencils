// Pinch-to-zoom maths for the preview box. No DOM, so it can be tested on its own.
//
// A view is { zoom, tx, ty }: a point p in the fitted (zoom 1) preview appears on
// screen at (tx + zoom * p.x, ty + zoom * p.y), in CSS pixels from the box's corner.

export const MAX_ZOOM = 4;

export const FIT_VIEW = Object.freeze({ zoom: 1, tx: 0, ty: 0 });

/** Where an image sits at zoom 1: as large as fits, centred in the box. */
export function fitRect(imageW, imageH, boxW, boxH) {
  const scale = Math.min(boxW / imageW, boxH / imageH);
  const w = imageW * scale;
  const h = imageH * scale;
  return { x: (boxW - w) / 2, y: (boxH - h) / 2, w, h };
}

/** Where the image sits on screen under a given view. */
export function imageRect(view, fit) {
  return {
    x: view.tx + view.zoom * fit.x,
    y: view.ty + view.zoom * fit.y,
    w: view.zoom * fit.w,
    h: view.zoom * fit.h,
  };
}

// One axis: an image narrower than the box stays centred; a wider one can't be
// dragged far enough to leave a gap at either edge.
function clampAxis(translate, zoom, fitStart, fitSize, boxSize) {
  const size = zoom * fitSize;
  const start = translate + zoom * fitStart;
  const wanted = size <= boxSize
    ? (boxSize - size) / 2
    : Math.min(0, Math.max(boxSize - size, start));
  return wanted - zoom * fitStart;
}

export function clampView(view, fit, boxW, boxH) {
  const zoom = Math.min(MAX_ZOOM, Math.max(1, view.zoom));
  return {
    zoom,
    tx: clampAxis(view.tx, zoom, fit.x, fit.w, boxW),
    ty: clampAxis(view.ty, zoom, fit.y, fit.h, boxH),
  };
}

/**
 * The view during a pinch. The point that was under the fingers when the pinch began
 * stays under the fingers' midpoint, so it zooms where you pinch, and moving both
 * fingers together moves the image.
 * start: { view, mid: {x, y}, distance }   now: { mid: {x, y}, distance }
 */
export function pinchView(start, now, fit, boxW, boxH) {
  const zoom = Math.min(MAX_ZOOM, Math.max(1, start.view.zoom * (now.distance / start.distance)));
  const px = (start.mid.x - start.view.tx) / start.view.zoom;
  const py = (start.mid.y - start.view.ty) / start.view.zoom;
  return clampView({ zoom, tx: now.mid.x - zoom * px, ty: now.mid.y - zoom * py }, fit, boxW, boxH);
}
