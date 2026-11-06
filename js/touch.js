// Touch behaviour for phones: no page zoom, and sliders that are easy to hit.

/**
 * Stops the page itself ever zooming. CSS touch-action does most of it, but iOS Safari
 * ignores the viewport's user-scalable setting and has its own gesture events, so both
 * are needed. The preview box's own pinch handling isn't affected, because that runs
 * from touch events and these only cancel the browser's default zoom.
 */
export function blockPageZoom() {
  for (const type of ['gesturestart', 'gesturechange', 'gestureend']) {
    document.addEventListener(type, (e) => e.preventDefault(), { passive: false });
  }
  document.addEventListener('touchmove', (e) => {
    if (e.touches.length > 1 && e.cancelable) e.preventDefault();
  }, { passive: false, capture: true });
  document.addEventListener('wheel', (e) => {
    if (e.ctrlKey || e.metaKey) e.preventDefault(); // how trackpad pinches arrive
  }, { passive: false });
}

const THUMB_PX = 32; // matches the thumb size in css/styles.css
const DRAG_START_PX = 6;

function showFill(input) {
  const min = Number(input.min);
  const max = Number(input.max);
  const ratio = max === min ? 0 : (Number(input.value) - min) / (max - min);
  // Offset by half a thumb so the fill ends under the thumb's centre.
  input.style.setProperty('--fill', `calc(${THUMB_PX / 2}px + (100% - ${THUMB_PX}px) * ${ratio})`);
}

function setFromX(input, clientX) {
  const rect = input.getBoundingClientRect();
  const min = Number(input.min);
  const max = Number(input.max);
  const step = Number(input.step) || 1;
  const ratio = Math.min(1, Math.max(0, (clientX - rect.left - THUMB_PX / 2) / (rect.width - THUMB_PX)));
  const value = String(min + Math.round((ratio * (max - min)) / step) * step);
  if (value !== input.value) {
    input.value = value;
    input.dispatchEvent(new Event('input', { bubbles: true }));
  }
}

/**
 * Safari's built-in slider only responds if your finger lands on the thumb. This makes
 * the whole row work: tap anywhere to jump there, or drag sideways. A vertical swipe
 * still scrolls the page and leaves the value alone.
 */
export function enhanceSlider(input) {
  showFill(input);
  input.addEventListener('input', () => showFill(input));

  let gesture = null;
  input.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse') return;
    gesture = { x: e.clientX, y: e.clientY, sideways: false };
  });
  input.addEventListener('pointermove', (e) => {
    if (!gesture) return;
    if (!gesture.sideways
      && Math.abs(e.clientX - gesture.x) > DRAG_START_PX
      && Math.abs(e.clientX - gesture.x) > Math.abs(e.clientY - gesture.y)) {
      gesture.sideways = true;
    }
    if (gesture.sideways) setFromX(input, e.clientX);
  });
  input.addEventListener('pointerup', (e) => {
    if (gesture && Math.abs(e.clientY - gesture.y) <= DRAG_START_PX) setFromX(input, e.clientX);
    gesture = null;
  });
  input.addEventListener('pointercancel', () => { gesture = null; }); // the page took it as a scroll
}
