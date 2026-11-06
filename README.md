# Stencil Generator

Turns finished tattoo artwork into a stencil, entirely on the phone: no server, no account, nothing uploaded anywhere. It's made for an iPhone and works offline once it has been opened online.

## How it's used

1. Design on a 2031 x 2870 px Procreate canvas (172 x 243 mm at 300 DPI) and export a PNG.
2. Upload it here. The download has exactly the upload's pixels, so it lands exactly on top of the design.
3. Download the stencil, which has a transparent background, and insert it into Procreate as a new layer.
4. Correct it by hand, hide the design layer, and print from Procreate. Lines should be black, and the white background layer switched on, before printing.

Width and DPI fill in from the upload. Editing either one changes the output from then on, until the next upload.

**Saving to Photos:** a website can't save straight to an iPhone's Photos. Download opens the share sheet instead, where Save Image puts it in Photos. The file is made ahead of time, because Safari only opens the sheet straight after a tap. If it isn't ready yet, the button asks for one more tap.

**Preview:** two fingers zoom where they are and move the image. One finger only ever scrolls the page, and pinching back out returns to the full view. Pinching anywhere else does nothing.

## Run it locally

Needs `http://`, not `file://`. Double-click `Start Stencil Generator.cmd`, or:

```
npx serve .
```

## Publish to GitHub Pages

Everything in this folder goes in the repo. In Settings > Pages, choose the branch and `/ (root)`. It's about 32 MB, nearly all of it the model and the engine, both within GitHub's file limits.

**When you change any file, bump `VERSION` at the top of `sw.js`.** That's the only version number in the app. Without the bump, installed copies keep the old version. Bumping it makes the app download the new files in the background and show an Update button. If you add a file, add it to `ASSETS` in `sw.js` too, or it won't be there offline.

GitHub's web upload replaces files but doesn't delete them, so remove renamed or deleted ones by hand.

Backups of earlier versions and the tests live next to this folder, in `App - Dodo stencils (dev)`, so they don't get published.

## How it works

- **Line art comes from a neural network, not an edge filter.** `models/informative-drawings.onnx` is trained to turn a picture into selective, confident linework.
- **One analysis per image.** The model runs once on a copy of the upload at most 1200 px on its longest side. Slider moves and downloads all draw from that result, so they don't re-run it, and the lines in a download match the preview.
- **Downloads are full size.** The stencil is enlarged from the model's soft output first and only then cut to solid lines, which keeps its edges clean at any size. The reference is made directly from the full upload, not the analysis copy.
- **The preview is capped at about 2.7 megapixels** so sliders stay quick on a phone, so its edges can differ very slightly from the download.

| File | What it does |
| --- | --- |
| `js/worker.js` | Runs the model and the image processing off the main thread. |
| `js/pipeline.js` | Image processing: line clean-up, reference, colours. No DOM. |
| `js/png.js` | Writes PNGs with the DPI recorded and, for stencils, transparency. |
| `js/viewport.js` | Pinch-to-zoom maths. No DOM. |
| `js/touch.js` | Blocks page zoom, and makes sliders respond to a tap anywhere on the strip. |
| `js/main.js` | The screen: upload, preview, tabs, downloads. |
| `sw.js` | Offline cache and updates. |

The three downloads:

- **Stencil:** lines in black or a chosen colour, on transparent.
- **Reference:** the design in greyscale.
- **Stencil + Reference:** coloured lines over the reference, on a solid background.

## Limits

- **iOS version:** the engine needs WebAssembly SIMD, so iPhones need iOS 16.4 or newer. Older ones get a message saying so.
- **Print size:** output above 36 megapixels is refused, because it would run a phone out of memory.
- **Line character:** the model won't reproduce hand-drawn conventions such as solid versus dashed lines. It gives a strong starting point, not a finished stencil.

## Credits

- `models/informative-drawings.onnx`: ONNX export by [Joseph Rocca](https://github.com/josephrocca/image-to-line-art-js) of **Informative Drawings** by Caroline Chan et al. (CVPR 2022), [original repo](https://github.com/carolineec/informative-drawings), MIT licensed.
- `lib/ort/`: [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) 1.31.1, MIT licensed.
- `fonts/`: Fraunces and Newsreader, SIL Open Font License (see the `OFL-*.txt` files).
