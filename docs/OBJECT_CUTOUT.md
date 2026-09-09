# Object Cutout — manual edge guidance and watershed

**Optional third method:** [AI-assisted cutout](AI_CUTOUT.md) now provides an
explicit local pretrained-model choice. Manual watershed remains the default;
it and GrabCut never fall back to AI. This guide below covers the original two
non-AI methods; see the AI guide for its separate setup and soft-mask contract.

The editor now defaults to **Edge-guided watershed**: our existing manual
Gaussian and Sobel functions generate the elevation that determines foreground
segmentation. **GrabCut** remains a separate comparison method. Both retain the
actual uploaded colours; neither generates replacement pixels or calls an
external image service. The mathematical modules and `/analyze` contract are
unchanged. Fourier analysis still works in Analyze but does not enter cutout.

**Brush-free update:** draw a rectangle with a background margin and click
**Extract object**. Starting regions use background colours and quiet Sobel
interiors, not closed-edge nesting. Best with a fairly uniform background.
Keep/Remove brushes are optional corrections, grouped under a collapsed panel.
Clear marks retains the rectangle and returns to automatic selection. This
updates the earlier requirement to paint Keep before every first extraction.

## Setup and startup

From `D:\SpectraEdge`, install the updated backend requirements into the existing
virtual environment. This milestone adds `scikit-image==0.26.0` for watershed;
it also installs its transitive dependencies (including SciPy). Those packages
are **not** used to replace manual Gaussian, convolution, Sobel, or Fourier.
The existing `opencv-python-headless==4.14.0.94` supplies GrabCut, brush
rasterization and seed-interior erosion. The brush-free update adds no new
dependencies. Do not install a second OpenCV package in this environment.
Verified with Python 3.14.5 and NumPy 2.5.3; `pip check` passes.

```powershell
cd D:\SpectraEdge
.\backend\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
```

If starting from a fresh checkout, first follow the root README's environment
creation and `npm.cmd run setup` instructions. No frontend dependency was added.

Backend terminal (restart it after updating the source):

```powershell
cd D:\SpectraEdge
.\backend\.venv\Scripts\python.exe -B -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Frontend terminal:

```powershell
cd D:\SpectraEdge
npm.cmd run dev
```

Open <http://localhost:3000/cutout>, or choose **Object Cutout** in the navigation.
The existing `NEXT_PUBLIC_API_URL` and `SPECTRAEDGE_CORS_ORIGINS` settings apply.
Keep the backend on loopback; this is not an authenticated public upload service.
No commit, push, branch change, or deployment is required.

## Demonstration

1. Choose or drop an original colour photo. Choosing a file immediately sends it
   to the configured Python preparation endpoint. Unlike Analyze, preparation
   does not wait for Process.
2. Select **Edge-guided watershed** (the default). Set Gaussian sigma and kernel
   size; start with sigma 1.2 and 5×5. Sigma 0 skips smoothing.
3. Draw a rectangle around the person/object in any direction, leaving a little
   background **inside** the box around it. Mouse and single-touch are supported.
4. Click **Extract object**, without painting. Background colours and quiet
   Sobel interiors supply automatic starting regions; the rectangle centre is
   not assumed to be foreground. A subject may continue beyond the photo edge.
5. Inspect **Edge guidance** (bright
   means strong Sobel magnitude) and **Mask** (white means retained foreground).
6. Only if needed, expand **Optional brush corrections**. Keep identifies your
   intended foreground; Remove rejects unwanted regions. Re-extract after edits.
   Later strokes win. Undo removes a stroke; Clear marks returns to automatic
   selection without losing the box; Reset selection removes box and marks.
   Drawing a new rectangle also clears marks.
7. Switch checkerboard/light/dark backgrounds to inspect boundaries. These are
   CSS-only preview backgrounds and never enter the downloaded image.
8. Optionally enable **Trim empty space** and download the transparent PNG.
   Download always exports the cutout, even in Mask or Edge guidance view.
   Those previews stay full-frame; trimming affects cutout preview/export only.
9. To compare methods, select **GrabCut** and extract with the same photo and
   marks. The previous result is cleared. The result header identifies the
   actual method; watershed results also show the parameters that were used.

Editing the photo, method, sigma, kernel size, rectangle, or marks immediately
invalidates the result and disables downloading it. New requests use new UUIDs
and deep-copied selections **and method/Gaussian parameters**;
late successes and errors are ignored. Changing tool, brush size for the next
stroke, preview background, or trimming does not invalidate unchanged geometry.
Escape cancels the current drawing gesture. Refreshing/leaving the page discards
its photo, selection, and result; nothing is persisted as session history.

## What to tell your teacher

> Our manual Gaussian and Sobel implementations generate the elevation map used by marker-controlled watershed to separate foreground and background.

Strong edges guide the boundary. Starting regions use a background-colour model;
optional Keep/Remove marks correct that interpretation.
An edge map alone is not an object mask. Watershed floods the continuous
gradient surface from inferred or user-labelled seeds; high ridges influence where those
regions meet. We use `skimage.segmentation.watershed`, which accepts an elevation
array directly. We do **not** pass the magnitude to `cv2.watershed`, whose input
is a colour image. [scikit-image watershed documentation](https://scikit-image.org/docs/0.26.x/api/skimage.segmentation.html#skimage.segmentation.watershed).

Gaussian and Sobel are manual course implementations; seed discovery and
watershed use library helpers. Closed edge loops do not identify a person or
distinguish clothing details from background holes. We no longer alternate
foreground/background labels based on nested contours.
FFT is not used in segmentation. GrabCut uses colour models and graph cuts, not
this elevation. [OpenCV GrabCut tutorial](https://docs.opencv.org/4.13.0/d8/d83/tutorial_py_grabcut.html).

Weak or missing edges, textured backgrounds, fine hair, shadows and complex
boundaries can require additional Keep/Remove seeds and parameter adjustment.
More smoothing can suppress noise but also erase useful boundaries. There is
no automatic person recognition or guarantee of
a perfect cutout. This is binary segmentation, not fine-hair alpha matting;
existing partial alpha is preserved, but new soft alpha edges are not generated.

## How edge-guided extraction works

1. Decode/orient the source and create an RGBA prepared image, with no white
   compositing. If needed, nearest-neighbour sampling fits it inside 512×512
   without upscaling. This retains sampled RGB/alpha values but can alias fine
   detail. The prepared PNG is the coordinate and colour reference thereafter.
2. Convert RGB to floating-point grayscale with BT.601 weights
   `0.299 R + 0.587 G + 0.114 B`, without alpha compositing. Call existing
   `gaussian_blur(gray, sigma, kernel_size)`, then `sobel(filtered)`.
   Gaussian and Sobel both use our true 2D convolution with reflection padding.
   Sobel returns signed `gx`, `gy` and `sqrt(gx² + gy²)`; no clipping occurs.
3. Initialize integer markers: outside the rectangle is background (`2`);
   inside is **unknown** (`0`). The centre is not assumed to be foreground.
4. Replay every brush stroke in order, directly into the label image. Keep is
   definite foreground (`1`); Remove is definite background (`2`). A Keep stroke may
   override background outside the rectangle. Round stroke ends are rasterized
   with OpenCV's integer drawing operations, shared with GrabCut. Browser brush
   overlays are guides: antialiasing can differ from the integer seed footprint
   by approximately one pixel at the boundary.
5. Force originally alpha-zero pixels to background, even under Keep. Exclude
   them from watershed's flooding domain (`mask=alpha>0`), so transparent holes
   cannot spread background through an otherwise uniform visible object.
6. Without any Keep strokes, infer confident foreground/background cores from
   RGB and the actual magnitude (details below), only in still-unknown pixels. Remove and
   transparency constraints win. If Keep strokes exist, use them instead of
   automatic foreground seeds. Overwritten/transparent Keep does not secretly
   recreate foreground. Both classes must have markers; an ambiguous automatic
   selection asks for a box with more uniform background, or optional brushes.
7. Pass that **same float magnitude array** directly to watershed. No binary
   edge mask, uint8 preview, positive multiplier, or FFT replaces its elevation.
   Use `connectivity=1` (four neighbours: up/down/left/right), `compactness=0`,
   `watershed_line=False`. Watershed uses elevation then queue-entry order for
   ties; with fixed arrays/settings and the pinned library, runs are deterministic
   and require no random seed. Constant/all-zero elevations work with explicit
   brush seeds, but automatic inference reports insufficient selection evidence.
8. Select the foreground-labelled basin and explicitly enforce all hard markers
   again. No automatic hole filling or largest-component filtering occurs.
   Disconnected foreground and background holes are allowed; unseeded ambiguous
   regions may need additional marks. Copy the original prepared RGBA array;
   preserve RGB exactly and preserve source alpha for selected pixels. Set only
   rejected alpha to zero. Return an error with refinement guidance if no visible
   foreground remains.
9. Find the bounding box of nonzero output alpha and slice the same cutout for
   the trimmed PNG. The mask is binary 0/255, separate from source partial alpha.

Only **after segmentation**, encode the exact magnitude as an optional grayscale
PNG with `round(255 * magnitude / maximum)` for display. Zero maximum produces a
black image. `guidance_scale` returns the actual maximum (zero for a flat image).
This preview necessarily quantizes floats into 256 levels; the numerical
watershed input retains full precision. Different runs may have different
preview maxima, so compare their reported scales as well as appearance.

### How automatic seeds are inferred

This is a bounded colour/edge heuristic, not automatic semantic recognition:

1. Otsu's threshold on the manual magnitude identifies quiet interiors (at or
   below the threshold). This classification is for seeds only, not watershed.
2. Sample visible, quiet pixels from a thin rim **inside** the selection. Its
   thickness is the smaller box dimension divided by 20, clamped to 1–5 pixels.
   Find the most frequent coarse RGB bin (16 channel values per bin), then its
   median colour. Group samples within Euclidean RGB distance 24 of that median.
   Require at least four rim samples and at least 55% support for this model;
   otherwise ask for a better selection or optional brushes.
3. Refine the background colour to the grouped samples' median. Background
   radius is `max(8, percentile95(sample distances) + 2)`. Foreground radius is
   `background_radius + max(12, background_radius)`. These are heuristic distances
   in 0–255 RGB units, not probabilities or learned semantic classifications.
4. Pixels near that colour are background candidates; pixels sufficiently far
   from it are foreground candidates. Erode each candidate mask with a 3×3
   square, using replicated borders. Seed only eligible, visible, quiet cores.
   Intermediate colours and edge pixels remain unknown. Multiple colour regions
   within a face, shirt and jacket can now all receive foreground seeds.
5. Never label the whole box rim background. A subject touching the photo edge
   can receive foreground seeds there. Remove marks and source transparency win.
   Watershed determines the remaining boundary using the **unchanged continuous
   manual magnitude**. No GrabCut, contour nesting, hole filling, or
   largest-component selection is hidden in this automatic path.

The model assumes the background dominates the quiet selection rim and has a
fairly uniform colour. A flat image or ambiguous rim fails clearly. A busy
background, similar object/background colours, shadows or very small features
can still give an incorrect mask; confidence in a colour model is not confidence
in semantic correctness. In particular, background-coloured clothing can be
mistaken for a hole. Use optional brushes or the separate GrabCut comparison
when needed. This remains binary segmentation, not soft-edge matting.

### Portrait regression: why the earlier automatic result collapsed

The supplied 547×365 portrait is prepared at 512×342. With the screenshot's box
`{x:132, y:4, width:243, height:338}`, sigma 1.2, kernel 5 and **no marks**, the
old contour-nesting seeder retained just **704 pixels**. It classified nested
clothing details as holes and imposed background along the bottom of the box,
even where the subject continues beyond the photo. A Keep stroke bypassed this
automatic seeder, explaining the much better brush result.

The new seeder retains **47,308 pixels** on the same prepared photo/settings.
Checks cover face, shirt, jacket, trousers, the bottom of the subject, and known
background points. These are regression checks, not a pixel-perfect ground-truth
mask or a general accuracy claim. The result was also inspected against a
checkerboard: the person is retained, with background removed beside the arms.
A live localhost API request returned the same 47,308-pixel mask (about 1.58 s
for extraction on this run). Full-frame and wider boxes, plus sigma 0, retained
the checked subject points too. Original RGB remains unchanged. The private
photograph is not stored in this repository; an opt-in API test reads it from a
local path and otherwise skips:

```powershell
$env:SPECTRAEDGE_TEST_PORTRAIT = 'C:\path\to\the-original-images.jpg'
.\backend\.venv\Scripts\python.exe -B -m unittest backend.tests.test_edge_cutout -q
Remove-Item Env:SPECTRAEDGE_TEST_PORTRAIT
```

### GrabCut comparison

GrabCut's existing initialization is unchanged: rectangle interior is probable
foreground, outside is definite background, and ordered Keep/Remove marks plus
alpha-zero constraints override those defaults. It needs at least five samples
per class, runs five `GC_INIT_WITH_MASK` iterations on original BGR colour, and
uses seed 220 under the shared serialized cutout worker. It does not call manual
Gaussian/Sobel or return edge guidance. Both methods use the same unchanged-RGB
mask application and cropped/full-frame PNG encoding.

## Performance and the prepared grid

Before choosing the limit, one timed pass per setting on this machine measured:

| Square side | Sigma / kernel | Gaussian (s) | Sobel (s) | Total (s) |
| --- | --- | --- | --- | --- |
| 256 | 1.2 / 5 | 0.159 | 0.342 | 0.501 |
| 256 | 5 / 31 | 0.299 | 0.438 | 0.737 |
| 512 | 1.2 / 5 | 0.697 | 1.280 | 1.976 |
| 512 | 5 / 31 | 0.942 | 1.666 | 2.607 |
| 1024 | 1.2 / 5 | 3.278 | 5.375 | 8.653 |
| 1024 | 5 / 31 | 3.763 | 5.378 | 9.140 |

These are filter-only wall times, not guarantees or full request timings.
Repeat the same benchmark from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -B -m backend.benchmarks.cutout_filters
```

The shared limit is now **512 pixels on the longest side** for quicker manual
refinement. Both methods use the same prepared grid for comparison. Uploads are
prepared at that size **before drawing**; extraction never resizes. Canvas,
markers, grayscale, elevation, full mask and full output share dimensions.
Export is at prepared resolution, not full original camera resolution. Existing
1024-pixel prepared sessions must upload again; they are rejected, not resized
silently. Nearest sampling preserves sampled RGB/alpha but can lose/alias fine
detail. Manual filters have not been substituted with library equivalents.

Alpha hides rejected pixels; it is **not secure redaction**. RGB underneath
transparent pixels remains present in the full-frame PNG. Trimming removes
pixels outside the foreground bounding box, not hidden RGB within that box.

## Stateless API contract

Both endpoints accept multipart uploads, return JSON, and echo a canonical UUID.
Every returned image is a `data:image/png;base64,...` URL. All processing times
are measured milliseconds including decode, segmentation/preparation, and PNG
encoding, excluding transfer and JSON serialization. `/docs` lists both endpoints.

### POST /cutout/prepare

Exactly two multipart fields:

| Field | Meaning |
| --- | --- |
| `image` | Actual PNG, JPEG, WebP, or GIF bytes; first frame only |
| `request_id` | Client UUID, at most 128 bytes |

Response: `request_id`, `image_id` (SHA-256 of exact prepared PNG bytes),
`prepared_image` (RGBA PNG), `width`, `height`, `source_dimensions`
(`{width,height}` after EXIF orientation), and `processing_time`.

Limits: 20 MiB uploaded image, 20 million decoded pixels, prepared maximum side
512, one file/one text field. The streamed body is limited to file allowance
plus text allowance plus 64 KiB multipart overhead. The image's actual decoded
format is checked independently of its filename or MIME type.

### POST /cutout/extract

Exactly two multipart fields:

| Field | Meaning |
| --- | --- |
| `image` | The **identical prepared RGBA PNG bytes**, not the original upload or a canvas screenshot |
| `settings` | JSON text, at most 1 MiB, with the structure below |

```json
{
  "request_id": "12345678-1234-1234-1234-123456789abc",
  "image_id": "<64-character SHA-256 from prepare>",
  "width": 512,
  "height": 384,
  "method": "edge-watershed",
  "sigma": 1.2,
  "kernel_size": 5,
  "rectangle": { "x": 100, "y": 40, "width": 350, "height": 300 },
  "marks": [
    { "mode": "keep", "size": 20, "points": [{ "x": 200, "y": 140 }, { "x": 210, "y": 160 }] },
    { "mode": "remove", "size": 8, "points": [{ "x": 120, "y": 80 }] }
  ]
}
```

`method` is `edge-watershed` or `grabcut`. The **frontend default is watershed**
and always sends the method explicitly. For compatibility, older API requests
that omit `method` still invoke GrabCut. Edge-guided requests require numeric
sigma 0–5 in steps of 0.1 and an odd integer kernel size 3–31, matching Analyze's
validation. Sigma 0 skips smoothing but still validates the kernel. GrabCut
requests must omit sigma/kernel fields entirely; unused parameters are rejected.

Coordinates are integer prepared-image pixels, with origin at top-left. Brush
points are pixel centres, `0 <= x < width`, `0 <= y < height`. The rectangle
uses exclusive right/bottom bounds, must stay inside the image, and must be at
least 2×2. The frontend includes both dragged endpoint pixels. Pointer mapping
uses the actual unbordered canvas bounds on each event, not the whole panel or
the original photo dimensions.

Brush diameter: 1–128 prepared pixels. Limits: 200 marks, 2000 points per mark,
20,000 points total; marks are ordered. Empty point lists, boolean/fractional
coordinates/sizes, unknown fields, and out-of-bounds values are rejected.
Extraction image upload limit is 6 MiB, with a streamed total limit of that
allowance plus 1 MiB settings plus 64 KiB overhead. Extraction rejects PNGs with
the wrong dimensions, colour mode, or hash and never resizes/reorients them.
The hash detects accidental mismatches; it is not an authentication token.

Response:

| Field | Meaning |
| --- | --- |
| `request_id`, `image_id` | Match request and prepared source |
| `width`, `height` | Full-frame prepared dimensions |
| `method` | Actual `"edge-watershed"` or `"grabcut"` |
| `algorithm` | `"skimage-watershed"` or `"opencv-grabcut"` |
| `parameters_used` | `{sigma,kernel_size}` for watershed; `{}` for GrabCut |
| `seed_mode` | `"automatic"` without Keep strokes, `"brush"` with Keep; `null` for GrabCut |
| `cutout_image` | Full-frame RGBA PNG |
| `cropped_image` | Trimmed RGBA PNG |
| `mask_image` | Full-frame grayscale binary mask PNG |
| `guidance_image` | Full-frame grayscale magnitude visualization for watershed; `null` for GrabCut |
| `guidance_scale` | `{min:0,max:<actual magnitude maximum>,mapping:"linear_grayscale"}` or `null` |
| `foreground_bounds` | `{x,y,width,height}` of visible foreground |
| `cropped_dimensions` | `{width,height}` matching the bounding box |
| `processing_time` | Measured milliseconds |

No photo cache or previous mask is kept server-side. Every refinement resends
the same prepared image, rectangle, and complete ordered stroke list. The
frontend checks identities, dimensions, crop bounds, actual method/parameters
against the immutable request snapshot, guidance metadata, timing, and PNG
IHDR dimensions/colour types before accepting a response.

### Resource bounds and errors

One cutout request per Python process is admitted **before upload parsing**;
extra requests receive 429 with retry guidance. CPU work runs off the async
request loop. Analyze retains its own existing lock, so at most one analysis
and one cutout can run concurrently per process. Do not add multiple Uvicorn
workers without reviewing aggregate memory/CPU limits.

The multipart spool threshold exceeds the streamed body limit; uploads remain
in memory and are closed on completion/error. There is no database, disk photo
storage, history, or external image service. Browser cancellation protects UI
state but does not interrupt Python filters/segmentation already executing;
the worker remains occupied until completion. Retry after it finishes. A
120-second frontend timeout bounds waiting, not the Python computation itself.

Errors return a readable `detail` string: 400 invalid image/multipart/oversized
text part, 403 disallowed browser origin, 413 file/body/pixel limit, 415 format,
422 invalid geometry/settings or insufficient samples, 429 busy, 500 safe
unexpected failure. Internal errors are logged without uploading image pixels
to any external service.

## Verification commands

From the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s backend/tests/signal_ops -q
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s backend/tests/detection -q
.\backend\.venv\Scripts\python.exe -B -m unittest backend.tests.test_api backend.tests.test_cutout backend.tests.test_edge_cutout -q
.\backend\.venv\Scripts\python.exe -m pip check
npm.cmd run check
npm.cmd run build
```

With Python running:

```powershell
npm.cmd --prefix frontend run test:integration
```

Verified on 2026-09-09: 74 signal-ops tests, 14 detection tests and 67 API/cutout
tests passed (155 Python tests total, with the opt-in portrait supplied). Without
that local fixture, one portrait test skips and the other 154 pass. `npm.cmd run check` passed ESLint,
TypeScript and all 39 frontend tests. All five live HTTP integration tests
passed; `pip check` reported no broken requirements. Production build passed;
Vinext printed its existing informational unknown-route-classification notice.
All four local routes (`/`, `/compare`, `/live`, `/cutout`) returned HTTP 200.
The benchmark script also completed. No browser-opening tool was available;
mouse/touch interaction, visual layout and browser download behavior were not
manually verified. Automated geometry, PNG encoding, request and stale-state
checks passed, but they are not substitutes for those browser checks.

The edge tests spy on the real manual Gaussian/Sobel path and verify that its
exact floating-point magnitude reaches watershed. A fixed-seed fixture with a
moved elevation barrier changes its foreground boundary: elevation causally
affects the result. Brush-free tests cover off-centre objects, deterministic
automatic seeds, holes, multiple objects, full-frame boxes, Remove-only
refinement, source alpha and ambiguous/flat backgrounds. New regressions cover
multicolour subjects reaching the photo bottom, nested printed details that are
not holes, nearby background colours straddling RGB bins, and the original
portrait's brush-free API result. Tests also cover
overwritten seeds, unknown centre,
outside Keep, ordered/continuous strokes, holes, disconnected foreground,
constant/zero gradients, original RGB/alpha, exact guidance encoding, crop and
grid dimensions, parameter bounds, failures and worker release.

The original cutout tests still run real GrabCut on deterministic colour fixtures and verify
alpha, unchanged RGB, outside-rectangle Keep, later-mark precedence, partial and
zero source alpha, cropping, orientation/resizing, errors, limits, and no disk
spooling. Frontend unit tests cover coordinates, drag directions, validation,
exact prepared PNG submission, Gaussian/method response matching, immutable
parameter snapshots, and stale success/error cancellation across both
endpoints. Live HTTP checks use the real frontend adapter. These automated checks
do not constitute a mouse/touch, visual-layout, or browser download test.

## This milestone's files and ownership

Added:

- `backend/edge_cutout.py`: manual elevation, automatic/brush seeds and library watershed.
- `backend/tests/test_edge_cutout.py`: causal numerical, constraint and API tests.
- `backend/benchmarks/cutout_filters.py`: reproducible manual-filter benchmark.

Updated:

- `backend/cutout.py`: share brush rasterization and alpha application; preserve GrabCut.
- `backend/cutout_api.py`: method validation/routing, 512-pixel preparation, response provenance/guidance.
- `backend/requirements.txt`: add pinned scikit-image.
- `backend/tests/test_cutout.py`: adapt existing resize assertions to the benchmarked cap.
- `frontend/lib/cutout.ts`, `cutout-api.ts`: method/parameter contracts, validation, snapshots and response matching.
- `frontend/components/use-cutout.ts`: default method, clear-marks action and parameter-change invalidation.
- `frontend/components/cutout-workspace.tsx`, `cutout.module.css`: existing editor's controls, guidance preview, result labels and instructions.
- `frontend/app/cutout/page.tsx`: accurate page metadata.
- `frontend/tests/cutout.test.ts`, `cutout-fixture.ts`, `integration.test.ts`: parameter, response, cancellation and real HTTP tests.
- `docs/OBJECT_CUTOUT.md`: this updated handoff.
- Root, backend, and frontend READMEs: discoverability and setup clarification.

The teammate's remaining detector, contour/object, experiment, and full-pipeline
work stays separate. Future API/UI changes should be coordinated through the
existing frontend ownership. Existing canvas/navigation, mathematical modules,
Analyze endpoint, frontend dependencies and hosting configuration were not
changed in this milestone. Existing uncommitted work is retained; no branch
change, commit, push or deployment is part of this work.
