# SpectraEdge Python backend

FastAPI service for manual Gaussian/Fourier modules, reproducible Gaussian and salt-and-pepper noise experiments, Sobel, Prewitt, and Laplacian detection, plus connected-component boundary analysis.

**[Object Cutout](../docs/OBJECT_CUTOUT.md)** exposes `/cutout/prepare` and `/cutout/extract`. Manual Gaussian/Sobel produce the elevation for scikit-image watershed; OpenCV GrabCut remains a comparison method. Reinstall requirements and restart Python. Its 512-pixel RGBA preparation, markers, method/parameter provenance and guidance contract are documented separately from the unchanged `/analyze` contract below. FFT remains analysis-only.

## Run

**Optional AI:** [AI-assisted cutout](../docs/AI_CUTOUT.md) is a third explicit
method, with separate `requirements-ai.txt` and photo-free model setup. Base
requirements and the manual mathematics are unchanged. Optional libraries and
the cached model load only on an explicit AI extraction, never during app startup.

Follow [the root README](../README.md) for dependency installation. Run `npm.cmd run dev` from the project root to start both services. Python 3.12+ is required. To start only the backend:

```powershell
.\backend\.venv\Scripts\python.exe -B -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Dependencies are pinned in `requirements.txt` and installed into `backend/.venv`. No global Python changes are needed. Restart Python after source edits.

`/health` reports readiness and `/docs` provides interactive request documentation. CORS permits `http://localhost:3000` and `http://127.0.0.1:3000`. Override with `SPECTRAEDGE_CORS_ORIGINS`, a comma-separated list of exact origins. Browser POSTs from other origins are rejected before parsing. See [FastAPI's CORS documentation](https://fastapi.tiangolo.com/tutorial/cors/) for the origin model.

## POST /live/frame

Accepts multipart `image`, `detector` (`Sobel`, `Prewitt`, or `Laplacian`),
`sigma` (0..5 in steps of 0.1), odd `kernel_size` (3..31), `threshold`
(0..1443), and optional `frame_id` (1..128 characters; otherwise a UUID).
Frames are limited to 2 MiB and 20 million decoded pixels, oriented and
composited on white, then fitted within 256 pixels per side without upscaling.

Returns `frame_id`, `settings_used`, `processing_time` in milliseconds, and
four PNG data URLs: `grayscale_image`, `filtered_image`, `edge_image`, and
`spectrum_image`. The spectrum uses the grayscale input before smoothing.
Live reuses the manual numerical modules and has one independent processing
slot; concurrent frames receive 429. It does not compute object measurements,
noise, or multi-scale fusion. Unexpected processing failures return a safe
JSON `detail` with status 500; diagnostics remain in the server terminal.

## POST /compare

Detector comparison accepts one in-memory multipart upload with `image`,
`sigma`, `kernel_size`, `sobel_threshold`, `prewitt_threshold`,
`laplacian_contrast_threshold`, and `request_id`. Limits and image preparation
match `/analyze`. The image is decoded and resized once, and
`backend.analysis.comparison` applies the existing Gaussian filter once before
passing the same floating-point array to Sobel, Prewitt, and Laplacian.

The response includes shared `parameters_used`, source/analyzed dimensions,
total request `processing_time`, and top-level `sobel`, `prewitt`, and
`laplacian` results. Each detector result contains its binary `edge_map`,
`object_list`, `object_count`, detector processing time, and metadata describing
the existing magnitude-threshold or zero-crossing decision. Per-detector time
includes its response, edge decision, and object analysis; shared decoding,
Gaussian filtering, and PNG encoding are represented only in total request
time. Sobel and Prewitt thresholds retain raw gradient-magnitude units;
Laplacian independently uses raw response difference across a zero crossing.
Each detector metadata object reports `threshold_type`, `threshold`, and units.

## POST /analyze request

Send the four required `multipart/form-data` fields below. Add **both** optional detection fields to select a detector. Let the client set the multipart boundary.

| Field | Value |
| --- | --- |
| `image` | One binary PNG, JPEG, WebP, or GIF file |
| `sigma` | Finite 0..5 inclusive, in increments of 0.1 |
| `kernel_size` | Odd integer 3..31 inclusive, even when sigma is zero |
| `request_id` | Client-generated UUID, echoed in canonical UUID form |
| `detector` | Optional; case-sensitive `Sobel`, `Prewitt`, or `Laplacian` |
| `threshold` | Required with any detector, forbidden without one; finite 0..1443 inclusive. Raw magnitude units for Sobel/Prewitt; raw response difference units for Laplacian |
| `multi_scale` | Optional boolean; when true, independently smooth and detect at each requested scale |
| `scale_sigmas` | Required with `multi_scale=true`; 1..5 unique comma-separated values from 0..5 in steps of 0.1 |
| `scale_support` | Optional integer from 1 through the scale count; defaults to 2 for multiple scales and 1 for one scale |
| `noise_model` | Optional; case-sensitive `Gaussian` or `Salt & Pepper` |
| `noise_strength` | Required with a noise model; Gaussian standard deviation from 0..100 intensity units, or Salt & Pepper corruption probability from 0..1 |
| `noise_seed` | Required with a noise model; integer from 0..4294967295 |
| `laplacian_min_component_area` | Optional for Laplacian only; integer 1..10000, default 2. Excludes smaller components from contour/object measurements, preserving raw edge and fusion maps |

Omitting all three noise fields preserves the clean grayscale pipeline. Noise fields must be supplied together. Omitting both detector and threshold preserves Gaussian/Fourier-only behavior. Unsupported models/detectors, incomplete groups, duplicates, and extra fields are rejected. Thresholds may be fractional; the UI slider uses step 1 and starts at 96. The UI suggests Gaussian support `2 * ceil(3 * sigma) + 1` (at least 3), while smaller kernels remain available for educational comparison.

Server-side limits, independent of browser validation:

- 20 MiB per file; the streamed total body is capped at 20 MiB + 64 KiB.
- One file, at most twelve text fields, at most 4096 bytes per text field.
- 20,000,000 decoded pixels, checked before decoding pixels.
- Format is checked from content, not just filename/MIME.
- The multipart parser memory threshold exceeds the bounded body, preventing temporary-file spooling. Uploads are never written to disk.

## Preparation and calculation

Pillow decodes frame zero, applies EXIF orientation, and fits the image within a 512-pixel maximum side without upscaling. Nearest-neighbour resizing avoids adding a hidden smoothing filter, but can alias fine detail. Transparency is composited on white; Pillow prepares RGB and grayscale images.

Grayscale becomes float64. Optional seeded noise creates a separate analysis input while the clean grayscale array remains unchanged. The API passes that analysis input to the existing `gaussian_blur()`, detector, multi-scale, and pre-smoothing Fourier paths. Mathematics stays in array-only `signal_ops` and `detection`; detector implementations, reflection padding, and public interfaces are unchanged.

```python
from backend.signal_ops import gaussian_blur
from backend.detection import sobel, threshold_edges

filtered = gaussian_blur(grayscale_array, sigma=1.2, kernel_size=7)
gx, gy, magnitude = sobel(filtered)  # New, same-shape floating-point arrays.
edges = threshold_edges(magnitude, threshold=96)  # New uint8 array: 0 or 255.
```

The numerical functions accept nonempty, real, finite 2D arrays and never mutate them. Magnitude must also be nonnegative. Invalid types raise `TypeError`; invalid shapes/values raise `ValueError`. Unlike the 8-bit API, the numerical helper accepts any finite nonnegative threshold and does not limit input intensity to 0..255. Sobel uses true-convolution kernel signs, documented with ramp and step examples in [the root README](../README.md#mathematics-to-explain).

## Response contract

Matching types: `AnalysisResult` and `ComputedAnalysisResult` in `frontend/lib/workspace.ts`. Every image field is a **PNG base64 data URL**, `data:image/png;base64,...`, with `analyzed_dimensions` dimensions.

| Field | Meaning |
| --- | --- |
| `provenance` | `"computed"` |
| `request_id` | Echoed UUID linking the result to the file/settings snapshot |
| `original_image` | Unfiltered RGB source at analyzed resolution, after orientation/resize/alpha handling |
| `grayscale_image` | Clean grayscale image, preserved when noise is enabled |
| `noisy_image` | Display PNG of the noisy analysis input; `null` when noise is disabled |
| `noise` | `{model, strength, units, seed}`; disabled noise reports `None`, zero, `none`, and `null` |
| `filtered_image` | Gaussian-smoothed analysis input; when sigma=0, identical to the noisy image if enabled, otherwise clean grayscale |
| `fft_image` | Centred log-magnitude spectrum of the clean or noisy analysis input **before smoothing** |
| `filtered_fft_image` | Centred log-magnitude spectrum **after smoothing** |
| `parameters_used` | Actual sigma/kernel and active detector, threshold, noise, and multi-scale settings; echoes an explicitly supplied Laplacian minimum component area |
| `source_dimensions` | `{width, height}` after orientation, before resizing |
| `analyzed_dimensions` | `{width, height}` after bounded preparation |
| `completed_stages` | Gaussian-only: `["input", "grayscale", "smooth", "fourier"]`; detector runs also include detector/decision, `contours`, and `objects` before `fourier` |
| `spectrum_scale` | `{min: 0, max: combined_maximum, mapping: "linear_grayscale"}` |
| `processing_time` | Measured **milliseconds**, including decode, preparation, numerical work, PNG encoding; excluding upload transfer and JSON serialization |
| `detection_status` | `"not_run"` for Gaussian-only; `"edges_computed"` for any detector |
| `detector_metadata` | Decision, threshold units/type/label, and minimum component area; `null` without detection |
| `gx`, `gy`, `gradient_magnitude` | PNG visualizations for Sobel/Prewitt; `null` for Laplacian and Gaussian-only |
| `laplacian_response` | Signed-response PNG for Laplacian; `null` otherwise |
| `edge_map` | Raw binary decision PNG for any detector; `null` for Gaussian-only |
| `contour_image` | Detector runs: outer boundary pixels are white, hole-boundary pixels gray, and pixels in both masks light gray; `null` without detection |
| `object_list` | Detector runs: connected foreground measurements, including an empty list when no foreground exists; `null` without detection |
| `multi_scale` | `null` unless enabled; otherwise scale sigmas, each scale edge PNG, persistence PNG/scale, support count, and fused binary edge PNG |
| `fps` | `null`: no video stream |

For both spectra, `M = max(original_log_spectrum.max(), filtered_log_spectrum.max())`. At encoding only, both use `round(255 * spectrum / M)`, clipped to 0..255 and cast to uint8. If M=0, use denominator 1 so both PNGs are black; metadata still records max=0. Spatial images use rounded/clipped intensity, not independent contrast stretching. Raw floating-point arrays stay unchanged.

### Fixed Sobel display mapping

`smooth` includes the sigma-zero bypass; each detector decision produces the binary edge map. `contours` and `objects` consume that same in-memory mask before any PNG encoding.

For 0..255 grayscale intensities, each Sobel component is bounded by ±`4 × 255 = 1020`. A conservative magnitude bound is `B = 1020 × sqrt(2) ≈ 1442.498`; the UI/API threshold ceiling of 1443 covers it.

- Gx/Gy PNG value: `round(127.5 × (component / 1020 + 1))`. Negative responses are dark, zero is gray 128, and positive responses are bright. Positive Gx points right; positive Gy points downward in image coordinates.
- Magnitude PNG value: `round(255 × magnitude / B)`. Neither this scale nor the signed scale is recomputed from each image's extrema.
- Edge PNG value: 255 when **raw** `magnitude > threshold`, otherwise 0. Equality is excluded. A flat image has no edges at threshold zero.

Clipping to 0..255 and conversion to uint8 happen only when encoding these display values. Do not feed a magnitude PNG back into thresholding. PNGs cannot recover the original signed floating-point data; a future numerical export needs its own explicit contract.

The browser retains the original File/local preview separately. Session JSON exports metadata; cards export images.

## Noise experiments

`backend.signal_ops.noise` provides array-only functions. Gaussian noise adds
seeded zero-mean samples with strength interpreted as intensity standard
deviation and does not clip its floating-point response. Salt & Pepper uses
strength as per-pixel corruption probability and replaces selected samples with
0 or 255, choosing each extreme with equal probability. Both return new float64
arrays and leave their input unchanged. The same noisy realization feeds every
multi-scale sigma, so persistence measures scale support rather than new random
draws. PNG clipping occurs only at the display boundary. The frontend uses seed
220 for repeatable experiments; API clients may supply any valid explicit seed.

## Multi-scale analysis

`backend.analysis.multiscale` receives the prepared grayscale array, selected
detector, scale sigmas, detector threshold, and optional support count. Every
scale starts from that same original grayscale array, selects Gaussian support
as `2 * ceil(3 * sigma) + 1` (bounded to 3..31), and calls the existing Gaussian
and detector functions. Sobel and Prewitt use the existing magnitude threshold;
Laplacian uses the existing zero-crossing decision. Detector mathematics are not
duplicated.

The persistence value at each pixel is the number of scale edge masks that are
nonzero there. The fused mask is 255 where `persistence >= scale_support`, and
zero elsewhere. Duplicate sigmas are rejected because they would add duplicate
votes. Single-scale mode defaults to support one and therefore matches its
normal edge decision exactly.

Laplacian contour/object analysis excludes components below the requested
minimum area (default two pixels), after fusion when enabled. The raw
single-scale and fused edge maps remain unchanged.

The ordinary `filtered_image`, derivative/response fields, and `edge_map` still
come from the request's single `sigma` and `kernel_size`. Only contours and
object measurements switch to the fused mask when multi-scale mode is enabled.
The cutout endpoints and their pipeline are unchanged. Fusion is pixel-aligned:
nearby edges at different coordinates do not support one another, and this is
not scale-space non-maximum suppression.

## Prewitt and Laplacian backend extension

Both detectors consume the same floating-point Gaussian output used by Sobel
and Fourier. They reuse `signal_ops.convolve2d`, including its reflect padding,
shape preservation, real/finite validation, and floating-point arithmetic.
Existing Sobel and Gaussian-only response fields and values are unchanged.

```python
from backend.detection import prewitt, laplacian, threshold_edges, zero_crossing_edges

gx, gy, magnitude = prewitt(filtered)
prewitt_edges = threshold_edges(magnitude, 96)
response = laplacian(filtered)
laplacian_edges = zero_crossing_edges(response, contrast_threshold=96)
```

Prewitt uses convolution kernels `[[1,0,-1],[1,0,-1],[1,0,-1]]` and
`[[1,1,1],[0,0,0],[-1,-1,-1]]`. After the convolution flip, positive Gx
points right and positive Gy points down. Returns signed Gx, signed Gy and
`sqrt(Gx*Gx + Gy*Gy)`, all new same-shape floats in raw units. Its API fields
match Sobel's, with `prewitt` replacing `sobel` in completed stages. Fixed
display bounds are +/-765 for components and `765*sqrt(2)` for magnitude.

Laplacian uses `[[0,1,0],[1,-4,1],[0,1,0]]` and returns a signed, same-shape
float response. Positive means the centre is darker than its four neighbours;
negative means brighter. `zero_crossing_edges` separately examines horizontal
and vertical pairs (four-connectivity, no diagonals or border wrapping). Both
endpoints of an opposite-sign pair are marked when `abs(a-b) > threshold`.
Equality is excluded. An exact-zero centre is marked only when its immediate
left/right or up/down neighbours have opposite signs and exceed that contrast.
Wider zero plateaus are not bridged, and weak signs are never rounded to zero.
The result is a new uint8 0/255 mask, without thinning or contour extraction.

Laplacian API responses keep `gx`, `gy`, and `gradient_magnitude` null. They
add `laplacian_response`, a signed-response **PNG visualization**, mapping
-1020 to black, zero to gray 128, and +1020 to white. This field is null
for other detectors. `edge_map` encodes the raw zero-crossing decision;
the PNG is never used for that decision. Completed stages include `laplacian`,
`zero_crossing`, `contours`, and `objects` before `fourier`. All detectors
report `edges_computed`, echo detector/threshold in `parameters_used`, and
return contours and connected foreground measurements. FPS remains null.

The shared API threshold range stays 0..1443 for compatibility. Unlike Sobel
and Prewitt, Laplacian contrast can reach 2040 for 8-bit input, so 1443 does
not guarantee an empty Laplacian mask. Numerical threshold helpers have no
upper bound. None of the numerical detector functions clip or normalize.

Use `/docs`, the frontend detector selector, or a multipart API client to select
any detector. The frontend labels first-order gradients separately from the
signed Laplacian response and its zero-crossing mask.

## Errors and cancellation

Errors contain a JSON `detail` string: 400 invalid image/multipart, 403 disallowed browser origin, 413 size/pixel limit, 415 unsupported request/image format, 422 invalid fields/settings, 429 engine busy, or 500 safe processing failure. Internal diagnostics stay in the terminal; uploaded pixels are not logged.

CPU work runs off the async request loop. Analyze and Compare share a process-local lock; Live and Cutout have separate processing slots. The browser aborts and invalidates requests when settings/images change, on reset, or unmount; already running numerical work finishes. No result history is retained. This is a bounded local demonstration, not an authenticated public upload service.

## Ownership handoff

Convolution, Gaussian filtering, Fourier analysis, all three detectors, connected-object analysis, multi-scale persistence fusion, detector comparison, noise experiments, and Live frames have API and frontend integration.

`backend.analysis` uses manual breadth-first labeling with eight-connected foreground and four-connected background. Area counts foreground pixels; perimeter counts exposed horizontal/vertical pixel sides; centroid is `[x,y]`; bounding boxes use `{x,y,width,height}`. Exterior and hole-boundary masks remain separate numerically. Public hosting/authentication/retention policies require a separate milestone.

## Tests

From the project root:

```powershell
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s backend/tests -p "test_*.py" -v
```

Detection tests use analytical ramp/step expectations, sign conventions, fractional constants, magnitude, raw thresholds, input preservation, and monotonic edge selection. API tests use HTTPX's in-process ASGI transport and generated uploads. They cover both request modes, actual outputs, same-array processing, fixed derivative/common spectrum scales, validation, multipart limits, no disk spooling, CORS, busy handling, and safe failures. The separate frontend `test:integration` script exercises its actual adapter against a running Python server over HTTP. Neither is a browser interaction test.

Full discovery also runs comparison, component, multi-scale, noise, Live, and
Cutout tests. Optional real-photo tests skip when their fixture environment
variables are absent. `npm.cmd --prefix frontend run test:live-integration`
checks the Live adapter over HTTP for all three detectors. Physical webcam
permission, rendering, and disconnect behavior still need a browser smoke test.
