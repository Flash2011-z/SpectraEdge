# SpectraEdge Python backend

FastAPI service for the existing manual Gaussian/Fourier modules plus manual Sobel and basic thresholding. No Prewitt, Laplacian, contours, object analysis, noise experiments, or complete future pipeline are implemented.

**[Object Cutout](../docs/OBJECT_CUTOUT.md)** exposes `/cutout/prepare` and `/cutout/extract`. Manual Gaussian/Sobel produce the elevation for scikit-image watershed; OpenCV GrabCut remains a comparison method. Reinstall requirements and restart Python. Its 512-pixel RGBA preparation, markers, method/parameter provenance and guidance contract are documented separately from the unchanged `/analyze` contract below. FFT remains analysis-only.

## Run

**Optional AI:** [AI-assisted cutout](../docs/AI_CUTOUT.md) is a third explicit
method, with separate `requirements-ai.txt` and photo-free model setup. Base
requirements and the manual mathematics are unchanged. Optional libraries and
the cached model load only on an explicit AI extraction, never during app startup.

Follow [the root README](../README.md) for dependency installation and two-terminal startup. Python 3.12+ is required. From the project root:

```powershell
.\backend\.venv\Scripts\python.exe -B -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Dependencies are pinned in `requirements.txt` and installed into `backend/.venv`. No global Python changes are needed. Restart Python after source edits.

`/health` reports readiness and `/docs` provides interactive request documentation. CORS permits `http://localhost:3000` and `http://127.0.0.1:3000`. Override with `SPECTRAEDGE_CORS_ORIGINS`, a comma-separated list of exact origins. Browser POSTs from other origins are rejected before parsing. See [FastAPI's CORS documentation](https://fastapi.tiangolo.com/tutorial/cors/) for the origin model.

## POST /analyze request

Send the four required `multipart/form-data` fields below. Add **both** optional detection fields for Sobel. Let the client set the multipart boundary.

| Field | Value |
| --- | --- |
| `image` | One binary PNG, JPEG, WebP, or GIF file |
| `sigma` | Finite 0..5 inclusive, in increments of 0.1 |
| `kernel_size` | Odd integer 3..31 inclusive, even when sigma is zero |
| `request_id` | Client-generated UUID, echoed in canonical UUID form |
| `detector` | Optional; only the case-sensitive value `Sobel` is supported |
| `threshold` | Required with Sobel, forbidden without detector; finite 0..1443 inclusive, in raw magnitude units |

Omitting both detector and threshold preserves the original Gaussian/Fourier-only behavior. Unsupported detectors, unused thresholds, duplicates, and extra fields (including object-area/noise) are rejected. Thresholds may be fractional; the UI slider uses step 1 and starts at 96. The UI suggests Gaussian support `2 * ceil(3 * sigma) + 1` (at least 3), while smaller kernels remain available for educational comparison.

Server-side limits, independent of browser validation:

- 20 MiB per file; the streamed total body is capped at 20 MiB + 64 KiB.
- One file, at most five text fields, at most 4096 bytes per text field.
- 20,000,000 decoded pixels, checked before decoding pixels.
- Format is checked from content, not just filename/MIME.
- The multipart parser memory threshold exceeds the bounded body, preventing temporary-file spooling. Uploads are never written to disk.

## Preparation and calculation

Pillow decodes frame zero, applies EXIF orientation, and fits the image within a 512-pixel maximum side without upscaling. Nearest-neighbour resizing avoids adding a hidden smoothing filter, but can alias fine detail. Transparency is composited on white; Pillow prepares RGB and grayscale images.

Grayscale becomes float64. The API calls the existing `gaussian_blur()`. If requested, it passes that same floating-point filtered image to `sobel()` and thresholds the raw magnitude. `fft_spectrum()` still runs on both original grayscale and the same filtered array. Mathematics stays in array-only `signal_ops` and `detection`; existing signal implementations, reflection padding, and public interfaces are unchanged.

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
| `grayscale_image` | Grayscale input to the numerical modules |
| `filtered_image` | Gaussian-smoothed grayscale; identical to grayscale when sigma=0 |
| `fft_image` | Centred log-magnitude spectrum of grayscale **before smoothing** |
| `filtered_fft_image` | Centred log-magnitude spectrum **after smoothing** |
| `parameters_used` | Actual `{sigma, kernel_size}`; Sobel adds `detector: "Sobel"` and `threshold` |
| `source_dimensions` | `{width, height}` after orientation, before resizing |
| `analyzed_dimensions` | `{width, height}` after bounded preparation |
| `completed_stages` | Gaussian-only: `["input", "grayscale", "smooth", "fourier"]`; Sobel: `["input", "grayscale", "smooth", "sobel", "threshold", "fourier"]` |
| `spectrum_scale` | `{min: 0, max: combined_maximum, mapping: "linear_grayscale"}` |
| `processing_time` | Measured **milliseconds**, including decode, preparation, numerical work, PNG encoding; excluding upload transfer and JSON serialization |
| `detection_status` | `"not_run"` for Gaussian-only; `"edges_computed"` for Sobel |
| `gx`, `gy`, `gradient_magnitude`, `edge_map` | PNG visualizations for Sobel; `null` for Gaussian-only |
| `contour_image` | Always `null`: contour analysis has not run |
| `object_list` | Always `null`: objects have not been analyzed, even when edges were computed |
| `fps` | `null`: no video stream |

For both spectra, `M = max(original_log_spectrum.max(), filtered_log_spectrum.max())`. At encoding only, both use `round(255 * spectrum / M)`, clipped to 0..255 and cast to uint8. If M=0, use denominator 1 so both PNGs are black; metadata still records max=0. Spatial images use rounded/clipped intensity, not independent contrast stretching. Raw floating-point arrays stay unchanged.

### Fixed Sobel display mapping

`smooth` includes the sigma-zero bypass; `sobel` includes Gx, Gy, and magnitude. `threshold` produces the binary edge map. These stages do **not** imply contours or objects have run.

For 0..255 grayscale intensities, each Sobel component is bounded by ±`4 × 255 = 1020`. A conservative magnitude bound is `B = 1020 × sqrt(2) ≈ 1442.498`; the UI/API threshold ceiling of 1443 covers it.

- Gx/Gy PNG value: `round(127.5 × (component / 1020 + 1))`. Negative responses are dark, zero is gray 128, and positive responses are bright. Positive Gx points right; positive Gy points downward in image coordinates.
- Magnitude PNG value: `round(255 × magnitude / B)`. Neither this scale nor the signed scale is recomputed from each image's extrema.
- Edge PNG value: 255 when **raw** `magnitude > threshold`, otherwise 0. Equality is excluded. A flat image has no edges at threshold zero.

Clipping to 0..255 and conversion to uint8 happen only when encoding these display values. Do not feed a magnitude PNG back into thresholding. PNGs cannot recover the original signed floating-point data; a future numerical export needs its own explicit contract.

The browser retains the original File/local preview separately. Session JSON exports metadata; cards export images.

## Errors and cancellation

Errors contain a JSON `detail` string: 400 invalid image/multipart, 403 disallowed browser origin, 413 size/pixel limit, 415 unsupported request/image format, 422 invalid fields/settings, 429 engine busy, or 500 safe processing failure. Internal diagnostics stay in the terminal; uploaded pixels are not logged.

CPU work runs off the async request loop. A process-local lock allows one calculation at a time. The browser aborts and invalidates requests when settings/images change, on reset, or unmount; already running numerical work finishes. No result history is retained. This is a bounded local demonstration, not an authenticated public upload service.

## Ownership handoff

The frontend/signal-operations contributor has completed convolution, Gaussian filtering, manual Fourier analysis, and now Sobel/basic thresholding with frontend integration. The teammate takes over `app.py` and API tests for the remaining Prewitt/Laplacian, contour/object, experiment, multi-scale, and pipeline work. Dependencies remain unchanged in this milestone.

Keep mathematics in array-only modules. Preserve existing image-field meanings, request IDs, timing units, common spectrum scaling, upload limits, and in-memory handling. Coordinate future contract/parser changes through the frontend owner. Return `object_list: []` only after **object analysis** actually runs and finds no objects; computing edges alone must leave it null. Public hosting/authentication/retention policies require a separate milestone.

## Tests

From the project root:

```powershell
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s backend/tests/signal_ops -p "test_*.py" -v
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s backend/tests/detection -p "test_*.py" -v
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s backend/tests -p "test_api.py" -v
```

Detection tests use analytical ramp/step expectations, sign conventions, fractional constants, magnitude, raw thresholds, input preservation, and monotonic edge selection. API tests use HTTPX's in-process ASGI transport and generated uploads. They cover both request modes, actual outputs, same-array processing, fixed derivative/common spectrum scales, validation, multipart limits, no disk spooling, CORS, busy handling, and safe failures. The separate frontend `test:integration` script exercises its actual adapter against a running Python server over HTTP. Neither is a browser interaction test.
