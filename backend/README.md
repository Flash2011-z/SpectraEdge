# SpectraEdge Python backend

Initial FastAPI service for the existing manual Gaussian/Fourier modules. No detectors, contours, noise experiments, or complete future pipeline are implemented.

## Run

Follow [the root README](../README.md) for dependency installation and two-terminal startup. Python 3.12+ is required. From the project root:

```powershell
.\backend\.venv\Scripts\python.exe -B -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Dependencies are pinned in `requirements.txt` and installed into `backend/.venv`. No global Python changes are needed. Restart Python after source edits.

`/health` reports readiness and `/docs` provides interactive request documentation. CORS permits `http://localhost:3000` and `http://127.0.0.1:3000`. Override with `SPECTRAEDGE_CORS_ORIGINS`, a comma-separated list of exact origins. Browser POSTs from other origins are rejected before parsing. See [FastAPI's CORS documentation](https://fastapi.tiangolo.com/tutorial/cors/) for the origin model.

## POST /analyze request

Send exactly four `multipart/form-data` fields. Let the client set the multipart boundary.

| Field | Value |
| --- | --- |
| `image` | One binary PNG, JPEG, WebP, or GIF file |
| `sigma` | Finite 0..5 inclusive, in increments of 0.1 |
| `kernel_size` | Odd integer 3..31 inclusive, even when sigma is zero |
| `request_id` | Client-generated UUID, echoed in canonical UUID form |

Detector, threshold, object-area, and noise parameters are not silently accepted. The UI suggests support `2 * ceil(3 * sigma) + 1` (at least 3), while smaller kernels remain available for educational comparison.

Server-side limits, independent of browser validation:

- 20 MiB per file; the streamed total body is capped at 20 MiB + 64 KiB.
- One file, three text fields, at most 4096 bytes per text field.
- 20,000,000 decoded pixels, checked before decoding pixels.
- Format is checked from content, not just filename/MIME.
- The multipart parser memory threshold exceeds the bounded body, preventing temporary-file spooling. Uploads are never written to disk.

## Preparation and calculation

Pillow decodes frame zero, applies EXIF orientation, and fits the image within a 512-pixel maximum side without upscaling. Nearest-neighbour resizing avoids adding a hidden smoothing filter, but can alias fine detail. Transparency is composited on white; Pillow prepares RGB and grayscale images.

Grayscale becomes float64. The API calls the existing `gaussian_blur()`, then `fft_spectrum()` on both grayscale and blurred arrays. Mathematics stays in `signal_ops`; reflection padding and public mathematical interfaces are unchanged.

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
| `parameters_used` | Actual `{sigma, kernel_size}` |
| `source_dimensions` | `{width, height}` after orientation, before resizing |
| `analyzed_dimensions` | `{width, height}` after bounded preparation |
| `completed_stages` | `["input", "grayscale", "smooth", "fourier"]`; smooth includes sigma-zero bypass |
| `spectrum_scale` | `{min: 0, max: combined_maximum, mapping: "linear_grayscale"}` |
| `processing_time` | Measured **milliseconds**, including decode, preparation, numerical work, PNG encoding; excluding upload transfer and JSON serialization |
| `detection_status` | `"not_run"` |
| `gx`, `gy`, `gradient_magnitude`, `edge_map`, `contour_image` | `null` |
| `object_list` | `null`: detection has not run, not a completed result with zero objects |
| `fps` | `null`: no video stream |

For both spectra, `M = max(original_log_spectrum.max(), filtered_log_spectrum.max())`. At encoding only, both use `round(255 * spectrum / M)`, clipped to 0..255 and cast to uint8. If M=0, use denominator 1 so both PNGs are black; metadata still records max=0. Spatial images use rounded/clipped intensity, not independent contrast stretching. Raw floating-point arrays stay unchanged.

The browser retains the original File/local preview separately. Session JSON exports metadata; cards export images.

## Errors and cancellation

Errors contain a JSON `detail` string: 400 invalid image/multipart, 403 disallowed browser origin, 413 size/pixel limit, 415 unsupported request/image format, 422 invalid fields/settings, 429 engine busy, or 500 safe processing failure. Internal diagnostics stay in the terminal; uploaded pixels are not logged.

CPU work runs off the async request loop. A process-local lock allows one calculation at a time. The browser aborts and invalidates requests when settings/images change, on reset, or unmount; already running numerical work finishes. No result history is retained. This is a bounded local demonstration, not an authenticated public upload service.

## Ownership handoff

The frontend/signal-operations contributor created this initial setup. The teammate takes ownership of `app.py`, `requirements.txt`, and `tests/test_api.py` for future API/pipeline work, then adds detection, contours, experiments, and multi-scale processing in separate modules.

Keep mathematics in `signal_ops`. Preserve existing image-field meanings, request IDs, timing units, common spectrum scaling, upload limits, and in-memory handling. When adding detectors, update the frontend contract/parser together: explicitly mark completed detection stages/status and return `object_list: []` only when detection actually ran and found no objects. Public hosting/authentication/retention policies require a separate milestone.

## Tests

From the project root:

```powershell
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s backend/tests/signal_ops -p "test_*.py" -v
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s backend/tests -p "test_api.py" -v
```

API tests use HTTPX's in-process ASGI transport and generated uploads. They cover manual-module calls, numerical/display equivalence, common scaling, validation, limits, no disk spooling, CORS, busy handling, and safe failures. The separate frontend `test:integration` script exercises its actual adapter against a running Python server over HTTP. Neither is a browser interaction test.
