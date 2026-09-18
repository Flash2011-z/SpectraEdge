# SpectraEdge frontend

The React/TypeScript/Vinext website connects to Python Analyze, Compare, Noise, Multi-scale, and Live workflows. Follow [the root README](../README.md) for Windows setup, startup, and a teacher demonstration.

**[Object Cutout](../docs/OBJECT_CUTOUT.md)** at `/cutout` defaults to Edge-guided watershed, with Gaussian controls and a preview of manual Sobel elevation. Draw a rectangle with background around the object and extract without painting. Brushes are optional corrections; Clear marks returns to automatic selection. GrabCut remains a comparison. Method/parameter or selection changes invalidate downloads. Upload prepares the photo in Python at up to 512 pixels per side before drawing. State is isolated from Analyze, in memory only, and discarded on leaving. No frontend dependency was added.

Existing commands remain `npm run dev`, `npm run build`, and `npm start`. Use `npm.cmd` in Windows PowerShell if execution policy blocks npm.ps1. Requires Node.js 22.13+.

## Functionality

Object Cutout also offers **AI-assisted cutout — optional**, backed by a local
pretrained portrait model. Manual watershed remains default. AI can start on the
whole photo without a drawn box; its result is explicitly labelled, Gaussian and
edge-guidance controls are unavailable, and brushes directly edit opacity.
Missing AI setup never triggers a fallback. See [setup and scope](../docs/AI_CUTOUT.md).

- Upload/drag a File, change sigma/kernel/threshold settings, and Process with the real Python backend and the selected detector.
- Original, grayscale, blurred, Gx, Gy, magnitude, edges, and before/after Fourier images are actual computed results. Dimensions, parameters used, and time come from the response.
- Both spectra use one shared grayscale display scale, shown below their cards.
- Inspect, expand, zoom, pan, and download images. Session export writes current metadata without embedding image bytes.
- Parameter changes clear outdated results. Image changes, reset, and unmount abort and invalidate requests. Late successes/errors are ignored even if transport cancellation arrives too late.
- Errors persist with a retry action; requests time out after 120 seconds.
- Sobel and threshold are enabled for uploaded images. Threshold uses **raw magnitude > threshold**, ranges 0..1443, and initially defaults to 96. Gx/Gy use fixed −1020..1020 display scales; magnitude uses 0..1020√2. Display brightness never determines the threshold.
- Sobel, Prewitt, and Laplacian are selectable. Detector runs show their edge map, connected-region contours, object count, and area/perimeter/centroid/bounding-box measurements.
- Noise controls add reproducible Gaussian noise in intensity standard-deviation units or Salt & Pepper noise as pixel corruption probability. The noisy input card, pipeline stage, model-specific units, strength, and seed are shown with computed results.
- Optional multi-scale controls run the selected detector at three independent Gaussian sigmas. Results show each scale edge mask, the persistence count image, and the fused mask. Support selects how many of the three masks must agree; fused edges drive object measurements while the normal detector cards remain single-scale.
- Compare runs Sobel, Prewitt, and Laplacian through one dedicated backend request and displays each edge map, object count, detector processing time, threshold type, and threshold value. Sobel and Prewitt expose independent magnitude thresholds; Laplacian exposes a separate zero-crossing contrast threshold.
- Live requests webcam permission on Start camera, captures frames at up to 256 pixels per side, and displays synchronized grayscale, Gaussian, edge, and input-spectrum outputs. It supports all three detectors, adjustable controls, timing/FPS, cancellation, busy backoff, and retry. Stop and unmount release camera tracks. Camera access requires localhost or HTTPS.
- The calibration example is separate and labelled illustrative, with processing disabled.

Only device preferences/settings persist in localStorage. Files, preview URLs, and results remain in session memory. Analyze/Compare upload on their processing actions; Live sends frames while the camera is active, and Cutout uploads during preparation. Python does not save images. See [the API contract](../backend/README.md) for limits, preparation, encoding, and timing.

## Architecture

- `lib/workspace.ts`: parameters, shared Gaussian/Sobel contracts and validation, active-settings snapshot construction, result-to-card mapping, real stages, and separate demo fixtures.
- `lib/api.ts`: configurable multipart adapter, response validation, timeout, AbortController, and generation-based stale-response protection.
- `components/workspace-provider.tsx`: File, analysis, and comparison lifecycles, parameter snapshots, cancellation, and preferences. Every invalidating action cancels active runners and clears stale results.
- `components/workspace.tsx`: routes, result grid, loading/error/status UI, dialogs, shortcuts, and metadata export.
- `components/visualization.tsx`: PNG/local preview, inspector, zoom/pan, downloads, and isolated illustrative renderer.
- `components/parameters.tsx`: Gaussian, detector, threshold, optional multi-scale, and noise experiment controls.
- `components/pipeline.tsx` and `object-information.tsx`: completed stages and connected-object measurements.
- `components/multiscale-results.tsx`: per-scale masks, persistence visualization, and fused edge output.
- `components/live-workspace.tsx`, `use-camera.ts`, and `use-live-analysis.ts`: webcam lifecycle, frame capture, output canvases, and timing; `lib/live.ts` and `live-api.ts` supply validation and bounded request handling.

`NEXT_PUBLIC_API_URL` defaults to `http://127.0.0.1:8000`. Copy `.env.example` to `.env.local` to change it; restart the frontend. This public configuration is not a secret. Align backend allowed origins if the website port changes.

## Checks

```powershell
npm.cmd run lint
npm.cmd run typecheck
npm.cmd test
npm.cmd run build
```

With Python running:

```powershell
npm.cmd run test:integration
npm.cmd run test:live-integration
```

Unit checks cover preferences/image validation, multipart request modes, all detector contracts, connected-object results, errors, parameter snapshots and matching, and stale cancellation. Live integration submits actual PNGs through the workspace adapter. These Node/HTTP checks do not interact with a browser.

## Ownership

The frontend supports all three detectors, connected-component results, optional three-scale persistence fusion, shared-preprocessing detector comparison, reproducible noise experiments, and Live webcam analysis while preserving Gaussian-only and single-scale API compatibility.

Existing Vinext/Vite scripts, hosting configuration, icons, and social previews are retained. Local development does not publish a website or expose the Python service.
