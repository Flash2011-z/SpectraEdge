# SpectraEdge frontend

The React/TypeScript/Vinext website connects to Python Gaussian, Sobel/threshold, and Fourier analysis. Follow [the root README](../README.md) for exact Windows setup, two-terminal startup, and a teacher demonstration.

Existing commands remain `npm run dev`, `npm run build`, and `npm start`. Use `npm.cmd` in Windows PowerShell if execution policy blocks npm.ps1. Requires Node.js 22.13+.

## Functionality

- Upload/drag a File, change sigma/kernel/threshold settings, and Process with the real Python backend and Sobel detector.
- Original, grayscale, blurred, Gx, Gy, magnitude, edges, and before/after Fourier images are actual computed results. Dimensions, parameters used, and time come from the response.
- Both spectra use one shared grayscale display scale, shown below their cards.
- Inspect, expand, zoom, pan, and download images. Session export writes current metadata without embedding image bytes.
- Parameter changes clear outdated results. Image changes, reset, and unmount abort and invalidate requests. Late successes/errors are ignored even if transport cancellation arrives too late.
- Errors persist with a retry action; requests time out after 120 seconds.
- Sobel and threshold are enabled for uploaded images. Threshold uses **raw magnitude > threshold**, ranges 0..1443, and initially defaults to 96. Gx/Gy use fixed −1020..1020 display scales; magnitude uses 0..1020√2. Display brightness never determines the threshold.
- Prewitt, Laplacian, object-area, and noise controls remain disabled. Status distinguishes **edges computed** from **objects analyzed**; object count stays unavailable.
- Compare and Live remain explicitly unfinished; no detector comparison or webcam processing runs.
- The calibration example is separate and labelled illustrative, with processing disabled.

Only device preferences/settings persist in localStorage. Files, preview URLs, and results remain in session memory. Upload occurs only on Process, and Python does not save images. See [the API contract](../backend/README.md) for limits, preparation, encoding, and timing.

## Architecture

- `lib/workspace.ts`: parameters, shared Gaussian/Sobel contracts and validation, active-settings snapshot construction, result-to-card mapping, real stages, and separate demo fixtures.
- `lib/api.ts`: configurable multipart adapter, response validation, timeout, AbortController, and generation-based stale-response protection.
- `components/workspace-provider.tsx`: File/result lifecycle, parameter snapshots, and preferences. Every invalidating action cancels the request runner and clears results.
- `components/workspace.tsx`: routes, result grid, loading/error/status UI, dialogs, shortcuts, and metadata export.
- `components/visualization.tsx`: PNG/local preview, inspector, zoom/pan, downloads, and isolated illustrative renderer.
- `components/parameters.tsx`: Gaussian/Sobel/threshold controls and disabled planned controls.
- `components/pipeline.tsx` and `object-information.tsx`: completed-stage and not-run displays.

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
```

Unit checks cover preferences/image validation, both multipart request modes, Gaussian/Sobel settings, both result contracts, errors, detector/threshold snapshots and matching, and stale success/error cancellation. Live integration submits a real checkerboard PNG through the workspace's adapter and checks Gaussian-only compatibility, smoothing, spectra, Sobel outputs, threshold-only changes, unavailable objects, and a decode error. These Node/HTTP checks do not interact with a browser.

## Ownership

Frontend, signal_ops, and the completed Sobel/basic threshold milestone belong to the frontend/signal-operations contributor. The teammate takes over the API for remaining detectors, contours/objects, experiments, and pipeline work. Coordinate `AnalysisSettings`, `ComputedAnalysisResult`, and `parseAnalysisResult` changes through the frontend owner; preserve Gaussian-only compatibility and never relabel demo fixtures as results.

Existing Vinext/Vite scripts, hosting configuration, icons, and social previews are retained. Local development does not publish a website or expose the Python service.
