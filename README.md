# SpectraEdge web workspace

SpectraEdge: A Multi-Scale Multi-Object Edge Detection and Frequency-Domain Analysis System.

This folder contains the browser edition of the university Signals and Linear Systems GUI prototype. The original Python/QML edition remains in the parent directory.

## Run locally

Requires Node.js 22.13 or newer.

```sh
npm install
npm run dev
```

Open the local URL printed in the terminal. On Windows PowerShell, use `npm.cmd` if execution policy blocks `npm.ps1`.

```sh
npm run build
npm start
```

## Current scope

- Analyze, Compare, and Live are separate browser routes.
- Local image selection, drag/drop, metadata, zoom, pan, and image export work.
- Parameters, display preferences, and reduced motion are saved locally on the device.
- All calibration images, edge/gradient/FFT artwork, object values, and comparison times are explicitly demo fixtures.
- Process Image walks through the demo pipeline. Uploaded image outputs remain empty.
- No edge detection, convolution, FFT, contour detection, camera capture, or machine-vision algorithms are implemented.
- No image bytes are uploaded or saved in browser storage. Reloading discards the selected image.
- The exported JSON file identifies all sample metrics as demo values.

## Architecture

- `app/`: Analyze, Compare, and Live routes and shared layout.
- `components/workspace-provider.tsx`: shared session, input validation, local preferences, and preview lifecycle.
- `components/workspace.tsx`: navigation, page orchestration, dialogs, shortcuts, session export.
- `components/parameters.tsx`: analysis controls.
- `components/visualization.tsx`: image cards, static demonstration renderer, inspector.
- `components/object-information.tsx` and `pipeline.tsx`: reusable analysis instruments.
- `lib/workspace.ts`: typed parameters, demo fixtures, and future backend result contract.

## Connecting the future Python backend

The `AnalysisResult` interface defines `original_image`, `filtered_image`, `gx`, `gy`, `gradient_magnitude`, `edge_map`, `contour_image`, `fft_image`, `object_list`, `processing_time`, and `fps`.

Add an HTTP API adapter when the processing phase begins. Keep a request identifier and parameter snapshot with each request so an outdated response cannot replace newer output. Return `provenance: "computed"` for real results. Use explicit busy, success, and error states. Replace demo rendering only when a validated result is available; never present demo fixtures as calculated output.

Sites builds this React/TypeScript application with Vinext and Vite into a Workers-compatible deployment. The retained build scripts and `.openai/hosting.json` belong to that deployment workflow.

## Checks

```sh
npm run lint
npx tsc --noEmit --incremental false
node --test tests/workspace.test.ts
npm run build
```

The icon set is Lucide. The generated social card is in `public/og.png`.
