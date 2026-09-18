# SpectraEdge final audit

Audit date: 2026-09-19. Scope: the current working tree, including existing
uncommitted and untracked work. No features added; no commits or deletions made.

The five requested workflows pass numerical/API and frontend adapter checks.
No blocking numerical or request-contract defect was found in the exercised
paths. Final submission remains conditional on a browser/webcam smoke test
and inclusion of the currently untracked source files in the submission.

## Issues found and corrected

1. **Live processing failures broke the JSON error contract.** Unexpected
   exceptions in `/live/frame` escaped to the default server error handler,
   unlike Analyze and Compare. Added a safe JSON 500 response with server-side
   diagnostics. A regression test reproduced the failure before the fix and
   now verifies the response and successful processing of the next frame.
2. **Documentation and visible camera help were stale.** READMEs claimed Noise
   and/or Live were unfinished; the camera button said SOON and its dialog
   claimed capture was disabled. Corrected reachable help text and documented
   Live, noise, endpoint limits, independent processing slots, and Laplacian
   component filtering. Clarified the example checkout path.
3. **Documented test commands missed substantial suites.** Replaced selective
   backend discovery with full recursive discovery and documented the separate
   Live HTTP integration command.

## Architecture consistency

- React components own presentation; the workspace provider owns uploaded-image
  state and Analyze/Compare cancellation. Live has its own camera and processing
  hooks. Frontend adapters validate settings and response provenance.
- FastAPI handles bounded uploads, decoding, PNG encoding, and response models.
  Signal operations, detectors, and object analysis remain array-only modules.
- Compare smooths one shared input before running all detectors. Noise feeds the
  analysis signal before smoothing. Multi-scale reuses the same source/noise
  realization across scales. Live reuses the detector and Fourier modules.
- Analyze/Compare share one processing lock; Live and Cutout use independent
  slots. This is consistent with their separate lifecycles, but is not a
  process-wide limit of one calculation.
- Laplacian raw edge maps retain small components; contour/object measurements
  filter them according to the reported minimum area. This is intentional and
  now documented. Live returns raw edges and no object measurements.

## Recommended cleanup (not required for functionality)

- Remove the obsolete `LiveView` implementation and unreachable live branch in
  `frontend/components/workspace.tsx`, plus its exclusive global CSS. Current
  `/live` renders `components/live-workspace.tsx`. Narrow the workspace's view
  type without removing Live from shared navigation.
- Remove `frontend/public/images/signal-welcome.png` (1,735,671 bytes): no source
  references found; the active welcome mapping uses `signal-welcome-sharp.png`.
  Other signal images are referenced through the visualization mapping.
- Remove the unused `multiscale` import at
  `backend/tests/analysis/test_multiscale.py:9`. The Python AST scan found no
  other unused imports outside intentional package re-exports. This scan is
  heuristic; frontend ESLint and TypeScript unused-local/parameter checks pass.
- Consider centralizing repeated upload decoding, PNG encoding, bounded stream
  parsing, detector dispatch, and threshold validation. Preserve endpoint-specific
  size limits and response contracts. Sobel and Prewitt share similar scaffolding
  but have distinct kernels; there is no duplicated convolution/FFT algorithm in
  the HTTP layer. A broad refactor is unnecessary immediately before submission.
- Ensure untracked `backend/analysis`, detector/noise/Live modules, response
  models, frontend Live/multi-scale components, and their tests are included in
  the final Git commit or submission archive. A checkout of HEAD alone will omit
  required implementation files. Existing user changes were preserved.

## Verification results

| Workflow | Evidence | Result |
| --- | --- | --- |
| Analyze | Numerical detector tests, ASGI request tests, real PNG frontend adapter uploads for all three detectors | Pass |
| Compare | Shared-preprocessing tests, threshold independence, response validation, actual HTTP adapter | Pass |
| Noise | Seed reproducibility, model bounds, same-array pipeline assertions, rendered cards/stages, actual HTTP adapter | Pass |
| Multi-scale | Independent scale passes, support/fusion validation, API metadata and actual HTTP adapter | Pass |
| Live | All detector outputs match numerical modules; bounded input, busy handling, cancellation, timeout and camera-session unit tests; three actual HTTP adapter tests | Pass at automated/API level |

- Backend: **242 tests run, 239 passed, 3 skipped** for optional external photos.
- Frontend unit/rendering: **75 passed**.
- Main HTTP adapter integration: **10 passed, 1 skipped** for optional AI photo.
- Live HTTP adapter integration: **3 passed**.
- ESLint, TypeScript (including `noUnusedLocals` and `noUnusedParameters`), and
  production build: **passed**.
- HTTP route smoke checks: `/`, `/compare`, and `/live` returned **200**.
- Git diff whitespace check: **passed**. Vinext emits a non-fatal unknown route
  classification notice during its successful build.
- Windows sandbox initially blocked some Node subprocesses and temporary-file
  cleanup. The affected checks passed when rerun with approved permissions.

## Coverage limits and final readiness

No line/branch coverage percentage was measured; test counts are not a coverage
percentage. Automated coverage is strongest in numerical operations, contracts,
validation, cancellation, and transport. There is no browser interaction suite.
Physical webcam permissions, disconnect/restart, canvas presentation, responsive
layout, file picking, and downloaded output still require a manual browser pass.
Optional real-photo AI cases were not verified and are outside the five requested
workflows.

The project is ready for final demonstration validation. Before declaring final
submission complete, exercise upload/process/download, all three comparison
cards, both noise models, multi-scale support changes, and Live start/stop,
permission denial, control changes, retry, and navigation away. Include all
required working-tree files in the submission. Optional deduplication and dead
code cleanup can wait; no unresolved blocking bug was found in tested paths.
