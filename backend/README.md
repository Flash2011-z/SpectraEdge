# SpectraEdge backend

This folder is reserved for the future **Python signal-processing API**.

No backend server, algorithm implementation, or Python dependency installation is needed for the current GUI prototype.

## Future responsibilities

- Accept a source image and an immutable analysis-parameter snapshot.
- Run the requested filtering, detector, threshold, and contour stages.
- Produce gradient and frequency-domain visualizations.
- Return object measurements and timing information.
- Add live-frame processing only when camera functionality is requested.

The shared result contract is currently documented in [frontend/lib/workspace.ts](../frontend/lib/workspace.ts) as `AnalysisResult`:

```text
original_image
filtered_image
gx
gy
gradient_magnitude
edge_map
contour_image
fft_image
object_list
processing_time
fps
```

Real results must carry `provenance: "computed"`. Existing demo fixtures carry `provenance: "demo"`.

## When implementation starts

Put the Python API entry point, request/response models, processing modules, tests, and actual Python dependencies here. Keep Python environments and generated output out of source control.

Connect the website through an HTTP adapter rather than embedding Python code in UI components. Use request IDs, cancellation, parameter snapshots, and explicit error states so outdated responses cannot overwrite newer output.

Keep image uploads opt-in: images currently stay in browser memory. Introduce upload limits, CORS configuration, and any server-side retention policy when that behavior is implemented.

For now, start only the frontend using the command in the [project README](../README.md).
