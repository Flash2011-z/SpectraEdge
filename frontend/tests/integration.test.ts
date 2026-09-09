// Explicit live HTTP check: start the Python service before running test:integration.
import test from "node:test";
import assert from "node:assert/strict";
import { analyzeImage } from "../lib/api.ts";

const pixels = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAACAAAAAYCAAAAAC+OKDoAAAAJklEQVR4nGNkYPjPwIgHMzEQAv//48cETWAcdQMYjIYDBAyGcAAAxx2Pub9i5ZMAAAAASUVORK5CYII=", "base64");
const file = new File([pixels], "fine-detail.png", { type: "image/png" });

test("frontend API adapter processes an actual uploaded PNG through the running Python server", async () => {
  const original = await analyzeImage(file, { sigma: 0, kernel_size: 7 }, crypto.randomUUID(), new AbortController().signal);
  const smoothed = await analyzeImage(file, { sigma: 1.2, kernel_size: 7 }, crypto.randomUUID(), new AbortController().signal);
  assert.equal(original.filtered_image, original.grayscale_image);
  assert.equal(original.fft_image, original.filtered_fft_image);
  assert.notEqual(smoothed.filtered_image, smoothed.grayscale_image);
  assert.notEqual(smoothed.fft_image, smoothed.filtered_fft_image);
  assert.equal(smoothed.spectrum_scale.mapping, "linear_grayscale");
  assert.equal(smoothed.spectrum_scale.min, 0);
  assert.deepEqual(smoothed.analyzed_dimensions, { width: 32, height: 24 });
  assert.equal(smoothed.detection_status, "not_run");
  assert.equal(smoothed.edge_map, null);
  console.log(`Actual Python processing: sigma=0 ${original.processing_time} ms; sigma=1.2 ${smoothed.processing_time} ms.`);
});

test("frontend adapter presents a real backend decode error", async () => {
  const invalid = new File(["not an image"], "invalid.png", { type: "image/png" });
  await assert.rejects(analyzeImage(invalid, { sigma: 0, kernel_size: 3 }, crypto.randomUUID(), new AbortController().signal), /valid image/);
});

test("real Sobel upload returns gradients and changing only threshold changes edges", async () => {
  const settings = { sigma: 1.2, kernel_size: 7, detector: "Sobel" as const, threshold: 0 };
  const weak = await analyzeImage(file, settings, crypto.randomUUID(), new AbortController().signal);
  const strong = await analyzeImage(file, { ...settings, threshold: 1443 }, crypto.randomUUID(), new AbortController().signal);
  assert.equal(weak.detection_status, "edges_computed");
  assert.equal(weak.parameters_used.detector, "Sobel");
  assert.equal(strong.parameters_used.threshold, 1443);
  for (const key of ["gx", "gy", "gradient_magnitude", "edge_map"] as const) assert.match(weak[key]!, /^data:image\/png;base64,/);
  assert.notEqual(weak.edge_map, strong.edge_map);
  for (const key of ["gx", "gy", "gradient_magnitude", "filtered_image", "fft_image", "filtered_fft_image"] as const)
    assert.equal(weak[key], strong[key]);
  assert.deepEqual(weak.completed_stages, ["input", "grayscale", "smooth", "sobel", "threshold", "fourier"]);
  assert.equal(weak.contour_image, null);
  assert.equal(weak.object_list, null);
  assert.equal(weak.fps, null);
  console.log(`Actual Python Sobel processing: threshold=0 ${weak.processing_time} ms; threshold=1443 ${strong.processing_time} ms.`);
});
