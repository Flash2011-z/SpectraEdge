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
