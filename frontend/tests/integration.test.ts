// Explicit live HTTP check: start the Python service before running test:integration.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { analyzeImage, compareImage } from "../lib/api.ts";
import { extractCutout, prepareCutout } from "../lib/cutout-api.ts";
import { photoFile, rectangle } from "./cutout-fixture.ts";
import { AI_MODEL, resolveCutoutRectangle } from "../lib/cutout.ts";

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
  assert.deepEqual(weak.completed_stages, ["input", "grayscale", "smooth", "sobel", "threshold", "contours", "objects", "fourier"]);
  assert.match(weak.contour_image!, /^data:image\/png;base64,/);
  assert.ok(Array.isArray(weak.object_list));
  assert.equal(weak.fps, null);
  console.log(`Actual Python Sobel processing: threshold=0 ${weak.processing_time} ms; threshold=1443 ${strong.processing_time} ms.`);
});

test("actual cutout preparation/extraction and brush refinement work over HTTP", async () => {
  const prepared = await prepareCutout(photoFile, crypto.randomUUID(), new AbortController().signal);
  assert.deepEqual({ width: prepared.width, height: prepared.height }, { width: 16, height: 12 });
  const original = await extractCutout({ prepared, rectangle, marks: [], settings: { method: "grabcut" } }, crypto.randomUUID(), new AbortController().signal);
  assert.deepEqual(original.foreground_bounds, { x: 4, y: 3, width: 8, height: 6 });
  const refined = await extractCutout({ prepared, rectangle, settings: { method: "grabcut" }, marks: [
    { mode: "keep", size: 1, points: [{ x: 8, y: 6 }] },
    { mode: "keep", size: 1, points: [{ x: 0, y: 0 }] },
  ] }, crypto.randomUUID(), new AbortController().signal);
  assert.notEqual(refined.cutout_image, original.cutout_image);
  assert.equal(refined.foreground_bounds.x, 0);
  assert.equal(refined.foreground_bounds.y, 0);
  assert.equal(refined.image_id, prepared.image_id);
  console.log(`Actual GrabCut: ${original.processing_time} ms; refined ${refined.processing_time} ms.`);
});

test("brush-free edge extraction, optional refinement and comparison work over HTTP", async () => {
  const prepared = await prepareCutout(photoFile, crypto.randomUUID(), new AbortController().signal);
  const settings = { method: "edge-watershed" as const, sigma: 0, kernel_size: 5 };
  const marks = [{ mode: "keep" as const, size: 1, points: [{ x: 8, y: 6 }] }];
  const extract = (sigma: number) => extractCutout({ prepared, rectangle, marks, settings: { ...settings, sigma } }, crypto.randomUUID(), new AbortController().signal);
  const automatic = await extractCutout({ prepared, rectangle, marks: [], settings }, crypto.randomUUID(), new AbortController().signal);
  assert.equal(automatic.seed_mode, "automatic");
  assert.equal(automatic.method, "edge-watershed");
  const original = await extract(0);
  const smoothed = await extract(1.2);
  assert.equal(original.method, "edge-watershed");
  assert.equal(original.seed_mode, "brush");
  assert.equal(original.algorithm, "skimage-watershed");
  assert.deepEqual(smoothed.parameters_used, { sigma: 1.2, kernel_size: 5 });
  assert.notEqual(original.guidance_image, smoothed.guidance_image);
  assert.deepEqual({ width: original.width, height: original.height }, { width: 16, height: 12 });
  const comparison = await extractCutout({ prepared, rectangle, marks, settings: { method: "grabcut" } }, crypto.randomUUID(), new AbortController().signal);
  assert.equal(comparison.method, "grabcut");
  assert.equal(comparison.guidance_image, null);
  assert.deepEqual(comparison.parameters_used, {});
  console.log(`Actual watershed: sigma=0 ${original.processing_time} ms; sigma=1.2 ${smoothed.processing_time} ms.`);
});

test("optional real AI photo uses local model and exact prepared grid over HTTP", {
  skip: !process.env.SPECTRAEDGE_TEST_AI_PHOTO,
  timeout: 180_000,
}, async () => {
  const bytes = await readFile(process.env.SPECTRAEDGE_TEST_AI_PHOTO!);
  const photo = new File([bytes], "local-portrait.jpg", { type: "image/jpeg" });
  const signal = AbortSignal.timeout(170_000);
  const prepared = await prepareCutout(photo, crypto.randomUUID(), signal);
  const output = await extractCutout({ prepared, rectangle: resolveCutoutRectangle(prepared, null, "ai-assisted")!,
    marks: [], settings: { method: "ai-assisted" } }, crypto.randomUUID(), signal);
  assert.equal(output.method, "ai-assisted");
  assert.equal(output.algorithm, "rembg-onnx");
  assert.deepEqual(output.parameters_used, { model: AI_MODEL });
  assert.equal(output.guidance_image, null);
  assert.equal(output.seed_mode, null);
  assert.equal(output.width, prepared.width);
  assert.equal(output.height, prepared.height);
  assert.ok(output.foreground_bounds.width > prepared.width / 4);
  assert.ok(output.foreground_bounds.height > prepared.height / 4);
  console.log(`Actual optional AI: ${output.processing_time} ms; ${output.width} x ${output.height}.`);
});

for (const detector of ["Prewitt", "Laplacian"] as const) {
  test(`real ${detector} upload returns detector-specific outputs`, async () => {
    const settings = { sigma: 1.2, kernel_size: 7, detector, threshold: 0 };
    const weak = await analyzeImage(file, settings, crypto.randomUUID(), new AbortController().signal);
    const strong = await analyzeImage(file, { ...settings, threshold: 1443 }, crypto.randomUUID(), new AbortController().signal);
    assert.equal(weak.parameters_used.detector, detector);
    assert.equal(weak.detection_status, "edges_computed");
    assert.notEqual(weak.edge_map, strong.edge_map);
    assert.equal(weak.filtered_image, strong.filtered_image);
    if (detector === "Prewitt") {
      for (const key of ["gx", "gy", "gradient_magnitude"] as const) {
        assert.match(weak[key]!, /^data:image\/png;base64,/);
        assert.equal(weak[key], strong[key]);
      }
    } else {
      assert.match(weak.laplacian_response!, /^data:image\/png;base64,/);
      assert.equal(weak.laplacian_response, strong.laplacian_response);
      assert.deepEqual([weak.gx, weak.gy, weak.gradient_magnitude], [null, null, null]);
    }
    assert.deepEqual(weak.completed_stages, ["input", "grayscale", "smooth", detector.toLowerCase(),
      detector === "Laplacian" ? "zero_crossing" : "threshold", "contours", "objects", "fourier"]);
    assert.match(weak.contour_image!, /^data:image\/png;base64,/);
    assert.ok(Array.isArray(weak.object_list));
  });
}

test("real multi-scale upload returns independent edges, persistence and fused objects", async () => {
  const settings = { sigma: 1.2, kernel_size: 7, detector: "Sobel" as const, threshold: 20,
    multi_scale: true, scale_sigmas: [0, 1.2, 2], scale_support: 2 };
  const result = await analyzeImage(file, settings, crypto.randomUUID(), new AbortController().signal);
  assert.deepEqual(result.parameters_used, settings);
  assert.deepEqual(result.multi_scale?.sigmas, [0, 1.2, 2]);
  assert.equal(result.multi_scale?.support_count, 2);
  assert.equal(result.multi_scale?.scales.length, 3);
  for (const scale of result.multi_scale!.scales) assert.match(scale.edge_map, /^data:image\/png;base64,/);
  assert.match(result.multi_scale!.persistence_map, /^data:image\/png;base64,/);
  assert.match(result.multi_scale!.fused_edge_map, /^data:image\/png;base64,/);
  assert.ok(Array.isArray(result.object_list));
  assert.ok(result.completed_stages.includes("multi_scale"));
});

test("real noise experiments are reproducible and expose the noisy analysis input", async () => {
  for (const settings of [
    { noise_model: "Gaussian" as const, noise_strength: 12, noise_seed: 220 },
    { noise_model: "Salt & Pepper" as const, noise_strength: 0.12, noise_seed: 220 },
  ]) {
    const request = { sigma: 1.2, kernel_size: 7, detector: "Sobel" as const, threshold: 96, ...settings };
    const first = await analyzeImage(file, request, crypto.randomUUID(), new AbortController().signal);
    const second = await analyzeImage(file, request, crypto.randomUUID(), new AbortController().signal);
    assert.equal(first.noisy_image, second.noisy_image);
    assert.equal(first.filtered_image, second.filtered_image);
    assert.notEqual(first.noisy_image, first.grayscale_image);
    assert.equal(first.noise.model, settings.noise_model);
    assert.equal(first.noise.seed, 220);
    assert.ok(first.completed_stages.includes("noise"));
  }
});

test("real comparison upload returns all detectors from one request", async () => {
  const settings = { sigma: 1.2, kernel_size: 7, sobel_threshold: 96,
    prewitt_threshold: 96, laplacian_contrast_threshold: 20 };
  const result = await compareImage(file, settings, crypto.randomUUID(), new AbortController().signal);
  const strictLaplacian = await compareImage(file, { ...settings, laplacian_contrast_threshold: 500 },
    crypto.randomUUID(), new AbortController().signal);
  assert.deepEqual(result.parameters_used, settings);
  assert.deepEqual(result.analyzed_dimensions, { width: 32, height: 24 });
  assert.ok(result.processing_time >= 0);
  for (const [key, detector] of ([
    ["sobel", "Sobel"], ["prewitt", "Prewitt"], ["laplacian", "Laplacian"],
  ] as const)) {
    const entry = result[key];
    assert.equal(entry.metadata.detector, detector);
    assert.equal(entry.metadata.threshold_type,
      detector === "Laplacian" ? "zero_crossing_contrast" : "gradient_magnitude");
    assert.match(entry.edge_map, /^data:image\/png;base64,/);
    assert.equal(entry.object_count, entry.object_list.length);
    assert.ok(entry.processing_time >= 0);
  }
  assert.equal(result.sobel.edge_map, strictLaplacian.sobel.edge_map);
  assert.equal(result.prewitt.edge_map, strictLaplacian.prewitt.edge_map);
  assert.notEqual(result.laplacian.edge_map, strictLaplacian.laplacian.edge_map);
});
