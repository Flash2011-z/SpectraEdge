import test from "node:test";
import assert from "node:assert/strict";
import { analyzeImage, compareImage, createAnalysisRunner, createComparisonRunner,
  parseAnalysisResult, parseComparisonResult } from "../lib/api.ts";
import { analysisSettings, analysisStages, comparisonEntries, comparisonSettings, detectorDisplay,
  detectorLabel, thresholdLabel, thresholdRule, restoreParameters, DEFAULT_PARAMETERS, resultImage,
  validateAnalysis, validateComparison, validateGaussian, type ComparisonResult, type Detector,
  type AnalysisSettings, type ComputedAnalysisResult } from "../lib/workspace.ts";

const png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAYCAAAAAC+OKDoAAAAJklEQVR4nGNkYPjPwIgHMzEQAv//48cETWAcdQMYjIYDBAyGcAAAxx2Pub9i5ZMAAAAASUVORK5CYII=";
const measurement = { id: 1, area: 12, perimeter: 14, centroid: [2.5, 2] as [number, number],
  bounding_box: { x: 1, y: 1, width: 4, height: 3 } };
function fixture(id = "test-request"): ComputedAnalysisResult {
  return {
    provenance: "computed", request_id: id, original_image: png,
    grayscale_image: png, noisy_image: null, filtered_image: png, fft_image: png, filtered_fft_image: png,
    gx: null, gy: null, gradient_magnitude: null, edge_map: null,
    contour_image: null, object_list: null, fps: null, processing_time: 10,
    parameters_used: { sigma: 1.2, kernel_size: 7 },
    source_dimensions: { width: 32, height: 24 }, analyzed_dimensions: { width: 32, height: 24 },
    completed_stages: ["input", "grayscale", "smooth", "fourier"], detection_status: "not_run",
    detector_metadata: null,
    multi_scale: null,
    noise: { model: "None", strength: 0, units: "none", seed: null },
    spectrum_scale: { min: 0, max: 12, mapping: "linear_grayscale" },
  };
}
const file = new File([new Uint8Array([1, 2, 3])], "input.png", { type: "image/png" });

function sobelFixture(id = "test-request", threshold = 96): ComputedAnalysisResult {
  return { ...fixture(id), gx: png, gy: png, gradient_magnitude: png, edge_map: png,
    contour_image: png, object_list: [measurement],
    parameters_used: { sigma: 1.2, kernel_size: 7, detector: "Sobel", threshold },
    detector_metadata: { detector: "Sobel", decision: "magnitude_threshold",
      threshold_type: "gradient_magnitude", threshold_label: "Gradient magnitude threshold",
      threshold, threshold_units: "raw_gradient_magnitude", minimum_component_area: null },
    detection_status: "edges_computed",
    completed_stages: ["input", "grayscale", "smooth", "sobel", "threshold", "contours", "objects", "fourier"] };
}

test("Sobel settings use raw 0..1443 units and reject unsupported or incomplete settings", () => {
  for (const threshold of [0, 96, 96.5, 1443])
    assert.equal(validateAnalysis({ sigma: 1.2, kernel_size: 7, detector: "Sobel", threshold }), null);
  for (const threshold of [-1, 1443.1, NaN, Infinity, undefined])
    assert.match(validateAnalysis({ sigma: 1.2, kernel_size: 7, detector: "Sobel", threshold })!, /threshold/);
  assert.match(validateAnalysis({ sigma: 1.2, kernel_size: 7, threshold: 96 })!, /requires/);
  for (const detector of ["Prewitt", "Laplacian"] as const)
    assert.equal(validateAnalysis({ sigma: 1.2, kernel_size: 7, detector, threshold: 96 }), null);
  assert.deepEqual(analysisSettings(DEFAULT_PARAMETERS), { sigma: 1.2, kernel_size: 5, detector: "Sobel", threshold: 96 });
});

test("Sobel response requires derivative, contour, and object outputs", () => {
  const result = parseAnalysisResult(sobelFixture(), "test-request");
  for (const kind of ["gx", "gy", "gradient", "edges"] as const) assert.equal(resultImage(result, kind), png);
  assert.deepEqual(result.object_list, [measurement]);
  assert.equal(resultImage(result, "contours"), png);
  const invalid = [
    { ...result, detection_status: "not_run" }, { ...result, completed_stages: fixture().completed_stages },
    { ...result, parameters_used: fixture().parameters_used }, { ...result, contour_image: null },
    { ...result, object_list: null }, { ...result, object_list: [{ ...measurement, area: 0 }] },
    { ...result, object_list: [{ ...measurement, bounding_box: { x: -1, y: 1, width: 4, height: 3 } }] },
    { ...result, fps: 30 },
    { ...result, parameters_used: { ...result.parameters_used, threshold: 1444 } },
    ...["gx", "gy", "gradient_magnitude", "edge_map"].map((field) => ({ ...result, [field]: null })),
  ];
  for (const value of invalid) assert.throws(() => parseAnalysisResult(value, "test-request"), /incompatible/);
});

test("workspace snapshot sends Sobel plus threshold and omits inactive noise controls", async (t) => {
  t.mock.method(globalThis, "fetch", async (_url: string | URL | Request, options?: RequestInit) => {
    const form = options?.body as FormData;
    assert.deepEqual([...form.keys()], ["image", "sigma", "kernel_size", "request_id", "detector", "threshold"]);
    assert.equal(form.get("image"), file);
    assert.equal(form.get("detector"), "Sobel");
    assert.equal(form.get("threshold"), "1443");
    return Response.json(sobelFixture("test-request", 1443));
  });
  const snapshot = analysisSettings({ ...DEFAULT_PARAMETERS, kernel: 7, threshold: 1443 });
  const result = await analyzeImage(file, snapshot, "test-request", new AbortController().signal);
  assert.equal(result.detection_status, "edges_computed");
});

test("adapter rejects different threshold or detector mode even for otherwise valid results", async (t) => {
  const mock = t.mock.method(globalThis, "fetch", async () => Response.json(sobelFixture("test-request", 200)));
  await assert.rejects(analyzeImage(file, sobelFixture().parameters_used, "test-request", new AbortController().signal), /different settings/);
  mock.mock.mockImplementation(async () => Response.json(fixture()));
  await assert.rejects(analyzeImage(file, sobelFixture().parameters_used, "test-request", new AbortController().signal), /different settings/);
  mock.mock.mockImplementation(async () => Response.json(sobelFixture()));
  await assert.rejects(analyzeImage(file, fixture().parameters_used, "test-request", new AbortController().signal), /different settings/);
});

test("invalid detector settings fail before sending an upload", async (t) => {
  const send = t.mock.method(globalThis, "fetch", async () => { throw new Error("must not send"); });
  await assert.rejects(analyzeImage(file, { sigma: 0, kernel_size: 3, detector: "Canny" as Detector, threshold: 96 }, "id", new AbortController().signal), /Unsupported detector/);
  await assert.rejects(analyzeImage(file, { sigma: 0, kernel_size: 3, detector: "Sobel" }, "id", new AbortController().signal), /threshold/);
  assert.equal(send.mock.callCount(), 0);
});

test("changing only threshold cancels the previous request and ignores its late result", async () => {
  const pending: { resolve: (result: ComputedAnalysisResult) => void; signal: AbortSignal; id: string; settings: AnalysisSettings }[] = [];
  const runner = createAnalysisRunner((_file, settings, id, signal) => new Promise((resolve) => pending.push({ resolve, signal, id, settings })));
  const accepted: number[] = [];
  const callbacks = { success: (result: ComputedAnalysisResult) => accepted.push(result.parameters_used.threshold!), error: assert.fail };
  const oldRequest = runner.run(file, sobelFixture().parameters_used, callbacks);
  const newRequest = runner.run(file, sobelFixture("test-request", 500).parameters_used, callbacks);
  assert.equal(pending[0].signal.aborted, true);
  assert.equal(pending[1].settings.threshold, 500);
  pending[1].resolve(sobelFixture(pending[1].id, 500));
  await newRequest;
  pending[0].resolve(sobelFixture(pending[0].id, 96));
  await oldRequest;
  assert.deepEqual(accepted, [500]);
});

test("detector and threshold are copied into the immutable submission snapshot", async () => {
  const settings = sobelFixture().parameters_used;
  const runner = createAnalysisRunner(async (_file, snapshot, id) => {
    settings.detector = "Prewitt";
    settings.threshold = 500;
    assert.deepEqual(snapshot, { sigma: 1.2, kernel_size: 7, detector: "Sobel", threshold: 96 });
    return sobelFixture(id);
  });
  await runner.run(file, settings, { success: () => {}, error: assert.fail });
});

test("cancelling for a detector change discards late success and error", async () => {
  for (const rejectLate of [false, true]) {
    let finish!: () => void;
    const runner = createAnalysisRunner((_file, _settings, id) => new Promise((resolve, reject) => {
      finish = () => rejectLate ? reject(new Error("stale detector error")) : resolve(sobelFixture(id));
    }));
    const request = runner.run(file, sobelFixture().parameters_used, { success: () => assert.fail("stale detector result"), error: assert.fail });
    // updateParameter cancels this runner for every change, including detector.
    runner.cancel();
    finish();
    await request;
  }
});

test("Gaussian bounds and odd kernels agree with the API", () => {
  for (const kernel_size of [3, 5, 15, 31]) assert.equal(validateGaussian({ sigma: 5, kernel_size }), null);
  for (const sigma of [-1, 5.1, NaN, Infinity, 0.25]) assert.match(validateGaussian({ sigma, kernel_size: 7 })!, /sigma/);
  for (const kernel_size of [0, 2, 33, 3.5]) assert.match(validateGaussian({ sigma: 0, kernel_size })!, /kernel/);
});

test("computed contracts keep unavailable detections distinct from zero objects", () => {
  const result = parseAnalysisResult(fixture(), "test-request");
  assert.equal(result.object_list, null);
  assert.equal(resultImage(result, "grayscale"), png);
  assert.equal(resultImage(result, "filtered-spectrum"), png);
  assert.equal(resultImage(result, "edges"), null);
  assert.equal(resultImage(null, "filtered"), null);
  for (const invalid of [null, {}, { ...fixture(), request_id: "old-request" },
    { ...fixture(), edge_map: png }, { ...fixture(), object_list: [] },
    { ...fixture(), analyzed_dimensions: { width: 513, height: 24 } },
    { ...fixture(), fft_image: "https://example.com/not-a-result" },
    { ...fixture(), spectrum_scale: { min: 0, max: NaN } }]) {
    assert.throws(() => parseAnalysisResult(invalid, "test-request"), /incompatible/);
  }
});

test("adapter uploads the actual File and a Gaussian-only snapshot", async (t) => {
  t.mock.method(globalThis, "fetch", async (url: string | URL | Request, options?: RequestInit) => {
    assert.match(String(url), /\/analyze$/);
    assert.equal(options?.method, "POST");
    const form = options?.body as FormData;
    assert.deepEqual([...form.keys()], ["image", "sigma", "kernel_size", "request_id"]);
    assert.equal((form.get("image") as File).name, "input.png");
    assert.equal(form.get("sigma"), "1.2");
    assert.equal(form.get("kernel_size"), "7");
    assert.equal(form.get("request_id"), "test-request");
    assert.ok(options?.signal instanceof AbortSignal);
    assert.equal(options?.headers, undefined); // Browser must supply the multipart boundary.
    return Response.json(fixture());
  });
  const result = await analyzeImage(file, { sigma: 1.2, kernel_size: 7 }, "test-request", new AbortController().signal);
  assert.equal(result.provenance, "computed");
});

test("network failures explain how to start the backend", async (t) => {
  t.mock.method(globalThis, "fetch", async () => { throw new TypeError("Failed to fetch"); });
  await assert.rejects(analyzeImage(file, { sigma: 1.2, kernel_size: 7 }, "id", new AbortController().signal), /Start it/);
});

test("API errors stay understandable and mismatched settings are rejected", async (t) => {
  const mocked = t.mock.method(globalThis, "fetch", async () => Response.json({ detail: "Choose an image at or below 20 megapixels." }, { status: 413 }));
  await assert.rejects(analyzeImage(file, { sigma: 1.2, kernel_size: 7 }, "test-request", new AbortController().signal), /20 megapixels/);
  mocked.mock.mockImplementation(async () => Response.json({ ...fixture(), parameters_used: { sigma: 2, kernel_size: 7 } }));
  await assert.rejects(analyzeImage(file, { sigma: 1.2, kernel_size: 7 }, "test-request", new AbortController().signal), /different settings/);
});

test("an older response cannot replace newer results even if transport ignores abort", async () => {
  const pending: { resolve: (result: ComputedAnalysisResult) => void; signal: AbortSignal; id: string }[] = [];
  const runner = createAnalysisRunner((_file, _settings, id, signal) => new Promise((resolve) => pending.push({ resolve, signal, id })));
  const accepted: string[] = [];
  const callbacks = { success: (result: ComputedAnalysisResult) => accepted.push(result.request_id), error: assert.fail };
  const oldRequest = runner.run(file, { sigma: 1.2, kernel_size: 7 }, callbacks);
  const newRequest = runner.run(file, { sigma: 2, kernel_size: 13 }, callbacks);
  assert.equal(pending[0].signal.aborted, true);
  pending[1].resolve(fixture(pending[1].id));
  await newRequest;
  pending[0].resolve(fixture(pending[0].id));
  await oldRequest;
  assert.deepEqual(accepted, [pending[1].id]);
});

test("cancellation on image/settings changes or reset discards late success and error", async () => {
  for (const completion of ["success", "error"]) {
    let resolve!: (result: ComputedAnalysisResult) => void;
    let reject!: (error: Error) => void;
    let signal!: AbortSignal;
    const runner = createAnalysisRunner((_file, _settings, _id, requestSignal) => {
      signal = requestSignal;
      return new Promise((success, failure) => { resolve = success; reject = failure; });
    });
    const request = runner.run(file, { sigma: 1.2, kernel_size: 7 }, {
      success: () => assert.fail("Cancelled result was accepted"), error: assert.fail,
    });
    runner.cancel();
    assert.equal(signal.aborted, true);
    if (completion === "success") resolve(fixture());
    else reject(new Error("Old request failed"));
    await request;
  }
});

test("request settings cannot change after submission", async () => {
  const settings = { sigma: 1.2, kernel_size: 7 };
  const runner = createAnalysisRunner(async (_file, snapshot, id) => {
    settings.sigma = 4;
    assert.deepEqual(snapshot, { sigma: 1.2, kernel_size: 7 });
    return fixture(id);
  });
  await runner.run(file, settings, { success: () => {}, error: assert.fail });
});

function detectorFixture(detector: Detector, id = "test-request"): ComputedAnalysisResult {
  const result = sobelFixture(id);
  const laplacian = detector === "Laplacian";
  return { ...result,
    parameters_used: { ...result.parameters_used, detector,
      ...(laplacian ? { laplacian_min_component_area: 2 } : {}) },
    detector_metadata: { detector, decision: laplacian ? "zero_crossing" : "magnitude_threshold",
      threshold_type: laplacian ? "zero_crossing_contrast" : "gradient_magnitude",
      threshold_label: laplacian ? "Zero-crossing contrast threshold" : "Gradient magnitude threshold",
      threshold: 96, threshold_units: laplacian ? "raw_response_difference" : "raw_gradient_magnitude",
      minimum_component_area: laplacian ? 2 : null },
    ...(laplacian ? { gx: null, gy: null, gradient_magnitude: null, laplacian_response: png } : {}),
    completed_stages: ["input", "grayscale", "smooth", detector.toLowerCase(),
      detector === "Laplacian" ? "zero_crossing" : "threshold", "contours", "objects", "fourier"] };
}

const comparisonDefaults = {
  sigma: 1.2, kernel_size: 7, sobel_threshold: 96, prewitt_threshold: 96,
  laplacian_contrast_threshold: 20,
};
function comparisonFixture(id = "test-request", settings = comparisonDefaults): ComparisonResult {
  const entry = (detector: Detector) => ({
    edge_map: png, object_list: [measurement], edge_pixel_count: 12,
    object_count: 1, average_object_area: 12, processing_time: 4.5,
    metadata: {
      detector,
      decision: detector === "Laplacian" ? "zero_crossing" as const : "magnitude_threshold" as const,
      threshold_type: detector === "Laplacian" ? "zero_crossing_contrast" as const : "gradient_magnitude" as const,
      threshold_label: detector === "Laplacian" ? "Zero-crossing contrast threshold" as const : "Gradient magnitude threshold" as const,
      threshold: detector === "Sobel" ? settings.sobel_threshold
        : detector === "Prewitt" ? settings.prewitt_threshold : settings.laplacian_contrast_threshold,
      threshold_units: detector === "Laplacian" ? "raw_response_difference" as const : "raw_gradient_magnitude" as const,
      minimum_component_area: detector === "Laplacian" ? 2 : null,
    },
  });
  return {
    provenance: "computed", request_id: id,
    parameters_used: settings,
    source_dimensions: { width: 32, height: 24 }, analyzed_dimensions: { width: 32, height: 24 },
    sobel: entry("Sobel"), prewitt: entry("Prewitt"), laplacian: entry("Laplacian"),
    processing_time: 18,
  };
}

test("comparison adapter posts shared settings and validates all detector results", async (t) => {
  const expected = comparisonFixture();
  t.mock.method(globalThis, "fetch", async (url: string | URL | Request, options?: RequestInit) => {
    assert.match(String(url), /\/compare$/);
    const form = options?.body as FormData;
    assert.deepEqual([...form.keys()], ["image", "sigma", "kernel_size", "sobel_threshold",
      "prewitt_threshold", "laplacian_contrast_threshold", "request_id"]);
    assert.equal(form.get("image"), file);
    assert.equal(form.get("sigma"), "1.2");
    assert.equal(form.get("kernel_size"), "7");
    assert.equal(form.get("sobel_threshold"), "96");
    assert.equal(form.get("prewitt_threshold"), "96");
    assert.equal(form.get("laplacian_contrast_threshold"), "20");
    return Response.json(expected);
  });
  const settings = comparisonSettings({ ...DEFAULT_PARAMETERS, kernel: 7 });
  assert.equal(validateComparison(settings), null);
  assert.deepEqual(await compareImage(file, settings, "test-request", new AbortController().signal), expected);
});

test("noise settings serialize and validate with model-specific units", async (t) => {
  const settings = analysisSettings({
    ...DEFAULT_PARAMETERS, noise: "Salt & Pepper", noiseStrength: 0.12,
  });
  assert.deepEqual(settings, {
    sigma: 1.2, kernel_size: 5, detector: "Sobel", threshold: 96,
    noise_model: "Salt & Pepper", noise_strength: 0.12, noise_seed: 220,
  });
  assert.equal(validateAnalysis(settings), null);
  assert.match(validateAnalysis({ ...settings, noise_strength: 1.01 })!, /probability/);
  assert.match(validateAnalysis({ ...settings, noise_seed: -1 })!, /seed/);
  t.mock.method(globalThis, "fetch", async (_url: string | URL | Request, options?: RequestInit) => {
    const form = options?.body as FormData;
    assert.equal(form.get("noise_model"), "Salt & Pepper");
    assert.equal(form.get("noise_strength"), "0.12");
    assert.equal(form.get("noise_seed"), "220");
    const result = sobelFixture("noise-request");
    return Response.json({
      ...result,
      noisy_image: png,
      noise: { model: "Salt & Pepper", strength: 0.12,
        units: "pixel corruption probability", seed: 220 },
      parameters_used: settings,
      completed_stages: ["input", "grayscale", "noise", "smooth", "sobel", "threshold",
        "contours", "objects", "fourier"],
    });
  });
  const result = await analyzeImage(file, settings, "noise-request", new AbortController().signal);
  assert.equal(result.noise.units, "pixel corruption probability");
  assert.equal(resultImage(result, "noisy"), png);
});

test("analysis response rejects inconsistent noise visualization and metadata", () => {
  const base = fixture();
  for (const invalid of [
    { ...base, noisy_image: png },
    { ...base, noise: { model: "Gaussian", strength: 12, units: "intensity standard deviation", seed: 220 } },
  ]) assert.throws(() => parseAnalysisResult(invalid, "test-request"), /incompatible/);
});

test("comparison parser rejects missing, inconsistent, or mislabeled detector data", () => {
  const result = comparisonFixture();
  assert.equal(parseComparisonResult(result, "test-request"), result);
  for (const invalid of [
    null,
    { ...result, request_id: "old" },
    { ...result, sobel: { ...result.sobel, edge_map: null } },
    { ...result, prewitt: { ...result.prewitt, object_count: 2 } },
    { ...result, laplacian: { ...result.laplacian, metadata: { ...result.laplacian.metadata, detector: "Sobel" } } },
    { ...result, parameters_used: { ...result.parameters_used, laplacian_contrast_threshold: -1 } },
    { ...result, sobel: { ...result.sobel, metadata: { ...result.sobel.metadata, threshold_type: "zero_crossing_contrast" } } },
  ]) assert.throws(() => parseComparisonResult(invalid, "test-request"), /incompatible/);
});

test("comparison cards map real results to Sobel, Prewitt, and Laplacian in display order", () => {
  const result = comparisonFixture();
  const cards = comparisonEntries(result);
  assert.deepEqual(cards.map(({ key, detector }) => [key, detector]), [
    ["sobel", "Sobel"], ["prewitt", "Prewitt"], ["laplacian", "Laplacian"],
  ]);
  assert.deepEqual(cards.map((card) => card.result?.edge_map), [png, png, png]);
  assert.deepEqual(comparisonEntries(null).map((card) => card.result), [null, null, null]);
});

test("comparison runner cancels and ignores stale success and errors", async () => {
  for (const rejectOld of [false, true]) {
    const pending: { resolve: (result: ComparisonResult) => void; reject: (error: Error) => void;
      id: string; signal: AbortSignal }[] = [];
    const runner = createComparisonRunner((_file, _settings, id, signal) =>
      new Promise((resolve, reject) => pending.push({ resolve, reject, id, signal })));
    const accepted: string[] = [];
    const callbacks = { success: (result: ComparisonResult) => accepted.push(result.request_id), error: assert.fail };
    const old = runner.run(file, comparisonDefaults, callbacks);
    const currentSettings = { ...comparisonDefaults, sigma: 2, kernel_size: 13 };
    const current = runner.run(file, currentSettings, callbacks);
    assert.equal(pending[0].signal.aborted, true);
    pending[1].resolve(comparisonFixture(pending[1].id, currentSettings));
    await current;
    if (rejectOld) pending[0].reject(new Error("stale comparison error"));
    else pending[0].resolve(comparisonFixture(pending[0].id));
    await old;
    assert.deepEqual(accepted, [pending[1].id]);
  }
});

function multiScaleFixture(detector: Detector = "Sobel", id = "test-request"): ComputedAnalysisResult {
  const result = detectorFixture(detector, id);
  return { ...result,
    parameters_used: { ...result.parameters_used, multi_scale: true,
      scale_sigmas: [0.8, 1.6, 3.2], scale_support: 2 },
    multi_scale: {
      sigmas: [0.8, 1.6, 3.2], support_count: 2,
      scales: [
        { sigma: 0.8, kernel_size: 7, edge_map: png },
        { sigma: 1.6, kernel_size: 11, edge_map: png },
        { sigma: 3.2, kernel_size: 21, edge_map: png },
      ],
      persistence_map: png,
      persistence_scale: { min: 0, max: 3, mapping: "linear_grayscale" },
      fused_edge_map: png,
    },
    completed_stages: ["input", "grayscale", "smooth", detector.toLowerCase(),
      detector === "Laplacian" ? "zero_crossing" : "threshold", "multi_scale",
      "contours", "objects", "fourier"] };
}

for (const detector of ["Sobel", "Prewitt", "Laplacian"] as const) {
  test(`${detector} request sends the selected detector and accepts its actual response`, async (t) => {
    const expected = detectorFixture(detector);
    t.mock.method(globalThis, "fetch", async (_url: string | URL | Request, options?: RequestInit) => {
      const form = options?.body as FormData;
      assert.equal(form.get("detector"), detector);
      assert.equal(form.get("threshold"), "96");
      assert.equal(form.get("image"), file);
      return Response.json(expected);
    });
    const settings = analysisSettings({ ...DEFAULT_PARAMETERS, detector, kernel: 7 });
    assert.deepEqual(await analyzeImage(file, settings, "test-request", new AbortController().signal), expected);
    assert.equal(restoreParameters({ detector }).detector, detector);
    for (const threshold of [-1, 1444, NaN, Infinity, undefined])
      assert.match(validateAnalysis({ ...settings, threshold })!, /threshold/);
  });

  test(`${detector} parser rejects missing or mixed detector outputs`, () => {
    const result = detectorFixture(detector);
    assert.equal(parseAnalysisResult(result, "test-request"), result);
    const required = detector === "Laplacian" ? ["laplacian_response", "edge_map", "contour_image"] : ["gx", "gy", "gradient_magnitude", "edge_map", "contour_image"];
    for (const field of required) {
      for (const value of [null, undefined, "not a PNG"])
        assert.throws(() => parseAnalysisResult({ ...result, [field]: value }, "test-request"), /incompatible/);
    }
    const wrongFields = detector === "Laplacian"
      ? ["gx", "gy", "gradient_magnitude"].map((field) => ({ ...result, [field]: png }))
      : [{ ...result, laplacian_response: png }];
    for (const invalid of [...wrongFields, { ...result, completed_stages: fixture().completed_stages }])
      assert.throws(() => parseAnalysisResult(invalid, "test-request"), /incompatible/);
    assert.equal(resultImage(result, "laplacian"), detector === "Laplacian" ? png : null);
    assert.equal(resultImage(result, "gradient"), detector === "Laplacian" ? null : png);
  });
}

test("detector stages, labels and export scales distinguish signed responses from gradients", () => {
  for (const detector of ["Sobel", "Prewitt", "Laplacian"] as const) {
    const stages = analysisStages(detector);
    const display = detectorDisplay(detector);
    assert.equal(stages[3].name, detector);
    assert.equal(stages[3].key, detector.toLowerCase());
    assert.equal(stages[3].view, detector === "Laplacian" ? "laplacian" : "gradient");
    assert.equal(stages[4].key, detector === "Laplacian" ? "zero_crossing" : "threshold");
    assert.equal(display.signed_max, detector === "Prewitt" ? 765 : 1020);
    assert.equal(display.magnitude_max, detector === "Laplacian" ? null : display.signed_max * Math.sqrt(2));
    if (detector !== "Laplacian") {
      assert.match(detectorLabel("gx", detector)!, new RegExp(`${detector} Gx`));
      assert.match(detectorLabel("gy", detector)!, new RegExp(`${detector} Gy`));
      assert.match(detectorLabel("gradient", detector)!, /magnitude/);
    }
  }
  assert.equal(detectorLabel("laplacian", "Laplacian"), "Laplacian signed response");
  assert.equal(detectorLabel("edges", "Laplacian"), "Zero-crossing edge map");
  assert.equal(detectorLabel("gradient", "Laplacian"), null);
  assert.equal(thresholdLabel("Sobel"), "Gradient magnitude threshold");
  assert.equal(thresholdLabel("Prewitt"), "Gradient magnitude threshold");
  assert.equal(thresholdLabel("Laplacian"), "Zero-crossing contrast threshold");
  assert.match(thresholdRule("Laplacian"), /Opposite-sign.*response difference > threshold/);
  assert.equal(thresholdRule("Sobel"), "raw Sobel magnitude > threshold");
  assert.equal(thresholdRule("Prewitt"), "raw Prewitt magnitude > threshold");
});

test("switching detectors accepts the new result and discards the old result", async () => {
  const pending: { resolve: (result: ComputedAnalysisResult) => void; id: string; signal: AbortSignal }[] = [];
  const runner = createAnalysisRunner((_file, _settings, id, signal) => new Promise((resolve) => pending.push({ resolve, id, signal })));
  const accepted: (Detector | undefined)[] = [];
  const callbacks = { success: (result: ComputedAnalysisResult) => accepted.push(result.parameters_used.detector), error: assert.fail };
  const old = runner.run(file, detectorFixture("Prewitt").parameters_used, callbacks);
  const current = runner.run(file, detectorFixture("Laplacian").parameters_used, callbacks);
  assert.equal(pending[0].signal.aborted, true);
  pending[1].resolve(detectorFixture("Laplacian", pending[1].id));
  await current;
  pending[0].resolve(detectorFixture("Prewitt", pending[0].id));
  await old;
  assert.deepEqual(accepted, ["Laplacian"]);
});

test("multi-scale settings validate and serialize without changing single-scale defaults", () => {
  const enabled = { ...DEFAULT_PARAMETERS, multiScale: true };
  assert.deepEqual(analysisSettings(enabled), {
    sigma: 1.2, kernel_size: 5, detector: "Sobel", threshold: 96,
    multi_scale: true, scale_sigmas: [0.8, 1.6, 3.2], scale_support: 2,
  });
  assert.equal(validateAnalysis(analysisSettings(enabled)), null);
  for (const invalid of [
    { ...analysisSettings(enabled), scale_sigmas: [] },
    { ...analysisSettings(enabled), scale_sigmas: [1, 1, 2] },
    { ...analysisSettings(enabled), scale_sigmas: [-1, 1, 2] },
    { ...analysisSettings(enabled), scale_sigmas: [0.25, 1, 2] },
    { ...analysisSettings(enabled), scale_support: 4 },
    { ...analysisSettings(DEFAULT_PARAMETERS), scale_sigmas: [1, 2] },
  ]) assert.match(validateAnalysis(invalid)!, /scale|sigma|support/i);
});

test("multi-scale request sends controls and accepts per-scale persistence outputs", async (t) => {
  const expected = multiScaleFixture();
  t.mock.method(globalThis, "fetch", async (_url: string | URL | Request, options?: RequestInit) => {
    const form = options?.body as FormData;
    assert.deepEqual([...form.keys()], ["image", "sigma", "kernel_size", "request_id", "detector",
      "threshold", "multi_scale", "scale_sigmas", "scale_support"]);
    assert.equal(form.get("multi_scale"), "true");
    assert.equal(form.get("scale_sigmas"), "0.8,1.6,3.2");
    assert.equal(form.get("scale_support"), "2");
    return Response.json(expected);
  });
  const result = await analyzeImage(file, expected.parameters_used, "test-request", new AbortController().signal);
  assert.deepEqual(result.multi_scale, expected.multi_scale);
});

test("multi-scale parser rejects inconsistent scale metadata and missing images", () => {
  const result = multiScaleFixture("Laplacian");
  assert.equal(parseAnalysisResult(result, "test-request"), result);
  const invalid = [
    { ...result, multi_scale: null },
    { ...result, completed_stages: detectorFixture("Laplacian").completed_stages },
    { ...result, multi_scale: { ...result.multi_scale!, support_count: 4 } },
    { ...result, multi_scale: { ...result.multi_scale!, persistence_map: null } },
    { ...result, multi_scale: { ...result.multi_scale!, persistence_scale: { min: 0, max: 2, mapping: "linear_grayscale" } } },
    { ...result, multi_scale: { ...result.multi_scale!, scales: result.multi_scale!.scales.slice(1) } },
    { ...result, parameters_used: { ...result.parameters_used, multi_scale: undefined } },
  ];
  for (const value of invalid)
    assert.throws(() => parseAnalysisResult(value, "test-request"), /incompatible/);
});
