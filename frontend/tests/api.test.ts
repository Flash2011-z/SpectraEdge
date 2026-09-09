import test from "node:test";
import assert from "node:assert/strict";
import { analyzeImage, createAnalysisRunner, parseAnalysisResult } from "../lib/api.ts";
import { analysisSettings, DEFAULT_PARAMETERS, resultImage, validateAnalysis, validateGaussian, type AnalysisSettings, type ComputedAnalysisResult } from "../lib/workspace.ts";

const png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAYCAAAAAC+OKDoAAAAJklEQVR4nGNkYPjPwIgHMzEQAv//48cETWAcdQMYjIYDBAyGcAAAxx2Pub9i5ZMAAAAASUVORK5CYII=";
function fixture(id = "test-request"): ComputedAnalysisResult {
  return {
    provenance: "computed", request_id: id, original_image: png,
    grayscale_image: png, filtered_image: png, fft_image: png, filtered_fft_image: png,
    gx: null, gy: null, gradient_magnitude: null, edge_map: null,
    contour_image: null, object_list: null, fps: null, processing_time: 10,
    parameters_used: { sigma: 1.2, kernel_size: 7 },
    source_dimensions: { width: 32, height: 24 }, analyzed_dimensions: { width: 32, height: 24 },
    completed_stages: ["input", "grayscale", "smooth", "fourier"], detection_status: "not_run",
    spectrum_scale: { min: 0, max: 12, mapping: "linear_grayscale" },
  };
}
const file = new File([new Uint8Array([1, 2, 3])], "input.png", { type: "image/png" });

function sobelFixture(id = "test-request", threshold = 96): ComputedAnalysisResult {
  return { ...fixture(id), gx: png, gy: png, gradient_magnitude: png, edge_map: png,
    parameters_used: { sigma: 1.2, kernel_size: 7, detector: "Sobel", threshold },
    detection_status: "edges_computed",
    completed_stages: ["input", "grayscale", "smooth", "sobel", "threshold", "fourier"] };
}

test("Sobel settings use raw 0..1443 units and reject unsupported or incomplete settings", () => {
  for (const threshold of [0, 96, 96.5, 1443])
    assert.equal(validateAnalysis({ sigma: 1.2, kernel_size: 7, detector: "Sobel", threshold }), null);
  for (const threshold of [-1, 1443.1, NaN, Infinity, undefined])
    assert.match(validateAnalysis({ sigma: 1.2, kernel_size: 7, detector: "Sobel", threshold })!, /threshold/);
  assert.match(validateAnalysis({ sigma: 1.2, kernel_size: 7, threshold: 96 })!, /requires/);
  for (const detector of ["Prewitt", "Laplacian"] as const)
    assert.match(validateAnalysis({ sigma: 1.2, kernel_size: 7, detector, threshold: 96 })!, /Only.*Sobel/);
  assert.deepEqual(analysisSettings(DEFAULT_PARAMETERS), { sigma: 1.2, kernel_size: 5, detector: "Sobel", threshold: 96 });
});

test("Sobel response requires all actual derivative images but never invents object measurements", () => {
  const result = parseAnalysisResult(sobelFixture(), "test-request");
  for (const kind of ["gx", "gy", "gradient", "edges"] as const) assert.equal(resultImage(result, kind), png);
  assert.equal(result.object_list, null);
  const invalid = [
    { ...result, detection_status: "not_run" }, { ...result, completed_stages: fixture().completed_stages },
    { ...result, parameters_used: fixture().parameters_used }, { ...result, contour_image: png },
    { ...result, object_list: [] }, { ...result, fps: 30 },
    { ...result, parameters_used: { ...result.parameters_used, threshold: 1444 } },
    ...["gx", "gy", "gradient_magnitude", "edge_map"].map((field) => ({ ...result, [field]: null })),
  ];
  for (const value of invalid) assert.throws(() => parseAnalysisResult(value, "test-request"), /incompatible/);
});

test("workspace snapshot sends Sobel plus threshold and omits all future controls", async (t) => {
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
  await assert.rejects(analyzeImage(file, { sigma: 0, kernel_size: 3, detector: "Prewitt", threshold: 96 }, "id", new AbortController().signal), /Only.*Sobel/);
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
