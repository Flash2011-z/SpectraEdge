import test from "node:test";
import assert from "node:assert/strict";
import { analyzeImage, createAnalysisRunner, parseAnalysisResult } from "../lib/api.ts";
import { resultImage, validateGaussian, type ComputedAnalysisResult } from "../lib/workspace.ts";

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
