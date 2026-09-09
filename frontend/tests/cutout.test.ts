import test from "node:test";
import assert from "node:assert/strict";
import { createCutoutRunner, extractCutout, prepareCutout } from "../lib/cutout-api.ts";
import { AI_MODEL, DEFAULT_CUTOUT_SETTINGS, imagePoint, parseCutoutResult, parsePreparedImage, rectangleFromPoints, resolveCutoutRectangle, snapshotCutout, validateCutoutSettings, validateSelection,
  type CutoutResult, type CutoutSettings, type CutoutSnapshot, type PreparedImage } from "../lib/cutout.ts";
import { photoBytes, photoFile, preparedFixture, rectangle, resultFixture, selectionFixture } from "./cutout-fixture.ts";

test("pointer coordinates use the current displayed canvas, including offsets and resized touch views", () => {
  const image = { width: 1024, height: 512 };
  assert.deepEqual(imagePoint({ x: 228, y: 114 }, { x: 100, y: 50, width: 512, height: 256 }, image), { x: 256, y: 128 });
  assert.deepEqual(imagePoint({ x: 164, y: 82 }, { x: 100, y: 50, width: 256, height: 128 }, image), { x: 256, y: 128 });
  assert.deepEqual(imagePoint({ x: -1, y: -1 }, { x: 100, y: 50, width: 512, height: 256 }, image), { x: 0, y: 0 });
  assert.deepEqual(imagePoint({ x: 2000, y: 2000 }, { x: 100, y: 50, width: 512, height: 256 }, image), { x: 1023, y: 511 });
  assert.throws(() => imagePoint({ x: 0, y: 0 }, { x: 0, y: 0, width: 0, height: 0 }, image), /size/);
});

test("rectangle selection includes both endpoints in all four drag directions", () => {
  for (const [start, end] of [[{ x: 2, y: 1 }, { x: 13, y: 10 }], [{ x: 13, y: 10 }, { x: 2, y: 1 }],
    [{ x: 13, y: 1 }, { x: 2, y: 10 }], [{ x: 2, y: 10 }, { x: 13, y: 1 }]]) {
    assert.deepEqual(rectangleFromPoints(start, end), rectangle);
  }
  assert.equal(validateSelection(preparedFixture(), rectangle, []), null);
  assert.match(validateSelection(preparedFixture(), rectangleFromPoints({ x: 2, y: 2 }, { x: 2, y: 2 }), [])!, /rectangle/);
});

test("selection validation agrees with image, brush, and payload bounds", () => {
  const snapshot = selectionFixture();
  assert.equal(validateSelection(snapshot.prepared, snapshot.rectangle, snapshot.marks), null);
  assert.match(validateSelection(snapshot.prepared, { ...rectangle, x: -1 }, [])!, /rectangle/);
  assert.match(validateSelection(snapshot.prepared, { ...rectangle, width: 16 }, [])!, /rectangle/);
  for (const size of [0, 129, 1.5, NaN])
    assert.match(validateSelection(snapshot.prepared, rectangle, [{ ...snapshot.marks[0], size }])!, /Brush/);
  assert.match(validateSelection(snapshot.prepared, rectangle, [{ ...snapshot.marks[0], points: [{ x: 16, y: 0 }] }])!, /points/);
  assert.match(validateSelection(snapshot.prepared, rectangle, Array(201).fill(snapshot.marks[0]))!, /200/);
  assert.match(validateSelection(snapshot.prepared, rectangle, [{ ...snapshot.marks[0], points: Array(2001).fill({ x: 1, y: 1 }) }])!, /2000/);
  assert.match(validateSelection(snapshot.prepared, rectangle, Array(11).fill({ ...snapshot.marks[0], points: Array(2000).fill({ x: 1, y: 1 }) }))!, /20000/);
});

test("snapshots deep-copy rectangles, marks, and points", () => {
  const original = selectionFixture();
  const snapshot = snapshotCutout(original);
  original.rectangle.x = 10;
  original.marks[0].mode = "remove";
  original.marks[0].points[0].x = 15;
  original.prepared.source_dimensions.width = 2000;
  assert.deepEqual(snapshot, selectionFixture());
});

test("preparation response validates identity, PNG RGBA type, dimensions, and timing", () => {
  assert.deepEqual(parsePreparedImage(preparedFixture(), "request-1"), preparedFixture());
  for (const value of [null, {}, { ...preparedFixture(), request_id: "old" }, { ...preparedFixture(), image_id: "wrong" },
    { ...preparedFixture(), width: 1025 }, { ...preparedFixture(), width: 15 }, { ...preparedFixture(), prepared_image: resultFixture().mask_image },
    { ...preparedFixture(), prepared_image: "data:image/png;base64,AAAA" }, { ...preparedFixture(), processing_time: NaN }])
    assert.throws(() => parsePreparedImage(value, "request-1"), /incompatible/);
});

test("extraction validates all three PNGs, alpha-capable colour type, crop bounds and image identity", () => {
  assert.deepEqual(parseCutoutResult(resultFixture(), "request-1", preparedFixture()), resultFixture());
  for (const value of [null, {}, { ...resultFixture(), request_id: "old" }, { ...resultFixture(), image_id: "0".repeat(64) },
    { ...resultFixture(), algorithm: "mock" }, { ...resultFixture(), width: 15 }, { ...resultFixture(), processing_time: -1 },
    { ...resultFixture(), foreground_bounds: { x: 15, y: 3, width: 8, height: 6 } },
    { ...resultFixture(), cropped_dimensions: { width: 7, height: 6 } }, { ...resultFixture(), cutout_image: resultFixture().mask_image },
    { ...resultFixture(), mask_image: resultFixture().cutout_image }, { ...resultFixture(), cropped_image: resultFixture().cutout_image }])
    assert.throws(() => parseCutoutResult(value, "request-1", preparedFixture()), /incompatible/);
});

test("preparation posts the actual uploaded file and request ID", async (t) => {
  t.mock.method(globalThis, "fetch", async (url: string | URL | Request, options?: RequestInit) => {
    assert.match(String(url), /\/cutout\/prepare$/);
    const form = options?.body as FormData;
    assert.deepEqual([...form.keys()], ["image", "request_id"]);
    assert.equal(form.get("image"), photoFile);
    assert.equal(form.get("request_id"), "request-1");
    assert.equal(options?.headers, undefined);
    assert.ok(options?.signal instanceof AbortSignal);
    return Response.json(preparedFixture());
  });
  await prepareCutout(photoFile, "request-1", new AbortController().signal);
});

test("extraction sends identical prepared PNG bytes and all marks in order", async (t) => {
  const snapshot = selectionFixture();
  snapshot.marks.push({ mode: "remove", size: 2, points: [{ x: 0, y: 0 }] });
  t.mock.method(globalThis, "fetch", async (url: string | URL | Request, options?: RequestInit) => {
    assert.match(String(url), /\/cutout\/extract$/);
    const form = options?.body as FormData;
    assert.deepEqual([...form.keys()], ["image", "settings"]);
    assert.deepEqual(Buffer.from(await (form.get("image") as File).arrayBuffer()), photoBytes);
    assert.deepEqual(JSON.parse(form.get("settings") as string), { request_id: "request-1", image_id: snapshot.prepared.image_id,
      width: 16, height: 12, rectangle: snapshot.rectangle, marks: snapshot.marks, method: "grabcut" });
    return Response.json(resultFixture());
  });
  await extractCutout(snapshot, "request-1", new AbortController().signal);
});

test("invalid selections fail before posting and network/API errors remain actionable", async (t) => {
  const send = t.mock.method(globalThis, "fetch", async () => { throw new TypeError("network"); });
  await assert.rejects(extractCutout({ ...selectionFixture(), rectangle: { ...rectangle, width: 0 } }, "request-1", new AbortController().signal), /rectangle/);
  assert.equal(send.mock.callCount(), 0);
  await assert.rejects(prepareCutout(photoFile, "request-1", new AbortController().signal), /Start it/);
  send.mock.mockImplementation(async () => Response.json({ detail: "Leave some background." }, { status: 422 }));
  await assert.rejects(extractCutout(selectionFixture(), "request-1", new AbortController().signal), /background/);
});

test("new uploads cancel older preparations even when the transport ignores abort", async () => {
  const pending: { resolve: (value: PreparedImage) => void; signal: AbortSignal; id: string }[] = [];
  const runner = createCutoutRunner((_file, id, signal) => new Promise((resolve) => pending.push({ resolve, id, signal })));
  const accepted: string[] = [];
  const callbacks = { success: (image: PreparedImage) => accepted.push(image.request_id), error: assert.fail };
  const old = runner.prepare(photoFile, callbacks);
  const latest = runner.prepare(photoFile, callbacks);
  assert.equal(pending[0].signal.aborted, true);
  pending[1].resolve(preparedFixture(pending[1].id)); await latest;
  pending[0].resolve(preparedFixture(pending[0].id)); await old;
  assert.deepEqual(accepted, [pending[1].id]);
});

test("brush/rectangle/reset cancellation discards late extraction success and errors", async () => {
  for (const failed of [false, true]) {
    let finish!: () => void;
    let signal!: AbortSignal;
    const runner = createCutoutRunner(prepareCutout, (_snapshot, id, active) => new Promise((resolve, reject) => {
      signal = active;
      finish = () => failed ? reject(new Error("old failure")) : resolve(resultFixture(id));
    }));
    const request = runner.extract(selectionFixture(), { success: () => assert.fail("outdated download enabled"), error: assert.fail });
    runner.cancel();
    assert.equal(signal.aborted, true);
    finish(); await request;
  }
});

test("preparation and extraction share one cancellation generation", async () => {
  let finish!: () => void;
  const runner = createCutoutRunner(async (_file, id) => preparedFixture(id), (_snapshot, id) => new Promise((resolve) => { finish = () => resolve(resultFixture(id)); }));
  const extraction = runner.extract(selectionFixture(), { success: () => assert.fail("previous photo result accepted"), error: assert.fail });
  await runner.prepare(photoFile, { success: () => {}, error: assert.fail });
  finish(); await extraction;
});

test("runner keeps an immutable selection and ignores older successful refinements", async () => {
  const input = selectionFixture();
  const pending: { resolve: (value: CutoutResult) => void; id: string }[] = [];
  const runner = createCutoutRunner(prepareCutout, (snapshot, id) => {
    input.marks[0].points[0].x = 15;
    assert.equal(snapshot.marks[0].points[0].x, 6);
    return new Promise((resolve) => pending.push({ resolve, id }));
  });
  const accepted: string[] = [];
  const callbacks = { success: (value: CutoutResult) => accepted.push(value.request_id), error: assert.fail };
  const first = runner.extract(input, callbacks);
  const second = runner.extract(selectionFixture(), callbacks);
  pending[1].resolve(resultFixture(pending[1].id)); await second;
  pending[0].resolve(resultFixture(pending[0].id)); await first;
  assert.deepEqual(accepted, [pending[1].id]);
});

function edgeResult(id = "request-1"): CutoutResult {
  return { ...resultFixture(id), method: "edge-watershed", algorithm: "skimage-watershed",
    seed_mode: "brush",
    parameters_used: { sigma: 1.2, kernel_size: 5 },
    // A grayscale fixture checks transport metadata, not segmentation accuracy.
    guidance_image: resultFixture().mask_image,
    guidance_scale: { min: 0, max: 345.67, mapping: "linear_grayscale" } };
}

test("watershed is the editor default, with existing Gaussian validation and 512-pixel bound", () => {
  assert.deepEqual(DEFAULT_CUTOUT_SETTINGS, { method: "edge-watershed", sigma: 1.2, kernel_size: 5 });
  assert.equal(validateCutoutSettings(DEFAULT_CUTOUT_SETTINGS), null);
  for (const sigma of [-1, 5.1, 0.15, NaN, Infinity])
    assert.match(validateCutoutSettings({ method: "edge-watershed", sigma, kernel_size: 5 })!, /sigma/);
  for (const kernel_size of [0, 2, 4, 33, 5.5])
    assert.match(validateCutoutSettings({ method: "edge-watershed", sigma: 1.2, kernel_size })!, /kernel/);
  assert.equal(validateCutoutSettings({ method: "edge-watershed", sigma: 0, kernel_size: 31 }), null);
  assert.equal(validateCutoutSettings({ method: "grabcut" }), null);
  assert.match(validateCutoutSettings({ method: "grabcut", sigma: 1.2 } as CutoutSettings)!, /only/);
  assert.match(validateSelection({ width: 513, height: 12 }, rectangle, [])!, /512/);
});

test("edge responses require matching actual method, Gaussian parameters and full-size grayscale guidance", () => {
  const expected = DEFAULT_CUTOUT_SETTINGS;
  assert.deepEqual(parseCutoutResult(edgeResult(), "request-1", preparedFixture(), expected), edgeResult());
  const zero = { ...edgeResult(), guidance_scale: { min: 0, max: 0, mapping: "linear_grayscale" } };
  assert.doesNotThrow(() => parseCutoutResult(zero, "request-1", preparedFixture(), expected));
  for (const value of [resultFixture(), { ...edgeResult(), method: "grabcut" },
    { ...edgeResult(), algorithm: "opencv-grabcut" }, { ...edgeResult(), parameters_used: null },
    { ...edgeResult(), parameters_used: { sigma: 0, kernel_size: 5 } },
    { ...edgeResult(), parameters_used: { sigma: 1.2, kernel_size: 7 } },
    { ...edgeResult(), guidance_image: null }, { ...edgeResult(), guidance_image: preparedFixture().prepared_image },
    { ...edgeResult(), guidance_scale: null }, { ...edgeResult(), guidance_scale: { min: 0, max: NaN, mapping: "linear_grayscale" } },
    { ...edgeResult(), guidance_scale: { min: 0, max: -1, mapping: "linear_grayscale" } },
    { ...edgeResult(), guidance_scale: { min: 0, max: 2, mapping: "log" } }])
    assert.throws(() => parseCutoutResult(value, "request-1", preparedFixture(), expected), /incompatible/);
  assert.throws(() => parseCutoutResult(edgeResult(), "request-1", preparedFixture(), { method: "grabcut" }), /incompatible/);
});

test("edge adapter submits the method and numerical snapshot, rejecting mismatched response parameters", async (t) => {
  const input = { ...selectionFixture(), settings: { ...DEFAULT_CUTOUT_SETTINGS } };
  const send = t.mock.method(globalThis, "fetch", async (_url: string | URL | Request, options?: RequestInit) => {
    const form = options?.body as FormData;
    const posted = JSON.parse(form.get("settings") as string);
    assert.equal(posted.method, "edge-watershed");
    assert.equal(posted.sigma, 1.2);
    assert.equal(posted.kernel_size, 5);
    assert.deepEqual(Buffer.from(await (form.get("image") as File).arrayBuffer()), photoBytes);
    return Response.json(edgeResult());
  });
  await extractCutout(input, "request-1", new AbortController().signal);
  send.mock.mockImplementation(async () => Response.json({ ...edgeResult(), parameters_used: { sigma: 0, kernel_size: 5 } }));
  await assert.rejects(extractCutout(input, "request-1", new AbortController().signal), /incompatible/);
  send.mock.resetCalls();
  await assert.rejects(extractCutout({ ...input, settings: { method: "edge-watershed", sigma: -1, kernel_size: 5 } }, "request-1", new AbortController().signal), /sigma/);
  assert.equal(send.mock.callCount(), 0);
});

test("method/sigma/kernel snapshots cannot be changed by edits while extraction is pending", async () => {
  const input: CutoutSnapshot = { ...selectionFixture(), settings: { ...DEFAULT_CUTOUT_SETTINGS } };
  let captured!: CutoutSnapshot;
  let finish!: () => void;
  const runner = createCutoutRunner(prepareCutout, (snapshot, id) => {
    captured = snapshot;
    return new Promise((resolve) => { finish = () => resolve(edgeResult(id)); });
  });
  const request = runner.extract(input, { success: () => {}, error: assert.fail });
  if (input.settings.method === "edge-watershed") {
    input.settings.sigma = 4;
    input.settings.kernel_size = 31;
  }
  input.settings = { method: "grabcut" };
  assert.deepEqual(captured.settings, DEFAULT_CUTOUT_SETTINGS);
  finish(); await request;
});

test("cancel on each method/Gaussian edit prevents late success or failure from restoring a download", async () => {
  const edits: CutoutSettings[] = [
    { method: "grabcut" }, { method: "ai-assisted" }, { method: "edge-watershed", sigma: 0, kernel_size: 5 },
    { method: "edge-watershed", sigma: 1.2, kernel_size: 7 },
  ];
  for (const settings of edits) {
    for (const failed of [false, true]) {
      let finish!: () => void;
      let active!: AbortSignal;
      const runner = createCutoutRunner(prepareCutout, (_snapshot, id, signal) => new Promise((resolve, reject) => {
        active = signal;
        finish = () => failed ? reject(new Error("stale failure")) : resolve(edgeResult(id));
      }));
      const input = { ...selectionFixture(), settings: { ...DEFAULT_CUTOUT_SETTINGS } };
      const request = runner.extract(input, { success: () => assert.fail("stale download enabled"), error: assert.fail });
      // useCutout.changeSettings invokes this same cancellation before editing.
      runner.cancel();
      input.settings = settings;
      assert.equal(active.aborted, true);
      finish(); await request;
    }
  }
});

test("no-brush extraction sends empty marks and accepts only automatic seed provenance", async (t) => {
  const automatic = { ...edgeResult(), seed_mode: "automatic" as const };
  const input = { ...selectionFixture(), settings: { ...DEFAULT_CUTOUT_SETTINGS }, marks: [] };
  const send = t.mock.method(globalThis, "fetch", async (_url: string | URL | Request, options?: RequestInit) => {
    const posted = JSON.parse((options?.body as FormData).get("settings") as string);
    assert.deepEqual(posted.marks, []);
    return Response.json(automatic);
  });
  const result = await extractCutout(input, "request-1", new AbortController().signal);
  assert.equal(result.seed_mode, "automatic");
  send.mock.mockImplementation(async () => Response.json(edgeResult()));
  await assert.rejects(extractCutout(input, "request-1", new AbortController().signal), /incompatible/);
});

test("adding or clearing optional marks cancels the pending automatic/brush result", async () => {
  for (const marks of [[], selectionFixture().marks]) {
    let finish!: () => void;
    const runner = createCutoutRunner(prepareCutout, (_snapshot, id) => new Promise((resolve) => {
      finish = () => resolve(edgeResult(id));
    }));
    const input = { ...selectionFixture(), settings: { ...DEFAULT_CUTOUT_SETTINGS }, marks };
    const request = runner.extract(input, { success: () => assert.fail("old seed mode enabled download"), error: assert.fail });
    runner.cancel();
    input.marks = marks.length ? [] : selectionFixture().marks;
    finish(); await request;
  }
});

function aiResult(id = "request-1"): CutoutResult {
  return { ...resultFixture(id), method: "ai-assisted", algorithm: "rembg-onnx",
    parameters_used: { model: AI_MODEL }, seed_mode: null, guidance_image: null, guidance_scale: null };
}

test("AI is opt-in; only AI supplies a full-frame rectangle when no selection exists", () => {
  assert.equal(DEFAULT_CUTOUT_SETTINGS.method, "edge-watershed");
  assert.equal(resolveCutoutRectangle(preparedFixture(), null, "edge-watershed"), null);
  assert.equal(resolveCutoutRectangle(preparedFixture(), null, "grabcut"), null);
  assert.equal(resolveCutoutRectangle(null, null, "ai-assisted"), null);
  assert.deepEqual(resolveCutoutRectangle(preparedFixture(), null, "ai-assisted"), { x: 0, y: 0, width: 16, height: 12 });
  assert.deepEqual(resolveCutoutRectangle(preparedFixture(), rectangle, "ai-assisted"), rectangle);
  assert.equal(validateCutoutSettings({ method: "ai-assisted" }), null);
  for (const extra of [{ sigma: 1.2 }, { kernel_size: 5 }, { model: "withoutbg" }])
    assert.ok(validateCutoutSettings({ method: "ai-assisted", ...extra } as CutoutSettings));
});

test("AI result requires explicit matching model, method, algorithm and no fake edge guidance", () => {
  const settings: CutoutSettings = { method: "ai-assisted" };
  assert.deepEqual(parseCutoutResult(aiResult(), "request-1", preparedFixture(), settings), aiResult());
  for (const value of [resultFixture(), edgeResult(), { ...aiResult(), algorithm: "opencv-grabcut" },
    { ...aiResult(), parameters_used: {} }, { ...aiResult(), parameters_used: { model: "other" } },
    { ...aiResult(), parameters_used: { model: AI_MODEL, sigma: 1.2 } },
    { ...aiResult(), seed_mode: "automatic" }, { ...aiResult(), guidance_image: resultFixture().mask_image },
    { ...aiResult(), guidance_scale: { min: 0, max: 255, mapping: "linear_grayscale" } }])
    assert.throws(() => parseCutoutResult(value, "request-1", preparedFixture(), settings), /incompatible/);
  assert.throws(() => parseCutoutResult(aiResult(), "request-1", preparedFixture(), DEFAULT_CUTOUT_SETTINGS), /incompatible/);
});

test("AI adapter submits only the explicit method and surfaces unavailable errors without fallback", async (t) => {
  const input: CutoutSnapshot = { ...selectionFixture(), settings: { method: "ai-assisted" }, marks: [],
    rectangle: resolveCutoutRectangle(preparedFixture(), null, "ai-assisted")! };
  const send = t.mock.method(globalThis, "fetch", async (_url: string | URL | Request, options?: RequestInit) => {
    const form = options?.body as FormData;
    const posted = JSON.parse(form.get("settings") as string);
    assert.equal(posted.method, "ai-assisted");
    assert.equal("sigma" in posted || "kernel_size" in posted || "model" in posted, false);
    assert.deepEqual(posted.rectangle, { x: 0, y: 0, width: 16, height: 12 });
    assert.deepEqual(posted.marks, []);
    assert.deepEqual(Buffer.from(await (form.get("image") as File).arrayBuffer()), photoBytes);
    return Response.json(aiResult());
  });
  assert.equal((await extractCutout(input, "request-1", new AbortController().signal)).method, "ai-assisted");
  send.mock.resetCalls();
  send.mock.mockImplementation(async () => Response.json({ detail: "AI model is not installed. Choose another method." }, { status: 503 }));
  await assert.rejects(extractCutout(input, "request-1", new AbortController().signal), /AI model is not installed/);
  assert.equal(send.mock.callCount(), 1);
});

test("switching away from AI discards its late success or error", async () => {
  for (const failed of [false, true]) {
    let finish!: () => void;
    const runner = createCutoutRunner(prepareCutout, (_snapshot, id) => new Promise((resolve, reject) => {
      finish = () => failed ? reject(new Error("old AI error")) : resolve(aiResult(id));
    }));
    const input: CutoutSnapshot = { ...selectionFixture(), settings: { method: "ai-assisted" } };
    const request = runner.extract(input, { success: () => assert.fail("Old AI result accepted"), error: assert.fail });
    runner.cancel();
    input.settings = { ...DEFAULT_CUTOUT_SETTINGS };
    finish(); await request;
  }
});
