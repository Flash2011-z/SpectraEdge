import test from "node:test";
import assert from "node:assert/strict";
import { cameraError, createCameraSession, DEFAULT_LIVE_SETTINGS, frameDimensions,
  validateLiveSettings, type LiveResult, type LiveSettings } from "../lib/live.ts";
import { createLiveRunner, LiveRequestError, parseLiveResult, sendLiveFrame } from "../lib/live-api.ts";

const settings = { ...DEFAULT_LIVE_SETTINGS };
const png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAYCAAAAAC+OKDoAAAAJklEQVR4nGNkYPjPwIgHMzEQAv//48cETWAcdQMYjIYDBAyGcAAAxx2Pub9i5ZMAAAAASUVORK5CYII=";
const blob = new Blob([new Uint8Array([1])], { type: "image/png" });
function fixture(frame_id = "frame-1", settings_used = settings): LiveResult {
  return { frame_id, settings_used, grayscale_image: png, filtered_image: png,
    edge_image: png, spectrum_image: png, processing_time: 12.5 };
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function fakeStream() {
  let stopped = 0;
  return { stream: { getTracks: () => [{ stop: () => stopped++ }, { stop: () => stopped++ }] } as unknown as MediaStream,
    stopped: () => stopped };
}

test("Live validates supported settings and capture dimensions", () => {
  for (const detector of ["Sobel", "Prewitt", "Laplacian"] as const)
    assert.equal(validateLiveSettings({ ...settings, detector }), null);
  for (const patch of [{ detector: "Canny" }, { sigma: NaN }, { sigma: 5.1 }, { kernel_size: 4 },
    { threshold: -1 }, { threshold: Infinity }])
    assert.ok(validateLiveSettings({ ...settings, ...patch } as LiveSettings));
  assert.deepEqual(frameDimensions(640, 480), { width: 256, height: 192 });
  assert.deepEqual(frameDimensions(480, 640), { width: 192, height: 256 });
  assert.deepEqual(frameDimensions(32, 24), { width: 32, height: 24 });
});

test("Live contract requires all four images, matching frame/settings, and finite timing", () => {
  assert.equal(parseLiveResult(fixture(), "frame-1", settings).frame_id, "frame-1");
  for (const patch of [{ frame_id: "old" }, { processing_time: -1 }, { processing_time: NaN },
    { settings_used: { ...settings, threshold: 10 } }, { settings_used: null },
    ...["grayscale_image", "filtered_image", "edge_image", "spectrum_image"].map((key) => ({ [key]: "bad" }))])
    assert.throws(() => parseLiveResult({ ...fixture(), ...patch }, "frame-1", settings), /incompatible/);
});

test("Live adapter sends only the multipart frame contract", async (t) => {
  const signal = new AbortController().signal;
  t.mock.method(globalThis, "fetch", async (url: string, init: RequestInit) => {
    assert.ok(url.endsWith("/live/frame"));
    assert.equal(init.method, "POST");
    assert.equal(init.signal, signal);
    const form = init.body as FormData;
    assert.deepEqual([...form.keys()], ["image", "frame_id", "detector", "sigma", "kernel_size", "threshold"]);
    assert.equal((form.get("image") as File).type, "image/png");
    assert.equal(form.get("frame_id"), "frame-1");
    for (const key of ["detector", "sigma", "kernel_size", "threshold"] as const)
      assert.equal(form.get(key), String(settings[key]));
    return Response.json(fixture());
  });
  assert.deepEqual(await sendLiveFrame(blob, settings, "frame-1", signal), fixture());
});

test("Live adapter reports busy, invalid responses, and network errors", async (t) => {
  const mock = t.mock.method(globalThis, "fetch", async () => Response.json({ detail: "Busy" }, { status: 429 }));
  const send = () => sendLiveFrame(blob, settings, "frame-1", new AbortController().signal);
  await assert.rejects(send(), (error: unknown) => error instanceof LiveRequestError && error.status === 429);
  mock.mock.mockImplementation(async () => new Response("bad json"));
  await assert.rejects(send(), /incompatible/);
  mock.mock.mockImplementation(async () => { throw new TypeError("network"); });
  await assert.rejects(send(), /Cannot reach/);
});

test("Runner drops incoming frames while capture or request is busy", async () => {
  const capture = deferred<Blob>();
  const response = deferred<LiveResult>();
  let sent = 0;
  let displayed = 0;
  const runner = createLiveRunner(async () => { sent++; return response.promise; });
  const callbacks = { success: () => displayed++, error: () => assert.fail("unexpected error") };
  const first = runner.run(() => capture.promise, settings, callbacks);
  const drop = () => runner.run(() => { assert.fail("busy capture should be dropped"); }, settings, callbacks);
  assert.equal(await drop(), false);
  capture.resolve(blob);
  await Promise.resolve();
  assert.equal(sent, 1);
  assert.equal(await drop(), false);
  response.resolve(fixture());
  await first;
  assert.equal(displayed, 1);
});

test("Stop during capture prevents upload and suppresses callbacks", async () => {
  const capture = deferred<Blob>();
  const runner = createLiveRunner(async () => { assert.fail("cancelled capture sent"); });
  const first = runner.run(() => capture.promise, settings, { success: () => assert.fail(), error: () => assert.fail() });
  runner.cancel();
  capture.resolve(blob);
  await first;
});

test("Cancellation holds the request gate and discards late success or failure", async () => {
  for (const fail of [false, true]) {
    const response = deferred<LiveResult>();
    let signal: AbortSignal | undefined;
    const runner = createLiveRunner(async (_blob, _settings, _id, nextSignal) => { signal = nextSignal; return response.promise; });
    const callbacks = { success: () => assert.fail("stale success"), error: () => assert.fail("stale error") };
    const first = runner.run(async () => blob, settings, callbacks);
    await Promise.resolve();
    runner.cancel();
    assert.equal(signal?.aborted, true);
    assert.equal(await runner.run(async () => { assert.fail("overlap"); }, settings, callbacks), false);
    if (fail) response.reject(new Error("late failure")); else response.resolve(fixture());
    await first;
  }
});

test("Runner snapshots controls and can process again after completion", async () => {
  const capture = deferred<Blob>();
  const draft = { ...settings };
  const seen: number[] = [];
  const runner = createLiveRunner(async (_blob, snapshot, id) => { seen.push(snapshot.threshold); return fixture(id, snapshot); });
  const callbacks = { success: () => {}, error: () => assert.fail() };
  const first = runner.run(() => capture.promise, draft, callbacks);
  draft.threshold = 200;
  capture.resolve(blob);
  await first;
  await runner.run(async () => blob, draft, callbacks);
  assert.deepEqual(seen, [96, 200]);
});

test("Runner times out a request and permits retry", async () => {
  const runner = createLiveRunner(async (_blob, snapshot, id, signal) => new Promise<LiveResult>((resolve, reject) => {
    if (snapshot.threshold === 0) resolve(fixture(id, snapshot));
    else signal.addEventListener("abort", () => reject(new Error("abort")), { once: true });
  }), 5);
  let error = "";
  await runner.run(async () => blob, settings, { success: () => assert.fail(), error: (cause) => { error = cause.message; } });
  assert.match(error, /timed out/);
  let success = false;
  await runner.run(async () => blob, { ...settings, threshold: 0 }, { success: () => { success = true; }, error: () => assert.fail() });
  assert.equal(success, true);
});

test("Camera stop releases every track and ignores late permission grants", async () => {
  const acquired = fakeStream();
  const session = createCameraSession(async () => acquired.stream);
  assert.equal(await session.start(), acquired.stream);
  session.stop();
  session.stop();
  assert.equal(acquired.stopped(), 2);
  const pending = deferred<MediaStream>();
  const late = fakeStream();
  const pendingSession = createCameraSession(() => pending.promise);
  const start = pendingSession.start();
  pendingSession.stop();
  pending.resolve(late.stream);
  assert.equal(await start, null);
  assert.equal(late.stopped(), 2);
});

test("Camera restart cannot be replaced by an older permission response", async () => {
  const old = deferred<MediaStream>();
  const oldStream = fakeStream();
  const current = fakeStream();
  let calls = 0;
  const session = createCameraSession(() => ++calls === 1 ? old.promise : Promise.resolve(current.stream));
  const first = session.start();
  assert.equal(await session.start(), current.stream);
  old.resolve(oldStream.stream);
  assert.equal(await first, null);
  assert.equal(oldStream.stopped(), 2);
  assert.equal(current.stopped(), 0);
  session.stop();
  assert.equal(current.stopped(), 2);
});

test("Camera permission and device failures produce actionable messages", async () => {
  assert.match(cameraError({ name: "NotAllowedError" }), /permission/);
  assert.match(cameraError({ name: "NotFoundError" }), /No camera/);
  assert.match(cameraError({ name: "NotReadableError" }), /in use/);
  const pending = deferred<MediaStream>();
  const session = createCameraSession(() => pending.promise);
  const start = session.start();
  session.stop();
  pending.reject(new Error("permission denied after unmount"));
  assert.equal(await start, null);
});
