// Start the Python backend before running test:live-integration.
import test from "node:test";
import assert from "node:assert/strict";
import { sendLiveFrame } from "../lib/live-api.ts";
import { DEFAULT_LIVE_SETTINGS, LIVE_IMAGES } from "../lib/live.ts";

const png = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAYCAAAAAC+OKDoAAAAJklEQVR4nGNkYPjPwIgHMzEQAv//48cETWAcdQMYjIYDBAyGcAAAxx2Pub9i5ZMAAAAASUVORK5CYII=";
for (const detector of ["Sobel", "Prewitt", "Laplacian"] as const) {
  test(`Live ${detector} adapter works against the Python endpoint`, async () => {
    const frameId = crypto.randomUUID();
    const result = await sendLiveFrame(new Blob([Buffer.from(png, "base64")], { type: "image/png" }),
      { ...DEFAULT_LIVE_SETTINGS, detector }, frameId, AbortSignal.timeout(15_000));
    assert.equal(result.frame_id, frameId);
    assert.equal(result.settings_used.detector, detector);
    for (const key of LIVE_IMAGES) {
      const bytes = Buffer.from(result[key].split(",")[1], "base64");
      assert.equal(bytes.subarray(1, 4).toString(), "PNG");
      assert.equal(bytes.readUInt32BE(16), 32);
      assert.equal(bytes.readUInt32BE(20), 24);
    }
    assert.ok(result.processing_time > 0);
  });
}
