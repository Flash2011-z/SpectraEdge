import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_PARAMETERS, restoreParameters, validateImage } from "../lib/workspace.ts";

test("corrupt stored preferences safely restore defaults", () => {
  assert.deepEqual(restoreParameters(null), DEFAULT_PARAMETERS);
  assert.deepEqual(restoreParameters("broken"), DEFAULT_PARAMETERS);
  assert.deepEqual(
    restoreParameters({ detector: "Canny", kernel: 99, threshold: NaN }),
    DEFAULT_PARAMETERS,
  );
});
test("restored numeric settings stay within supported bounds and steps", () => {
  const restored = restoreParameters({
    sigma: 99,
    threshold: -2,
    laplacianMinimumComponentArea: 476,
    noiseStrength: Infinity,
  });
  assert.equal(restored.sigma, 5);
  assert.equal(restored.threshold, 0);
  assert.equal(restored.laplacianMinimumComponentArea, 476);
  assert.equal(restoreParameters({ laplacianMinimumComponentArea: 0 }).laplacianMinimumComponentArea, 1);
  assert.equal(restored.noiseStrength, 12);
  assert.equal(restoreParameters({ noise: "Salt & Pepper", noiseStrength: 0.12 }).noiseStrength, 0.12);
  assert.equal(restoreParameters({ threshold: 2000 }).threshold, 1443);
  assert.equal(restoreParameters({ threshold: 900 }).threshold, 900);
  assert.deepEqual(restoreParameters({ multiScale: true, scaleSigmas: [0.8, 1.6, 3.2], scaleSupport: 3 }).scaleSigmas,
    [0.8, 1.6, 3.2]);
  assert.equal(restoreParameters({ multiScale: true, scaleSigmas: [1, 1, 2], scaleSupport: 9 }).scaleSupport, 2);
});
test("image validation rejects empty, unsupported, and oversized files", () => {
  assert.match(validateImage({ type: "image/svg+xml", size: 100 })!, /PNG/);
  assert.match(validateImage({ type: "image/png", size: 0 })!, /empty/);
  assert.match(validateImage({ type: "image/jpeg", size: 20 * 1024 * 1024 + 1 })!, /20 MB/);
  for (const type of ["image/png", "image/jpeg", "image/webp", "image/gif"]) {
    assert.equal(validateImage({ type, size: 100 }), null);
  }
});
