// Deterministic RGBA photo: an orange rectangle on a blue background, 16×12.
// No test fixture is imported by the UI.
import { createHash } from "node:crypto";
import type { CutoutResult, CutoutSnapshot, PreparedImage } from "../lib/cutout.ts";

export const photoBytes = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAABAAAAAMCAYAAABr5z2BAAAAMElEQVR4nGMUsan4z0ABYKJEM1UMYEEXOCG3Gq8Gi0eh1HUB06gBDBixgB7Kgz8QAQedBlKmx7X9AAAAAElFTkSuQmCC", "base64");
export const photoFile = new File([photoBytes], "colour-fixture.png", { type: "image/png" });
export const rectangle = { x: 2, y: 1, width: 12, height: 10 };

export function preparedFixture(id = "request-1"): PreparedImage {
  return { request_id: id, image_id: createHash("sha256").update(photoBytes).digest("hex"),
    prepared_image: `data:image/png;base64,${photoBytes.toString("base64")}`,
    width: 16, height: 12, source_dimensions: { width: 16, height: 12 }, processing_time: 1 };
}
export function resultFixture(id = "request-1"): CutoutResult {
  return { request_id: id, image_id: preparedFixture().image_id, algorithm: "opencv-grabcut",
    method: "grabcut", parameters_used: {}, guidance_image: null, guidance_scale: null, seed_mode: null,
    width: 16, height: 12, cutout_image: preparedFixture().prepared_image,
    cropped_image: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAGCAYAAAD+Bd/7AAAAFklEQVR4nGO8E6X8nwEPYMInOVgUAADPlAJkygEqHwAAAABJRU5ErkJggg==",
    mask_image: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAMCAAAAABOjGJdAAAAGUlEQVR4nGNgIBkwgoj/SBwmdBXUECADAACZWQEM9TELngAAAABJRU5ErkJggg==",
    foreground_bounds: { x: 4, y: 3, width: 8, height: 6 }, cropped_dimensions: { width: 8, height: 6 }, processing_time: 12 };
}
export function selectionFixture(): CutoutSnapshot {
  return { prepared: preparedFixture(), rectangle: { ...rectangle }, settings: { method: "grabcut" },
    marks: [{ mode: "keep", size: 3, points: [{ x: 6, y: 5 }, { x: 7, y: 5 }] }] };
}
