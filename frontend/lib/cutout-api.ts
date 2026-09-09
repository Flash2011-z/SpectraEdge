import { API_BASE_URL } from "./api.ts";
import { parseCutoutResult, parsePreparedImage, snapshotCutout, validateSelection, validateCutoutSettings,
  type CutoutResult, type CutoutSnapshot, type PreparedImage } from "./cutout.ts";

async function post(path: string, body: FormData, signal: AbortSignal): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL.replace(/\/$/, "")}/cutout/${path}`, { method: "POST", body, signal });
  } catch (error) {
    if (signal.aborted) throw error;
    throw new Error(`Cannot reach the Python backend at ${API_BASE_URL}. Start it and try again.`);
  }
  const data: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data && typeof data === "object" && "detail" in data ? data.detail : null;
    throw new Error(typeof detail === "string" ? detail : `Cutout request failed (HTTP ${response.status}). Try again.`);
  }
  return data;
}

export async function prepareCutout(file: File, requestId: string, signal: AbortSignal): Promise<PreparedImage> {
  const body = new FormData();
  body.append("image", file);
  body.append("request_id", requestId);
  return parsePreparedImage(await post("prepare", body, signal), requestId);
}

export async function extractCutout(input: CutoutSnapshot, requestId: string, signal: AbortSignal): Promise<CutoutResult> {
  const snapshot = snapshotCutout(input);
  const invalid = validateSelection(snapshot.prepared, snapshot.rectangle, snapshot.marks) || validateCutoutSettings(snapshot.settings);
  if (invalid) throw new Error(invalid);
  const body = new FormData();
  // Decode the exact prepared PNG bytes. No canvas re-encoding, original-file
  // resubmission, rescaling, or browser-composited preview enters extraction.
  const bytes = Uint8Array.from(atob(snapshot.prepared.prepared_image.split(",")[1]), (c) => c.charCodeAt(0));
  body.append("image", new Blob([bytes], { type: "image/png" }), "prepared.png");
  body.append("settings", JSON.stringify({ request_id: requestId, image_id: snapshot.prepared.image_id,
    width: snapshot.prepared.width, height: snapshot.prepared.height,
    rectangle: snapshot.rectangle, marks: snapshot.marks, ...snapshot.settings }));
  const seedMode = snapshot.settings.method === "edge-watershed"
    ? snapshot.marks.some((mark) => mark.mode === "keep") ? "brush" : "automatic" : undefined;
  return parseCutoutResult(await post("extract", body, signal), requestId, snapshot.prepared, snapshot.settings, seedMode);
}

interface Callbacks<T> { success: (value: T) => void; error: (message: string) => void }

// One generation spans BOTH endpoints: an old preparation cannot replace a
// newer upload, and an old extraction cannot survive any selection/parameter edit.
export function createCutoutRunner(prepare = prepareCutout, extract = extractCutout) {
  let generation = 0;
  let controller: AbortController | null = null;
  const cancel = () => { generation++; controller?.abort(); controller = null; };
  async function run<T>(action: (id: string, signal: AbortSignal) => Promise<T>, callbacks: Callbacks<T>) {
    cancel();
    const current = generation;
    const active = new AbortController();
    controller = active;
    const timer = setTimeout(() => active.abort(), 120_000);
    try {
      const result = await action(crypto.randomUUID(), active.signal);
      if (current === generation) {
        if (active.signal.aborted) callbacks.error("Cutout processing timed out. Try a smaller image or restart Python.");
        else callbacks.success(result);
      }
    } catch (error) {
      if (current === generation) callbacks.error(active.signal.aborted ? "Cutout processing timed out. Try again."
        : error instanceof Error ? error.message : "Cutout processing failed. Try again.");
    } finally {
      clearTimeout(timer);
      if (current === generation) controller = null;
    }
  }
  return { cancel,
    prepare: (file: File, callbacks: Callbacks<PreparedImage>) => run((id, signal) => prepare(file, id, signal), callbacks),
    extract: (input: CutoutSnapshot, callbacks: Callbacks<CutoutResult>) => {
      const snapshot = snapshotCutout(input);
      return run((id, signal) => extract(snapshot, id, signal), callbacks);
    },
  };
}
