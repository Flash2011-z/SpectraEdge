import { API_BASE_URL } from "./api.ts";
import { LIVE_IMAGES, sameLiveSettings, validateLiveSettings, type LiveResult, type LiveSettings } from "./live.ts";

export class LiveRequestError extends Error {
  status: number;
  constructor(message: string, status: number) { super(message); this.status = status; }
}

export function parseLiveResult(value: unknown, frameId: string, settings: LiveSettings): LiveResult {
  const fail = () => { throw new Error("The backend returned an incompatible live frame."); };
  if (!value || typeof value !== "object") return fail();
  const result = value as LiveResult;
  if (result.frame_id !== frameId || !Number.isFinite(result.processing_time) || result.processing_time < 0 ||
      !result.settings_used || validateLiveSettings(result.settings_used) || !sameLiveSettings(result.settings_used, settings) ||
      !LIVE_IMAGES.every((key) => typeof result[key] === "string" &&
        /^data:image\/png;base64,[A-Za-z0-9+/]+=*$/.test(result[key]))) return fail();
  return result;
}

export async function sendLiveFrame(frame: Blob, settings: LiveSettings, frameId: string, signal: AbortSignal) {
  const invalid = validateLiveSettings(settings);
  if (invalid) throw new Error(invalid);
  const body = new FormData();
  body.append("image", frame, "frame.png");
  body.append("frame_id", frameId);
  for (const key of ["detector", "sigma", "kernel_size", "threshold"] as const) body.append(key, String(settings[key]));
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL.replace(/\/$/, "")}/live/frame`, { method: "POST", body, signal });
  } catch (error) {
    if (signal.aborted) throw error;
    throw new Error(`Cannot reach the Python backend at ${API_BASE_URL}. Check that it is running.`);
  }
  const data: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data && typeof data === "object" && "detail" in data ? data.detail : null;
    throw new LiveRequestError(typeof detail === "string" ? detail : `Live processing failed (HTTP ${response.status}).`, response.status);
  }
  return parseLiveResult(data, frameId, settings);
}

// The busy gate covers capture, transport, and presentation preparation. No queue.
// Cancelling invalidates results immediately, but retains the gate until settled.
export function createLiveRunner(send = sendLiveFrame, timeoutMs = 15_000) {
  let generation = 0;
  let busy = false;
  let controller: AbortController | null = null;
  return {
    cancel() { generation++; controller?.abort(); },
    async run(capture: () => Promise<Blob | null>, settings: LiveSettings, callbacks: {
      success: (result: LiveResult) => void;
      error: (error: Error) => void;
    }) {
      if (busy) return false;
      busy = true;
      const current = generation;
      const snapshot = { ...settings };
      const request = new AbortController();
      controller = request;
      const timeout = setTimeout(() => request.abort(), timeoutMs);
      try {
        const frame = await capture();
        if (!frame || current !== generation) return false;
        if (request.signal.aborted) throw new Error("Live frame timed out.");
        const result = await send(frame, snapshot, crypto.randomUUID(), request.signal);
        if (current === generation && !request.signal.aborted) callbacks.success(result);
        else if (current === generation) callbacks.error(new Error("Live frame timed out."));
      } catch (error) {
        if (current === generation) callbacks.error(request.signal.aborted ? new Error("Live frame timed out. Check the backend and retry.")
          : error instanceof Error ? error : new Error("Live processing failed."));
      } finally {
        clearTimeout(timeout);
        controller = null;
        busy = false;
      }
      return true;
    },
  };
}
