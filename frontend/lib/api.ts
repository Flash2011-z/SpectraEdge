import { validateGaussian, type ComputedAnalysisResult, type GaussianSettings } from "./workspace.ts";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export function parseAnalysisResult(value: unknown, requestId: string): ComputedAnalysisResult {
  const fail = () => { throw new Error("The backend returned an incompatible analysis response. Check that both services use the current contract."); };
  if (!value || typeof value !== "object") return fail();
  const result = value as ComputedAnalysisResult;
  const png = (url: unknown) => typeof url === "string" && /^data:image\/png;base64,[A-Za-z0-9+/]+=*$/.test(url);
  const dimensions = (size: { width: number; height: number } | undefined, max: number) =>
    size && [size.width, size.height].every((n) => Number.isInteger(n) && n > 0 && n <= max);
  if (result.provenance !== "computed" || result.request_id !== requestId ||
      ![result.original_image, result.grayscale_image, result.filtered_image,
        result.fft_image, result.filtered_fft_image].every(png) ||
      !dimensions(result.analyzed_dimensions, 512) || !dimensions(result.source_dimensions, 20_000_000) ||
      !result.parameters_used || validateGaussian(result.parameters_used) ||
      !Number.isFinite(result.processing_time) || result.processing_time < 0 ||
      result.detection_status !== "not_run" ||
      ![result.gx, result.gy, result.edge_map, result.gradient_magnitude,
        result.contour_image, result.object_list, result.fps].every((field) => field === null) ||
      !Array.isArray(result.completed_stages) ||
      result.completed_stages.join(",") !== "input,grayscale,smooth,fourier" ||
      result.spectrum_scale?.min !== 0 || !Number.isFinite(result.spectrum_scale?.max) ||
      result.spectrum_scale.max < 0 || result.spectrum_scale.mapping !== "linear_grayscale") return fail();
  return result;
}

export async function analyzeImage(
  file: File, settings: GaussianSettings, requestId: string, signal: AbortSignal,
): Promise<ComputedAnalysisResult> {
  const invalid = validateGaussian(settings);
  if (invalid) throw new Error(invalid);
  const body = new FormData();
  body.append("image", file);
  body.append("sigma", String(settings.sigma));
  body.append("kernel_size", String(settings.kernel_size));
  body.append("request_id", requestId);
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL.replace(/\/$/, "")}/analyze`, { method: "POST", body, signal });
  } catch (error) {
    if (signal.aborted) throw error;
    throw new Error(`Cannot reach the Python backend at ${API_BASE_URL}. Start it and try Process again.`);
  }
  const data: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data && typeof data === "object" && "detail" in data ? data.detail : null;
    throw new Error(typeof detail === "string" ? detail : `Analysis failed (HTTP ${response.status}). Check the backend terminal and retry.`);
  }
  const result = parseAnalysisResult(data, requestId);
  if (result.parameters_used.sigma !== settings.sigma || result.parameters_used.kernel_size !== settings.kernel_size)
    throw new Error("The backend used different settings. Results were discarded; please retry.");
  return result;
}

// Cancellation alone is insufficient: a response can already be in flight.
// This controller also gates every callback by generation, including errors.
export function createAnalysisRunner(send = analyzeImage) {
  let generation = 0;
  let controller: AbortController | null = null;
  const cancel = () => { generation++; controller?.abort(); controller = null; };
  return {
    cancel,
    async run(file: File, settings: GaussianSettings, callbacks: {
      success: (result: ComputedAnalysisResult) => void;
      error: (message: string) => void;
    }) {
      cancel();
      const current = generation;
      const requestController = new AbortController();
      controller = requestController;
      const snapshot = { ...settings };
      const timeout = setTimeout(() => requestController.abort(), 120_000);
      try {
        const result = await send(file, snapshot, crypto.randomUUID(), requestController.signal);
        if (current === generation && !requestController.signal.aborted) callbacks.success(result);
        else if (current === generation) callbacks.error("Analysis timed out. Try a smaller image or restart the backend.");
      } catch (error) {
        if (current === generation) callbacks.error(requestController.signal.aborted
          ? "Analysis timed out. Try a smaller image or restart the backend."
          : error instanceof Error ? error.message : "Analysis failed. Please retry.");
      } finally {
        clearTimeout(timeout);
        if (current === generation) controller = null;
      }
    },
  };
}
