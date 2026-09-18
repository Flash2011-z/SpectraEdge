import { validateAnalysis, validateComparison, type ComputedAnalysisResult, type AnalysisSettings,
  type ComparisonDetectorResult, type ComparisonResult, type ComparisonSettings, type Detector,
  type ObjectMeasurement } from "./workspace.ts";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export function parseAnalysisResult(value: unknown, requestId: string): ComputedAnalysisResult {
  const fail = () => { throw new Error("The backend returned an incompatible analysis response. Check that both services use the current contract."); };
  if (!value || typeof value !== "object") return fail();
  const result = value as ComputedAnalysisResult;
  const png = (url: unknown) => typeof url === "string" && /^data:image\/png;base64,[A-Za-z0-9+/]+=*$/.test(url);
  const dimensions = (size: { width: number; height: number } | undefined, max: number) =>
    size && [size.width, size.height].every((n) => Number.isInteger(n) && n > 0 && n <= max);
  const object = (value: unknown) => {
    if (!value || typeof value !== "object") return false;
    const item = value as ObjectMeasurement;
    const box = item.bounding_box;
    return Number.isInteger(item.id) && item.id > 0 && Number.isInteger(item.area) && item.area > 0 &&
      Number.isInteger(item.perimeter) && item.perimeter > 0 && Array.isArray(item.centroid) &&
      item.centroid.length === 2 && item.centroid.every(Number.isFinite) && box &&
      [box.x, box.y, box.width, box.height].every(Number.isInteger) &&
      box.x >= 0 && box.y >= 0 && box.width > 0 && box.height > 0;
  };
  const objects = () => Array.isArray(result.object_list) && result.object_list.every((item, index) => {
    if (!object(item) || item.id !== index + 1) return false;
    const box = item.bounding_box;
    return box.x + box.width <= result.analyzed_dimensions.width &&
      box.y + box.height <= result.analyzed_dimensions.height &&
      item.centroid[0] >= box.x && item.centroid[0] < box.x + box.width &&
      item.centroid[1] >= box.y && item.centroid[1] < box.y + box.height;
  });
  const multiScale = () => {
    const multi = result.multi_scale;
    if (!multi || !Array.isArray(multi.sigmas) || multi.sigmas.length < 1 || multi.sigmas.length > 5 ||
        !Number.isInteger(multi.support_count) || multi.support_count < 1 ||
        multi.support_count > multi.sigmas.length || !Array.isArray(multi.scales) ||
        multi.scales.length !== multi.sigmas.length || !png(multi.persistence_map) ||
        !png(multi.fused_edge_map) || multi.persistence_scale?.min !== 0 ||
        multi.persistence_scale?.max !== multi.sigmas.length ||
        multi.persistence_scale?.mapping !== "linear_grayscale") return false;
    return multi.scales.every((scale, index) => scale.sigma === multi.sigmas[index] &&
      Number.isInteger(scale.kernel_size) && scale.kernel_size >= 3 && scale.kernel_size <= 31 &&
      scale.kernel_size % 2 === 1 && png(scale.edge_map));
  };
  if (result.provenance !== "computed" || result.request_id !== requestId ||
      ![result.original_image, result.grayscale_image, result.filtered_image,
        result.fft_image, result.filtered_fft_image].every(png) ||
      !dimensions(result.analyzed_dimensions, 512) || !dimensions(result.source_dimensions, 20_000_000) ||
      !result.parameters_used || validateAnalysis(result.parameters_used) ||
      !Number.isFinite(result.processing_time) || result.processing_time < 0 ||
      result.fps !== null ||
      !Array.isArray(result.completed_stages) ||
      result.spectrum_scale?.min !== 0 || !Number.isFinite(result.spectrum_scale?.max) ||
      result.spectrum_scale.max < 0 || result.spectrum_scale.mapping !== "linear_grayscale") return fail();
  const noiseModel = result.parameters_used.noise_model;
  const expectedNoiseUnits = noiseModel === "Gaussian"
    ? "intensity standard deviation"
    : noiseModel === "Salt & Pepper" ? "pixel corruption probability" : null;
  if (expectedNoiseUnits) {
    if (!png(result.noisy_image) || result.noise?.model !== noiseModel ||
        result.noise.units !== expectedNoiseUnits ||
        result.noise.strength !== result.parameters_used.noise_strength ||
        result.noise.seed !== result.parameters_used.noise_seed) return fail();
  } else if (result.noisy_image !== null || result.noise?.model !== "None" ||
      result.noise.strength !== 0 || result.noise.units !== "none" || result.noise.seed !== null) return fail();
  const derivatives = [result.gx, result.gy, result.gradient_magnitude, result.edge_map];
  const detector = result.parameters_used.detector;
  const hasMultiScale = result.parameters_used.multi_scale === true;
  const expectedDecision = detector === "Laplacian" ? "zero_crossing" : "threshold";
  const detectorMetadata = () => {
    const metadata = result.detector_metadata;
    if (!detector) return metadata === null;
    const laplacian = detector === "Laplacian";
    return metadata?.detector === detector &&
      metadata.decision === (laplacian ? "zero_crossing" : "magnitude_threshold") &&
      metadata.threshold_type === (laplacian ? "zero_crossing_contrast" : "gradient_magnitude") &&
      metadata.threshold_label === (laplacian
        ? "Zero-crossing contrast threshold" : "Gradient magnitude threshold") &&
      metadata.threshold === result.parameters_used.threshold &&
      metadata.threshold_units === (laplacian ? "raw_response_difference" : "raw_gradient_magnitude") &&
      metadata.minimum_component_area === (laplacian
        ? result.parameters_used.laplacian_min_component_area ?? 2 : null);
  };
  const noiseStage = noiseModel ? ["noise"] : [];
  const expectedStages = detector
    ? ["input", "grayscale", ...noiseStage, "smooth", detector.toLowerCase(), expectedDecision,
      ...(hasMultiScale ? ["multi_scale"] : []), "contours", "objects", "fourier"].join(",")
    : ["input", "grayscale", ...noiseStage, "smooth", "fourier"].join(",");
  if (!detectorMetadata() || (hasMultiScale ? !multiScale() : result.multi_scale !== null)) return fail();
  if (detector === "Sobel" || detector === "Prewitt") {
    if (result.detection_status !== "edges_computed" || !derivatives.every(png) ||
        result.laplacian_response != null || !png(result.contour_image) ||
        !objects() ||
        result.completed_stages.join(",") !== expectedStages) return fail();
  } else if (detector === "Laplacian") {
    if (result.detection_status !== "edges_computed" || !png(result.laplacian_response) || !png(result.edge_map) ||
        ![result.gx, result.gy, result.gradient_magnitude].every((field) => field === null) ||
        !png(result.contour_image) || !objects() ||
        result.completed_stages.join(",") !== expectedStages) return fail();
  } else if (result.detection_status !== "not_run" || !derivatives.every((field) => field === null) ||
      result.laplacian_response != null || result.contour_image !== null || result.object_list !== null ||
      result.completed_stages.join(",") !== expectedStages) return fail();
  return result;
}

export async function analyzeImage(
  file: File, settings: AnalysisSettings, requestId: string, signal: AbortSignal,
): Promise<ComputedAnalysisResult> {
  const invalid = validateAnalysis(settings);
  if (invalid) throw new Error(invalid);
  const body = new FormData();
  body.append("image", file);
  body.append("sigma", String(settings.sigma));
  body.append("kernel_size", String(settings.kernel_size));
  body.append("request_id", requestId);
  if (settings.detector !== undefined) {
    body.append("detector", settings.detector);
    body.append("threshold", String(settings.threshold));
    if (settings.detector === "Laplacian" && settings.laplacian_min_component_area !== undefined)
      body.append("laplacian_min_component_area", String(settings.laplacian_min_component_area));
    if (settings.multi_scale) {
      body.append("multi_scale", "true");
      body.append("scale_sigmas", settings.scale_sigmas!.join(","));
      body.append("scale_support", String(settings.scale_support));
    }
  }
  if (settings.noise_model !== undefined) {
    body.append("noise_model", settings.noise_model);
    body.append("noise_strength", String(settings.noise_strength));
    body.append("noise_seed", String(settings.noise_seed));
  }
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
  if (result.parameters_used.sigma !== settings.sigma || result.parameters_used.kernel_size !== settings.kernel_size ||
      result.parameters_used.detector !== settings.detector || result.parameters_used.threshold !== settings.threshold ||
      result.parameters_used.multi_scale !== settings.multi_scale ||
      result.parameters_used.scale_support !== settings.scale_support ||
      JSON.stringify(result.parameters_used.scale_sigmas) !== JSON.stringify(settings.scale_sigmas) ||
      result.parameters_used.noise_model !== settings.noise_model ||
      result.parameters_used.noise_strength !== settings.noise_strength ||
      result.parameters_used.noise_seed !== settings.noise_seed ||
      result.parameters_used.laplacian_min_component_area !== settings.laplacian_min_component_area)
    throw new Error("The backend used different settings. Results were discarded; please retry.");
  return result;
}

export function parseComparisonResult(value: unknown, requestId: string): ComparisonResult {
  const fail = () => { throw new Error("The backend returned an incompatible comparison response. Check that both services use the current contract."); };
  if (!value || typeof value !== "object") return fail();
  const result = value as ComparisonResult;
  const png = (url: unknown) => typeof url === "string" && /^data:image\/png;base64,[A-Za-z0-9+/]+=*$/.test(url);
  const dimensions = (size: { width: number; height: number } | undefined, max: number) =>
    size && [size.width, size.height].every((n) => Number.isInteger(n) && n > 0 && n <= max);
  const validObject = (value: unknown) => {
    if (!value || typeof value !== "object") return false;
    const item = value as ObjectMeasurement;
    const box = item.bounding_box;
    return Number.isInteger(item.id) && item.id > 0 && Number.isInteger(item.area) && item.area > 0 &&
      Number.isInteger(item.perimeter) && item.perimeter > 0 && Array.isArray(item.centroid) &&
      item.centroid.length === 2 && item.centroid.every(Number.isFinite) && box &&
      [box.x, box.y, box.width, box.height].every(Number.isInteger) && box.x >= 0 && box.y >= 0 &&
      box.width > 0 && box.height > 0;
  };
  const detector = (entry: ComparisonDetectorResult, name: Detector) => {
    const laplacian = name === "Laplacian";
    const threshold = name === "Sobel" ? result.parameters_used.sobel_threshold
      : name === "Prewitt" ? result.parameters_used.prewitt_threshold
      : result.parameters_used.laplacian_contrast_threshold;
    return entry && png(entry.edge_map) && Array.isArray(entry.object_list) &&
      entry.object_list.every((object, index) => validObject(object) && object.id === index + 1) &&
      Number.isInteger(entry.edge_pixel_count) && entry.edge_pixel_count >= 0 &&
      Number.isInteger(entry.object_count) && entry.object_count === entry.object_list.length &&
      Number.isFinite(entry.average_object_area) && entry.average_object_area >= 0 &&
      entry.average_object_area === (entry.object_count
        ? entry.object_list.reduce((sum, object) => sum + object.area, 0) / entry.object_count : 0) &&
      Number.isFinite(entry.processing_time) && entry.processing_time >= 0 &&
      entry.metadata?.detector === name && entry.metadata.threshold === threshold &&
      entry.metadata.decision === (laplacian ? "zero_crossing" : "magnitude_threshold") &&
      entry.metadata.threshold_type === (laplacian ? "zero_crossing_contrast" : "gradient_magnitude") &&
      entry.metadata.threshold_label === (laplacian
        ? "Zero-crossing contrast threshold" : "Gradient magnitude threshold") &&
      entry.metadata.threshold_units === (laplacian ? "raw_response_difference" : "raw_gradient_magnitude") &&
      entry.metadata.minimum_component_area === (laplacian ? 2 : null);
  };
  if (result.provenance !== "computed" || result.request_id !== requestId ||
      validateComparison(result.parameters_used) ||
      !dimensions(result.source_dimensions, 20_000_000) || !dimensions(result.analyzed_dimensions, 512) ||
      !Number.isFinite(result.processing_time) || result.processing_time < 0 ||
      !detector(result.sobel, "Sobel") || !detector(result.prewitt, "Prewitt") ||
      !detector(result.laplacian, "Laplacian")) return fail();
  return result;
}

export async function compareImage(
  file: File, settings: ComparisonSettings, requestId: string, signal: AbortSignal,
): Promise<ComparisonResult> {
  const invalid = validateComparison(settings);
  if (invalid) throw new Error(invalid);
  const body = new FormData();
  body.append("image", file);
  body.append("sigma", String(settings.sigma));
  body.append("kernel_size", String(settings.kernel_size));
  body.append("sobel_threshold", String(settings.sobel_threshold));
  body.append("prewitt_threshold", String(settings.prewitt_threshold));
  body.append("laplacian_contrast_threshold", String(settings.laplacian_contrast_threshold));
  body.append("request_id", requestId);
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL.replace(/\/$/, "")}/compare`, { method: "POST", body, signal });
  } catch (error) {
    if (signal.aborted) throw error;
    throw new Error(`Cannot reach the Python backend at ${API_BASE_URL}. Start it and try Compare again.`);
  }
  const data: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data && typeof data === "object" && "detail" in data ? data.detail : null;
    throw new Error(typeof detail === "string" ? detail : `Comparison failed (HTTP ${response.status}). Check the backend terminal and retry.`);
  }
  const result = parseComparisonResult(data, requestId);
  if (result.parameters_used.sigma !== settings.sigma ||
      result.parameters_used.kernel_size !== settings.kernel_size ||
      result.parameters_used.sobel_threshold !== settings.sobel_threshold ||
      result.parameters_used.prewitt_threshold !== settings.prewitt_threshold ||
      result.parameters_used.laplacian_contrast_threshold !== settings.laplacian_contrast_threshold)
    throw new Error("The backend used different comparison settings. Results were discarded; please retry.");
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
    async run(file: File, settings: AnalysisSettings, callbacks: {
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

export function createComparisonRunner(send = compareImage) {
  let generation = 0;
  let controller: AbortController | null = null;
  const cancel = () => { generation++; controller?.abort(); controller = null; };
  return {
    cancel,
    async run(file: File, settings: ComparisonSettings, callbacks: {
      success: (result: ComparisonResult) => void;
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
        else if (current === generation) callbacks.error("Comparison timed out. Try a smaller image or restart the backend.");
      } catch (error) {
        if (current === generation) callbacks.error(requestController.signal.aborted
          ? "Comparison timed out. Try a smaller image or restart the backend."
          : error instanceof Error ? error.message : "Comparison failed. Please retry.");
      } finally {
        clearTimeout(timeout);
        if (current === generation) controller = null;
      }
    },
  };
}
