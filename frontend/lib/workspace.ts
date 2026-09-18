export type Detector = "Sobel" | "Prewitt" | "Laplacian";
export type View = "analyze" | "compare" | "live";
export type Visualization =
  | "original"
  | "grayscale"
  | "noisy"
  | "filtered"
  | "edges"
  | "gradient"
  | "gx"
  | "gy"
  | "laplacian"
  | "contours"
  | "spectrum"
  | "filtered-spectrum";
export interface Parameters {
  detector: Detector;
  sigma: number;
  kernel: number;
  threshold: number;
  comparisonSobelThreshold: number;
  comparisonPrewittThreshold: number;
  comparisonLaplacianContrastThreshold: number;
  multiScale: boolean;
  scaleSigmas: number[];
  scaleSupport: number;
  laplacianMinimumComponentArea: number;
  noise: "None" | "Gaussian" | "Salt & Pepper";
  noiseStrength: number;
}
export interface Preferences {
  grid: boolean;
  demoVisuals: boolean;
  reducedMotion: boolean;
}
export interface SourceImage {
  kind: "demo" | "image" | "empty";
  name: string;
  url: string;
  width: number;
  height: number;
}
export interface ObjectMeasurement {
  id: number;
  area: number;
  perimeter: number;
  centroid: [number, number];
  bounding_box: { x: number; y: number; width: number; height: number };
}
// PNG data URLs are display artifacts; numerical arrays stay in Python.
export interface AnalysisResult {
  provenance: "demo" | "computed";
  original_image: string;
  noisy_image?: string | null;
  filtered_image: string | null;
  gx: string | null;
  gy: string | null;
  gradient_magnitude: string | null;
  laplacian_response?: string | null;
  edge_map: string | null;
  contour_image: string | null;
  fft_image: string | null;
  object_list: ObjectMeasurement[] | null;
  processing_time: number | null;
  fps: number | null;
}
export interface GaussianSettings {
  sigma: number;
  kernel_size: number;
}
export interface AnalysisSettings extends GaussianSettings {
  detector?: Detector;
  threshold?: number;
  multi_scale?: boolean;
  scale_sigmas?: number[];
  scale_support?: number;
  noise_model?: "Gaussian" | "Salt & Pepper";
  noise_strength?: number;
  noise_seed?: number;
  laplacian_min_component_area?: number;
}
export interface DetectorMetadata {
  detector: Detector;
  decision: "magnitude_threshold" | "zero_crossing";
  threshold_type: "gradient_magnitude" | "zero_crossing_contrast";
  threshold_label: "Gradient magnitude threshold" | "Zero-crossing contrast threshold";
  threshold: number;
  threshold_units: "raw_gradient_magnitude" | "raw_response_difference";
  minimum_component_area: number | null;
}
export type NoiseMetadata = {
  model: "None" | "Gaussian" | "Salt & Pepper";
  strength: number;
  units: "none" | "intensity standard deviation" | "pixel corruption probability";
  seed: number | null;
};
export interface ComparisonSettings extends GaussianSettings {
  sobel_threshold: number;
  prewitt_threshold: number;
  laplacian_contrast_threshold: number;
}
export interface ComparisonDetectorResult {
  edge_map: string;
  object_list: ObjectMeasurement[];
  edge_pixel_count: number;
  object_count: number;
  average_object_area: number;
  processing_time: number;
  metadata: DetectorMetadata;
}
export interface ComparisonResult {
  provenance: "computed";
  request_id: string;
  parameters_used: ComparisonSettings;
  source_dimensions: { width: number; height: number };
  analyzed_dimensions: { width: number; height: number };
  sobel: ComparisonDetectorResult;
  prewitt: ComparisonDetectorResult;
  laplacian: ComparisonDetectorResult;
  processing_time: number;
}
export interface MultiScaleAnalysis {
  sigmas: number[];
  support_count: number;
  scales: { sigma: number; kernel_size: number; edge_map: string }[];
  persistence_map: string;
  persistence_scale: { min: 0; max: number; mapping: "linear_grayscale" };
  fused_edge_map: string;
}
export interface ComputedAnalysisResult extends AnalysisResult {
  provenance: "computed";
  request_id: string;
  grayscale_image: string;
  filtered_image: string;
  fft_image: string;
  filtered_fft_image: string;
  parameters_used: AnalysisSettings;
  source_dimensions: { width: number; height: number };
  analyzed_dimensions: { width: number; height: number };
  completed_stages: string[];
  detection_status: "not_run" | "edges_computed";
  detector_metadata: DetectorMetadata | null;
  multi_scale: MultiScaleAnalysis | null;
  noisy_image: string | null;
  noise: NoiseMetadata;
  spectrum_scale: { min: number; max: number; mapping: "linear_grayscale" };
  processing_time: number;
}
export const KERNEL_SIZES = Array.from({ length: 15 }, (_, index) => 3 + index * 2);
export const MAX_THRESHOLD = 1443;
export const MAX_DECODED_PIXELS = 20_000_000;
export const DEFAULT_NOISE_SEED = 220;
export const EMPTY_SOURCE: SourceImage = {
  kind: "empty", name: "No image selected", url: "", width: 0, height: 0,
};
export const DEFAULT_PARAMETERS: Parameters = {
  detector: "Sobel",
  sigma: 1.2,
  kernel: 5,
  threshold: 96,
  comparisonSobelThreshold: 96,
  comparisonPrewittThreshold: 96,
  comparisonLaplacianContrastThreshold: 20,
  multiScale: false,
  scaleSigmas: [0.8, 1.6, 3.2],
  scaleSupport: 2,
  laplacianMinimumComponentArea: 2,
  noise: "None",
  noiseStrength: 12,
};
export const DEFAULT_PREFERENCES: Preferences = {
  grid: true,
  demoVisuals: true,
  reducedMotion: false,
};
export const DEMO_SOURCE: SourceImage = {
  kind: "demo",
  name: "Geometric calibration",
  url: "",
  width: 1920,
  height: 1080,
};
export const DEMO_OBJECTS: ObjectMeasurement[] = [
  { id: 1, area: 18432, perimeter: 544, centroid: [355, 480], bounding_box: { x: 263, y: 388, width: 184, height: 184 } },
  { id: 2, area: 22316, perimeter: 530, centroid: [770, 480], bounding_box: { x: 686, y: 396, width: 169, height: 169 } },
  { id: 3, area: 12430, perimeter: 512, centroid: [1180, 480], bounding_box: { x: 1096, y: 370, width: 168, height: 221 } },
  { id: 4, area: 9612, perimeter: 694, centroid: [1600, 480], bounding_box: { x: 1525, y: 405, width: 150, height: 150 } },
];
export const STAGES = [
  {
    name: "Input",
    detail: "Source image",
    description: "Inspect the original image before any transformation.",
    view: "original",
  },
  {
    name: "Smooth",
    detail: "Noise reduction",
    description: "Gaussian smoothing will reduce high-frequency noise before detection.",
    view: "filtered",
  },
  {
    name: "Gradient",
    detail: "Spatial derivative",
    description: "The selected operator will measure changes in image intensity.",
    view: "gradient",
  },
  {
    name: "Threshold",
    detail: "Edge selection",
    description: "A threshold will select significant intensity changes.",
    view: "edges",
  },
  {
    name: "Edges",
    detail: "Edge map",
    description: "The binary map will isolate the detected image boundaries.",
    view: "edges",
  },
  {
    name: "Objects",
    detail: "Contour analysis",
    description: "Connected contours will supply object measurements.",
    view: "contours",
  },
] as const;
export const ANALYSIS_STAGES = [
  { name: "Input", detail: "Bounded preview", description: "Orient and fit the source within 512 pixels per side; never upscale.", view: "original", key: "input" },
  { name: "Grayscale", detail: "Intensity signal", description: "Convert the analyzed image to grayscale intensity values.", view: "grayscale", key: "grayscale" },
  { name: "Smooth", detail: "Manual convolution", description: "Apply the selected Gaussian kernel with reflection boundaries; sigma zero bypasses smoothing.", view: "filtered", key: "smooth" },
  { name: "Sobel", detail: "Gradient magnitude", description: "True convolution produces signed Gx and Gy; their Euclidean length measures the strength of intensity changes.", view: "gradient", key: "sobel" },
  { name: "Threshold", detail: "Binary edges", description: "Select raw Sobel magnitude strictly greater than the threshold. White pixels are edges, not identified objects.", view: "edges", key: "threshold" },
  { name: "Fourier", detail: "Manual FFT", description: "Compare original and smoothed log-magnitude spectra on one shared display scale.", view: "spectrum", key: "fourier" },
  { name: "Objects", detail: "Connected regions", description: "Measure each eight-connected foreground region and preserve outer and hole boundaries.", view: "contours", key: "objects" },
] as const;

export function analysisStages(
  detector: Detector,
  multiScale = false,
  noiseModel: Parameters["noise"] = "None",
): {
  name: string; detail: string; description: string; view: Visualization; key: string;
}[] {
  const stages: { name: string; detail: string; description: string; view: Visualization; key: string }[] = ANALYSIS_STAGES.map((stage) => {
    if (stage.key === "sobel") return { ...stage, name: detector, key: detector.toLowerCase(),
      view: detector === "Laplacian" ? "laplacian" as const : "gradient" as const,
      detail: detector === "Laplacian" ? "Signed response" : "Gradient magnitude",
      description: detector === "Laplacian" ? "Four-neighbour second derivative: signed neighbour sum minus four times the centre." : stage.description };
    if (stage.key === "threshold") return { ...stage,
      name: detector === "Laplacian" ? "Zero crossings" : "Threshold",
      key: detector === "Laplacian" ? "zero_crossing" : "threshold",
      description: thresholdRule(detector) };
    return stage;
  });
  if (noiseModel !== "None") stages.splice(2, 0, {
    name: "Noise", detail: noiseModel,
    description: noiseModel === "Gaussian"
      ? "Add seeded zero-mean Gaussian noise with the selected intensity standard deviation."
      : "Replace seeded random pixels with black or white at the selected corruption probability.",
    view: "noisy", key: "noise",
  });
  if (multiScale) stages.splice(stages.findIndex((stage) => stage.key === "fourier"), 0, {
    name: "Multi-scale", detail: "Edge persistence",
    description: "Count aligned edge decisions across independent Gaussian scales and retain the configured support.",
    view: "edges", key: "multi_scale",
  });
  return stages;
}

export function thresholdRule(detector: Detector): string {
  return detector === "Laplacian"
    ? "Opposite-sign horizontal/vertical neighbours with raw response difference > threshold"
    : `raw ${detector} magnitude > threshold`;
}

export function thresholdLabel(detector: Detector): DetectorMetadata["threshold_label"] {
  return detector === "Laplacian"
    ? "Zero-crossing contrast threshold"
    : "Gradient magnitude threshold";
}

export function detectorDisplay(detector: Detector) {
  const limit = detector === "Prewitt" ? 765 : 1020;
  return { signed_min: -limit, signed_max: limit,
    magnitude_min: detector === "Laplacian" ? null : 0,
    magnitude_max: detector === "Laplacian" ? null : limit * Math.sqrt(2),
    mapping: "linear_grayscale" as const };
}

export function detectorLabel(kind: Visualization, detector: Detector): string | null {
  if (kind === "laplacian") return "Laplacian signed response";
  if (kind === "gx") return `${detector} Gx · horizontal`;
  if (kind === "gy") return `${detector} Gy · vertical`;
  if (kind === "gradient" && detector !== "Laplacian") return `${detector} gradient magnitude`;
  if (kind === "edges" && detector === "Laplacian") return "Zero-crossing edge map";
  return null;
}

export function validateGaussian(settings: GaussianSettings): string | null {
  if (!Number.isFinite(settings.sigma) || settings.sigma < 0 || settings.sigma > 5 ||
      Math.abs(settings.sigma * 10 - Math.round(settings.sigma * 10)) > 1e-8)
    return "Gaussian sigma must be from 0 to 5 in steps of 0.1.";
  if (!KERNEL_SIZES.includes(settings.kernel_size))
    return "Choose an odd kernel size from 3 through 31.";
  return null;
}

export function validateAnalysis(settings: AnalysisSettings): string | null {
  const gaussianError = validateGaussian(settings);
  if (gaussianError) return gaussianError;
  const noiseFields = settings.noise_strength !== undefined || settings.noise_seed !== undefined;
  if (settings.noise_model === undefined) {
    if (noiseFields) return "Noise strength and seed require a noise model.";
  } else {
    if (!["Gaussian", "Salt & Pepper"].includes(settings.noise_model)) return "Unsupported noise model.";
    const maximum = settings.noise_model === "Gaussian" ? 100 : 1;
    if (typeof settings.noise_strength !== "number" || !Number.isFinite(settings.noise_strength) ||
        settings.noise_strength < 0 || settings.noise_strength > maximum)
      return settings.noise_model === "Gaussian"
        ? "Gaussian strength must be from 0 to 100 intensity standard deviation units."
        : "Salt & Pepper strength must be a pixel corruption probability from 0 to 1.";
    if (!Number.isInteger(settings.noise_seed) || settings.noise_seed! < 0 || settings.noise_seed! > 4294967295)
      return "Noise seed must be an integer from 0 through 4294967295.";
  }
  if (settings.detector === undefined)
    return settings.threshold === undefined && settings.multi_scale === undefined &&
      settings.scale_sigmas === undefined && settings.scale_support === undefined &&
      settings.laplacian_min_component_area === undefined
      ? null : "Threshold requires a detector; multi-scale settings require one too.";
  if (!["Sobel", "Prewitt", "Laplacian"].includes(settings.detector)) return "Unsupported detector.";
  if (typeof settings.threshold !== "number" || !Number.isFinite(settings.threshold) ||
      settings.threshold < 0 || settings.threshold > MAX_THRESHOLD)
    return `${settings.detector} threshold must be a finite number from 0 to ${MAX_THRESHOLD} in raw ${settings.detector === "Laplacian" ? "response difference" : "magnitude"} units.`;
  if (settings.detector === "Laplacian") {
    if (settings.laplacian_min_component_area !== undefined &&
        (!Number.isInteger(settings.laplacian_min_component_area) ||
         settings.laplacian_min_component_area < 1 || settings.laplacian_min_component_area > 10000))
      return "Laplacian minimum component area must be an integer from 1 to 10000 pixels.";
  } else if (settings.laplacian_min_component_area !== undefined) {
    return "Laplacian minimum component area requires the Laplacian detector.";
  }
  if (settings.multi_scale === true) {
    const sigmas = settings.scale_sigmas;
    if (!Array.isArray(sigmas) || sigmas.length < 1 || sigmas.length > 5 ||
        sigmas.some((sigma) => !Number.isFinite(sigma) || sigma < 0 || sigma > 5 ||
          Math.abs(sigma * 10 - Math.round(sigma * 10)) > 1e-8) ||
        new Set(sigmas).size !== sigmas.length)
      return "Multi-scale sigma values must be 1–5 unique numbers from 0 to 5 in steps of 0.1.";
    if (!Number.isInteger(settings.scale_support) || settings.scale_support! < 1 ||
        settings.scale_support! > sigmas.length)
      return "Multi-scale support must be an integer from 1 through the number of scales.";
  } else if (settings.multi_scale !== undefined || settings.scale_sigmas !== undefined || settings.scale_support !== undefined) {
    return "Scale sigma and support settings require multi_scale=true.";
  }
  return null;
}

// Only active numerical settings cross the API boundary; future controls do not.
export function analysisSettings(parameters: Parameters): AnalysisSettings {
  return { sigma: parameters.sigma, kernel_size: parameters.kernel,
    detector: parameters.detector, threshold: parameters.threshold,
    ...(parameters.detector === "Laplacian" ? {
      laplacian_min_component_area: parameters.laplacianMinimumComponentArea,
    } : {}),
    ...(parameters.noise !== "None" ? {
      noise_model: parameters.noise,
      noise_strength: parameters.noiseStrength,
      noise_seed: DEFAULT_NOISE_SEED,
    } : {}),
    ...(parameters.multiScale ? { multi_scale: true, scale_sigmas: [...parameters.scaleSigmas],
      scale_support: parameters.scaleSupport } : {}) };
}

export function comparisonSettings(parameters: Parameters): ComparisonSettings {
  return {
    sigma: parameters.sigma,
    kernel_size: parameters.kernel,
    sobel_threshold: parameters.comparisonSobelThreshold,
    prewitt_threshold: parameters.comparisonPrewittThreshold,
    laplacian_contrast_threshold: parameters.comparisonLaplacianContrastThreshold,
  };
}

export function validateComparison(settings: ComparisonSettings): string | null {
  const gaussianError = validateGaussian(settings);
  if (gaussianError) return gaussianError;
  const thresholds = [settings.sobel_threshold, settings.prewitt_threshold,
    settings.laplacian_contrast_threshold];
  if (thresholds.some((threshold) => !Number.isFinite(threshold) || threshold < 0 ||
      threshold > MAX_THRESHOLD))
    return `Comparison thresholds must be finite numbers from 0 to ${MAX_THRESHOLD}.`;
  return null;
}

export function comparisonEntries(result: ComparisonResult | null) {
  return ([
    ["sobel", "Sobel"], ["prewitt", "Prewitt"], ["laplacian", "Laplacian"],
  ] as const).map(([key, detector]) => ({ key, detector, result: result?.[key] ?? null }));
}

export function resultImage(result: ComputedAnalysisResult | null, kind: Visualization): string | null {
  if (!result) return null;
  const images: Record<Visualization, string | null> = {
    original: result.original_image, grayscale: result.grayscale_image,
    noisy: result.noisy_image,
    filtered: result.filtered_image, spectrum: result.fft_image,
    "filtered-spectrum": result.filtered_fft_image, edges: result.edge_map,
    gradient: result.gradient_magnitude, gx: result.gx, gy: result.gy, contours: result.contour_image,
    laplacian: result.laplacian_response ?? null,
  };
  return images[kind];
}
export const DETECTOR_DESCRIPTIONS: Record<Detector, string> = {
  Sobel: "Weighted first-order gradient",
  Prewitt: "Uniform first-order gradient",
  Laplacian: "Second-order spatial derivative",
};

export function restoreParameters(value: unknown): Parameters {
  const p = value && typeof value === "object" ? (value as Partial<Parameters>) : {};
  const bounded = (n: unknown, min: number, max: number, fallback: number, step: number) =>
    typeof n === "number" && Number.isFinite(n)
      ? Math.round(Math.max(min, Math.min(max, n)) / step) * step
      : fallback;
  return {
    detector: ["Sobel", "Prewitt", "Laplacian"].includes(p.detector ?? "") ? p.detector! : "Sobel",
    sigma: Number(bounded(p.sigma, 0, 5, 1.2, 0.1).toFixed(1)),
    kernel: KERNEL_SIZES.includes(p.kernel ?? 0) ? p.kernel! : 5,
    threshold: bounded(p.threshold, 0, MAX_THRESHOLD, 96, 1),
    comparisonSobelThreshold: bounded(p.comparisonSobelThreshold, 0, MAX_THRESHOLD, 96, 1),
    comparisonPrewittThreshold: bounded(p.comparisonPrewittThreshold, 0, MAX_THRESHOLD, 96, 1),
    comparisonLaplacianContrastThreshold: bounded(
      p.comparisonLaplacianContrastThreshold, 0, MAX_THRESHOLD, 20, 1,
    ),
    multiScale: typeof p.multiScale === "boolean" ? p.multiScale : false,
    scaleSigmas: Array.isArray(p.scaleSigmas) && p.scaleSigmas.length === 3 &&
      p.scaleSigmas.every((value) => typeof value === "number" && Number.isFinite(value) &&
        value >= 0 && value <= 5 && Math.abs(value * 10 - Math.round(value * 10)) <= 1e-8) &&
      new Set(p.scaleSigmas).size === 3
      ? p.scaleSigmas.map((value) => Number((Math.round(value * 10) / 10).toFixed(1)))
      : [0.8, 1.6, 3.2],
    scaleSupport: Number.isInteger(p.scaleSupport) && p.scaleSupport! >= 1 && p.scaleSupport! <= 3
      ? p.scaleSupport! : 2,
    laplacianMinimumComponentArea: bounded(
      p.laplacianMinimumComponentArea, 1, 10000, 2, 1,
    ),
    noise: ["None", "Gaussian", "Salt & Pepper"].includes(p.noise ?? "") ? p.noise! : "None",
    noiseStrength: p.noise === "Salt & Pepper"
      ? bounded(p.noiseStrength, 0, 1, 0.12, 0.01)
      : bounded(p.noiseStrength, 0, 100, 12, 1),
  };
}
export function validateImage(file: Pick<File, "size" | "type">): string | null {
  if (!["image/png", "image/jpeg", "image/webp", "image/gif"].includes(file.type))
    return "Choose a PNG, JPG, WebP, or GIF image.";
  if (file.size > 20 * 1024 * 1024)
    return "This image is larger than 20 MB. Choose a smaller file.";
  if (file.size === 0) return "This image is empty. Choose another file.";
  return null;
}
