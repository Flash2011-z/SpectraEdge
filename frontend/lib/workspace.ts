export type Detector = "Sobel" | "Prewitt" | "Laplacian";
export type View = "analyze" | "compare" | "live";
export type Visualization =
  | "original"
  | "grayscale"
  | "filtered"
  | "edges"
  | "gradient"
  | "gx"
  | "gy"
  | "contours"
  | "spectrum"
  | "filtered-spectrum";
export interface Parameters {
  detector: Detector;
  sigma: number;
  kernel: number;
  threshold: number;
  minimumArea: number;
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
  bounding_box: [number, number];
}
// PNG data URLs are display artifacts; numerical arrays stay in Python.
export interface AnalysisResult {
  provenance: "demo" | "computed";
  original_image: string;
  filtered_image: string | null;
  gx: string | null;
  gy: string | null;
  gradient_magnitude: string | null;
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
  spectrum_scale: { min: number; max: number; mapping: "linear_grayscale" };
  processing_time: number;
}
export const KERNEL_SIZES = Array.from({ length: 15 }, (_, index) => 3 + index * 2);
export const MAX_THRESHOLD = 1443;
export const MAX_DECODED_PIXELS = 20_000_000;
export const EMPTY_SOURCE: SourceImage = {
  kind: "empty", name: "No image selected", url: "", width: 0, height: 0,
};
export const DEFAULT_PARAMETERS: Parameters = {
  detector: "Sobel",
  sigma: 1.2,
  kernel: 5,
  threshold: 96,
  minimumArea: 450,
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
  { id: 1, area: 18432, perimeter: 544, centroid: [355, 480], bounding_box: [184, 184] },
  { id: 2, area: 22316, perimeter: 530, centroid: [770, 480], bounding_box: [169, 169] },
  { id: 3, area: 12430, perimeter: 512, centroid: [1180, 480], bounding_box: [168, 221] },
  { id: 4, area: 9612, perimeter: 694, centroid: [1600, 480], bounding_box: [150, 150] },
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
  { name: "Objects", detail: "Not implemented", description: "Contours and object measurements have not run.", view: "contours", key: "objects" },
] as const;

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
  if (settings.detector === undefined)
    return settings.threshold === undefined ? null : "Threshold requires the Sobel detector.";
  if (settings.detector !== "Sobel") return "Only the Sobel detector is supported.";
  if (typeof settings.threshold !== "number" || !Number.isFinite(settings.threshold) ||
      settings.threshold < 0 || settings.threshold > MAX_THRESHOLD)
    return `Sobel threshold must be a finite number from 0 to ${MAX_THRESHOLD} in raw magnitude units.`;
  return null;
}

// Only active numerical settings cross the API boundary; future controls do not.
export function analysisSettings(parameters: Parameters): AnalysisSettings {
  return { sigma: parameters.sigma, kernel_size: parameters.kernel,
    detector: parameters.detector, threshold: parameters.threshold };
}

export function resultImage(result: ComputedAnalysisResult | null, kind: Visualization): string | null {
  if (!result) return null;
  const images: Record<Visualization, string | null> = {
    original: result.original_image, grayscale: result.grayscale_image,
    filtered: result.filtered_image, spectrum: result.fft_image,
    "filtered-spectrum": result.filtered_fft_image, edges: result.edge_map,
    gradient: result.gradient_magnitude, gx: result.gx, gy: result.gy, contours: result.contour_image,
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
    minimumArea: bounded(p.minimumArea, 0, 5000, 450, 50),
    noise: ["None", "Gaussian", "Salt & Pepper"].includes(p.noise ?? "") ? p.noise! : "None",
    noiseStrength: bounded(p.noiseStrength, 0, 100, 12, 1),
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
