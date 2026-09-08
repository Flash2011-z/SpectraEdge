export type Detector = "Sobel" | "Prewitt" | "Laplacian";
export type View = "analyze" | "compare" | "live";
export type Visualization =
  | "original"
  | "filtered"
  | "edges"
  | "gradient"
  | "contours"
  | "spectrum";
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
// The future Python API can fill this contract without changing the workspace.
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
  object_list: ObjectMeasurement[];
  processing_time: number | null;
  fps: number | null;
}
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
    kernel: [3, 5, 7].includes(p.kernel ?? 0) ? p.kernel! : 5,
    threshold: bounded(p.threshold, 0, 255, 96, 1),
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
