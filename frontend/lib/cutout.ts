import { validateGaussian, type GaussianSettings } from "./workspace.ts";

export interface Dimensions { width: number; height: number }
export interface Point { x: number; y: number }
export interface Rectangle extends Point, Dimensions {}
export type CutoutTool = "rectangle" | "keep" | "remove";
export interface BrushMark {
  mode: "keep" | "remove";
  size: number; // Diameter in prepared-image pixels, not CSS pixels.
  points: Point[];
}
export interface PreparedImage extends Dimensions {
  request_id: string;
  image_id: string;
  prepared_image: string;
  source_dimensions: Dimensions;
  processing_time: number;
}
export type CutoutMethod = "edge-watershed" | "grabcut" | "ai-assisted";
export type CutoutSettings = ({ method: "edge-watershed" } & GaussianSettings) | { method: "grabcut" | "ai-assisted" };
export const DEFAULT_CUTOUT_SETTINGS: CutoutSettings = { method: "edge-watershed", sigma: 1.2, kernel_size: 5 };
export const CUTOUT_METHOD_LABELS = { "edge-watershed": "Edge-guided watershed", grabcut: "GrabCut", "ai-assisted": "AI-assisted cutout" };
export const AI_MODEL = "birefnet-portrait";
export const MAX_PREPARED_SIDE = 512;
interface CutoutImages extends Dimensions {
  request_id: string;
  image_id: string;
  cutout_image: string;
  cropped_image: string;
  mask_image: string;
  cropped_dimensions: Dimensions;
  foreground_bounds: Rectangle;
  processing_time: number;
}
export type CutoutResult = CutoutImages & (
  { method: "edge-watershed"; algorithm: "skimage-watershed"; parameters_used: GaussianSettings;
    seed_mode: "automatic" | "brush";
    guidance_image: string; guidance_scale: { min: number; max: number; mapping: "linear_grayscale" } } |
  { method: "grabcut"; algorithm: "opencv-grabcut"; parameters_used: Record<string, never>;
    seed_mode: null;
    guidance_image: null; guidance_scale: null } |
  { method: "ai-assisted"; algorithm: "rembg-onnx"; parameters_used: { model: typeof AI_MODEL };
    seed_mode: null;
    guidance_image: null; guidance_scale: null }
);
export interface CutoutSnapshot {
  prepared: PreparedImage;
  rectangle: Rectangle;
  marks: BrushMark[];
  settings: CutoutSettings;
}
export const MAX_MARKS = 200;
export const MAX_STROKE_POINTS = 2000;
export const MAX_TOTAL_POINTS = 20_000;

export function resolveCutoutRectangle(image: Dimensions | null, rectangle: Rectangle | null, method: CutoutMethod): Rectangle | null {
  // AI can start on the whole photo; manual methods still require a user box.
  return rectangle ?? (image && method === "ai-assisted" ? { x: 0, y: 0, width: image.width, height: image.height } : null);
}

export function imagePoint(client: Point, canvas: Rectangle, image: Dimensions): Point {
  // The canvas has the image's aspect ratio and no border/padding. Its actual
  // bounding rectangle is measured on every event, including after resizing.
  if (canvas.width <= 0 || canvas.height <= 0) throw new Error("Canvas has no displayed size.");
  return {
    x: Math.max(0, Math.min(image.width - 1, Math.floor((client.x - canvas.x) * image.width / canvas.width))),
    y: Math.max(0, Math.min(image.height - 1, Math.floor((client.y - canvas.y) * image.height / canvas.height))),
  };
}

export function rectangleFromPoints(start: Point, end: Point): Rectangle {
  // Include both endpoint pixels; right/bottom in the API are exclusive.
  return { x: Math.min(start.x, end.x), y: Math.min(start.y, end.y),
    width: Math.abs(end.x - start.x) + 1, height: Math.abs(end.y - start.y) + 1 };
}

const integer = (n: unknown, min: number, max: number): n is number =>
  typeof n === "number" && Number.isInteger(n) && n >= min && n <= max;
const dimensionsValid = (size: Dimensions | undefined, max = MAX_PREPARED_SIDE) => Boolean(size &&
  integer(size.width, 1, max) && integer(size.height, 1, max));

export function validateSelection(image: Dimensions, rectangle: Rectangle | null, marks: BrushMark[]): string | null {
  if (!dimensionsValid(image)) return `Prepare an image first (maximum side ${MAX_PREPARED_SIDE} pixels).`;
  if (!rectangle || !integer(rectangle.x, 0, image.width - 1) || !integer(rectangle.y, 0, image.height - 1) ||
      !integer(rectangle.width, 2, image.width) || !integer(rectangle.height, 2, image.height) ||
      rectangle.x + rectangle.width > image.width || rectangle.y + rectangle.height > image.height)
    return "Draw a rectangle at least 2 × 2 pixels, inside the prepared image.";
  if (marks.length > MAX_MARKS) return `Use at most ${MAX_MARKS} brush marks. Undo a mark or reset the selection.`;
  let total = 0;
  for (const mark of marks) {
    if (!["keep", "remove"].includes(mark.mode) || !integer(mark.size, 1, 128))
      return "Brush marks need Keep/Remove mode and a size from 1 to 128 pixels.";
    if (mark.points.length < 1 || mark.points.length > MAX_STROKE_POINTS)
      return `Each mark needs 1–${MAX_STROKE_POINTS} points. Use shorter strokes.`;
    total += mark.points.length;
    if (mark.points.some((p) => !integer(p.x, 0, image.width - 1) || !integer(p.y, 0, image.height - 1)))
      return "Brush points must stay within the prepared image.";
  }
  return total > MAX_TOTAL_POINTS ? `Use at most ${MAX_TOTAL_POINTS} brush points. Undo a mark or reset.` : null;
}

export function snapshotCutout(value: CutoutSnapshot): CutoutSnapshot {
  return { prepared: { ...value.prepared, source_dimensions: { ...value.prepared.source_dimensions } },
    settings: { ...value.settings },
    rectangle: { ...value.rectangle },
    marks: value.marks.map((mark) => ({ ...mark, points: mark.points.map((p) => ({ ...p })) })) };
}

export function validateCutoutSettings(settings: CutoutSettings): string | null {
  if (settings.method === "edge-watershed") return validateGaussian(settings);
  if (settings.method !== "grabcut" && settings.method !== "ai-assisted") return "Choose Edge-guided watershed, GrabCut, or AI-assisted cutout.";
  return Object.keys(settings).length !== 1 ? "Additional settings apply only to Edge-guided watershed." : null;
}

function resultMethodMatches(data: CutoutResult, expected: CutoutSettings): boolean {
  if (data.method !== expected.method || !data.parameters_used || typeof data.parameters_used !== "object") return false;
  if (data.method === "grabcut") return data.algorithm === "opencv-grabcut" &&
    Object.keys(data.parameters_used).length === 0 && data.guidance_image === null && data.guidance_scale === null && data.seed_mode === null;
  if (data.method === "ai-assisted") return data.algorithm === "rembg-onnx" &&
    Object.keys(data.parameters_used).length === 1 && data.parameters_used.model === AI_MODEL &&
    data.guidance_image === null && data.guidance_scale === null && data.seed_mode === null;
  return expected.method === "edge-watershed" && data.algorithm === "skimage-watershed" &&
    (data.seed_mode === "automatic" || data.seed_mode === "brush") &&
    validateGaussian(data.parameters_used) === null && Object.keys(data.parameters_used).length === 2 &&
    data.parameters_used.sigma === expected.sigma && data.parameters_used.kernel_size === expected.kernel_size &&
    pngMatches(data.guidance_image, data, 0) && Boolean(data.guidance_scale && data.guidance_scale.min === 0 &&
      Number.isFinite(data.guidance_scale.max) && data.guidance_scale.max >= 0 && data.guidance_scale.mapping === "linear_grayscale");
}

function pngMatches(value: unknown, size: Dimensions, colorType: number): value is string {
  if (typeof value !== "string" || value.length > 9 * 1024 * 1024 ||
      !/^data:image\/png;base64,[A-Za-z0-9+/]+={0,2}$/.test(value)) return false;
  try {
    // Check PNG's IHDR dimensions/type, not just the data-URL prefix. The
    // browser subsequently decodes the complete image before displaying it.
    const bytes = Uint8Array.from(atob(value.split(",")[1].slice(0, 44)), (c) => c.charCodeAt(0));
    if (bytes.length < 33 || ![137, 80, 78, 71, 13, 10, 26, 10].every((b, i) => bytes[i] === b)) return false;
    const header = new DataView(bytes.buffer);
    return header.getUint32(8) === 13 && header.getUint32(12) === 0x49484452 &&
      header.getUint32(16) === size.width && header.getUint32(20) === size.height && bytes[24] === 8 && bytes[25] === colorType;
  } catch { return false; }
}

export function parsePreparedImage(value: unknown, requestId: string): PreparedImage {
  const data = value as PreparedImage | null;
  if (!data || data.request_id !== requestId || !/^[a-f0-9]{64}$/.test(data.image_id) ||
      !dimensionsValid(data) || !dimensionsValid(data.source_dimensions, 20_000_000) ||
      data.source_dimensions.width * data.source_dimensions.height > 20_000_000 ||
      data.width > data.source_dimensions.width || data.height > data.source_dimensions.height ||
      !Number.isFinite(data.processing_time) || data.processing_time < 0 || !pngMatches(data.prepared_image, data, 6))
    throw new Error("The backend returned an incompatible prepared image. Check both services and upload again.");
  return data;
}

export function parseCutoutResult(value: unknown, requestId: string, prepared: PreparedImage,
  settings: CutoutSettings = { method: "grabcut" }, seedMode?: "automatic" | "brush"): CutoutResult {
  const data = value as CutoutResult | null;
  const box = data?.foreground_bounds;
  if (!data || data.request_id !== requestId || data.image_id !== prepared.image_id ||
      !resultMethodMatches(data, settings) || data.width !== prepared.width || data.height !== prepared.height ||
      (seedMode !== undefined && data.seed_mode !== seedMode) ||
      !box || !integer(box.x, 0, data.width - 1) || !integer(box.y, 0, data.height - 1) ||
      !dimensionsValid(box) || box.x + box.width > data.width || box.y + box.height > data.height ||
      !dimensionsValid(data.cropped_dimensions) || data.cropped_dimensions.width !== box.width || data.cropped_dimensions.height !== box.height ||
      !pngMatches(data.cutout_image, data, 6) || !pngMatches(data.cropped_image, box, 6) || !pngMatches(data.mask_image, data, 0) ||
      !Number.isFinite(data.processing_time) || data.processing_time < 0)
    throw new Error("The backend returned an incompatible cutout. Results were discarded; extract again.");
  return data;
}
