import { validateGaussian, MAX_THRESHOLD, type Detector } from "./workspace.ts";

export interface LiveSettings {
  detector: Detector;
  sigma: number;
  kernel_size: number;
  threshold: number;
}
export interface LiveResult {
  frame_id: string;
  grayscale_image: string;
  filtered_image: string;
  edge_image: string;
  spectrum_image: string;
  processing_time: number;
  settings_used: LiveSettings;
}
export const DEFAULT_LIVE_SETTINGS: LiveSettings = {
  detector: "Sobel", sigma: 1.2, kernel_size: 5, threshold: 96,
};
export const LIVE_IMAGES = ["grayscale_image", "filtered_image", "edge_image", "spectrum_image"] as const;

export function validateLiveSettings(settings: LiveSettings): string | null {
  if (!["Sobel", "Prewitt", "Laplacian"].includes(settings.detector)) return "Choose a supported detector.";
  return validateGaussian(settings) || (!Number.isFinite(settings.threshold) || settings.threshold < 0 ||
    settings.threshold > MAX_THRESHOLD ? "Threshold must be from 0 to 1443." : null);
}

export function sameLiveSettings(a: LiveSettings, b: LiveSettings) {
  return a.detector === b.detector && a.sigma === b.sigma &&
    a.kernel_size === b.kernel_size && a.threshold === b.threshold;
}

export function frameDimensions(width: number, height: number) {
  const scale = Math.min(1, 256 / Math.max(width, height));
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}

export function stopCameraTracks(stream: MediaStream | null) {
  stream?.getTracks().forEach((track) => track.stop());
}

// Generation protection also covers permission prompts resolved after Stop/unmount.
export function createCameraSession(acquire: () => Promise<MediaStream>) {
  let generation = 0;
  let stream: MediaStream | null = null;
  const stop = () => { generation++; stopCameraTracks(stream); stream = null; };
  return {
    stop,
    async start() {
      stop();
      const current = generation;
      try {
        const next = await acquire();
        if (current !== generation) { stopCameraTracks(next); return null; }
        stream = next;
        return next;
      } catch (error) {
        if (current !== generation) return null;
        throw error;
      }
    },
  };
}

export function cameraError(error: unknown): string {
  const name = error && typeof error === "object" && "name" in error ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError")
    return "Camera permission was denied. Allow camera access in your browser, then start again.";
  if (name === "NotFoundError") return "No camera was found. Connect a webcam and try again.";
  if (name === "NotReadableError") return "The camera is unavailable or in use by another application.";
  return error instanceof Error ? error.message : "Could not start the camera.";
}
