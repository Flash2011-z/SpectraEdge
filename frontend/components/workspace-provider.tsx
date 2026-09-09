"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  DEFAULT_PARAMETERS,
  DEFAULT_PREFERENCES,
  DEMO_SOURCE,
  EMPTY_SOURCE,
  MAX_DECODED_PIXELS,
  analysisSettings,
  restoreParameters,
  validateImage,
  type Parameters,
  type Preferences,
  type SourceImage,
  type ComputedAnalysisResult,
} from "@/lib/workspace";
import { createAnalysisRunner } from "@/lib/api";

function useWorkspaceState() {
  const [parameters, setParameters] = useState<Parameters>(DEFAULT_PARAMETERS);
  const [preferences, setPreferences] = useState<Preferences>(DEFAULT_PREFERENCES);
  const [source, setSource] = useState<SourceImage>(EMPTY_SOURCE);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<ComputedAnalysisResult | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error" | "outdated">("idle");
  const [error, setError] = useState("");
  const [requests] = useState(createAnalysisRunner);
  const [stage, setStage] = useState(0);
  const [selectedObject, setSelectedObject] = useState(3);
  const busy = status === "loading";
  const [loadingImage, setLoadingImage] = useState(false);
  const [notice, setNotice] = useState("");
  const [hydrated, setHydrated] = useState(false);
  const openImagePicker = useCallback(
    () => document.getElementById("source-image-input")?.click(),
    [],
  );
  const objectUrl = useRef("");
  const imageRequest = useRef(0);
  const invalidate = useCallback(() => {
    requests.cancel();
    setResult(null);
    setError("");
    setStatus((previous) => ["success", "loading", "outdated"].includes(previous) ? "outdated" : "idle");
  }, [requests]);
  const notify = useCallback((message: string) => setNotice(message), []);
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("spectraedge.preferences.v1") || "{}");
      // Device-only preferences must hydrate after the server-rendered first frame.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setParameters(restoreParameters(saved.parameters));
      const p = saved.preferences;
      if (p && typeof p === "object")
        setPreferences({
          grid: typeof p.grid === "boolean" ? p.grid : true,
          demoVisuals: typeof p.demoVisuals === "boolean" ? p.demoVisuals : true,
          reducedMotion: typeof p.reducedMotion === "boolean" ? p.reducedMotion : false,
        });
    } catch {
      /* An unavailable or old local preference never prevents startup. */
    }
    setHydrated(true);
    return () => {
      requests.cancel();
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
      // This counter cancels decoding callbacks; it is not a rendered DOM ref.
      // eslint-disable-next-line react-hooks/exhaustive-deps
      imageRequest.current++;
    };
  }, [requests]);
  useEffect(() => {
    if (!hydrated) return;
    try {
      localStorage.setItem(
        "spectraedge.preferences.v1",
        JSON.stringify({ parameters, preferences }),
      );
    } catch {
      /* Storage is optional. */
    }
  }, [hydrated, parameters, preferences]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 4200);
    return () => clearTimeout(timer);
  }, [notice]);
  const updateParameter = useCallback(
    <K extends keyof Parameters>(key: K, value: Parameters[K]) => {
      invalidate();
      setParameters((p) => ({ ...p, [key]: value }));
    },
    [invalidate],
  );
  const reset = useCallback(() => {
    imageRequest.current++;
    setLoadingImage(false);
    invalidate();
    setParameters({ ...DEFAULT_PARAMETERS });
    setStage(0);
    notify("Parameters restored to defaults.");
  }, [invalidate, notify]);
  const loadDemo = useCallback(() => {
    imageRequest.current++;
    setLoadingImage(false);
    invalidate();
    setFile(null);
    setStatus("idle");
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = "";
    setSource(DEMO_SOURCE);
    setStage(0);
    notify("Calibration demo loaded. All measurements are illustrative.");
  }, [invalidate, notify]);
  const loadImage = useCallback(
    (file?: File) => {
      if (!file) return;
      const invalid = validateImage(file);
      if (invalid) {
        notify(invalid);
        return;
      }
      invalidate();
      setFile(null);
      const request = ++imageRequest.current;
      const url = URL.createObjectURL(file);
      const image = new Image();
      setLoadingImage(true);
      image.onload = () => {
        if (request !== imageRequest.current) {
          URL.revokeObjectURL(url);
          return;
        }
        if (image.naturalWidth * image.naturalHeight > MAX_DECODED_PIXELS) {
          URL.revokeObjectURL(url);
          setLoadingImage(false);
          notify("Choose an image at or below 20 megapixels.");
          return;
        }
        if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
        objectUrl.current = url;
        setSource({
          kind: "image",
          name: file.name,
          url,
          width: image.naturalWidth,
          height: image.naturalHeight,
        });
        setFile(file);
        // An old illustrative Prewitt/Laplacian preference cannot select an
        // unimplemented detector for a real upload.
        setParameters((previous) => ({ ...previous, detector: "Sobel" }));
        setStatus("idle");
        setLoadingImage(false);
        setStage(0);
        notify("Image ready. Process sends it to your configured Python backend; it is not saved.");
      };
      image.onerror = () => {
        URL.revokeObjectURL(url);
        if (request === imageRequest.current) {
          setLoadingImage(false);
          notify("This image could not be decoded. Try a different file.");
        }
      };
      image.src = url;
    },
    [invalidate, notify],
  );
  const process = useCallback(() => {
    if (busy || loadingImage) return;
    if (!file || source.kind !== "image") {
      notify("Upload an image to run real Gaussian, Sobel, and Fourier analysis. The calibration example is illustrative only.");
      return;
    }
    setStatus("loading");
    setError("");
    setResult(null);
    setStage(0);
    void requests.run(file, analysisSettings(parameters), {
      success: (computed) => {
        setResult(computed);
        setStatus("success");
        setStage(4);
        notify("Sobel edges and Fourier analysis complete. Objects have not been analyzed.");
      },
      error: (message) => { setStatus("error"); setError(message); },
    });
  }, [busy, loadingImage, file, source.kind, parameters, requests, notify]);
  return {
    parameters,
    preferences,
    setPreferences,
    source,
    stage,
    setStage,
    selectedObject,
    setSelectedObject,
    busy,
    result,
    status,
    error,
    canProcess: file !== null && !loadingImage && !busy,
    loadingImage,
    notice,
    notify,
    openImagePicker,
    updateParameter,
    reset,
    loadDemo,
    loadImage,
    process,
  };
}
type WorkspaceState = ReturnType<typeof useWorkspaceState>;
const WorkspaceContext = createContext<WorkspaceState | null>(null);
export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const state = useWorkspaceState();
  return (
    <WorkspaceContext.Provider value={state}>
      <div data-reduced-motion={state.preferences.reducedMotion ? "true" : undefined}>
        {children}
        <input
          id="source-image-input"
          type="file"
          accept="image/png,image/jpeg,image/webp,image/gif"
          hidden
          aria-label="Choose an input image"
          onChange={(event) => {
            state.loadImage(event.target.files?.[0]);
            event.target.value = "";
          }}
        />
      </div>
    </WorkspaceContext.Provider>
  );
}
export function useWorkspace() {
  const context = useContext(WorkspaceContext);
  if (!context) throw new Error("WorkspaceProvider is required.");
  return context;
}
