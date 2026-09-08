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
  restoreParameters,
  validateImage,
  type Parameters,
  type Preferences,
  type SourceImage,
} from "@/lib/workspace";

function useWorkspaceState() {
  const [parameters, setParameters] = useState<Parameters>(DEFAULT_PARAMETERS);
  const [preferences, setPreferences] = useState<Preferences>(DEFAULT_PREFERENCES);
  const [source, setSource] = useState<SourceImage>(DEMO_SOURCE);
  const [stage, setStage] = useState(0);
  const [selectedObject, setSelectedObject] = useState(3);
  const [busy, setBusy] = useState(false);
  const [previewed, setPreviewed] = useState(false);
  const [loadingImage, setLoadingImage] = useState(false);
  const [notice, setNotice] = useState("");
  const [hydrated, setHydrated] = useState(false);
  const openImagePicker = useCallback(
    () => document.getElementById("source-image-input")?.click(),
    [],
  );
  const processTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const objectUrl = useRef("");
  const imageRequest = useRef(0);
  const cancelPreview = useCallback(() => {
    if (processTimer.current) clearInterval(processTimer.current);
    processTimer.current = null;
    setBusy(false);
    setPreviewed(false);
  }, []);
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
      if (processTimer.current) clearInterval(processTimer.current);
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
      // This counter cancels decoding callbacks; it is not a rendered DOM ref.
      // eslint-disable-next-line react-hooks/exhaustive-deps
      imageRequest.current++;
    };
  }, []);
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
      cancelPreview();
      setParameters((p) => ({ ...p, [key]: value }));
    },
    [cancelPreview],
  );
  const reset = useCallback(() => {
    cancelPreview();
    setParameters({ ...DEFAULT_PARAMETERS });
    setStage(0);
    notify("Parameters restored to defaults.");
  }, [cancelPreview, notify]);
  const loadDemo = useCallback(() => {
    imageRequest.current++;
    setLoadingImage(false);
    cancelPreview();
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = "";
    setSource(DEMO_SOURCE);
    setStage(0);
    notify("Calibration demo loaded. All measurements are illustrative.");
  }, [cancelPreview, notify]);
  const loadImage = useCallback(
    (file?: File) => {
      if (!file) return;
      const invalid = validateImage(file);
      if (invalid) {
        notify(invalid);
        return;
      }
      cancelPreview();
      const request = ++imageRequest.current;
      const url = URL.createObjectURL(file);
      const image = new Image();
      setLoadingImage(true);
      image.onload = () => {
        if (request !== imageRequest.current) {
          URL.revokeObjectURL(url);
          return;
        }
        if (image.naturalWidth * image.naturalHeight > 80_000_000) {
          URL.revokeObjectURL(url);
          setLoadingImage(false);
          notify("Choose an image below 80 megapixels.");
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
        setLoadingImage(false);
        setStage(0);
        notify("Image opened locally. Processing will be available in the next phase.");
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
    [cancelPreview, notify],
  );
  const process = useCallback(() => {
    if (busy || loadingImage) return;
    if (source.kind !== "demo") {
      notify(
        "Image processing is not connected yet. Load the calibration demo to explore the pipeline.",
      );
      return;
    }
    setBusy(true);
    setPreviewed(false);
    setStage(0);
    let next = 0;
    processTimer.current = setInterval(
      () => {
        next++;
        setStage(Math.min(next, 5));
        if (next >= 5) {
          if (processTimer.current) clearInterval(processTimer.current);
          processTimer.current = null;
          setBusy(false);
          setPreviewed(true);
          notify("Demo walkthrough complete. Visuals and measurements are preset examples.");
        }
      },
      preferences.reducedMotion ? 20 : 180,
    );
  }, [busy, loadingImage, source.kind, preferences.reducedMotion, notify]);
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
    previewed,
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
