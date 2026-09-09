"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  Camera,
  Check,
  CircleHelp,
  Cpu,
  FlaskConical,
  GitCompareArrows,
  Info,
  Keyboard,
  LoaderCircle,
  LockKeyhole,
  Play,
  Radio,
  ScanLine,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  X,
} from "lucide-react";
import {
  DEMO_OBJECTS,
  ANALYSIS_STAGES,
  DETECTOR_DESCRIPTIONS,
  STAGES,
  type Preferences,
  type View,
} from "@/lib/workspace";
import { useWorkspace } from "./workspace-provider";
import { ParametersPanel } from "./parameters";
import { VisualizationCard } from "./visualization";
import { ObjectInformation } from "./object-information";
import { Pipeline } from "./pipeline";
import { downloadJson, IconButton, Modal } from "./ui";

const PAGE_COPY = {
  analyze: {
    eyebrow: "SPATIAL + FREQUENCY DOMAIN",
    title: "Analysis workspace",
    subtitle: "See the structure behind the signal.",
  },
  compare: {
    eyebrow: "ONE INPUT. THREE PERSPECTIVES.",
    title: "Detector comparison",
    subtitle: "Explore the same scene through different spatial operators.",
  },
  live: {
    eyebrow: "REAL-TIME ANALYSIS",
    title: "Live workspace",
    subtitle: "A dedicated workspace for your future camera pipeline.",
  },
};
type Dialog = "settings" | "about" | "shortcuts" | "camera" | null;

function SettingsContent({ onClose }: { onClose: () => void }) {
  const { preferences, setPreferences, notify } = useWorkspace();
  const [draft, setDraft] = useState<Preferences>(preferences);
  return (
    <>
      <div className="modal-body">
        <p className="dialog-description">
          Make the workspace your own. Preferences are saved on this device.
        </p>
        {(
          [
            ["grid", "Instrument grid", "Show a coordinate grid behind each visualization."],
            [
              "demoVisuals",
              "Demo visuals",
              "Display clearly marked calibration artwork and sample metrics.",
            ],
            [
              "reducedMotion",
              "Reduce motion",
              "Limit workspace transitions and animations.",
            ],
          ] as const
        ).map(([key, title, detail]) => (
          <label className="settings-option" key={key}>
            <span>
              <strong>{title}</strong>
              <small>{detail}</small>
            </span>
            <input
              type="checkbox"
              role="switch"
              checked={draft[key]}
              onChange={(e) => setDraft((p) => ({ ...p, [key]: e.target.checked }))}
            />
            <span className="switch-track" aria-hidden="true" />
          </label>
        ))}
        <div className="privacy-note">
          <ShieldCheck size={16} />
          <p>
            Process sends the selected image to your configured Python backend. Images and results
            stay in memory and are not saved there. Only preferences and settings are stored on this device.
          </p>
        </div>
      </div>
      <footer className="modal-actions">
        <button className="button subtle" onClick={onClose}>
          Cancel
        </button>
        <button
          className="button primary"
          onClick={() => {
            setPreferences(draft);
            notify("Workspace settings saved.");
            onClose();
          }}
        >
          Save preferences
          <Check size={13} />
        </button>
      </footer>
    </>
  );
}

function CompareView() {
  const { source, preferences, parameters } = useWorkspace();
  const demo = source.kind === "demo" && preferences.demoVisuals;
  return (
    <div className="compare-view">
      <div className="comparison-source">
        <VisualizationCard kind="original" index="01" large />
        <div className="comparison-brief">
          <span className="eyebrow">CONTROLLED COMPARISON</span>
          <h2>
            Same signal.
            <br />
            Different operators.
          </h2>
          <p>
            Each detector will use the same source and filter settings, so you can compare its
            response directly.
          </p>
          <dl>
            <div>
              <dt>Gaussian sigma</dt>
              <dd>{parameters.sigma.toFixed(1)}</dd>
            </div>
            <div>
              <dt>Kernel size</dt>
              <dd>
                {parameters.kernel} × {parameters.kernel}
              </dd>
            </div>
            <div>
              <dt>Threshold</dt>
              <dd>{parameters.threshold}</dd>
            </div>
          </dl>
          <button className="button primary" disabled>
            <Play size={13} />Detector comparison not implemented
          </button>
        </div>
      </div>
      <div className="comparison-results">
        {(["Sobel", "Prewitt", "Laplacian"] as const).map((detector, i) => (
          <section className="detector-result" key={detector}>
            <header>
              <span className="detector-number">0{i + 1}</span>
              <div>
                <h2>{detector}</h2>
                <p>{DETECTOR_DESCRIPTIONS[detector]}</p>
              </div>
              <span className="tag">{demo ? "DEMO" : "IDLE"}</span>
            </header>
            <VisualizationCard kind="edges" index={`0${i + 2}`} detector={detector} />
            <dl className="comparison-metrics">
              <div>
                <dt>Objects detected</dt>
                <dd>{demo ? "04" : "—"}</dd>
              </div>
              <div>
                <dt>Processing time</dt>
                <dd>
                  {demo ? [18, 16, 12][i] : "—"}
                  <small> ms</small>
                </dd>
              </div>
            </dl>
            <p className="result-caption">
              {demo
                ? "Preset visual and metrics · not a measured result"
                : "Detector comparison is not implemented"}
            </p>
          </section>
        ))}
      </div>
      <div className="page-note">
        <Info size={14} />
        <span>
          This prototype demonstrates the comparison layout. Demo results are illustrative and
          cannot establish which detector performs best.
        </span>
      </div>
    </div>
  );
}
function LiveView({ onCamera }: { onCamera: () => void }) {
  const { parameters } = useWorkspace();
  return (
    <div className="live-view">
      <div className="live-toolbar">
        <span className="standby">
          <span />
          CAMERA STANDBY
        </span>
        <span className="muted">No camera connected</span>
        <button className="button" onClick={onCamera}>
          <Camera size={14} />
          Camera information
          <ArrowRight size={12} />
        </button>
      </div>
      <div className="live-panels">
        {[
          {
            title: "Live camera",
            detail: "Camera preview will appear here",
            icon: Camera,
            label: "INPUT STREAM",
          },
          {
            title: "Live edge / contour output",
            detail: "Processed frames will appear here",
            icon: ScanLine,
            label: "ANALYSIS STREAM",
          },
        ].map(({ title, detail, icon: Icon, label }, i) => (
          <section className="instrument live-instrument" key={title}>
            <header className="instrument-header">
              <div>
                <span className="panel-index">0{i + 1}</span>
                <h2>{title}</h2>
              </div>
              <span className="tag">{label}</span>
            </header>
            <div className="viewport live-viewport">
              <div className="camera-reticle">
                <span />
                <Icon size={31} strokeWidth={1} />
                <span />
              </div>
              <strong>{detail}</strong>
              <p>Available when live processing is connected.</p>
              <span className="viewport-label">NO SIGNAL</span>
            </div>
            <footer className="instrument-footer">
              <span>{i === 0 ? "Source / camera" : "Output / spatial domain"}</span>
              <span>— × —</span>
            </footer>
          </section>
        ))}
      </div>
      <div className="live-metrics">
        {[
          { label: "DETECTOR", value: parameters.detector },
          { label: "THRESHOLD", value: parameters.threshold },
          { label: "GAUSSIAN SIGMA", value: parameters.sigma.toFixed(1) },
          { label: "OBJECTS", value: "—" },
          { label: "FRAME RATE", value: "—", suffix: " FPS" },
        ].map((item) => (
          <div key={item.label}>
            <span className="eyebrow">{item.label}</span>
            <strong>
              {item.value}
              <small>{item.suffix}</small>
            </strong>
          </div>
        ))}
      </div>
      <div className="live-bottom">
        <div className="privacy-note">
          <LockKeyhole size={18} />
          <div>
            <strong>You control camera access.</strong>
            <p>Camera access is disabled in this prototype. No video is captured or transmitted.</p>
          </div>
        </div>
        <Link href="/" className="text-button">
          Configure analysis parameters
          <ArrowRight size={13} />
        </Link>
      </div>
    </div>
  );
}

export default function Workspace({ view }: { view: View }) {
  const {
    parameters,
    preferences,
    source,
    stage,
    notice,
    notify,
    openImagePicker,
    reset,
    process,
    loadDemo,
    selectedObject,
    result,
    status,
    error,
    busy,
    canProcess,
    loadingImage,
  } = useWorkspace();
  const [dialog, setDialog] = useState<Dialog>(null);
  const [controlsOpen, setControlsOpen] = useState(false);
  const demo = source.kind === "demo" && preferences.demoVisuals;
  const stages = source.kind === "demo" ? STAGES : ANALYSIS_STAGES;
  const edgesComputed = result?.detection_status === "edges_computed";
  const statusText = busy ? "Processing in Python…" : status === "success" ? edgesComputed ? "Edges computed · objects not analyzed" : "Gaussian + Fourier complete · detection not run"
    : status === "error" ? "Analysis failed" : status === "outdated" ? "Settings changed · process again" : "Ready for an image analysis";
  const exportSession = useCallback(() => {
    downloadJson(
      {
        schema_version: 3,
        application: "SpectraEdge",
        exported_at: new Date().toISOString(),
        mode: demo ? "illustrative-example" : result ? result.detection_status === "edges_computed" ? "sobel-analysis" : "gaussian-fourier" : "unprocessed-session",
        source: {
          name: source.name,
          kind: source.kind,
          width: source.width,
          height: source.height,
        },
        parameters,
        result: result ? {
          provenance: result.provenance,
          request_id: result.request_id,
          parameters_used: result.parameters_used,
          source_dimensions: result.source_dimensions,
          analyzed_dimensions: result.analyzed_dimensions,
          completed_stages: result.completed_stages,
          detection_status: result.detection_status,
          object_list: result.object_list,
          processing_time: result.processing_time,
          timing_unit: "ms",
          spectrum_scale: result.spectrum_scale,
          threshold_rule: result.detection_status === "edges_computed" ? "raw Sobel magnitude > threshold" : null,
          derivative_display: result.detection_status === "edges_computed" ? {
            signed_min: -1020, signed_max: 1020, magnitude_min: 0,
            magnitude_max: 1020 * Math.sqrt(2), mapping: "linear_grayscale",
          } : null,
        } : demo
          ? {
              provenance: "demo",
              object_list: DEMO_OBJECTS,
              selected_object: selectedObject,
              processing_time: 18,
              fps: null,
            }
          : null,
        status,
        note: result ? "Actual manual signal-processing results; see completed_stages and detection_status. Contours and object analysis have not run. Cards export display PNGs, not raw arrays."
          : "No computed results for the current image/settings. Any demo measurements are illustrative only.",
      },
      "spectraedge-session.json",
    );
    notify("Session configuration exported. Image data is not included.");
  }, [source, parameters, selectedObject, demo, result, status, notify]);
  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (
        document.querySelector("dialog[open]") ||
        target.matches("input, select, textarea, button") ||
        target.isContentEditable
      )
        return;
      const key = event.key.toLowerCase(),
        modifier = event.ctrlKey || event.metaKey;
      if (modifier && key === "o") {
        event.preventDefault();
        openImagePicker();
      } else if (modifier && key === ",") {
        event.preventDefault();
        setDialog("settings");
      } else if (modifier && key === "r" && view === "analyze") {
        event.preventDefault();
        reset();
      } else if (event.code === "Space" && view === "analyze") {
        event.preventDefault();
        process();
      } else if (key === "?") {
        event.preventDefault();
        setDialog("shortcuts");
      }
    };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [openImagePicker, reset, process, view]);
  const copy = PAGE_COPY[view];
  return (
    <div className="application">
      <a className="skip-link" href="#workspace-main">
        Skip to workspace
      </a>
      <header className="topbar">
        <Link className="brand" href="/">
          <span className="brand-mark">
            <Activity size={23} strokeWidth={1.5} />
          </span>
          <strong>
            Spectra<span>Edge</span>
          </strong>
          <span className="version">WEB / 01</span>
        </Link>
        <nav aria-label="Workspace">
          {[
            { view: "analyze", path: "/", label: "Analyze", icon: ScanLine },
            { view: "compare", path: "/compare", label: "Compare", icon: GitCompareArrows },
            { view: "live", path: "/live", label: "Live", icon: Radio },
          ].map(({ view: page, path, label, icon: Icon }) => (
            <Link
              key={page}
              className={`nav-link ${view === page ? "active" : ""}`}
              href={path}
              aria-current={view === page ? "page" : undefined}
            >
              <Icon size={14} strokeWidth={1.6} />
              {label}
            </Link>
          ))}
        </nav>
        <div className="top-actions">
          <span className="mode-label">
            <i />
            Sobel + Fourier
          </span>
          <span className="top-divider" />
          <IconButton label="Workspace settings" onClick={() => setDialog("settings")}>
            <Settings2 size={16} />
          </IconButton>
          <IconButton label="About SpectraEdge" onClick={() => setDialog("about")}>
            <CircleHelp size={16} />
          </IconButton>
        </div>
      </header>
      <div className={`workstation ${view !== "analyze" ? "full-workstation" : ""}`}>
        {view === "analyze" && (
          <>
            <button
              className="mobile-controls button"
              aria-expanded={controlsOpen}
              aria-controls="parameters"
              onClick={() => setControlsOpen((value) => !value)}
            >
              <SlidersHorizontal size={14} />
              {controlsOpen ? "Hide parameters" : "Analysis parameters"}
              {controlsOpen ? <X size={14} /> : <span className="tag">{parameters.detector}</span>}
            </button>
            <ParametersPanel open={controlsOpen} onCamera={() => setDialog("camera")} />
          </>
        )}
        <main className="workspace" id="workspace-main">
          <div className="workspace-heading">
            <div>
              <div className="eyebrow">{copy.eyebrow}</div>
              <h1>
                {copy.title}
                <span className="heading-dot">.</span>
              </h1>
              <p>{copy.subtitle}</p>
            </div>
            <div className="heading-actions">
              <span className="badge">
                <FlaskConical size={11} />
                {source.kind === "demo" ? "ILLUSTRATIVE EXAMPLE" : result ? "COMPUTED SESSION" : "LOCAL SESSION"}
              </span>
              <button
                className="button"
                onClick={exportSession}
                title="Export current configuration and computed result metadata"
              >
                <ArrowDownToLine size={13} />
                <span>Export session</span>
              </button>
            </div>
          </div>
          {view === "analyze" ? (
            <>
              <div className="session-strip">
                <div>
                  <span className="status-dot" />
                  <span>
                    {loadingImage
                      ? "Opening local image…"
                      : source.kind === "demo"
                        ? "Calibration study"
                        : source.name}
                  </span>
                  <span className="strip-separator">/</span>
                  <span className="muted">
                    {source.kind === "demo"
                      ? "Explore a four-object reference scene"
                      : result ? `Analyzed ${result.analyzed_dimensions.width} × ${result.analyzed_dimensions.height} · ${result.processing_time.toFixed(1)} ms`
                        : "Preview ≤512 px · Gaussian → Sobel → edges + Fourier"}
                  </span>
                </div>
                <button
                  className="text-button"
                  onClick={source.kind === "demo" ? () => openImagePicker() : loadDemo}
                >
                  {source.kind === "demo" ? "Use your own image" : "Illustrative example"}
                  <ArrowRight size={12} />
                </button>
              </div>
              <div className={`analysis-notice ${error ? "analysis-error" : ""}`} role={error ? "alert" : "status"}>
                {busy ? <LoaderCircle size={15} className="spin" /> : <Info size={15} />}
                <span>{error || (source.kind === "demo" ? "Separate illustrative example · no calculations or measured results." : statusText)}</span>
                {error && <button className="text-button" disabled={!canProcess} onClick={process}>Try again</button>}
              </div>
              <VisualizationCard kind="original" index="01" large selected={stage === 0} />
              <div className="outputs-heading">
                <span className="eyebrow">ANALYSIS OUTPUTS</span>
                <span>
                  {demo ? "Illustrative previews" : result ? "Computed in Python" : "No current results"}
                  <span className="subtle-dot" />
                  {demo ? `${parameters.detector} example` : edgesComputed ? "Sobel edges · objects not analyzed" : "Detection not run"}
                </span>
              </div>
              <div className={`output-grid ${demo ? "" : "computed-grid"}`}>
                {!demo && <VisualizationCard kind="grayscale" index="02" selected={stages[stage].view === "grayscale"} />}
                <VisualizationCard
                  kind="filtered"
                  index="03"
                  selected={stages[stage].view === "filtered"}
                />
                {!demo && <>
                  <VisualizationCard kind="gx" index="04" selected={stages[stage].view === "gradient"} />
                  <VisualizationCard kind="gy" index="05" selected={stages[stage].view === "gradient"} />
                  <VisualizationCard kind="gradient" index="06" selected={stages[stage].view === "gradient"} />
                  <VisualizationCard kind="edges" index="07" selected={stages[stage].view === "edges"} />
                  <VisualizationCard kind="spectrum" index="08" selected={stages[stage].view === "spectrum"} />
                  <VisualizationCard kind="filtered-spectrum" index="09" selected={stages[stage].view === "spectrum"} />
                </>}
                {demo && <>
                <VisualizationCard
                  kind="edges"
                  index="03"
                  selected={stages[stage].view === "edges"}
                />
                <VisualizationCard
                  kind="gradient"
                  index="04"
                  selected={stages[stage].view === "gradient"}
                />
                <VisualizationCard
                  kind="contours"
                  index="05"
                  selected={stages[stage].view === "contours"}
                />
                <VisualizationCard kind="spectrum" index="06" />
                </>}
                <ObjectInformation />
              </div>
              {!demo && <p className="spectrum-scale-note">
                Both spectra share one grayscale display range: 0 to {result ? result.spectrum_scale.max.toFixed(4) : "—"} in log(1 + magnitude).
                {result && ` Used σ=${result.parameters_used.sigma}, kernel ${result.parameters_used.kernel_size} × ${result.parameters_used.kernel_size}.`}
                {edgesComputed && ` Sobel edges use raw magnitude > ${result.parameters_used.threshold}. Derivatives use −1020…1020; magnitude uses 0…1442.5 for display only.`}
                {" "}Contours, objects, and noise experiments have not run.
              </p>}
              <Pipeline />
            </>
          ) : view === "compare" ? (
            <CompareView />
          ) : (
            <LiveView onCamera={() => setDialog("camera")} />
          )}
        </main>
      </div>
      <footer className="statusbar">
        <span>
          <i className="status-dot" />
          {source.kind === "demo" ? "Illustrative example" : view === "analyze" ? statusText : "Planned workspace · not implemented"}
        </span>
        <span className="status-metrics">
          Objects <b>{demo && view !== "live" ? "4" : "—"}</b>
          <em />
          Time <b>{result && view === "analyze" ? `${result.processing_time.toFixed(1)} ms` : demo && view !== "live" ? "18 ms (demo)" : "—"}</b>
          <em />
          Detector <b>{demo ? `${parameters.detector} (example)` : edgesComputed && view === "analyze" ? "Sobel" : "not run"}</b>
          <em />
          {view === "live" ? "No stream" : result ? `${result.analyzed_dimensions.width} × ${result.analyzed_dimensions.height} analyzed` : `${source.width} × ${source.height} source`}
        </span>
        <button className="text-button" onClick={() => setDialog("shortcuts")}>
          <Keyboard size={12} />
          Keyboard shortcuts
        </button>
      </footer>
      {notice && (
        <div className="toast" key={notice} role="status">
          <Info size={16} />
          <span>{notice}</span>
        </div>
      )}
      {dialog === "settings" && (
        <Modal
          title="Workspace settings"
          eyebrow="DISPLAY & INTERACTION"
          onClose={() => setDialog(null)}
        >
          <SettingsContent onClose={() => setDialog(null)} />
        </Modal>
      )}
      {dialog === "about" && (
        <Modal
          title="About SpectraEdge"
          eyebrow="SIGNALS & LINEAR SYSTEMS"
          onClose={() => setDialog(null)}
        >
          <div className="modal-body about-content">
            <div className="about-logo">
              <Activity size={32} strokeWidth={1.5} />
            </div>
            <h3>
              Spectra<span>Edge</span>
            </h3>
            <p className="about-tagline">See the structure behind the signal.</p>
            <p>
              SpectraEdge: A Multi-Scale Multi-Object Edge Detection and Frequency-Domain Analysis
              System.
            </p>
            <p>
              A university Signals and Linear Systems project, reimagined as a workspace you can
              open in your browser.
            </p>
            <dl className="about-facts">
              <div>
                <dt>Current phase</dt>
                <dd>Manual Sobel + threshold demonstration</dd>
              </div>
              <div>
                <dt>Analysis engine</dt>
                <dd>Python · manual convolution and FFT</dd>
              </div>
              <div>
                <dt>Image handling</dt>
                <dd>Opt-in processing · in memory only</dd>
              </div>
            </dl>
            <div className="privacy-note">
              <Cpu size={17} />
              <p>
                Upload an image to calculate grayscale, Gaussian smoothing, Sobel gradients, binary edges, and two Fourier spectra.
                Prewitt, Laplacian, contours, object analysis, detector comparison, and live camera processing are not implemented.
                Calibration artwork and its measurements remain a separate illustrative example.
              </p>
            </div>
          </div>
          <footer className="modal-actions">
            <button className="button primary" onClick={() => setDialog(null)}>
              Back to workspace
              <ArrowRight size={13} />
            </button>
          </footer>
        </Modal>
      )}
      {dialog === "camera" && (
        <Modal title="Camera input" eyebrow="LIVE WORKSPACE" onClose={() => setDialog(null)}>
          <div className="modal-body">
            <div className="camera-info-icon">
              <Camera size={30} strokeWidth={1.3} />
            </div>
            <h3>Ready for the next phase.</h3>
            <p className="dialog-description">
              The Live page provides a dedicated camera and output workspace. Camera capture is
              currently disabled while the analysis engine is being developed.
            </p>
            <div className="privacy-note">
              <LockKeyhole size={17} />
              <p>No camera permissions are requested and no video is recorded.</p>
            </div>
          </div>
          <footer className="modal-actions">
            <button className="button subtle" onClick={() => setDialog(null)}>
              Close
            </button>
            <Link href="/live" className="button primary" onClick={() => setDialog(null)}>
              Explore Live
              <ArrowRight size={13} />
            </Link>
          </footer>
        </Modal>
      )}
      {dialog === "shortcuts" && (
        <Modal
          title="A faster way to explore"
          eyebrow="KEYBOARD SHORTCUTS"
          onClose={() => setDialog(null)}
        >
          <div className="modal-body shortcuts-list">
            {[
              ["Open an image", "Ctrl / ⌘", "O"],
              ["Reset Analyze parameters", "Ctrl / ⌘", "R"],
              ["Process Analyze image", "", "Space"],
              ["Workspace settings", "Ctrl / ⌘", ","],
              ["Keyboard shortcuts", "", "?"],
              ["Close a dialog", "", "Esc"],
            ].map(([label, modifier, key]) => (
              <div key={label}>
                <span>{label}</span>
                <span>
                  {modifier && <kbd>{modifier}</kbd>}
                  <kbd>{key}</kbd>
                </span>
              </div>
            ))}
            <p className="control-hint">
              Workspace shortcuts apply when you are not editing a control or using a dialog.
            </p>
          </div>
        </Modal>
      )}
    </div>
  );
}
