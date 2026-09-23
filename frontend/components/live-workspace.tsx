"use client";

import { useEffect, useRef, useState } from "react";
import { Camera, Square } from "lucide-react";
import { DEFAULT_LIVE_SETTINGS, LIVE_IMAGES, sameLiveSettings, validateLiveSettings, type LiveSettings } from "../lib/live.ts";
import { KERNEL_SIZES, thresholdLabel, type Detector } from "../lib/workspace.ts";
import { WorkspaceBrand, WorkspaceNavigation } from "./workspace-navigation";
import { useCamera } from "./use-camera";
import { useLiveAnalysis } from "./use-live-analysis";
import styles from "./live-workspace.module.css";

const TITLES = ["Grayscale", "Gaussian filtered", "Edge map", "Fourier spectrum"];

export default function LiveWorkspace() {
  const { videoRef, stream, status: cameraStatus, error: cameraMessage, start, stop } = useCamera();
  const [settings, setSettings] = useState<LiveSettings>({ ...DEFAULT_LIVE_SETTINGS });
  const analysis = useLiveAnalysis(videoRef, stream, settings);
  const canvases = useRef<(HTMLCanvasElement | null)[]>([]);
  const [displayed, setDisplayed] = useState<{
    stream: MediaStream; settings: LiveSettings; processing: number; latency: number; fps: number; frameId: string;
  } | null>(null);
  const [displayError, setDisplayError] = useState<string | null>(null);
  const [resolution, setResolution] = useState<{ stream: MediaStream | null; width: number; height: number } | null>(null);
  const cadence = useRef<{ stream: MediaStream; settings: LiveSettings; times: number[] } | null>(null);
  const invalid = validateLiveSettings(settings);
  const current = displayed && displayed.stream === stream && sameLiveSettings(displayed.settings, settings) ? displayed : null;

  useEffect(() => {
    const output = analysis.result;
    if (!output || output.stream !== stream || !sameLiveSettings(output.frame.settings_used, settings)) return;
    let cancelled = false;
    const started = performance.now();
    const images = LIVE_IMAGES.map((key) => {
      const image = new Image();
      image.src = output.frame[key];
      return image;
    });
    // Canvas draws occur together only after all four PNGs have decoded.
    void Promise.all(images.map((image) => image.decode())).then(() => {
      if (cancelled) return;
      if (!images.every((image) => image.naturalWidth > 0 && image.naturalWidth <= 256 &&
          image.naturalHeight > 0 && image.naturalHeight <= 256 &&
          image.naturalWidth === images[0].naturalWidth && image.naturalHeight === images[0].naturalHeight))
        throw new Error("Live output dimensions are inconsistent.");
      const surfaces = canvases.current.map((canvas) => ({ canvas, context: canvas?.getContext("2d") }));
      if (surfaces.length !== 4 || surfaces.some(({ canvas, context }) => !canvas || !context))
        throw new Error("Your browser could not create the output canvases.");
      surfaces.forEach(({ canvas, context }, index) => {
        canvas!.width = images[index].naturalWidth;
        canvas!.height = images[index].naturalHeight;
        context!.drawImage(images[index], 0, 0);
      });
      const now = performance.now();
      if (!cadence.current || cadence.current.stream !== output.stream || !sameLiveSettings(cadence.current.settings, settings))
        cadence.current = { stream: output.stream, settings: { ...settings }, times: [] };
      const times = cadence.current.times;
      times.push(now);
      if (times.length > 10) times.shift();
      const fps = times.length > 1 ? (times.length - 1) * 1000 / (now - times[0]) : 0;
      setDisplayed({ stream: output.stream, settings: output.frame.settings_used,
        processing: output.frame.processing_time, latency: output.roundTrip + now - started,
        fps, frameId: output.frame.frame_id });
      setDisplayError(null);
    }).catch((error: unknown) => {
      if (!cancelled) setDisplayError(error instanceof Error ? error.message : "Could not display the live frame.");
    });
    return () => { cancelled = true; };
  }, [analysis.result, stream, settings]);

  const update = <K extends keyof LiveSettings>(key: K, value: LiveSettings[K]) =>
    setSettings((previous) => ({ ...previous, [key]: value }));
  const active = cameraStatus === "active";
  const processingStatus = !active ? (cameraStatus === "requesting" ? "Waiting for camera" : "Idle")
    : invalid ? "Paused · invalid parameters"
    : analysis.error || displayError ? "Processing error"
    : current ? "Processing" : "Waiting for frame";
  const metrics = active && !invalid && !analysis.error && !displayError ? current : null;
  const frameResolution = active && resolution?.stream === stream && resolution.width > 0 && resolution.height > 0
    ? `${resolution.width} × ${resolution.height}` : "--";
  const readResolution = (video: HTMLVideoElement) => setResolution({
    stream: video.srcObject as MediaStream | null, width: video.videoWidth, height: video.videoHeight,
  });
  return <div className="application">
    <a className="skip-link" href="#live-main">Skip to live analyzer</a>
    <header className="topbar"><WorkspaceBrand /><WorkspaceNavigation active="live" /></header>
    <main id="live-main" className={styles.main}>
      <div className="workspace-heading"><div>
        <div className="eyebrow">SPATIAL + FREQUENCY DOMAIN</div>
        <h1>Live Signal Analyzer</h1>
        <p>Explore webcam frames as image signals.</p>
      </div></div>
      <div className={styles.toolbar}>
        <span role="status">{active ? "Camera active" : cameraStatus === "requesting" ? "Waiting for camera permission…" : "Camera stopped"}</span>
        <button className="button primary" onClick={() => void start()} disabled={active || cameraStatus === "requesting"}>
          <Camera size={15} /> Start camera
        </button>
        <button className="button" onClick={stop} disabled={!active && cameraStatus !== "requesting"}>
          <Square size={14} /> Stop camera
        </button>
      </div>
      {cameraMessage && <p className="analysis-error" role="alert">{cameraMessage}</p>}
      <section className={styles.controls} aria-label="Live processing controls">
        <label>Detector<select value={settings.detector} onChange={(event) => update("detector", event.target.value as Detector)}>
          {["Sobel", "Prewitt", "Laplacian"].map((detector) => <option key={detector}>{detector}</option>)}
        </select></label>
        <label>Gaussian sigma<input type="number" min={0} max={5} step={0.1} value={Number.isNaN(settings.sigma) ? "" : settings.sigma}
          onChange={(event) => update("sigma", event.target.valueAsNumber)} /></label>
        <label>Kernel size<select value={settings.kernel_size} onChange={(event) => update("kernel_size", Number(event.target.value))}>
          {KERNEL_SIZES.map((size) => <option key={size} value={size}>{size} × {size}</option>)}
        </select></label>
        <label>{thresholdLabel(settings.detector)}<input type="number" min={0} max={1443} step={1}
          value={Number.isNaN(settings.threshold) ? "" : settings.threshold}
          onChange={(event) => update("threshold", event.target.valueAsNumber)} /></label>
      </section>
      {invalid && <p className="analysis-error" role="alert">{invalid}</p>}
      {active && (analysis.error || displayError) && <div className={styles.error} role="alert">
        <span>{analysis.error || displayError}</span>
        <button className="button" onClick={analysis.retry}>Retry processing</button>
      </div>}
      <div className={styles.content}>
        <div className={styles.inputColumn}>
        <section className={`instrument ${styles.preview}`}>
          <header className="instrument-header"><h2>Camera preview</h2><span className="tag">INPUT</span></header>
          <video ref={videoRef} autoPlay muted playsInline aria-label="Live webcam preview" className={active ? "" : styles.hidden}
            onLoadedMetadata={(event) => readResolution(event.currentTarget)} onResize={(event) => readResolution(event.currentTarget)} />
          {!active && <p className={styles.placeholder}>Start the camera to see a preview.</p>}
          <footer className="instrument-footer">Source preview · analysis fits within 256 × 256</footer>
        </section>
        <section className={`instrument ${styles.monitor}`} aria-labelledby="live-monitor-title">
          <header className="instrument-header"><h2 id="live-monitor-title">Live Signal Monitor</h2><span className="tag">LIVE</span></header>
          <div className={styles.monitorBody}>
            <div className={styles.readouts}>
              <section aria-labelledby="signal-status-title">
                <h3 id="signal-status-title">Signal Status</h3>
                <dl className={styles.readout}>
                  <div><dt>Camera</dt><dd>{active ? "Active" : "Inactive"}</dd></div>
                  <div><dt>Frame resolution</dt><dd>{frameResolution}</dd></div>
                  <div><dt>Processing status</dt><dd role="status">{processingStatus}</dd></div>
                </dl>
              </section>
              <section aria-labelledby="processing-parameters-title">
                <h3 id="processing-parameters-title">Current Processing Parameters</h3>
                <dl className={styles.readout}>
                  <div><dt>Selected detector</dt><dd>{settings.detector}</dd></div>
                  <div><dt>Gaussian sigma</dt><dd>{Number.isFinite(settings.sigma) ? settings.sigma : "--"}</dd></div>
                  <div><dt>Kernel size</dt><dd>{settings.kernel_size} × {settings.kernel_size}</dd></div>
                  <div><dt>{thresholdLabel(settings.detector)}</dt><dd>{Number.isFinite(settings.threshold) ? settings.threshold : "--"}</dd></div>
                </dl>
              </section>
              <section aria-labelledby="performance-title">
                <h3 id="performance-title">Performance Metrics</h3>
                <dl className={styles.readout}>
                  <div><dt>Backend processing</dt><dd>{metrics ? `${metrics.processing.toFixed(1)} ms` : "--"}</dd></div>
                  <div><dt>Processing latency</dt><dd>{metrics ? `${metrics.latency.toFixed(0)} ms` : "--"}</dd></div>
                  <div><dt>Approximate FPS</dt><dd>{metrics && metrics.fps > 0 ? metrics.fps.toFixed(1) : "--"}</dd></div>
                </dl>
                <p className={styles.monitorNote}>Latency includes capture, request and display. FPS measures processed output.</p>
              </section>
            </div>
            <section className={styles.pipeline} aria-labelledby="pipeline-title">
              <h3 id="pipeline-title">Signal Pipeline</h3>
              <ol>
                {["Camera Frame", "Grayscale Conversion", "Gaussian Filtering", "Edge Detector", "Fourier Spectrum"].map((stage, index) =>
                  <li key={stage}>{index > 0 && <span className={styles.flowArrow} aria-hidden="true">↓</span>}<span className={styles.flowNode}>{stage}</span></li>)}
              </ol>
              <p className={styles.monitorNote}>Fourier spectrum uses grayscale input.</p>
            </section>
          </div>
        </section>
        </div>
        <div className={styles.outputs} aria-label="Synchronized signal outputs">
          {TITLES.map((title, index) => <section className="instrument" key={title}>
            <header className="instrument-header"><h2>{title}</h2></header>
            <div className={styles.surface}>
              <canvas ref={(element) => { canvases.current[index] = element; }} role="img" aria-label={`${title} of the same captured frame`}
                className={current ? "" : styles.hidden} />
              {!current && <p className={styles.placeholder}>{active ? "Waiting for a processed frame…" : "No live signal"}</p>}
            </div>
            <footer className="instrument-footer">{index === 3 ? "Grayscale input · centered log magnitude"
              : index === 2 ? `${current?.settings.detector ?? settings.detector} · ${settings.detector === "Laplacian" ? "zero crossings" : "magnitude threshold"}`
              : index === 1 ? "Gaussian smoothing" : "Grayscale intensity"}</footer>
          </section>)}
        </div>
      </div>
      <p className={styles.note}>Frames are sent to your configured Python backend for processing and kept in memory.
        Stop camera ends capture and releases the webcam. The four outputs share one frame; the camera preview runs independently.</p>
    </main>
  </div>;
}
