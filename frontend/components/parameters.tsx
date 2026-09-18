"use client";

import {
  Camera,
  ChevronDown,
  FileImage,
  FlaskConical,
  Play,
  RotateCcw,
  Upload,
  LoaderCircle,
} from "lucide-react";
import { useState } from "react";
import { KERNEL_SIZES, MAX_THRESHOLD, thresholdLabel, type Detector, type Parameters } from "@/lib/workspace";
import { useWorkspace } from "./workspace-provider";
import { Help, RangeControl } from "./ui";

export function ParametersPanel({ open, onCamera }: { open: boolean; onCamera: () => void }) {
  const {
    parameters: p,
    updateParameter: update,
    openImagePicker,
    loadImage,
    source,
    loadDemo,
    process,
    busy,
    reset,
    loadingImage,
    canProcess,
  } = useWorkspace();
  const suggestedKernel = Math.max(3, 2 * Math.ceil(3 * p.sigma) + 1);
  const [dragging, setDragging] = useState(false);
  return (
    <aside
      className={`sidebar ${open ? "mobile-open" : ""}`}
      aria-label="Analysis parameters"
      id="parameters"
    >
      <div className="sidebar-heading">
        <span className="eyebrow">CONFIGURATION</span>
        <span className="mono">01—05</span>
      </div>
      <section className="control-section">
        <h2>
          <span>Input source</span>
          <span className="section-number">01</span>
        </h2>
        <button
          className={`upload-zone ${dragging ? "dragging" : ""}`}
          disabled={loadingImage}
          onClick={() => openImagePicker()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            loadImage(e.dataTransfer.files[0]);
          }}
        >
          {loadingImage ? (
            <LoaderCircle className="spin" size={23} />
          ) : (
            <Upload size={23} strokeWidth={1.4} />
          )}
          <strong>{loadingImage ? "Opening image…" : "Choose an image"}</strong>
          <span>or drag and drop it here</span>
          <small>PNG, JPG, WEBP, GIF · up to 20 MB</small>
        </button>
        <button className="button subtle full" onClick={onCamera}>
          <Camera size={13} />
          Camera input<span className="soon">LIVE</span>
        </button>
        <div className="source-chip">
          <FileImage size={12} />
          <span title={source.name}>{source.name}</span>
          <span className="tag">{source.kind === "demo" ? "DEMO" : "LOCAL"}</span>
        </div>
        <button className="text-button" onClick={loadDemo}>
          <FlaskConical size={12} />
          Open illustrative example
        </button>
      </section>
      <section className="control-section">
        <h2>
          <span>Edge detector</span>
          <span className="section-number">02</span>
        </h2>
        <div className="segmented" role="group" aria-label="Edge detector">
          {(["Sobel", "Prewitt", "Laplacian"] as Detector[]).map((name) => (
            <button
              key={name}
              className={p.detector === name ? "selected" : ""}
              aria-pressed={p.detector === name}
              onClick={() => update("detector", name)}
            >
              {name}
            </button>
          ))}
        </div>
        <p className="control-hint">{p.detector === "Laplacian" ? "Signed second derivative with zero-crossing edge selection." : "Signed Gx and Gy with gradient magnitude edge selection."}</p>
      </section>
      <section className="control-section">
        <h2>
          <span>Filtering</span>
          <span className="section-number">03</span>
        </h2>
        <RangeControl
          label="Gaussian sigma"
          help="Smoothing strength. Higher values will suppress more high-frequency noise."
          min={0}
          max={5}
          step={0.1}
          value={p.sigma}
          onChange={(v) => update("sigma", v)}
        />
        <div className="select-row">
          <label htmlFor="kernel-size">Kernel size</label>
          <Help text="The neighborhood used by the smoothing filter." />
          <span className="select-wrap">
            <select
              id="kernel-size"
              value={p.kernel}
              onChange={(e) => update("kernel", Number(e.target.value))}
            >
              {KERNEL_SIZES.map((n) => (
                <option key={n} value={n}>
                  {n} × {n}
                </option>
              ))}
            </select>
            <ChevronDown size={12} aria-hidden="true" />
          </span>
        </div>
        <p className="control-hint">
          {p.sigma === 0 ? "Sigma 0 returns the analysis input unchanged, including any selected noise." : `Suggested support: ${suggestedKernel} × ${suggestedKernel} (about ±3σ).`}
        </p>
        {p.sigma > 0 && p.kernel !== suggestedKernel && (
          <button className="text-button" onClick={() => update("kernel", suggestedKernel)}>Use suggested kernel</button>
        )}
        <label className="select-row">
          <span>Multi-scale analysis</span>
          <input
            type="checkbox"
            checked={p.multiScale}
            onChange={(event) => update("multiScale", event.target.checked)}
          />
        </label>
        {p.multiScale && <>
          {p.scaleSigmas.map((sigma, index) => (
            <div className="select-row" key={index}>
              <label htmlFor={`scale-sigma-${index}`}>Scale {index + 1} sigma</label>
              <input
                id={`scale-sigma-${index}`}
                type="number"
                className="scale-input"
                min={0}
                max={5}
                step={0.1}
                value={sigma}
                onChange={(event) => {
                  const values = [...p.scaleSigmas];
                  values[index] = Number(event.target.value);
                  update("scaleSigmas", values);
                }}
              />
            </div>
          ))}
          <div className="select-row">
            <label htmlFor="scale-support">Required scale support</label>
            <span className="select-wrap">
              <select id="scale-support" value={p.scaleSupport}
                onChange={(event) => update("scaleSupport", Number(event.target.value))}>
                {p.scaleSigmas.map((_, index) => <option key={index + 1} value={index + 1}>{index + 1}</option>)}
              </select>
              <ChevronDown size={12} aria-hidden="true" />
            </span>
          </div>
          <p className="control-hint">Each scale starts from the same grayscale analysis input, including any selected noise. A fused edge needs support from at least {p.scaleSupport} of {p.scaleSigmas.length} scales.</p>
        </>}
      </section>
      <section className="control-section">
        <h2>
          <span>Detection</span>
          <span className="section-number">04</span>
        </h2>
        <RangeControl
          label={thresholdLabel(p.detector)}
          help={p.detector === "Laplacian"
            ? "Horizontal/vertical opposite-sign responses must differ by more than this value. Both endpoints are marked. An exact-zero centre is marked only between opposite-sign neighbours; wider zero plateaus are not bridged. Equality is excluded."
            : `Raw ${p.detector} magnitude units. A pixel is an edge only when sqrt(Gx² + Gy²) is strictly greater than this value.`}
          min={0}
          max={MAX_THRESHOLD}
          value={p.threshold}
          disabled={source.kind !== "image"}
          onChange={(v) => update("threshold", v)}
        />
        <p className="control-hint">{p.detector === "Laplacian"
          ? "Sign change and raw response difference > threshold · 0–1443. The maximum may retain strong crossings."
          : "Raw magnitude > threshold · 0–1443. Higher values keep stronger edges."} Process again after changing it.</p>
        <RangeControl
          label="Laplacian minimum component area"
          help="Before contour and object analysis, remove eight-connected Laplacian edge fragments smaller than this area. The raw zero-crossing edge map remains unchanged. One disables cleanup."
          min={1}
          max={100}
          step={1}
          unit=" px"
          value={p.laplacianMinimumComponentArea}
          disabled={source.kind !== "image" || p.detector !== "Laplacian"}
          onChange={(v) => update("laplacianMinimumComponentArea", v)}
        />
      </section>
      <details className="control-section noise-section" open>
        <summary>
          <h2>
            <span>Noise model</span>
            <span className="section-number">05</span>
          </h2>
          <ChevronDown size={12} />
        </summary>
        <div className="segmented noise-selector" role="group" aria-label="Noise model">
          {(["None", "Gaussian", "Salt & Pepper"] as Parameters["noise"][]).map((name) => (
            <button
              key={name}
              className={p.noise === name ? "selected" : ""}
              aria-pressed={p.noise === name}
              onClick={() => {
                update("noise", name);
                if (name === "Gaussian") update("noiseStrength", 12);
                if (name === "Salt & Pepper") update("noiseStrength", 0.12);
              }}
            >
              {name}
            </button>
          ))}
        </div>
        <RangeControl
          label="Noise strength"
          help={p.noise === "Salt & Pepper"
            ? "Probability that each pixel is replaced with black or white."
            : "Standard deviation of zero-mean Gaussian noise in intensity units."}
          min={0}
          max={p.noise === "Salt & Pepper" ? 1 : 100}
          step={p.noise === "Salt & Pepper" ? 0.01 : 1}
          value={p.noiseStrength}
          unit={p.noise === "Salt & Pepper" ? " probability" : " intensity σ"}
          disabled={p.noise === "None"}
          onChange={(v) => update("noiseStrength", v)}
        />
        <p className="control-hint">
          {p.noise === "None" ? "Noise is disabled." :
            `Units: ${p.noise === "Gaussian" ? "intensity standard deviation" : "pixel corruption probability"}.`}
        </p>
      </details>
      <button
        className="button primary full process-button"
        onClick={process}
        disabled={!canProcess}
      >
        {busy ? (
          <LoaderCircle size={14} className="spin" />
        ) : (
          <Play size={14} fill="currentColor" />
        )}{" "}
        {busy ? "Processing in Python…" : "Process image"}
        {!busy && <kbd>Space</kbd>}
      </button>
      <button className="button subtle full" onClick={reset}>
        <RotateCcw size={12} />
        Reset parameters
      </button>
      <div className="sidebar-foot">
        <span className="status-dot" />
        Process sends to Python · memory only.
      </div>
    </aside>
  );
}
