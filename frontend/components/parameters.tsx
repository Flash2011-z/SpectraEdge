"use client";

import { Camera, ChevronDown, FileImage, FlaskConical, Play, RotateCcw, Upload, LoaderCircle } from "lucide-react";
import { useState } from "react";
import { DETECTOR_DESCRIPTIONS, type Detector, type Parameters } from "@/lib/workspace";
import { useWorkspace } from "./workspace-provider";
import { Help, RangeControl } from "./ui";

export function ParametersPanel({ open, onCamera }: { open: boolean; onCamera: () => void }) {
  const { parameters: p, updateParameter: update, openImagePicker, loadImage, source, loadDemo, process, busy, reset, loadingImage } = useWorkspace();
  const [dragging, setDragging] = useState(false);
  return <aside className={`sidebar ${open ? "mobile-open" : ""}`} aria-label="Analysis parameters" id="parameters">
    <div className="sidebar-heading"><span className="eyebrow">CONFIGURATION</span><span className="mono">01—05</span></div>
    <section className="control-section"><h2><span>Input source</span><span className="section-number">01</span></h2>
      <button className={`upload-zone ${dragging ? "dragging" : ""}`} disabled={loadingImage} onClick={() => openImagePicker()} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); loadImage(e.dataTransfer.files[0]); }}>
        {loadingImage ? <LoaderCircle className="spin" size={23} /> : <Upload size={23} strokeWidth={1.4} />}
        <strong>{loadingImage ? "Opening image…" : "Choose an image"}</strong><span>or drag and drop it here</span><small>PNG, JPG, WEBP, GIF · up to 20 MB</small>
      </button>
      <button className="button subtle full" onClick={onCamera}><Camera size={13} />Camera input<span className="soon">SOON</span></button>
      <div className="source-chip"><FileImage size={12} /><span title={source.name}>{source.name}</span><span className="tag">{source.kind === "demo" ? "DEMO" : "LOCAL"}</span></div>
      <button className="text-button" onClick={loadDemo}><FlaskConical size={12} />Load calibration demo</button>
    </section>
    <section className="control-section"><h2><span>Edge detector</span><span className="section-number">02</span></h2>
      <div className="segmented" role="group" aria-label="Edge detector">{(["Sobel", "Prewitt", "Laplacian"] as Detector[]).map(name => <button key={name} className={p.detector === name ? "selected" : ""} aria-pressed={p.detector === name} onClick={() => update("detector", name)}>{name}</button>)}</div>
      <p className="control-hint">{DETECTOR_DESCRIPTIONS[p.detector]}</p>
    </section>
    <section className="control-section"><h2><span>Filtering</span><span className="section-number">03</span></h2>
      <RangeControl label="Gaussian sigma" help="Smoothing strength. Higher values will suppress more high-frequency noise." min={0} max={5} step={.1} value={p.sigma} onChange={v => update("sigma", v)} />
      <div className="select-row"><label htmlFor="kernel-size">Kernel size</label><Help text="The neighborhood used by the smoothing filter." /><span className="select-wrap"><select id="kernel-size" value={p.kernel} onChange={e => update("kernel", Number(e.target.value))}>{[3, 5, 7].map(n => <option key={n} value={n}>{n} × {n}</option>)}</select><ChevronDown size={12} aria-hidden="true" /></span></div>
    </section>
    <section className="control-section"><h2><span>Detection</span><span className="section-number">04</span></h2>
      <RangeControl label="Threshold" help="The future detection stage will reject gradient magnitudes below this threshold." min={0} max={255} value={p.threshold} onChange={v => update("threshold", v)} />
      <RangeControl label="Minimum area" help="Contours with a smaller enclosed area will be ignored." min={0} max={5000} step={50} unit=" px²" value={p.minimumArea} onChange={v => update("minimumArea", v)} />
    </section>
    <details className="control-section noise-section" open><summary><h2><span>Noise model</span><span className="section-number">05</span></h2><ChevronDown size={12} /></summary>
      <div className="segmented noise-selector" role="group" aria-label="Noise model">{(["None", "Gaussian", "Salt & Pepper"] as Parameters["noise"][]).map(name => <button key={name} className={p.noise === name ? "selected" : ""} aria-pressed={p.noise === name} onClick={() => update("noise", name)}>{name}</button>)}</div>
      <RangeControl label="Noise strength" help="The amount of synthetic noise to add in a future noise experiment." min={0} max={100} value={p.noiseStrength} unit="%" disabled={p.noise === "None"} onChange={v => update("noiseStrength", v)} />
    </details>
    <button className="button primary full process-button" onClick={process} disabled={busy || loadingImage}>{busy ? <LoaderCircle size={14} className="spin" /> : <Play size={14} fill="currentColor" />} {busy ? "Walking through demo…" : "Process image"}{!busy && <kbd>Space</kbd>}</button>
    <button className="button subtle full" onClick={reset}><RotateCcw size={12} />Reset parameters</button>
    <div className="sidebar-foot"><span className="status-dot" />Images stay in your browser.</div>
  </aside>;
}
