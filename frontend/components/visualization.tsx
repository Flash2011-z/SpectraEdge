"use client";

/* Local blob URLs must stay client-side; no remote image optimization is involved. */
/* eslint-disable @next/next/no-img-element */

import { useEffect, useRef, useState } from "react";
import { Download, Focus, ImageOff, Maximize2, Minus, Plus, ScanLine, ZoomIn } from "lucide-react";
import { type Visualization } from "@/lib/workspace";
import { useWorkspace } from "./workspace-provider";
import { downloadFile, IconButton, Modal } from "./ui";

const TITLES: Record<Visualization, string> = { original: "Original image", filtered: "Filtered image", edges: "Edge map", gradient: "Gradient magnitude", contours: "Object contours", spectrum: "FFT spectrum" };
const DESCRIPTIONS: Record<Visualization, string> = {
  original: "Input / spatial domain", filtered: "Gaussian smoothing", edges: "Binary edge representation",
  gradient: "Spatial intensity variation", contours: "Connected object boundaries", spectrum: "Log magnitude / frequency domain",
};

// Static educational artwork only: no input pixels are processed by this renderer.
function drawIllustration(canvas: HTMLCanvasElement, kind: Visualization, selected: number, grid: boolean) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = kind === "original" || kind === "filtered" ? "#202c37" : "#0e1721";
  ctx.fillRect(0, 0, w, h);
  if (grid) {
    ctx.strokeStyle = "#5e7b8d12"; ctx.lineWidth = 1;
    for (let x = 0; x <= w; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let y = 0; y <= h; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  }
  if (kind === "spectrum") {
    const cx = w / 2, cy = h / 2;
    const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, h * .42);
    glow.addColorStop(0, "#d4faffe0"); glow.addColorStop(.035, "#70d7ed98"); glow.addColorStop(.2, "#43acca38"); glow.addColorStop(1, "#0e172100");
    ctx.fillStyle = glow; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#69d4e748"; ctx.lineWidth = 1;
    for (const radius of [18, 36, 65, 100, 146]) {
      ctx.beginPath(); ctx.ellipse(cx, cy, radius * 1.6, radius, -.35, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.strokeStyle = "#6fc3d562";
    ctx.beginPath(); ctx.moveTo(w * .1, cy); ctx.lineTo(w * .9, cy); ctx.moveTo(cx, h * .12); ctx.lineTo(cx, h * .88); ctx.stroke();
    const points = [[-.2, .18], [.2, -.18], [-.34, -.15], [.34, .15], [-.13, -.26], [.13, .26]];
    for (const [x, y] of points) {
      ctx.fillStyle = "#a6e0e7"; ctx.beginPath(); ctx.arc(cx + x * w, cy + y * h, 2.5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = "#7293a4"; ctx.font = "14px monospace"; ctx.fillText("fᵧ", cx + 10, 35); ctx.fillText("fₓ", w - 38, cy - 12);
  } else {
    const centers = [.18, .40, .62, .84];
    for (let i = 0; i < 4; i++) {
      const x = centers[i] * w, y = h * .49, size = h * .28;
      ctx.save(); ctx.translate(x, y);
      ctx.beginPath();
      if (i === 0) { ctx.rotate(-.19); ctx.rect(-size / 2, -size / 2, size, size); }
      if (i === 1) ctx.arc(0, 0, size * .57, 0, Math.PI * 2);
      if (i === 2) { ctx.moveTo(0, -size * .64); ctx.lineTo(size * .63, size * .5); ctx.lineTo(-size * .63, size * .5); ctx.closePath(); }
      if (i === 3) { ctx.arc(0, 0, size * .5, 0, Math.PI * 2); ctx.arc(0, 0, size * .28, 0, Math.PI * 2, true); }
      if (kind === "original" || kind === "filtered") {
        const fill = ctx.createLinearGradient(-size, -size, size, size);
        fill.addColorStop(0, "#ecf0ef"); fill.addColorStop(.5, "#acbbc6"); fill.addColorStop(1, "#657d8d");
        ctx.fillStyle = fill; ctx.shadowColor = "#0007"; ctx.shadowBlur = 16; ctx.shadowOffsetX = 10; ctx.shadowOffsetY = 14;
        if (kind === "filtered") { ctx.shadowBlur = 26; ctx.globalAlpha = .8; }
        ctx.fill();
      } else {
        ctx.strokeStyle = kind === "edges" ? "#cce5e9" : "#67cde2";
        ctx.lineWidth = kind === "gradient" ? 8 : 2;
        if (kind === "gradient") { ctx.shadowColor = "#4bbfdf"; ctx.shadowBlur = 24; }
        ctx.stroke();
      }
      ctx.restore();
      if (kind === "contours") {
        const chosen = selected === i + 1;
        ctx.strokeStyle = chosen ? "#71d9ed" : "#467788"; ctx.lineWidth = chosen ? 2 : 1;
        ctx.setLineDash(chosen ? [] : [6, 6]); ctx.strokeRect(x - size * .75, y - size * .8, size * 1.5, size * 1.6); ctx.setLineDash([]);
        ctx.fillStyle = chosen ? "#71d9ed" : "#7297a5"; ctx.font = "17px monospace"; ctx.fillText("0" + (i + 1), x - size * .75, y - size * .8 - 10);
        ctx.fillRect(x - 6, y, 12, 1); ctx.fillRect(x, y - 6, 1, 12);
      }
    }
    if (kind === "original" || kind === "filtered") {
      ctx.fillStyle = "#7c91a0"; ctx.font = "13px monospace"; ctx.fillText("GEOMETRIC CALIBRATION   /   04 SPECIMENS", 28, h - 24);
      ctx.strokeStyle = "#7c91a0"; ctx.lineWidth = 1;
      for (const x of [22, w - 22]) for (const y of [22, h - 22]) {
        ctx.beginPath(); ctx.moveTo(x - 6, y); ctx.lineTo(x + 6, y); ctx.moveTo(x, y - 6); ctx.lineTo(x, y + 6); ctx.stroke();
      }
    }
  }
}

export function VisualSurface({ kind, zoom = 1, interactive = false, exportRef }: { kind: Visualization; zoom?: number; interactive?: boolean; exportRef?: React.RefObject<HTMLCanvasElement | null> }) {
  const { source, preferences, selectedObject, setSelectedObject } = useWorkspace();
  const ownRef = useRef<HTMLCanvasElement>(null);
  const canvasRef = exportRef ?? ownRef;
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const demo = source.kind === "demo" && preferences.demoVisuals;
  const uploaded = source.kind === "image" && kind === "original";
  useEffect(() => {
    if (demo && canvasRef.current) drawIllustration(canvasRef.current, kind, selectedObject, preferences.grid);
  }, [demo, canvasRef, kind, selectedObject, preferences.grid]);
  const clampedPan = (x: number, y: number) => ({ x: Math.max(-1000, Math.min(1000, x)), y: Math.max(-1000, Math.min(1000, y)) });
  return <div className={`viewport visual-surface ${kind === "original" ? "source-surface" : ""} ${preferences.grid ? "" : "no-grid"} ${interactive && zoom > 1 ? "pannable" : ""}`}
    tabIndex={interactive ? 0 : undefined} role={interactive ? "region" : undefined} aria-label={interactive ? "Image inspector. Use arrow keys to pan when zoomed." : undefined}
    onKeyDown={e => {
      if (!interactive || zoom <= 1) return;
      const moves: Record<string, [number, number]> = { ArrowLeft: [25, 0], ArrowRight: [-25, 0], ArrowUp: [0, 25], ArrowDown: [0, -25] };
      if (moves[e.key]) { e.preventDefault(); const [x, y] = moves[e.key]; setPan(p => clampedPan(p.x + x, p.y + y)); }
    }}
    onPointerDown={e => {
      if (!interactive || zoom <= 1) return;
      drag.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
      e.currentTarget.setPointerCapture(e.pointerId);
    }}
    onPointerUp={e => { drag.current = null; if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId); }}
    onPointerCancel={() => { drag.current = null; }}
    onPointerMove={e => {
      if (drag.current) setPan(clampedPan(drag.current.px + e.clientX - drag.current.x, drag.current.py + e.clientY - drag.current.y));
      // The readout reports display coordinates, never invented pixel intensities.
      if (interactive) { const rect = e.currentTarget.getBoundingClientRect(); setCursor({ x: Math.round(e.clientX - rect.left), y: Math.round(e.clientY - rect.top) }); }
    }} onPointerLeave={() => setCursor(null)}>
    {demo ? <canvas ref={canvasRef} width={960} height={480} className="sample-canvas" role="img" aria-label={`Illustrative ${TITLES[kind].toLowerCase()} of four calibration shapes; not a calculated result.`} style={{ transform: `translate(${pan.x}px,${pan.y}px) scale(${zoom})` }} /> :
      uploaded ? <img className="local-image" src={source.url} alt={source.name} style={{ transform: `translate(${pan.x}px,${pan.y}px) scale(${zoom})` }} draggable={false} /> :
      <div className="empty-state"><ImageOff size={24} strokeWidth={1.25} /><strong>{source.kind === "image" ? "Awaiting processing" : "No signal loaded"}</strong><span>{source.kind === "image" ? "Results will appear when analysis is connected." : "Enable demo visuals in Settings to explore."}</span></div>}
    <span className="viewport-label">{demo ? "ILLUSTRATIVE SAMPLE" : uploaded ? "LOCAL IMAGE" : "NO OUTPUT"}</span>
    {kind === "contours" && demo && !interactive && <div className="object-hotspots" role="group" aria-label="Select an object">{[1,2,3,4].map(id => <button key={id} aria-label={`Select object 0${id}`} aria-pressed={selectedObject === id} onClick={() => setSelectedObject(id)} />)}</div>}
    {interactive && cursor && <span className="cursor-readout">View x {cursor.x} / y {cursor.y}</span>}
    {!interactive && <span className="axis-label">{kind === "spectrum" ? "fₓ →" : "x →"}</span>}
  </div>;
}

export function VisualizationCard({ kind, index, large = false, detector, selected = false }: { kind: Visualization; index: string; large?: boolean; detector?: string; selected?: boolean }) {
  const { source, preferences, notify } = useWorkspace();
  const [inspect, setInspect] = useState(false);
  const [zoom, setZoom] = useState(1);
  const exportRef = useRef<HTMLCanvasElement>(null);
  const canView = source.kind === "demo" ? preferences.demoVisuals : kind === "original" && source.kind === "image";
  const save = () => {
    if (source.kind === "image" && kind === "original") { downloadFile(source.url, source.name); return; }
    if (!exportRef.current) return;
    downloadFile(exportRef.current.toDataURL("image/png"), `spectraedge-demo-${kind}.png`);
    notify("Illustrative demo image exported.");
  };
  return <>
    <section className={`instrument ${large ? "original" : ""} ${selected ? "stage-highlight" : ""}`}>
      <header className="instrument-header"><div><span className="panel-index">{index}</span><h2>{TITLES[kind]}</h2>{large && <span className="tag">{source.kind === "demo" ? "REFERENCE" : "SOURCE"}</span>}</div>
        <div className="card-tools"><IconButton label={`Inspect ${TITLES[kind].toLowerCase()}`} disabled={!canView} onClick={() => { setZoom(1.5); setInspect(true); }}><ZoomIn size={13} /></IconButton><IconButton label={`Expand ${TITLES[kind].toLowerCase()}`} disabled={!canView} onClick={() => { setZoom(1); setInspect(true); }}><Maximize2 size={13} /></IconButton></div>
      </header>
      <VisualSurface key={source.url + kind} kind={kind} />
      <footer className="instrument-footer"><span>{large ? source.name : DESCRIPTIONS[kind]}</span><span>{large ? `${source.width} × ${source.height}` : detector ?? (source.kind === "demo" ? "DEMO" : "—")}</span></footer>
    </section>
    {inspect && <Modal title={TITLES[kind]} eyebrow={source.kind === "demo" ? "DEMO INSPECTOR · ILLUSTRATIVE DATA" : "SOURCE INSPECTOR · LOCAL IMAGE"} wide onClose={() => setInspect(false)}>
      <div className="inspector-toolbar"><span className="mono muted">{source.name}</span><div className="toolbar-group"><IconButton label="Zoom out" disabled={zoom <= .5} onClick={() => setZoom(z => Math.max(.5, z - .25))}><Minus size={14} /></IconButton><output className="zoom-value">{Math.round(zoom * 100)}%</output><IconButton label="Zoom in" disabled={zoom >= 4} onClick={() => setZoom(z => Math.min(4, z + .25))}><Plus size={14} /></IconButton><IconButton label="Fit to view" onClick={() => setZoom(1)}><Focus size={15} /></IconButton><button className="button" onClick={save}><Download size={13} />Export image</button></div></div>
      <VisualSurface key={source.url + kind + zoom} kind={kind} zoom={zoom} interactive exportRef={exportRef} />
      <div className="inspector-footer"><ScanLine size={13} /><span>Drag or use arrow keys to pan when zoomed.</span><span>{source.kind === "demo" ? "Static demo artwork · not a computed output" : `${source.width} × ${source.height} pixels`}</span></div>
    </Modal>}
  </>;
}
