"use client";
/* PNG data URLs are local response artifacts, not remote optimized images. */
/* eslint-disable @next/next/no-img-element */

import { useRef, useState } from "react";
import { Check, CircleMinus, CirclePlus, Download, ImagePlus, Info, LoaderCircle,
  MousePointer2, RotateCcw, Scissors, ShieldCheck, SquareDashed, Undo2, Upload } from "lucide-react";
import { CUTOUT_METHOD_LABELS, DEFAULT_CUTOUT_SETTINGS, MAX_PREPARED_SIDE, type CutoutTool } from "@/lib/cutout";
import { KERNEL_SIZES } from "@/lib/workspace";
import { CutoutCanvas } from "./cutout-canvas";
import { useCutout } from "./use-cutout";
import { downloadFile, RangeControl } from "./ui";
import { WorkspaceBrand, WorkspaceNavigation } from "./workspace-navigation";
import styles from "./cutout.module.css";

export default function CutoutWorkspace() {
  const state = useCutout();
  const { prepared, rectangle, marks, settings, result, status, error } = state;
  const input = useRef<HTMLInputElement>(null);
  const [tool, setTool] = useState<CutoutTool>("rectangle");
  const [size, setSize] = useState(20);
  const [drawing, setDrawing] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<"cutout" | "mask" | "guidance">("cutout");
  const [background, setBackground] = useState<"checker" | "light" | "dark">("checker");
  const [trim, setTrim] = useState(false);
  const busy = status === "preparing" || status === "extracting";
  const edgeGuided = settings.method === "edge-watershed";
  const aiAssisted = settings.method === "ai-assisted";
  const activePreview = !edgeGuided && preview === "guidance" ? "cutout" : preview;
  const upload = (file?: File) => { setTool("rectangle"); setDrawing(false); state.upload(file); };
  const previewUrl = result ? activePreview === "guidance" ? result.guidance_image
    : activePreview === "mask" ? result.mask_image : trim ? result.cropped_image : result.cutout_image : null;
  const previewSize = result && activePreview === "cutout" && trim ? result.cropped_dimensions : prepared;
  const message = status === "preparing" ? "Preparing the colour photo…"
    : status === "extracting" ? aiAssisted ? "Running local AI segmentation… The first run also loads the model and may take longer." : "Separating foreground and background…"
    : status === "success" ? "Cutout ready. Inspect the boundary and refine if needed."
    : status === "outdated" ? "Selection or settings changed. Extract again before downloading."
    : prepared ? aiAssisted ? "AI mode selected. Extract directly, or draw a rectangle to limit the retained area. No brush is required."
      : edgeGuided ? "Draw a rectangle around the object, leave a background margin, then click Extract object. No brush needed to start."
      : "Draw a rectangle around the object, leaving some background outside."
    : "Upload a colour photo to begin. AI is optional and runs only when you select AI-assisted cutout and extract.";

  return <div className={`application ${styles.application}`}
    onDragOver={(event) => { event.preventDefault(); if (event.dataTransfer.types.includes("Files")) setDragging(true); }}
    onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node)) setDragging(false); }}
    onDrop={(event) => { event.preventDefault(); setDragging(false); upload(event.dataTransfer.files[0]); }}>
    <a className="skip-link" href="#cutout-main">Skip to Object Cutout</a>
    <header className="topbar"><WorkspaceBrand /><WorkspaceNavigation active="cutout" />
      <span className={styles.extensionBadge}><Scissors size={13} /> OBJECT CUTOUT</span>
    </header>
    <main id="cutout-main" className={styles.main}>
      <div className={styles.heading}>
        <div><span className="eyebrow">INTERACTIVE FOREGROUND SEGMENTATION</span>
          <h1>Object cutout<span>.</span></h1><p>Your photo. A precise selection. A transparent result.</p></div>
        <span className={styles.libraryTag}>{aiAssisted ? "Optional pretrained AI · local CPU" : edgeGuided ? "Manual Gaussian + Sobel · library watershed" : "OpenCV GrabCut · comparison method"}</span>
      </div>
      <div className={`${styles.notice} ${error ? styles.error : ""}`} role={error ? "alert" : "status"}>
        {busy ? <LoaderCircle className="spin" size={16} /> : result ? <Check size={16} /> : <Info size={16} />}
        <span>{error || message}</span>
        {error && !prepared && state.file && <button className="text-button" onClick={() => upload(state.file!)}>Retry upload</button>}
      </div>
      <div className={styles.layout}>
        <aside className={styles.tools} aria-label="Cutout tools">
          <section><span className="eyebrow">01 / SOURCE PHOTO</span>
            <button className={styles.upload} onClick={() => input.current?.click()}>
              <Upload size={22} strokeWidth={1.4} /><strong>Choose a colour photo</strong><span>or drop it anywhere here</span>
              <small>PNG, JPEG, WebP, GIF · up to 20 MiB</small>
            </button>
            <input ref={input} type="file" hidden aria-label="Upload a cutout photo" accept="image/png,image/jpeg,image/webp,image/gif"
              onChange={(event) => { upload(event.target.files?.[0]); event.target.value = ""; }} />
            {state.file && <p className={styles.filename} title={state.file.name}>{state.file.name}</p>}
            {prepared && <p className={styles.hint}>Prepared {prepared.width} × {prepared.height} px<br />Source {prepared.source_dimensions.width} × {prepared.source_dimensions.height} px</p>}
            <p className={styles.hint}>Prepared at up to {MAX_PREPARED_SIDE} px per side before selection. Exports use this resolution.</p>
          </section>
          <section><span className="eyebrow">02 / SEGMENTATION METHOD</span>
            <label className={styles.field}>Method
              <select value={settings.method} disabled={status === "preparing" || drawing}
                onChange={(event) => state.changeSettings(event.target.value === "ai-assisted" ? { method: "ai-assisted" }
                  : event.target.value === "grabcut" ? { method: "grabcut" } : { ...DEFAULT_CUTOUT_SETTINGS })}>
                <option value="edge-watershed">Edge-guided watershed</option>
                <option value="grabcut">GrabCut — comparison</option>
                <option value="ai-assisted">AI-assisted cutout — optional</option>
              </select>
            </label>
            {settings.method === "edge-watershed" && <>
              <RangeControl label="Gaussian sigma" help="Manual Gaussian smoothing before Sobel. Zero skips smoothing. Changing this requires extraction again."
                min={0} max={5} step={0.1} value={settings.sigma} disabled={status === "preparing" || drawing}
                onChange={(sigma) => state.changeSettings({ ...settings, sigma })} />
              <label className={styles.field}>Kernel size
                <select value={settings.kernel_size} disabled={status === "preparing" || drawing}
                  onChange={(event) => state.changeSettings({ ...settings, kernel_size: Number(event.target.value) })}>
                  {KERNEL_SIZES.map((value) => <option key={value} value={value}>{value} × {value}</option>)}
                </select>
              </label>
            </>}
            <p className={styles.hint}>{aiAssisted ? "BiRefNet portrait model, running on your Python backend. Best for people. No manual Gaussian/Sobel or cloud image service is used. Other methods never switch to AI automatically."
              : edgeGuided ? "Background colours identify starting regions; your Sobel edges guide the boundary. Best with a fairly uniform background. Brushes are optional."
              : "Colour models and graph cuts. Gaussian/Sobel are not used by this comparison method."}</p>
          </section>
          <section><span className="eyebrow">03 / SELECT & REFINE</span>
            <div className={styles.toolButtons} role="group" aria-label="Selection tool">
              <button disabled={!prepared || drawing} className={`${tool === "rectangle" ? styles.selected : ""} ${styles.rectangle}`}
                aria-pressed={tool === "rectangle"} onClick={() => setTool("rectangle")}><SquareDashed size={16} />Rectangle</button>
            </div>
            <p className={styles.hint}>{aiAssisted ? "The model sees the whole photo. A rectangle limits the output; it is not an AI prompt. Reset selection uses the whole image again. A new rectangle clears marks."
              : "Draw around the object with a little background inside the box. A new rectangle clears old brush marks."}</p>
            <details className={styles.corrections} onToggle={(event) => { if (!event.currentTarget.open) setTool("rectangle"); }}>
              <summary>Optional brush corrections{marks.length ? ` (${marks.length})` : ""}</summary>
              <div className={styles.toolButtons} role="group" aria-label="Optional correction tool">
              {([
                { value: "keep", label: "Keep brush", icon: CirclePlus },
                { value: "remove", label: "Remove brush", icon: CircleMinus },
              ] as const).map(({ value, label, icon: Icon }) => <button key={value} disabled={!prepared || drawing || !rectangle}
                className={`${tool === value ? styles.selected : ""} ${styles[value]}`}
                aria-pressed={tool === value} onClick={() => setTool(value)}><Icon size={16} />{label}</button>)}
            </div>
            <p className={styles.hint}>{aiAssisted ? "AI corrections paint transparency directly: Keep restores original opacity; Remove makes pixels transparent. They do not retrain or guide the model. Later marks win."
              : tool === "rectangle" ? "Use these only if the automatic result needs correction. Keep identifies your intended foreground; Remove rejects unwanted areas."
              : tool === "keep" ? "Paint missed areas green—even outside the rectangle. Transparent source pixels stay transparent."
              : "Paint unwanted areas red. Later marks override earlier marks."}</p>
            <RangeControl label="Brush size" help="Brush diameter in prepared-image pixels. Changing size affects the next stroke only."
              min={1} max={128} value={size} unit=" px" disabled={!prepared || drawing} onChange={setSize} />
            <div className={styles.editActions}>
              <button className="button" disabled={!marks.length || drawing} onClick={state.undo}><Undo2 size={13} />Undo mark</button>
              <button className="button" disabled={!marks.length || drawing} onClick={() => { state.clearMarks(); setTool("rectangle"); }}>Clear marks</button>
            </div>
            <p className={styles.hint}>{marks.length} brush marks · changes require extraction. Clear all marks to return to automatic selection.</p>
            </details>
            <div className={styles.editActions}>
              <button className="button" disabled={!prepared || drawing} onClick={() => { state.reset(); setTool("rectangle"); }}><RotateCcw size={13} />Reset selection</button>
            </div>
          </section>
          <section><button className={`button primary ${styles.full}`} onClick={state.extract} disabled={!prepared || !rectangle || busy || drawing}>
            {status === "extracting" ? <LoaderCircle className="spin" size={15} /> : <Scissors size={15} />}
            {status === "extracting" ? "Extracting…" : marks.length ? "Refine cutout" : "Extract object"}
          </button>
            <p className={styles.hint}>{aiAssisted ? "The optional model loads only on an AI extraction. If unavailable, choose either non-AI method. Nothing switches on your behalf."
              : edgeGuided ? marks.some((mark) => mark.mode === "keep")
              ? "Your Keep marks identify foreground. Clear marks to return to automatic colour + edge seeds."
              : "Include background inside the box. Objects may touch the photo edge. No Keep stroke is required."
              : "Five GrabCut iterations on the original prepared colour image."}</p>
          </section>
          <div className={styles.privacy}><ShieldCheck size={16} /><p>Choosing a photo sends it to your configured Python backend. Processing is in memory; no photos are saved.</p></div>
        </aside>

        <section className={styles.panel} aria-label="Selection editor">
          <header><div><span className="panel-index">01</span><h2>Original photo</h2></div><span className={styles.panelTag}>SELECTION EDITOR</span></header>
          <div className={styles.editor}>
            {prepared ? <CutoutCanvas key={prepared.request_id} prepared={prepared} rectangle={rectangle} marks={marks} tool={tool} size={size}
              onBegin={state.invalidate} onRectangle={state.select} onMark={state.addMark} onDrawing={setDrawing} onError={state.setError} />
              : <div className={styles.empty}><ImagePlus size={36} strokeWidth={1} /><strong>{status === "preparing" ? "Preparing your photo…" : "Start with your original photo"}</strong>
                <p>Draw around a person or object.<br />Keep colour and detail from the source.</p></div>}
          </div>
          <footer><MousePointer2 size={12} /><span>{rectangle ? `x ${rectangle.x} · y ${rectangle.y} · ${rectangle.width} × ${rectangle.height} px` : "Mouse or touch · drag to select"}</span></footer>
          <div className={styles.legend}><span><i className={styles.keepDot} />Keep foreground</span><span><i className={styles.removeDot} />Remove background</span></div>
        </section>

        <section className={styles.panel} aria-label="Extracted object preview">
          <header><div><span className="panel-index">02</span><h2>Extracted object</h2></div><span className={styles.resultMethod}>{result ? CUTOUT_METHOD_LABELS[result.method] : "NO RESULT"}</span></header>
          {result?.method === "edge-watershed" && <p className={styles.resultParameters}>{result.seed_mode === "automatic" ? "Automatic colour + edge seeds" : "Brush-guided seeds"} · σ {result.parameters_used.sigma.toFixed(1)} · {result.parameters_used.kernel_size} × {result.parameters_used.kernel_size} kernel</p>}
          {result?.method === "ai-assisted" && <p className={styles.resultParameters}>Pretrained AI · {result.parameters_used.model} · local CPU · soft mask</p>}
          <div className={styles.previewControls}>
            <div className={styles.toggle} role="group" aria-label="Preview output">
              {(["cutout", "mask", "guidance"] as const).map((value) => <button key={value} aria-pressed={activePreview === value}
                disabled={value === "guidance" && !edgeGuided}
                className={activePreview === value ? styles.active : ""} onClick={() => setPreview(value)}>
                {value === "cutout" ? "Cutout" : value === "mask" ? "Mask" : "Edge guidance"}</button>)}
            </div>
            <div className={styles.backgrounds} role="group" aria-label="Preview background">
              {(["checker", "light", "dark"] as const).map((value) => <button key={value} className={styles[value]}
                aria-label={`${value === "checker" ? "Checkerboard" : value === "light" ? "Light" : "Dark"} background`}
                title={`${value} preview only`} aria-pressed={background === value} onClick={() => setBackground(value)} />)}
            </div>
          </div>
          <div className={`${styles.preview} ${styles[background]}`}>
            {previewUrl ? <img key={previewUrl} src={previewUrl} alt={activePreview === "guidance" ? "Manual Sobel magnitude used as watershed elevation, brighter means stronger gradient"
              : activePreview === "mask" ? "Computed foreground mask, white means retained" : "Extracted object with original colours and alpha transparency"}
              width={previewSize!.width} height={previewSize!.height} />
              : <div className={styles.empty}><Scissors size={34} strokeWidth={1} /><strong>{status === "extracting" ? "Extracting your selection…" : status === "outdated" ? "Extract again to update" : "Your cutout will appear here"}</strong>
                <p>{status === "outdated" ? "The previous result is no longer downloadable." : "Draw a rectangle, then choose Extract object."}</p></div>}
          </div>
          <footer><span>{activePreview === "guidance" ? "Sobel magnitude · bright = strong gradient"
            : activePreview === "mask" ? aiAssisted ? "Soft mask · white = retained, gray = partial" : "Binary mask · white = foreground" : "Alpha transparency · original RGB"}</span><span>{result ? `${result.processing_time.toFixed(0)} ms` : "—"}</span></footer>
          {activePreview === "guidance" && result?.method === "edge-watershed" && <p className={styles.guidanceNote}>
            Exact elevation visualized on a linear 0–{result.guidance_scale.max.toFixed(2)} scale. Watershed used full floating-point values, not this 8-bit preview. {result.guidance_scale.max === 0 && "No edges: the seeds alone divide this flat image."}
          </p>}
          <div className={styles.download}>
            <label><input type="checkbox" checked={trim} onChange={(event) => setTrim(event.target.checked)} />Trim empty space</label>
            <button className={`button ${styles.full}`} disabled={!result || busy || drawing}
              onClick={() => { if (result) downloadFile(trim ? result.cropped_image : result.cutout_image, `spectraedge-cutout${trim ? "-trimmed" : ""}.png`); }}>
              <Download size={14} />Download transparent PNG
            </button>
            <p className={styles.hint}>Backgrounds are preview-only. Download always exports the RGBA cutout, including in Mask or Edge guidance view.</p>
          </div>
        </section>
      </div>
      <p className={styles.explanation}><Info size={15} />{aiAssisted
        ? "AI-assisted cutout is an optional pretrained-model extension, not our manually implemented detector. It predicts a soft foreground mask and changes only source transparency. Photos stay on your configured Python backend; the app does not download models or send photos to a cloud AI service during extraction. Fine hair, multiple people and unfamiliar objects may still need corrections. The manual method remains the default."
        : edgeGuided
        ? "Our manual Gaussian and Sobel implementations generate the elevation map used by marker-controlled watershed. Automatic seeds use the dominant background colour and low-gradient interiors, not closed-edge nesting. Brushes can correct ambiguous selections. Seed discovery uses library helpers; watershed is library-backed. Similar object/background colours, texture and fine hair remain difficult. This is not object recognition or alpha matting. FFT is not used here."
        : "GrabCut is a separate, library-backed colour-model comparison. It does not use the manual Gaussian/Sobel elevation or FFT. Similar colours and fine hair may need more Keep/Remove marks."}</p>
    </main>
    {dragging && <div className={styles.dropOverlay}><Upload size={38} /><strong>Drop your colour photo</strong><span>It will replace the current cutout session.</span></div>}
  </div>;
}
