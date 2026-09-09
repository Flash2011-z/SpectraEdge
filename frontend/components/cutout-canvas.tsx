"use client";

import { useEffect, useRef, useState, type PointerEvent } from "react";
import { imagePoint, MAX_STROKE_POINTS, rectangleFromPoints,
  type BrushMark, type CutoutTool, type Point, type PreparedImage, type Rectangle } from "@/lib/cutout";
import styles from "./cutout.module.css";

interface Gesture {
  id: number;
  tool: CutoutTool;
  size: number;
  points: Point[];
  end: Point;
}

export function CutoutCanvas({ prepared, rectangle, marks, tool, size, onBegin, onRectangle,
  onMark, onDrawing, onError }: {
  prepared: PreparedImage; rectangle: Rectangle | null; marks: BrushMark[];
  tool: CutoutTool; size: number; onBegin: () => void; onRectangle: (r: Rectangle) => void;
  onMark: (m: BrushMark) => void; onDrawing: (drawing: boolean) => void; onError: (message: string) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const gesture = useRef<Gesture | null>(null);
  const [photo, setPhoto] = useState<HTMLImageElement | null>(null);
  const [draft, setDraft] = useState<Gesture | null>(null);
  useEffect(() => {
    const image = new Image();
    image.onload = () => setPhoto(image);
    image.onerror = () => onError("The prepared preview could not be decoded. Upload the image again.");
    image.src = prepared.prepared_image;
    return () => { image.onload = null; image.onerror = null; };
  }, [prepared.prepared_image, onError]);

  useEffect(() => {
    const element = canvas.current;
    const ctx = element?.getContext("2d");
    if (!element || !ctx || !photo) return;
    ctx.clearRect(0, 0, prepared.width, prepared.height);
    ctx.drawImage(photo, 0, 0);
    const selection = draft?.tool === "rectangle" ? rectangleFromPoints(draft.points[0], draft.end) : rectangle;
    if (selection) {
      ctx.fillStyle = "rgba(0, 0, 0, 0.45)";
      ctx.beginPath();
      ctx.rect(0, 0, prepared.width, prepared.height);
      ctx.rect(selection.x, selection.y, selection.width, selection.height);
      ctx.fill("evenodd");
      ctx.strokeStyle = "#52c6e3";
      ctx.lineWidth = Math.max(1, prepared.width / element.getBoundingClientRect().width);
      ctx.strokeRect(selection.x, selection.y, selection.width, selection.height);
    }
    const allMarks = [...marks];
    if (draft && draft.tool !== "rectangle") allMarks.push({ mode: draft.tool, size: draft.size, points: draft.points });
    // Paint labels onto their own layer so the latest mark replaces earlier
    // colour, then blend that layer over the photo for a readable overlay.
    const overlay = document.createElement("canvas");
    overlay.width = prepared.width;
    overlay.height = prepared.height;
    const ink = overlay.getContext("2d")!;
    for (const mark of allMarks) {
      ink.fillStyle = ink.strokeStyle = mark.mode === "keep" ? "#4ade80" : "#fb7185";
      ink.lineWidth = mark.size;
      ink.lineCap = "round";
      ink.lineJoin = "round";
      ink.beginPath();
      mark.points.forEach((p, i) => i === 0 ? ink.moveTo(p.x + 0.5, p.y + 0.5) : ink.lineTo(p.x + 0.5, p.y + 0.5));
      ink.stroke();
      for (const p of [mark.points[0], mark.points[mark.points.length - 1]]) {
        ink.beginPath(); ink.arc(p.x + 0.5, p.y + 0.5, mark.size / 2, 0, Math.PI * 2); ink.fill();
      }
    }
    ctx.globalAlpha = 0.65;
    ctx.drawImage(overlay, 0, 0);
    ctx.globalAlpha = 1;
  }, [photo, prepared, rectangle, marks, draft]);

  const point = (event: PointerEvent<HTMLCanvasElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    return imagePoint({ x: event.clientX, y: event.clientY },
      { x: bounds.left, y: bounds.top, width: bounds.width, height: bounds.height }, prepared);
  };
  const move = (event: PointerEvent<HTMLCanvasElement>) => {
    const active = gesture.current;
    if (!active || active.id !== event.pointerId) return;
    const end = point(event);
    active.end = end;
    const last = active.points[active.points.length - 1];
    if (active.tool !== "rectangle" && (last.x !== end.x || last.y !== end.y) && active.points.length < MAX_STROKE_POINTS)
      active.points.push(end);
    setDraft({ ...active, points: [...active.points] });
  };
  const cancel = () => { gesture.current = null; setDraft(null); onDrawing(false); };
  return <div className={styles.canvasFrame}>
    {!photo && <span className={styles.canvasLoading}>Opening prepared image…</span>}
    <canvas ref={canvas} width={prepared.width} height={prepared.height}
      className={styles.canvas} tabIndex={0}
      aria-label={`Original photo. ${tool === "rectangle" ? "Drag a rectangle around the object" : `Paint ${tool} marks`}. Mouse and touch supported. Escape cancels a stroke.`}
      onKeyDown={(event) => { if (event.key === "Escape") cancel(); }}
      onPointerDown={(event) => {
        if (!photo || event.button !== 0 || !event.isPrimary || gesture.current) return;
        if (tool !== "rectangle" && !rectangle) { onError("Draw a rectangle before adding brush marks."); return; }
        event.preventDefault();
        onBegin();
        const start = point(event);
        gesture.current = { id: event.pointerId, tool, size, points: [start], end: start };
        setDraft({ ...gesture.current });
        onDrawing(true);
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={move}
      onPointerUp={(event) => {
        if (gesture.current?.id !== event.pointerId) return;
        move(event);
        const active = gesture.current;
        if (active.tool === "rectangle") onRectangle(rectangleFromPoints(active.points[0], active.end));
        else onMark({ mode: active.tool, size: active.size, points: [...active.points] });
        cancel();
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
      }}
      onPointerCancel={cancel}
      onLostPointerCapture={() => { if (gesture.current) cancel(); }}
    />
  </div>;
}
