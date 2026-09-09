"use client";

import { useCallback, useEffect, useState } from "react";
import { createCutoutRunner } from "@/lib/cutout-api";
import { DEFAULT_CUTOUT_SETTINGS, resolveCutoutRectangle, validateCutoutSettings, validateSelection,
  type BrushMark, type CutoutResult, type CutoutSettings, type PreparedImage, type Rectangle } from "@/lib/cutout";
import { validateImage } from "@/lib/workspace";

export function useCutout() {
  const [runner] = useState(createCutoutRunner);
  const [prepared, setPrepared] = useState<PreparedImage | null>(null);
  const [selectedRectangle, setRectangle] = useState<Rectangle | null>(null);
  const [marks, setMarks] = useState<BrushMark[]>([]);
  const [settings, setSettings] = useState<CutoutSettings>({ ...DEFAULT_CUTOUT_SETTINGS });
  const rectangle = resolveCutoutRectangle(prepared, selectedRectangle, settings.method);
  const [result, setResult] = useState<CutoutResult | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<"empty" | "preparing" | "ready" | "extracting" | "outdated" | "success" | "error">("empty");
  const [error, setError] = useState("");
  useEffect(() => () => runner.cancel(), [runner]);

  const invalidate = useCallback(() => {
    runner.cancel();
    setResult(null);
    setError("");
    setStatus(prepared ? "outdated" : "empty");
  }, [runner, prepared]);

  const changeSettings = (next: CutoutSettings) => {
    invalidate();
    setSettings({ ...next });
  };

  const upload = useCallback((next?: File) => {
    if (!next) return;
    invalidate();
    setPrepared(null);
    setRectangle(null);
    setMarks([]);
    setFile(next);
    const invalid = validateImage(next);
    if (invalid) { setError(invalid); setStatus("error"); return; }
    setStatus("preparing");
    void runner.prepare(next, {
      success: (image) => { setPrepared(image); setStatus("ready"); },
      error: (message) => { setError(message); setStatus("error"); },
    });
  }, [invalidate, runner]);

  const select = (next: Rectangle) => {
    invalidate();
    setMarks([]);
    if (next.width < 2 || next.height < 2) {
      setRectangle(null);
      setError("Drag a rectangle at least 2 × 2 pixels around the object.");
    } else setRectangle(next);
  };
  const addMark = (mark: BrushMark) => {
    invalidate();
    const next = [...marks, mark];
    const invalid = prepared ? validateSelection(prepared, rectangle, next) : "Prepare an image first.";
    if (invalid) setError(invalid);
    else setMarks(next);
  };
  const undo = () => { invalidate(); setMarks((previous) => previous.slice(0, -1)); };
  const clearMarks = () => { invalidate(); setMarks([]); };
  const reset = () => { invalidate(); setRectangle(null); setMarks([]); };
  const extract = () => {
    if (!prepared || !rectangle || status === "extracting") return;
    const invalid = validateSelection(prepared, rectangle, marks) || validateCutoutSettings(settings);
    if (invalid) { setError(invalid); return; }
    setResult(null);
    setError("");
    setStatus("extracting");
    void runner.extract({ prepared, rectangle, marks, settings }, {
      success: (output) => { setResult(output); setStatus("success"); },
      error: (message) => { setError(message); setStatus("error"); },
    });
  };
  return { prepared, rectangle, marks, settings, changeSettings, result, status, error, file,
    upload, invalidate, select, addMark, undo, clearMarks, reset, extract, setError };
}
