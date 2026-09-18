"use client";

import { useEffect, useState, type RefObject } from "react";
import { createLiveRunner, LiveRequestError } from "../lib/live-api.ts";
import { frameDimensions, validateLiveSettings, type LiveResult, type LiveSettings } from "../lib/live.ts";

export function useLiveAnalysis(videoRef: RefObject<HTMLVideoElement | null>, stream: MediaStream | null, settings: LiveSettings) {
  const [runner] = useState(createLiveRunner);
  const [result, setResult] = useState<{ frame: LiveResult; stream: MediaStream; roundTrip: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    if (!stream || validateLiveSettings(settings)) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    let lastVideoTime = -1;
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    const capture = async () => {
      const video = videoRef.current;
      if (disposed) return null;
      if (!context) throw new Error("Your browser could not create the frame capture canvas.");
      if (!video || video.readyState < 2 || !video.videoWidth || video.currentTime === lastVideoTime) return null;
      lastVideoTime = video.currentTime;
      const size = frameDimensions(video.videoWidth, video.videoHeight);
      canvas.width = size.width;
      canvas.height = size.height;
      context.drawImage(video, 0, 0, size.width, size.height);
      return new Promise<Blob>((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob)
        : reject(new Error("Could not capture a camera frame.")), "image/png"));
    };
    const tick = async () => {
      if (disposed) return;
      let delay = 50;
      let failed = false;
      const started = performance.now();
      await runner.run(capture, settings, {
        success: (frame) => {
          if (disposed) return;
          setError(null);
          setResult({ frame, stream, roundTrip: performance.now() - started });
        },
        error: (cause) => {
          if (disposed) return;
          if (cause instanceof LiveRequestError && cause.status === 429) delay = 500;
          else failed = true;
          setError(cause.message);
        },
      });
      if (!disposed && !failed) timer = setTimeout(tick, delay);
    };
    timer = setTimeout(tick, 0);
    return () => { disposed = true; clearTimeout(timer); runner.cancel(); };
  }, [videoRef, stream, settings, runner, retry]);
  return { result, error, retry: () => { setError(null); setRetry((value) => value + 1); } };
}
