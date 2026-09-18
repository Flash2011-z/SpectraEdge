"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { cameraError, createCameraSession } from "../lib/live.ts";

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [status, setStatus] = useState<"idle" | "requesting" | "active" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const generation = useRef(0);
  const [session] = useState(() => createCameraSession(async () => {
    if (!navigator.mediaDevices?.getUserMedia)
      throw new Error("Camera access requires HTTPS or localhost and a browser with webcam support.");
    return navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: 640 }, height: { ideal: 480 } } });
  }));
  const release = useCallback(() => {
    generation.current++;
    session.stop();
  }, [session]);
  const stop = useCallback(() => {
    release();
    if (videoRef.current) videoRef.current.srcObject = null;
    setStream(null);
    setStatus("idle");
    setError(null);
  }, [release]);
  const start = useCallback(async () => {
    const current = ++generation.current;
    setStatus("requesting");
    setError(null);
    try {
      const next = await session.start();
      if (!next || current !== generation.current) return;
      const video = videoRef.current;
      if (!video) { session.stop(); return; }
      video.srcObject = next;
      await video.play();
      if (current !== generation.current) return;
      setStream(next);
      setStatus("active");
    } catch (cause) {
      if (current !== generation.current) return;
      session.stop();
      if (videoRef.current) videoRef.current.srcObject = null;
      setStream(null);
      setStatus("error");
      setError(cameraError(cause));
    }
  }, [session]);
  useEffect(() => {
    if (!stream) return;
    const ended = () => { stop(); setStatus("error"); setError("Camera disconnected. Reconnect it and start again."); };
    const tracks = stream.getVideoTracks();
    tracks.forEach((track) => track.addEventListener("ended", ended));
    return () => tracks.forEach((track) => track.removeEventListener("ended", ended));
  }, [stream, stop]);
  useEffect(() => {
    const video = videoRef.current;
    return () => { release(); if (video) video.srcObject = null; };
  }, [release]);
  return { videoRef, stream, status, error, start, stop };
}
