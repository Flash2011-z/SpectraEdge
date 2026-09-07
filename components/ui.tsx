"use client";

import { useEffect, useId, useRef, type CSSProperties, type ReactNode } from "react";
import { HelpCircle, X } from "lucide-react";

export function IconButton({ label, children, onClick, disabled = false }: { label: string; children: ReactNode; onClick?: () => void; disabled?: boolean }) {
  return <button type="button" className="icon-button" aria-label={label} title={label} onClick={onClick} disabled={disabled}>{children}</button>;
}
export function Help({ text }: { text: string }) {
  return <span className="help" tabIndex={0} role="note" aria-label={text}><HelpCircle size={12} aria-hidden="true" /><span className="help-content">{text}</span></span>;
}
export function RangeControl({ label, help, value, min, max, step = 1, unit = "", disabled = false, onChange }: { label: string; help: string; value: number; min: number; max: number; step?: number; unit?: string; disabled?: boolean; onChange: (value: number) => void }) {
  const id = useId();
  return <div className={disabled ? "range-control disabled" : "range-control"}>
    <div className="slider-label"><label htmlFor={id}>{label}</label><Help text={help} /><output htmlFor={id}>{step < 1 ? value.toFixed(1) : value.toLocaleString("en-US")}{unit}</output></div>
    <input id={id} type="range" min={min} max={max} step={step} value={value} disabled={disabled} aria-valuetext={`${value}${unit}`} onChange={e => onChange(Number(e.target.value))} style={{ "--range-progress": `${(value - min) / (max - min) * 100}%` } as CSSProperties} />
  </div>;
}
export function Modal({ title, eyebrow, children, onClose, wide = false }: { title: string; eyebrow?: string; children: ReactNode; onClose: () => void; wide?: boolean }) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  useEffect(() => {
    const dialog = ref.current;
    dialog?.showModal();
    return () => dialog?.close();
  }, []);
  return <dialog ref={ref} className={wide ? "modal modal-wide" : "modal"} aria-labelledby={titleId} onCancel={e => { e.preventDefault(); onClose(); }} onClick={e => {
    if (e.target !== e.currentTarget) return;
    const rect = e.currentTarget.getBoundingClientRect();
    if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) onClose();
  }}>
    <header className="modal-header"><div>{eyebrow && <div className="eyebrow">{eyebrow}</div>}<h2 id={titleId}>{title}</h2></div><IconButton label="Close dialog" onClick={onClose}><X size={18} /></IconButton></header>{children}
  </dialog>;
}
export function downloadFile(url: string, name: string) {
  const anchor = document.createElement("a");
  anchor.href = url; anchor.download = name; anchor.click();
}
export function downloadJson(data: unknown, filename: string) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  downloadFile(url, filename);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
