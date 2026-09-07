"use client";

import { ArrowUpRight, BoxSelect } from "lucide-react";
import { DEMO_OBJECTS } from "@/lib/workspace";
import { useWorkspace } from "./workspace-provider";

export function ObjectInformation() {
  const { source, selectedObject, setSelectedObject, preferences } = useWorkspace();
  const demo = source.kind === "demo" && preferences.demoVisuals;
  const object = DEMO_OBJECTS.find(item => item.id === selectedObject)!;
  return <section className="instrument object-information">
    <header className="instrument-header"><div><span className="panel-index">07</span><h2>Object information</h2></div><span className="tag">{demo ? "DEMO VALUES" : "NO DATA"}</span></header>
    <div className="object-body"><div className="object-overview"><div><span className="object-count">{demo ? "04" : "—"}</span><span>objects detected</span></div><BoxSelect size={23} strokeWidth={1.2} /></div>
      <div className="object-select"><span>Selected object</span><div role="group" aria-label="Selected object">{[1,2,3,4].map(id => <button key={id} disabled={!demo} className={selectedObject === id && demo ? "selected" : ""} aria-label={`Object 0${id}`} aria-pressed={selectedObject === id && demo} onClick={() => setSelectedObject(id)}>0{id}</button>)}</div></div>
      <dl className="object-metrics">{[
        ["Area", `${object.area.toLocaleString("en-US")} px²`],
        ["Perimeter", `${object.perimeter} px`],
        ["Centroid", `(${object.centroid.join(", ")})`],
        ["Bounding box", object.bounding_box.join(" × ")],
      ].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{demo ? value : "—"}</dd></div>)}</dl>
    </div><footer className="instrument-footer"><span>{demo ? "Select a contour to inspect" : "Load the demo to explore"}</span><ArrowUpRight size={11} /></footer>
  </section>;
}
