"use client";

import { ArrowUpRight, BoxSelect } from "lucide-react";
import { DEMO_OBJECTS } from "@/lib/workspace";
import { useWorkspace } from "./workspace-provider";

export function ObjectInformation() {
  const { source, selectedObject, setSelectedObject, preferences, result } = useWorkspace();
  const demo = source.kind === "demo" && preferences.demoVisuals;
  const objects = demo ? DEMO_OBJECTS : result?.object_list ?? [];
  const object = objects.find((item) => item.id === selectedObject) ?? objects[0] ?? null;
  const computed = !demo && result?.object_list !== null && result?.object_list !== undefined;
  const box = object?.bounding_box;
  return (
    <section className="instrument object-information">
      <header className="instrument-header">
        <div>
          <span className="panel-index">{demo ? "07" : "11"}</span>
          <h2>Object information</h2>
        </div>
        <span className="tag">{demo ? "DEMO VALUES" : computed ? "COMPUTED" : "NOT RUN"}</span>
      </header>
      <div className="object-body">
        <div className="object-overview">
          <div>
            <span className="object-count">{demo ? "04" : computed ? String(objects.length).padStart(2, "0") : "—"}</span>
            <span>{demo ? "objects detected (example)" : computed ? "connected foreground regions" : "object analysis has not run"}</span>
          </div>
          <BoxSelect size={23} strokeWidth={1.2} />
        </div>
        <div className="object-select">
          <span>Selected object</span>
          <div role="group" aria-label="Selected object">
            {objects.map(({ id }) => (
              <button
                key={id}
                disabled={!demo && !computed}
                className={object?.id === id ? "selected" : ""}
                aria-label={`Object 0${id}`}
                aria-pressed={object?.id === id}
                onClick={() => setSelectedObject(id)}
              >
                {String(id).padStart(2, "0")}
              </button>
            ))}
          </div>
        </div>
        <dl className="object-metrics">
          {[
            ["Area", object ? `${object.area.toLocaleString("en-US")} px²` : "—"],
            ["Perimeter", object ? `${object.perimeter} px` : "—"],
            ["Centroid", object ? `(${object.centroid.map((value) => Number(value.toFixed(2))).join(", ")})` : "—"],
            ["Bounding box", box ? `x ${box.x}, y ${box.y}, ${box.width} × ${box.height}` : "—"],
          ].map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      </div>
      <footer className="instrument-footer">
        <span>{demo ? "Select a contour to inspect" : computed ? "Eight-connected foreground components" : "Run a detector to analyze objects"}</span>
        <ArrowUpRight size={11} />
      </footer>
    </section>
  );
}
