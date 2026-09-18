"use client";

/* Computed PNG data URLs are supplied by the local Python backend. */
/* eslint-disable @next/next/no-img-element */

import { useWorkspace } from "./workspace-provider";

export function MultiScaleResults() {
  const { result } = useWorkspace();
  const multi = result?.multi_scale;
  if (!multi) return null;
  const outputs = [
    ...multi.scales.map((scale) => ({
      key: `scale-${scale.sigma}`,
      title: `Scale σ=${scale.sigma}`,
      detail: `${scale.kernel_size} × ${scale.kernel_size} Gaussian · individual edge decision`,
      url: scale.edge_map,
    })),
    {
      key: "persistence",
      title: "Edge persistence",
      detail: `0–${multi.persistence_scale.max} supporting scales · linear grayscale`,
      url: multi.persistence_map,
    },
    {
      key: "fused",
      title: "Fused persistent edges",
      detail: `Persistence ≥ ${multi.support_count} · binary 0 or 255`,
      url: multi.fused_edge_map,
    },
  ];
  return <section aria-label="Multi-scale edge analysis">
    <div className="outputs-heading">
      <span className="eyebrow">MULTI-SCALE OUTPUTS</span>
      <span>{multi.sigmas.length} independent scales · support {multi.support_count}</span>
    </div>
    <div className="output-grid computed-grid">
      {outputs.map((output, index) => <section className="instrument" key={output.key}>
        <header className="instrument-header">
          <div>
            <span className="panel-index">M{String(index + 1).padStart(2, "0")}</span>
            <h2>{output.title}</h2>
          </div>
          <span className="tag">COMPUTED</span>
        </header>
        <div className="viewport visual-surface">
          <img className="local-image" src={output.url} alt={`${output.title} computed by Python`} draggable={false} />
          <span className="viewport-label">COMPUTED · PYTHON</span>
          <span className="axis-label">x →</span>
        </div>
        <footer className="instrument-footer">
          <span>{output.detail}</span>
          <span>{result.analyzed_dimensions.width} × {result.analyzed_dimensions.height}</span>
        </footer>
      </section>)}
    </div>
  </section>;
}
