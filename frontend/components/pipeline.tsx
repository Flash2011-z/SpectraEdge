"use client";

import { Check, ChevronRight } from "lucide-react";
import { analysisStages, STAGES } from "@/lib/workspace";
import { useWorkspace } from "./workspace-provider";

export function Pipeline() {
  const { stage, setStage, busy, source, result, parameters } = useWorkspace();
  const stages = source.kind === "demo" ? STAGES : analysisStages(
    result?.parameters_used.detector ?? parameters.detector,
    result?.parameters_used.multi_scale ?? parameters.multiScale,
    result?.noise.model ?? parameters.noise,
  );
  return (
    <section className="pipeline" aria-label="Processing pipeline">
      <div className="pipeline-title">
        <span className="eyebrow">PROCESSING PIPELINE</span>
        <span className="pipeline-description">{stages[stage].description}</span>
      </div>
      <div className="pipeline-nodes">
        {stages.map((item, i) => (
          <div className="pipeline-step" key={item.name}>
            <button
              onClick={() => setStage(i)}
              disabled={busy || ("key" in item && item.key === "objects" && result?.object_list === null)}
              className={stage === i ? "selected" : ""}
              aria-pressed={stage === i}
            >
              <span className="step-number">
                {result && "key" in item && result.completed_stages.includes(item.key) ? <Check size={11} /> : `0${i + 1}`}
              </span>
              <span className="step-text">
                {item.name}
                <small>{item.detail}</small>
              </span>
            </button>
            {i < stages.length - 1 && <ChevronRight className="pipeline-arrow" size={11} />}
          </div>
        ))}
      </div>
    </section>
  );
}
