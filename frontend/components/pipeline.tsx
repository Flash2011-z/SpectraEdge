"use client";

import { Check, ChevronRight } from "lucide-react";
import { STAGES } from "@/lib/workspace";
import { useWorkspace } from "./workspace-provider";

export function Pipeline() {
  const { stage, setStage, busy } = useWorkspace();
  return <section className="pipeline" aria-label="Processing pipeline"><div className="pipeline-title"><span className="eyebrow">PROCESSING PIPELINE</span><span className="pipeline-description">{STAGES[stage].description}</span></div>
    <div className="pipeline-nodes">{STAGES.map((item, i) => <div className="pipeline-step" key={item.name}><button onClick={() => setStage(i)} disabled={busy} className={stage === i ? "selected" : ""} aria-pressed={stage === i}><span className="step-number">{busy && stage > i ? <Check size={11} /> : `0${i+1}`}</span><span className="step-text">{item.name}<small>{item.detail}</small></span></button>{i < 5 && <ChevronRight className="pipeline-arrow" size={11} />}</div>)}</div>
  </section>;
}
