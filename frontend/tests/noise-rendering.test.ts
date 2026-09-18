import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { runInThisContext } from "node:vm";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";
import * as workspace from "../lib/workspace.ts";

const require = createRequire(import.meta.url);
const png = "data:image/png;base64,iVBORw0KGgo=";

// Compile the actual TSX using the installed compiler. Only the workspace hook
// is substituted, so these tests render real cards, images, and pipeline nodes.
function components(state: object) {
  const cache = new Map<string, unknown>();
  function load(name: string): unknown {
    if (cache.has(name)) return cache.get(name);
    const source = readFileSync(new URL(`../components/${name}.tsx`, import.meta.url), "utf8");
    const { outputText } = ts.transpileModule(source, {
      compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX,
        target: ts.ScriptTarget.ES2022, esModuleInterop: true },
    });
    const compiled = { exports: {} };
    const resolve = (id: string): unknown => {
      if (id === "./workspace-provider") return { useWorkspace: () => state };
      if (id === "@/lib/workspace") return workspace;
      if (id.startsWith("./")) return load(id.slice(2));
      return require(id);
    };
    runInThisContext(`(function(require, module, exports) {${outputText}\n})`,
      { filename: `${name}.tsx` })(resolve, compiled, compiled.exports);
    cache.set(name, compiled.exports);
    return compiled.exports;
  }
  return {
    ...load("visualization") as typeof import("../components/visualization"),
    ...load("pipeline") as typeof import("../components/pipeline"),
  };
}

for (const model of ["None", "Gaussian", "Salt & Pepper"] as const) {
  test(`${model} renders the correct noise image, metadata, and pipeline stage`, () => {
    const enabled = model !== "None";
    const units = model === "Gaussian" ? "intensity standard deviation" : "pixel corruption probability";
    const state = {
      source: { kind: "image", name: "sample.png", width: 32, height: 24 },
      preferences: workspace.DEFAULT_PREFERENCES,
      parameters: { ...workspace.DEFAULT_PARAMETERS, noise: model },
      stage: 2, busy: false, status: "success",
      result: {
        noisy_image: enabled ? png : null,
        parameters_used: { detector: "Sobel", multi_scale: true },
        analyzed_dimensions: { width: 32, height: 24 },
        completed_stages: enabled ? ["input", "grayscale", "noise"] : ["input", "grayscale"],
        noise: { model, strength: model === "Gaussian" ? 12 : 0.12, units, seed: 220 },
      },
    };
    const { VisualizationCard, Pipeline } = components(state);
    const card = renderToStaticMarkup(createElement(VisualizationCard, { kind: "noisy", index: "03" }));
    const pipeline = renderToStaticMarkup(createElement(Pipeline));
    assert.match(card, /Noisy grayscale image/);
    if (enabled) {
      assert.ok(card.includes(`src="${png}"`));
      assert.match(card, /computed by Python/);
      assert.ok(card.includes(units));
      assert.match(card, /seed 220/);
      assert.ok(pipeline.indexOf("Grayscale") < pipeline.indexOf(">Noise<"));
      assert.ok(pipeline.indexOf(">Noise<") < pipeline.indexOf(">Smooth<"));
      assert.ok(pipeline.indexOf("Multi-scale") < pipeline.indexOf(">Objects<"));
    } else {
      assert.ok(!card.includes(`src="${png}"`));
      assert.ok(!pipeline.includes(">Noise<"));
    }
  });
}
