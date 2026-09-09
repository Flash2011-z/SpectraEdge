import type { Metadata } from "next";
import CutoutWorkspace from "@/components/cutout-workspace";

export const metadata: Metadata = {
  title: "Object Cutout — SpectraEdge",
  description: "Choose manual Gaussian/Sobel edge guidance, GrabCut, or optional local AI-assisted cutout, and download a transparent PNG.",
};

export default function CutoutPage() {
  return <CutoutWorkspace />;
}
