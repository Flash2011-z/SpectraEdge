import type { Metadata } from "next";
import "./globals.css";
import { WorkspaceProvider } from "@/components/workspace-provider";

export const metadata: Metadata = {
  metadataBase: new URL("https://spectraedge.beige-elk-2785.chatgpt.site"),
  title: "SpectraEdge — Signal Analysis Workspace",
  description: "An interactive workspace for multi-scale edge detection, object inspection, and frequency-domain analysis. Explore the SpectraEdge web prototype.",
  icons: { icon: "/favicon.svg" },
  openGraph: {
    type: "website",
    title: "SpectraEdge — Signal Analysis Workspace",
    description: "See the structure behind the signal. Explore the interactive SpectraEdge web prototype.",
    images: [{ url: "https://spectraedge.beige-elk-2785.chatgpt.site/og.png", width: 1730, height: 909, alt: "SpectraEdge — See the structure behind the signal." }],
  },
  twitter: {
    card: "summary_large_image",
    title: "SpectraEdge — Signal Analysis Workspace",
    description: "See the structure behind the signal.",
    images: ["https://spectraedge.beige-elk-2785.chatgpt.site/og.png"],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body><WorkspaceProvider>{children}</WorkspaceProvider></body></html>;
}
