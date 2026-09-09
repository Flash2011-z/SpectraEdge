import Link from "next/link";
import { Activity, GitCompareArrows, Radio, ScanLine, Scissors } from "lucide-react";
import type { View } from "@/lib/workspace";
import styles from "./workspace-navigation.module.css";

export function WorkspaceBrand() {
  return <Link className="brand" href="/">
    <span className="brand-mark"><Activity size={23} strokeWidth={1.5} /></span>
    <strong>Spectra<span>Edge</span></strong><span className="version">WEB / 01</span>
  </Link>;
}

export function WorkspaceNavigation({ active }: { active: View | "cutout" }) {
  return <nav className={styles.navigation} aria-label="Workspace">
    {[
      { view: "analyze", path: "/", label: "Analyze", icon: ScanLine },
      { view: "compare", path: "/compare", label: "Compare", icon: GitCompareArrows },
      { view: "live", path: "/live", label: "Live", icon: Radio },
      { view: "cutout", path: "/cutout", label: "Object Cutout", icon: Scissors },
    ].map(({ view, path, label, icon: Icon }) => <Link key={view}
      className={`nav-link ${active === view ? "active" : ""}`} href={path}
      aria-current={active === view ? "page" : undefined}>
      <Icon size={14} strokeWidth={1.6} />{label}
    </Link>)}
  </nav>;
}
