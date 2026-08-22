import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "SPEAR Lab — Symbolic Pareto Evolutionary Algorithm for Research",
  description:
    "Découvrez des formules symboliques ultra-rapides pour remplacer le calcul lourd des LLM et de vos jeux de données grâce à SPEAR.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="fr">
      <body className="bg-slate-950 text-slate-100 antialiased">{children}</body>
    </html>
  );
}
