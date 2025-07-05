import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Semantic ArXiv - AI-Powered Academic Paper Search",
  description:
    "Search academic papers by meaning, not just keywords. Discover research through intelligent semantic understanding.",
  keywords: [
    "arXiv",
    "academic papers",
    "semantic search",
    "AI",
    "machine learning",
    "research",
  ],
  authors: [{ name: "Your Name" }],
  openGraph: {
    title: "Semantic ArXiv - AI-Powered Academic Paper Search",
    description:
      "Search academic papers by meaning, not just keywords. Discover research through intelligent semantic understanding.",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "Semantic ArXiv - AI-Powered Academic Paper Search",
    description:
      "Search academic papers by meaning, not just keywords. Discover research through intelligent semantic understanding.",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
