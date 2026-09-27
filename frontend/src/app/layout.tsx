import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Niva — Your factual guide to HDFC mutual funds",
  description:
    "Ask Niva for clear, factual answers about HDFC Mutual Fund schemes — expense ratio, exit load, minimum SIP, benchmark, riskometer and lock-in — with the source included. No investment advice.",
  applicationName: "Niva",
  formatDetection: { telephone: false },
  openGraph: {
    title: "Niva — Your factual guide to HDFC mutual funds",
    description:
      "Clear, factual answers about HDFC mutual funds — with the source included.",
    siteName: "Niva",
    type: "website",
  },
};

export const viewport: Viewport = {
  themeColor: "#ffffff",
  colorScheme: "light",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${inter.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="flex min-h-full flex-col bg-canvas font-sans">
        {children}
      </body>
    </html>
  );
}
