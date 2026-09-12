import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { THEME_BOOTSTRAP_SCRIPT } from "@/lib/theme";

// shadcn's theme reads --font-sans and --font-mono, so the loaded faces are
// bound to those names rather than to Geist-specific ones.
const geistSans = Geist({
  variable: "--font-sans",
  subsets: ["latin", "cyrillic"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin", "cyrillic"],
});

export const metadata: Metadata = {
  title: "contract-rag",
  description:
    "Чат по юридическим договорам с цитированием пункта и страницы, и загрузкой своих документов.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="ru"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      // The bootstrap script below adds `dark` to this element before React
      // hydrates, which is the whole point of it; without this React reports
      // the class it did not render as a hydration mismatch.
      suppressHydrationWarning
    >
      <head>
        {/* Before the first paint, so a dark-theme reader is not flashed a
            white page on every navigation. */}
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOTSTRAP_SCRIPT }} />
      </head>
      {/* h-full, not min-h-full: the page must not grow past the viewport,
          or the chat pushes the header out of reach instead of scrolling
          its own message list. */}
      <body className="flex h-full flex-col overflow-hidden">{children}</body>
    </html>
  );
}
