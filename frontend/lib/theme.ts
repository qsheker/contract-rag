// Theme is a class on <html>, which is what the shadcn tokens in globals.css
// already key off (`@custom-variant dark (&:is(.dark *))`). Nothing else here
// needs to know about it.

export type Theme = "light" | "dark" | "system";

export const THEME_STORAGE_KEY = "contract-rag.theme";

// A tiny store rather than component state: the chosen theme lives in
// localStorage, which is an external system, and React reads it through
// useSyncExternalStore. That keeps the server render and the first client
// render agreeing on "system" while the bootstrap script below has already
// painted the right colours.
const listeners = new Set<() => void>();

export function readStoredTheme(): Theme {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    return stored === "light" || stored === "dark" ? stored : "system";
  } catch {
    return "system";
  }
}

export function subscribeToTheme(listener: () => void): () => void {
  listeners.add(listener);
  // Another tab switching the theme counts as a change here too.
  window.addEventListener("storage", listener);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", listener);
  };
}

export function storeTheme(theme: Theme): void {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Storage can be blocked; the theme still applies for this session.
  }
  applyTheme(theme);
  for (const listener of listeners) {
    listener();
  }
}

export function applyTheme(theme: Theme): void {
  const isDark =
    theme === "dark" ||
    (theme === "system" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", isDark);
}

// Runs before the first paint, inlined into the document head: without it the
// page renders light and then flips, which is worse than having no dark theme
// at all. Written as a string because it has to execute ahead of hydration.
export const THEME_BOOTSTRAP_SCRIPT = `
(function () {
  try {
    var stored = localStorage.getItem("${THEME_STORAGE_KEY}");
    var dark = stored === "dark" || (stored !== "light" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.classList.toggle("dark", dark);
  } catch (error) {}
})();
`;
