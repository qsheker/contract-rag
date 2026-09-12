"use client";

import { useEffect, useSyncExternalStore } from "react";
import { Monitor, Moon, Sun } from "lucide-react";

import {
  applyTheme,
  readStoredTheme,
  storeTheme,
  subscribeToTheme,
  type Theme,
} from "@/lib/theme";
import { cn } from "cn";

const OPTIONS: { value: Theme; label: string; Icon: typeof Sun }[] = [
  { value: "light", label: "Светлая", Icon: Sun },
  { value: "dark", label: "Тёмная", Icon: Moon },
  { value: "system", label: "Системная", Icon: Monitor },
];

export function ThemeSwitch() {
  const theme = useSyncExternalStore(
    subscribeToTheme,
    readStoredTheme,
    // The server has no localStorage, and neither has the first client render:
    // both say "system", so hydration matches whatever the bootstrap painted.
    () => "system" as Theme,
  );

  // Only "system" follows the OS afterwards; an explicit choice must not be
  // overwritten when the machine switches at sunset.
  useEffect(() => {
    if (theme !== "system") {
      return;
    }
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const follow = () => applyTheme("system");
    query.addEventListener("change", follow);
    return () => query.removeEventListener("change", follow);
  }, [theme]);

  return (
    <div
      className="flex rounded-lg bg-sidebar-accent/60 p-0.5"
      role="radiogroup"
      aria-label="Тема оформления"
    >
      {OPTIONS.map(({ value, label, Icon }) => (
        <button
          key={value}
          type="button"
          role="radio"
          aria-checked={theme === value}
          title={label}
          onClick={() => storeTheme(value)}
          className={cn(
            "flex flex-1 items-center justify-center gap-1.5 rounded-md px-2 py-1.5 text-xs transition-colors",
            theme === value
              ? "bg-sidebar text-sidebar-foreground shadow-sm"
              : "text-sidebar-foreground/60 hover:text-sidebar-foreground",
          )}
        >
          <Icon className="size-3.5" />
          <span className="hidden xl:inline">{label}</span>
        </button>
      ))}
    </div>
  );
}
