import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** "0 min", "Under a minute", "12 min", "3 h 5 min". */
export function formatSaved(seconds: number): string {
  if (seconds <= 0) return "0 min";
  const minutes = Math.round(seconds / 60);
  if (minutes < 1) return "Under a minute";
  const hours = Math.floor(minutes / 60);
  return hours ? `${hours.toLocaleString()} h ${minutes % 60} min` : `${minutes} min`;
}

/** "18% more than usual by Wednesday", "about usual by Wednesday"; null with nothing usual to compare with. */
export function versusUsual(now: number, usual: number | null, by: Date = new Date()): string | null {
  if (!usual) return null;
  const weekday = by.toLocaleDateString(undefined, { weekday: "long" });
  const change = Math.round(((now - usual) / usual) * 100);
  if (Math.abs(change) <= 5) return `About usual by ${weekday}`;
  return `${Math.abs(change)}% ${change > 0 ? "more" : "less"} than usual by ${weekday}`;
}
