/** What setup writes to install-progress.json while an in-app update installs. */
export type InstallStep = "closing" | "copying" | "runtime" | "app" | "notices" | "starting" | "failed";

export interface InstallProgress {
  step: InstallStep;
  percent: number;
  version: string;
  log: string;
}

export interface InstallEstimate {
  fraction: number;
  label: string;
  secondsLeft: number;
}

type RunningStep = Exclude<InstallStep, "failed">;

const STEPS: RunningStep[] = ["closing", "copying", "runtime", "app", "notices", "starting"];

// ponytail: how long each step took in the 0.3.0 update on Huz's desktop. A slower disk or
// a fresh PyTorch download runs over, and the estimate then holds near the end of that step.
const EXPECTED_SECONDS: Record<RunningStep, number> = { closing: 3, copying: 10, runtime: 14, app: 9, notices: 3, starting: 4 };
const TOTAL_SECONDS = STEPS.reduce((sum, step) => sum + EXPECTED_SECONDS[step], 0);
// A step with no progress of its own never shows done before setup says it is.
const MOST_OF_A_STEP = 0.9;

const LABELS: Record<RunningStep, string> = {
  closing: "Closing Skryb",
  copying: "Copying files",
  runtime: "Checking the transcription runtime",
  app: "Installing Skryb",
  notices: "Finishing up",
  starting: "Opening the new version",
};

/** The progress file as setup wrote it; null for no file, or one caught half written. */
export function parseInstallProgress(text: string | null | undefined): InstallProgress | null {
  if (!text) return null;
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    return null;
  }
  if (typeof data !== "object" || data === null) return null;
  const { step, percent, version, log } = data as Record<string, unknown>;
  if (typeof step !== "string" || !(step === "failed" || (STEPS as string[]).includes(step))) return null;
  return {
    step: step as InstallStep,
    percent: typeof percent === "number" && Number.isFinite(percent) ? Math.min(Math.max(percent, 0), 100) : 0,
    version: typeof version === "string" ? version : "",
    log: typeof log === "string" ? log : "",
  };
}

/** How far the whole install has got, from setup's step and how long that step has run. */
export function estimateInstall(progress: InstallProgress, stepSeconds: number): InstallEstimate {
  if (progress.step === "failed") return { fraction: 0, label: "The update didn't finish", secondsLeft: 0 };
  const step = progress.step;
  const before = STEPS.slice(0, STEPS.indexOf(step)).reduce((sum, earlier) => sum + EXPECTED_SECONDS[earlier], 0);
  const expected = EXPECTED_SECONDS[step];
  const share = step === "copying" ? progress.percent / 100 : Math.min(Math.max(stepSeconds, 0) / expected, MOST_OF_A_STEP);
  const done = before + expected * share;
  return { fraction: done / TOTAL_SECONDS, label: LABELS[step], secondsLeft: Math.max(1, Math.ceil(TOTAL_SECONDS - done)) };
}

/** "About 25 seconds left", rounded so it doesn't flicker every poll. */
export function timeLeft(seconds: number): string {
  if (seconds <= 5) return "A few seconds left";
  if (seconds < 60) return `About ${Math.ceil(seconds / 5) * 5} seconds left`;
  const minutes = Math.round(seconds / 60);
  return minutes <= 1 ? "About a minute left" : `About ${minutes} minutes left`;
}
