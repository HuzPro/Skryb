import { LoaderCircle } from "lucide-react";
import { createContext, useContext, useEffect, useRef, useState, type RefObject } from "react";
import { estimateInstall, parseInstallProgress, timeLeft, type InstallEstimate, type InstallProgress, type InstallStep } from "../lib/install";
import { handOver, inTauri, readUpdateProgress } from "../lib/platform";
import type { AppInfo } from "../lib/types";
import { ProgressBar } from "./parts";

export interface InstallView {
  progress: InstallProgress;
  estimate: InstallEstimate;
}

/** The update installing right now, or null; Skryb itself is closed while it runs. */
export const InstallContext = createContext<InstallView | null>(null);

export function useInstall(): InstallView | null {
  return useContext(InstallContext);
}

const POLL_MS = 300;
const RESTORE_SCROLL_FOR_MS = 3000;

/**
 * Follows setup's progress file while an update installs, and once setup
 * reaches its last step hands this window over to the new version, which
 * reopens it with `where()`'s page and scroll.
 */
export function useInstallWatch(app: AppInfo | undefined, lost: boolean, where: () => { route: string; scroll: number }): InstallView | null {
  // Skryb quits before setup starts, so a window that can't reach it may be watching an install.
  const watching = inTauri && (lost || app === undefined || app.update?.state === "installing");
  const running = app?.version;
  const latestWhere = useRef(where);
  latestWhere.current = where;
  const [view, setView] = useState<InstallView | null>(null);

  useEffect(() => {
    if (!watching) {
      setView(null);
      return;
    }
    let step: InstallStep | null = null;
    let stepStarted = Date.now();
    let handedOver = false;
    const poll = async () => {
      const progress = parseInstallProgress(await readUpdateProgress().catch(() => null));
      // Nothing to show, a file caught half written, or one left by the update this copy came from.
      if (!progress || progress.version === running) return;
      if (progress.step !== step) {
        step = progress.step;
        stepStarted = Date.now();
      }
      setView({ progress, estimate: estimateInstall(progress, (Date.now() - stepStarted) / 1000) });
      if (progress.step === "starting" && !handedOver) {
        handedOver = true;
        const { route, scroll } = latestWhere.current();
        void handOver(route, scroll).catch(() => undefined);
      }
    };
    void poll();
    const timer = setInterval(() => void poll(), POLL_MS);
    return () => clearInterval(timer);
  }, [watching, running]);

  return watching ? view : null;
}

/** Scrolls the restored page back to where the old window was, once its content has loaded. */
export function useRestoredScroll(main: RefObject<HTMLElement | null>, scroll: number | undefined): void {
  useEffect(() => {
    if (!scroll || scroll <= 0) return;
    const giveUpAt = Date.now() + RESTORE_SCROLL_FOR_MS;
    let frame = 0;
    const tryScroll = () => {
      const element = main.current;
      if (element && element.scrollHeight - element.clientHeight >= scroll) {
        element.scrollTop = scroll;
        return;
      }
      if (Date.now() < giveUpAt) frame = requestAnimationFrame(tryScroll);
      else if (element) element.scrollTop = scroll;
    };
    frame = requestAnimationFrame(tryScroll);
    return () => cancelAnimationFrame(frame);
    // Only the first render's page is the restored one.
  }, []);
}

export function installTitle(view: InstallView): string {
  return view.progress.step === "failed" ? "The update didn't finish" : `Installing Skryb ${view.progress.version}`;
}

export function installDescription(view: InstallView): string {
  if (view.progress.step === "failed") {
    const log = view.progress.log ? ` The installer's log is ${view.progress.log}.` : "";
    return `Skryb ${view.progress.version} couldn't be installed. Start Skryb from the Start menu to keep using this version.${log}`;
  }
  return `${view.estimate.label} · ${timeLeft(view.estimate.secondsLeft)}`;
}

/** Shown above the other pages while the update installs; the Updates page shows it in full. */
export function InstallBanner({ view }: { view: InstallView }) {
  const failed = view.progress.step === "failed";
  return (
    <div role="status" className="absolute inset-x-0 top-10 z-30 flex items-center gap-3 border-b bg-card px-4 py-2.5 shadow-md">
      {failed ? null : <LoaderCircle className="size-4 shrink-0 animate-spin text-primary" />}
      <span className="shrink-0 text-[13px] font-medium">{installTitle(view)}</span>
      {failed ? (
        <span className="selectable min-w-0 flex-1 truncate text-[12.5px] text-muted-foreground">{installDescription(view)}</span>
      ) : (
        <>
          <ProgressBar value={view.estimate.fraction} className="min-w-24 flex-1" />
          <span className="shrink-0 text-[12.5px] text-muted-foreground">{timeLeft(view.estimate.secondsLeft)}</span>
        </>
      )}
    </div>
  );
}
