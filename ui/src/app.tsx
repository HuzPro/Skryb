import { QueryClientProvider, useIsFetching, useQueryClient } from "@tanstack/react-query";
import { AudioLines, BookA, ClipboardPaste, History, House, Keyboard, Mic } from "lucide-react";
import { createContext, startTransition, useCallback, useContext, useEffect, useMemo, useRef, useState, ViewTransition } from "react";
import { createMemoryRouter, Outlet, RouterProvider, useLocation, useNavigate } from "react-router";
import { InstallBanner, InstallContext, useInstallWatch, useRestoredScroll } from "./components/install";
import { ConnectionBanner, LostContext, Prompts, SaveBar, Sidebar, TopBar, useAppInfo, type Mode, type NavItem } from "./components/shell";
import { applyTheme, guardClose, onRelaunch, runBeforeClose, showWindow, windowControls, type LaunchArgs, type Page } from "./lib/platform";
import { prefetchPages, queryClient } from "./lib/queries";
import { useServerEvents } from "./lib/sse";
import { StoreProvider, useStore, useStoreRef } from "./lib/store";
import { DictionaryPage } from "./pages/dictionary";
import { HistoryPage } from "./pages/history";
import { DictationHomePage } from "./pages/home";
import {
  MeetingHistoryPage,
  MeetingHomePage,
  MeetingMicrophonePage,
  MeetingModelPage,
  MeetingOutputPage,
  SavedMeetingPage,
  useMeetingEvents,
} from "./pages/meetings";
import { MicrophonePage } from "./pages/microphone";
import { OutputPage } from "./pages/output";
import { SetupPage } from "./pages/setup";
import { ShortcutsPage } from "./pages/shortcuts";
import { TranscriptionPage } from "./pages/transcription";
import { SettingsPage } from "./pages/updates";

/** The same items in the same order in both modes; only the page behind each changes with the mode. */
export const NAV: NavItem[] = [
  { label: "Home", icon: House, dictation: "/home", meetings: "/meeting-home" },
  { label: "History", icon: History, dictation: "/history", meetings: "/meetings" },
  { label: "Dictionary", icon: BookA, dictation: "/dictionary", meetings: "/dictionary" },
  { label: "Shortcuts", icon: Keyboard, dictation: "/shortcuts", meetings: "/shortcuts" },
  { label: "Microphone & sounds", icon: Mic, dictation: "/microphone", meetings: "/meeting-microphone" },
  { label: "Output", icon: ClipboardPaste, dictation: "/output", meetings: "/meeting-output" },
  { label: "Model", icon: AudioLines, dictation: "/transcription", meetings: "/meeting-model" },
];

// Pages both modes share: opening one keeps the mode it was opened from.
const SHARED = new Set(["/dictionary", "/shortcuts", "/updates", "/setup"]);
// A saved meeting sits under History in meeting mode.
const MEETING = /^\/meetings\/./;

/** Whether `pathname` is one of `item`'s pages. */
export function isNavActive(item: NavItem, pathname: string): boolean {
  return pathname === item.dictation || pathname === item.meetings || (item.meetings === "/meetings" && MEETING.test(pathname));
}

/** The mode a page belongs to; null for a page both modes share. */
export function modeOf(pathname: string): Mode | null {
  if (SHARED.has(pathname)) return null;
  return MEETING.test(pathname) || NAV.some((item) => item.meetings === pathname) ? "meetings" : "dictation";
}

/** The page behind the same sidebar item in `mode`; from a page outside the sidebar, such as Settings, that mode's Home. */
export function pageInMode(pathname: string, mode: Mode): string {
  const item = NAV.find((candidate) => isNavActive(candidate, pathname)) ?? NAV[0]!;
  return item[mode];
}

/** The window opens on Home; the tray's Meetings opens meeting mode's Home. */
export function launchPath(page: Page): string {
  if (page === "settings") return "/home";
  if (page === "meetings") return "/meeting-home";
  return `/${page}`;
}

/** The page the last version's window showed when it handed over, if it is one this version has. */
export function restoredPath(route: string | null | undefined): string | null {
  return route && PAGES.some((page) => `/${page.path}` === route) ? route : null;
}

const LaunchContext = createContext<LaunchArgs | null>(null);
// Rust shows the window after four seconds whatever happens; this is the shell's own limit.
const SHOW_ANYWAY_MS = 1500;

/**
 * Shows the window once nothing the first page asked for is still loading, then fetches the other
 * pages' data. Rendered after the page, so its effect runs after the page's queries have started;
 * a hidden window gets no animation frames and throttled timers, so it shows from the effect itself.
 */
function ShowWhenReady() {
  const client = useQueryClient();
  const fetching = useIsFetching();
  useEffect(() => {
    const timer = setTimeout(showWindow, SHOW_ANYWAY_MS);
    return () => clearTimeout(timer);
  }, []);
  useEffect(() => {
    if (fetching || client.isFetching()) return;
    showWindow();
    prefetchPages(client);
  }, [fetching, client]);
  return null;
}

/**
 * Shows the theme being picked before it is saved; discarding puts the saved one back. The whole window
 * cross-fades as one picture, with colour transitions held off so no part of it fades on its own.
 */
function ThemeSync() {
  const draft = useStore().settings?.draft;
  const theme = draft?.theme;
  const colourway = theme ? draft?.colourways[theme] : undefined;
  useEffect(() => {
    const root = document.documentElement;
    if (!theme || (root.dataset.theme === theme && root.dataset.colourway === colourway)) return;
    const swap = () => {
      root.setAttribute("data-theme-switch", "");
      applyTheme(theme, colourway);
    };
    const fade = document.startViewTransition?.(swap);
    if (!fade) swap();
    void (fade?.ready ?? Promise.resolve()).finally(() => root.removeAttribute("data-theme-switch"));
  }, [theme, colourway]);
  return null;
}

function Shell() {
  const launch = useContext(LaunchContext);
  const lost = useServerEvents();
  const store = useStoreRef();
  const navigate = useNavigate();
  const navigateRef = useRef(navigate);
  navigateRef.current = navigate;
  const { pathname } = useLocation();
  const main = useRef<HTMLElement>(null);
  const install = useInstallWatch(useAppInfo(), lost, () => ({ route: pathname, scroll: main.current?.scrollTop ?? 0 }));
  useRestoredScroll(main, launch?.restored?.scroll);
  const setup = pathname === "/setup";
  const [chosenMode, setChosenMode] = useState<Mode>(() => modeOf(pathname) ?? "dictation");
  const mode = modeOf(pathname) ?? chosenMode;
  useMeetingEvents();
  useEffect(() => {
    const owner = modeOf(pathname);
    if (owner) setChosenMode(owner);
  }, [pathname]);
  const switchMode = useCallback(
    (next: Mode) => {
      startTransition(() => setChosenMode(next));
      const target = pageInMode(pathname, next);
      if (target !== pathname) void navigateRef.current(target);
    },
    [pathname],
  );

  const mayClose = useCallback(async (): Promise<boolean> => {
    if (!store.current.anyDirty()) return true;
    const choice = await store.current.askUnsaved();
    if (choice === "cancel") return false;
    return choice === "discard" || store.current.saveAll();
  }, [store]);

  const close = useCallback(async () => {
    if (!(await mayClose())) return;
    await runBeforeClose();
    windowControls.close();
  }, [mayClose]);

  useEffect(() => {
    guardClose(mayClose);
    onRelaunch((next) => {
      if (next.port !== launch?.port || next.token !== launch?.token) {
        window.location.reload();
        return;
      }
      void navigateRef.current(launchPath(next.page));
    });
  }, [mayClose, launch]);

  return (
    <LostContext.Provider value={lost}>
      <InstallContext.Provider value={install}>
        <ThemeSync />
        <div className="relative flex h-full">
          {setup ? null : (
            <Sidebar
              items={NAV}
              mode={mode}
              onMode={switchMode}
              pathname={pathname}
              isActive={isNavActive}
            />
          )}
          <div className="relative flex min-w-0 flex-1 flex-col">
            <TopBar nativeTitlebar={launch?.native_titlebar ?? false} onClose={() => void close()} showStatus={setup}>
              {setup ? <span className="pointer-events-none text-[13px] font-semibold">Skryb</span> : null}
            </TopBar>
            {/* Each page gets its own scroll area, so only a change of page animates; the panel stays put. */}
            <div className="content-panel flex min-h-0 flex-1 flex-col overflow-hidden">
              <ViewTransition key={pathname} name="page" update="none">
                <main ref={main} className="min-h-0 flex-1 overflow-y-auto [scrollbar-gutter:stable]">
                  <Outlet />
                </main>
              </ViewTransition>
            </div>
            {setup ? null : <SaveBar />}
            {install ? (
              pathname === "/updates" ? null : <InstallBanner view={install} />
            ) : lost ? (
              <ConnectionBanner onQuit={windowControls.close} />
            ) : null}
          </div>
          <Prompts />
        </div>
        <ShowWhenReady />
      </InstallContext.Provider>
    </LostContext.Provider>
  );
}

const PAGES = [
  { path: "home", element: <DictationHomePage /> },
  { path: "meeting-home", element: <MeetingHomePage /> },
  { path: "history", element: <HistoryPage /> },
  { path: "dictionary", element: <DictionaryPage /> },
  { path: "shortcuts", element: <ShortcutsPage /> },
  { path: "microphone", element: <MicrophonePage /> },
  { path: "output", element: <OutputPage /> },
  { path: "transcription", element: <TranscriptionPage /> },
  { path: "updates", element: <SettingsPage /> },
  { path: "setup", element: <SetupPage /> },
  { path: "meetings", element: <MeetingHistoryPage /> },
  { path: "meetings/:id", element: <SavedMeetingPage /> },
  { path: "meeting-microphone", element: <MeetingMicrophonePage /> },
  { path: "meeting-output", element: <MeetingOutputPage /> },
  { path: "meeting-model", element: <MeetingModelPage /> },
];

export function App({ launch }: { launch: LaunchArgs }) {
  const router = useMemo(
    () => createMemoryRouter([{ path: "/", element: <Shell />, children: PAGES }], { initialEntries: [restoredPath(launch.restored?.route) ?? launchPath(launch.page)] }),
    [launch],
  );
  return (
    <QueryClientProvider client={queryClient}>
      <LaunchContext.Provider value={launch}>
        <StoreProvider>
          <RouterProvider router={router} />
        </StoreProvider>
      </LaunchContext.Provider>
    </QueryClientProvider>
  );
}
