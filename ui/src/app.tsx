import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AudioLines, BookA, ClipboardPaste, Download, History, Keyboard, Mic, PenLine } from "lucide-react";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef } from "react";
import { createMemoryRouter, Outlet, RouterProvider, useLocation, useNavigate } from "react-router";
import { InstallBanner, InstallContext, useInstallWatch, useRestoredScroll } from "./components/install";
import { ConnectionBanner, LostContext, Prompts, SaveBar, Sidebar, TopBar, useAppInfo, type NavItem } from "./components/shell";
import { guardClose, onRelaunch, runBeforeClose, windowControls, type LaunchArgs, type Page } from "./lib/platform";
import { useServerEvents } from "./lib/sse";
import { StoreProvider, useStore } from "./lib/store";
import { DictionaryPage } from "./pages/dictionary";
import { HistoryPage } from "./pages/history";
import { MicrophonePage } from "./pages/microphone";
import { OutputPage } from "./pages/output";
import { SetupPage } from "./pages/setup";
import { ShortcutsPage } from "./pages/shortcuts";
import { TranscriptionPage } from "./pages/transcription";
import { UpdatesPage } from "./pages/updates";
import { WritingPage } from "./pages/writing";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: Infinity, retry: 1, refetchOnWindowFocus: false } },
});

const NAV: NavItem[] = [
  { path: "history", label: "History", icon: History },
  { path: "dictionary", label: "Dictionary", icon: BookA },
  { path: "writing", label: "Writing", icon: PenLine },
  { path: "shortcuts", label: "Shortcuts", icon: Keyboard },
  { path: "microphone", label: "Microphone & sounds", icon: Mic },
  { path: "output", label: "Output", icon: ClipboardPaste },
  { path: "transcription", label: "Transcription", icon: AudioLines },
  { path: "updates", label: "Updates", icon: Download },
];

const LAST_SETTINGS_PAGE = "last-settings-page";
const SETTINGS_PAGES = NAV.map((item) => item.path).filter((path) => path !== "history");

type PageMemory = Pick<Storage, "getItem" | "setItem">;

function browserStorage(): PageMemory | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

/** `--page settings` reopens the settings page shown last, Transcription the first time. */
export function launchPath(page: Page, memory: PageMemory | null = browserStorage()): string {
  if (page === "history" || page === "setup" || page === "updates") return `/${page}`;
  let last: string | null = null;
  try {
    last = memory?.getItem(LAST_SETTINGS_PAGE) ?? null;
  } catch {
    last = null;
  }
  return `/${last && SETTINGS_PAGES.includes(last) ? last : "transcription"}`;
}

export function rememberPage(pathname: string, memory: PageMemory | null = browserStorage()): void {
  const page = pathname.slice(1);
  if (!SETTINGS_PAGES.includes(page)) return;
  try {
    memory?.setItem(LAST_SETTINGS_PAGE, page);
  } catch {
    // Storage blocked: the next launch starts on Transcription.
  }
}

/** The page the last version's window showed when it handed over, if it is one this version has. */
export function restoredPath(route: string | null | undefined): string | null {
  return route && PAGES.some((page) => `/${page.path}` === route) ? route : null;
}

const LaunchContext = createContext<LaunchArgs | null>(null);

function Shell() {
  const launch = useContext(LaunchContext);
  const lost = useServerEvents();
  const store = useStore();
  const storeRef = useRef(store);
  storeRef.current = store;
  const navigate = useNavigate();
  const navigateRef = useRef(navigate);
  navigateRef.current = navigate;
  const location = useLocation();
  const main = useRef<HTMLElement>(null);
  const install = useInstallWatch(useAppInfo(), lost, () => ({ route: location.pathname, scroll: main.current?.scrollTop ?? 0 }));
  useRestoredScroll(main, launch?.restored?.scroll);
  const setup = location.pathname === "/setup";
  useEffect(() => rememberPage(location.pathname), [location.pathname]);

  const mayClose = useCallback(async (): Promise<boolean> => {
    if (!storeRef.current.anyDirty()) return true;
    const choice = await storeRef.current.askUnsaved();
    if (choice === "cancel") return false;
    return choice === "discard" || storeRef.current.saveAll();
  }, []);

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
        <div className="relative flex h-full">
          {setup ? null : <Sidebar items={NAV} />}
          <div className="relative flex min-w-0 flex-1 flex-col">
            <TopBar nativeTitlebar={launch?.native_titlebar ?? false} onClose={() => void close()}>
              {setup ? <span className="pointer-events-none text-[13px] font-semibold">Skryb</span> : null}
            </TopBar>
            <main ref={main} key={location.pathname} className="min-h-0 flex-1 overflow-y-auto">
              <Outlet />
            </main>
            {setup ? null : <SaveBar />}
            {install ? (
              location.pathname === "/updates" ? null : <InstallBanner view={install} />
            ) : lost ? (
              <ConnectionBanner onQuit={windowControls.close} />
            ) : null}
          </div>
          <Prompts />
        </div>
      </InstallContext.Provider>
    </LostContext.Provider>
  );
}

const PAGES = [
  { path: "history", element: <HistoryPage /> },
  { path: "dictionary", element: <DictionaryPage /> },
  { path: "writing", element: <WritingPage /> },
  { path: "shortcuts", element: <ShortcutsPage /> },
  { path: "microphone", element: <MicrophonePage /> },
  { path: "output", element: <OutputPage /> },
  { path: "transcription", element: <TranscriptionPage /> },
  { path: "updates", element: <UpdatesPage /> },
  { path: "setup", element: <SetupPage /> },
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
