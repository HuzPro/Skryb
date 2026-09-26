import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AudioLines, BookA, ClipboardPaste, History, Keyboard, Mic, PenLine } from "lucide-react";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { createMemoryRouter, Outlet, RouterProvider, useLocation, useNavigate } from "react-router";
import { ConnectionBanner, Prompts, SaveBar, Sidebar, TopBar, type NavItem } from "./components/shell";
import { connectTo } from "./lib/api";
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
];

const PAGE_PATHS: Record<Page, string> = { history: "/history", settings: "/dictionary", setup: "/setup" };

const LaunchContext = createContext<LaunchArgs | null>(null);

function Shell() {
  const launch = useContext(LaunchContext);
  const [connection, setConnection] = useState(`${launch?.port}:${launch?.token}`);
  const lost = useServerEvents(connection);
  const store = useStore();
  const storeRef = useRef(store);
  storeRef.current = store;
  const navigate = useNavigate();
  const navigateRef = useRef(navigate);
  navigateRef.current = navigate;
  const location = useLocation();
  const setup = location.pathname === PAGE_PATHS.setup;

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
      connectTo(next.port, next.token);
      setConnection(`${next.port}:${next.token}`);
      void navigateRef.current(PAGE_PATHS[next.page] ?? PAGE_PATHS.settings);
    });
  }, [mayClose]);

  return (
    <div className="relative flex h-full">
      {setup ? null : <Sidebar items={NAV} />}
      <div className="relative flex min-w-0 flex-1 flex-col">
        <TopBar nativeTitlebar={launch?.native_titlebar ?? false} onClose={() => void close()}>
          {setup ? <span className="pointer-events-none text-[13px] font-semibold">VoiceToText</span> : null}
        </TopBar>
        <main key={location.pathname} className="min-h-0 flex-1 overflow-y-auto">
          <Outlet />
        </main>
        {setup ? null : <SaveBar />}
        {lost ? <ConnectionBanner onQuit={windowControls.close} /> : null}
      </div>
      <Prompts />
    </div>
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
  { path: "setup", element: <SetupPage /> },
];

export function App({ launch }: { launch: LaunchArgs }) {
  const router = useMemo(
    () => createMemoryRouter([{ path: "/", element: <Shell />, children: PAGES }], { initialEntries: [PAGE_PATHS[launch.page] ?? PAGE_PATHS.settings] }),
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
