import { createRoot } from "react-dom/client";
import { App } from "./app";
import "./index.css";
import { connectTo } from "./lib/api";
import { applyTheme, readLaunchArgs } from "./lib/platform";
import { queries, queryClient } from "./lib/queries";

async function start(): Promise<void> {
  const args = await readLaunchArgs();
  connectTo(args.port, args.token);
  // The theme and the shell's own data arrive before the first render, so nothing pops in; a core
  // that isn't running fails these at once and the window says so.
  const [settings] = await Promise.all([
    queryClient.fetchQuery({ ...queries.settings, retry: false }).catch(() => undefined),
    queryClient.fetchQuery({ ...queries.app, retry: false }).catch(() => undefined),
  ]);
  applyTheme(settings?.theme);
  const root = document.getElementById("root");
  if (root) createRoot(root).render(<App launch={args} />);
}

void start();
