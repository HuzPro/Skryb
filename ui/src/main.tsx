import { createRoot } from "react-dom/client";
import { App } from "./app";
import "./index.css";
import { connectTo } from "./lib/api";
import { readLaunchArgs } from "./lib/platform";

async function start(): Promise<void> {
  const args = await readLaunchArgs();
  connectTo(args.port, args.token);
  const root = document.getElementById("root");
  if (root) createRoot(root).render(<App launch={args} />);
}

void start();
