import { useQuery } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { Link } from "react-router";
import { Card, KeyCaps, PAGE_EDGE, PageTitle, Section, StatusDot } from "../components/parts";
import { useEngineStatus } from "../components/shell";
import { Button } from "../components/ui/button";
import { queries } from "../lib/queries";
import { useStore } from "../lib/store";
import type { BindingMode, Usage } from "../lib/types";
import { cn } from "../lib/utils";
import { Detail } from "./meetings";
import { LevelMeter, useMicrophones } from "./microphone";

/** The typing speed time saved is measured against, in words a minute. */
export const TYPING_WPM = 40;

const HOW_IT_FIRES: Record<BindingMode, string> = {
  toggle: "Press to start, press again to stop",
  hold: "Hold while you talk",
  press: "Press once",
};

/** How long typing the words at 40 words a minute takes, less the time spent saying them; never below nothing. */
export function timeSaved(usage: Usage): number {
  return Math.max(0, (usage.words / TYPING_WPM) * 60 - usage.seconds);
}

/** "0 min", "Under a minute", "12 min", "3 h 5 min". */
export function formatSaved(seconds: number): string {
  if (seconds <= 0) return "0 min";
  const minutes = Math.round(seconds / 60);
  if (minutes < 1) return "Under a minute";
  const hours = Math.floor(minutes / 60);
  return hours ? `${hours.toLocaleString()} h ${minutes % 60} min` : `${minutes} min`;
}

function MoreLink({ to, children }: { to: string; children: ReactNode }) {
  return (
    <Link to={to} draggable={false} className="mt-3 inline-block text-[12.5px] font-medium text-primary hover:underline">
      {children}
    </Link>
  );
}

function EngineCard() {
  const status = useEngineStatus();
  const saved = useQuery(queries.settings).data;
  const engine = status.app?.engine;
  return (
    <Card className="px-5 py-4">
      <h2 className="mb-2.5 text-[13px] font-semibold text-muted-foreground">Status</h2>
      <p className="flex items-center gap-2 text-[15px] font-medium">
        <StatusDot tone={status.tone} />
        {status.label}
      </p>
      <dl className="mt-3 grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-[12.5px]">
        <Detail term="Model">{engine?.model || saved?.model_size || "No model"}</Detail>
        <Detail term="Runs on">{engine?.device ?? (engine?.ready ? "Not reported by this engine" : "Not loaded yet")}</Detail>
      </dl>
      {engine?.device_note ? <p className="mt-2 text-[12.5px] text-warning">{engine.device_note}</p> : null}
      <MoreLink to="/transcription">Change the model</MoreLink>
    </Card>
  );
}

function DictateCard() {
  const { settings } = useStore();
  const { microphones } = useMicrophones();
  const [testing, setTesting] = useState(false);
  const saved = settings?.saved;
  const shortcuts = saved?.hotkey_bindings.filter((binding) => binding.action === "dictate" && binding.keys.trim()) ?? [];
  const device = saved?.input_device ?? "";
  const microphone = device ? (microphones.find((candidate) => candidate.name === device)?.label ?? device) : "Windows default";
  return (
    <Card className="px-5 py-4">
      <h2 className="mb-2.5 text-[13px] font-semibold text-muted-foreground">Dictate</h2>
      {shortcuts.length ? (
        <ul className="flex flex-col gap-2">
          {shortcuts.map((binding) => (
            <li key={`${binding.keys}-${binding.mode}`} className="flex flex-wrap items-center gap-x-3 gap-y-1">
              <KeyCaps combo={binding.keys} />
              <span className="text-[12.5px] text-muted-foreground">{HOW_IT_FIRES[binding.mode]}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[12.5px] text-muted-foreground">{saved ? "No dictation shortcut is set." : "Loading…"}</p>
      )}
      <div className="mt-4 border-t pt-3">
        <div className="flex items-center gap-3">
          <span className="min-w-0 flex-1 truncate text-[12.5px]" title={microphone}>
            <span className="text-muted-foreground">Microphone </span>
            {microphone}
          </span>
          <Button variant="outline" size="sm" onClick={() => setTesting((on) => !on)} disabled={!saved}>
            {testing ? "Stop the test" : "Test the microphone"}
          </Button>
        </div>
        {testing ? (
          <div className="mt-2.5">
            <LevelMeter device={device} />
          </div>
        ) : null}
      </div>
      <MoreLink to="/shortcuts">Change the shortcuts</MoreLink>
    </Card>
  );
}

function Stat({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <Card className="px-5 py-4">
      <p className="text-[12.5px] text-muted-foreground">{label}</p>
      <p className="mt-1 text-[24px] leading-tight font-semibold tabular-nums">{value}</p>
      {detail ? <p className="mt-0.5 text-[12.5px] text-muted-foreground">{detail}</p> : null}
    </Card>
  );
}

function UsageSection() {
  const usage = useQuery(queries.usage).data;
  const { settings } = useStore();
  const all = usage?.all ?? { words: 0, seconds: 0 };
  const week = usage?.week ?? { words: 0, seconds: 0 };
  const words = (count: number) => (usage ? `${count.toLocaleString()} ${count === 1 ? "word" : "words"}` : "…");
  const historyOff = settings?.saved.capture_content === false;
  return (
    <Section title="Your dictation">
      <div className="grid gap-3 min-[900px]:grid-cols-3">
        <Stat label="Words dictated" value={usage ? all.words.toLocaleString() : "…"} />
        <Stat label="Time saved" value={usage ? formatSaved(timeSaved(all)) : "…"} />
        <Stat label="This week" value={words(week.words)} detail={usage ? `${formatSaved(timeSaved(week))} saved since Monday` : undefined} />
      </div>
      <p className="mt-2.5 max-w-[760px] text-[12.5px] text-muted-foreground">
        {usage && !all.words ? "Your numbers start with your first dictation. " : ""}
        Time saved is how long typing these words takes at {TYPING_WPM} words a minute, minus the time you spent saying them. Counted from the dictations kept in{" "}
        <Link to="/history" draggable={false} className="font-medium text-primary hover:underline">
          History
        </Link>
        {historyOff ? ", which is not saving new ones." : "."}
      </p>
    </Section>
  );
}

/** Dictation's Home: whether it is ready, how to start it, and how much it has saved you. */
export function DictationHomePage() {
  return (
    <div className={cn(PAGE_EDGE, "w-full max-w-[1400px] pb-10")}>
      <PageTitle title="Dictation" />
      <div className="mt-4 grid gap-3 min-[900px]:grid-cols-2">
        <EngineCard />
        <DictateCard />
      </div>
      <UsageSection />
    </div>
  );
}
