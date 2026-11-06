import { useQuery } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { Link } from "react-router";
import { Card, DailyBars, KeyCaps, PAGE_EDGE, PageTitle, Section, Stat, StatusDot } from "../components/parts";
import { useEngineStatus } from "../components/shell";
import { Button } from "../components/ui/button";
import { queries } from "../lib/queries";
import { useStore } from "../lib/store";
import type { BindingMode, Usage } from "../lib/types";
import { cn, formatSaved } from "../lib/utils";
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
export function timeSaved(usage: Pick<Usage, "words" | "seconds">): number {
  return Math.max(0, (usage.words / TYPING_WPM) * 60 - usage.seconds);
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

/** "1 word", "1,912 words". */
function count(value: number, one: string, many: string): string {
  return `${value.toLocaleString()} ${value === 1 ? one : many}`;
}

/** Words a minute while speaking; null until there is speech to measure. */
export function speakingSpeed(usage: Pick<Usage, "words" | "seconds">): number | null {
  return usage.seconds >= 1 ? Math.round(usage.words / (usage.seconds / 60)) : null;
}

/** What the streak tile says under the number. */
export function streakNote(streak: number, today: number): string {
  if (!streak) return "Starts with your next dictation";
  return today ? "Dictated today" : "Dictate today to keep it";
}

function UsageSection() {
  const usage = useQuery(queries.usage).data;
  const { settings } = useStore();
  const historyOff = settings?.saved.capture_content === false;
  if (!usage) {
    return (
      <Section title="Your dictation">
        <p className="text-[12.5px] text-muted-foreground">Loading…</p>
      </Section>
    );
  }
  const { all, week, today, streak, days } = usage;
  const speed = speakingSpeed(all);
  return (
    <Section title="Your dictation">
      <div className="grid grid-cols-2 gap-3 min-[1100px]:grid-cols-4">
        <Stat label="Today" value={count(today.words, "word", "words")} detail={count(today.dictations, "dictation", "dictations")} />
        <Stat label="Streak" value={count(streak, "day", "days")} detail={streakNote(streak, today.dictations)} />
        <Stat label="This week" value={count(week.words, "word", "words")} detail={`${formatSaved(timeSaved(week))} saved`} />
        <Stat label="All time" value={count(all.words, "word", "words")} detail={`${formatSaved(timeSaved(all))} saved`} />
      </div>
      <div className="mt-3 grid gap-3 min-[1100px]:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]">
        <DailyBars title="Words a day, last 30 days" values={days} format={(value) => count(value, "word", "words")} />
        <div className="grid gap-3 max-[1099px]:grid-cols-2">
          <Stat label="Speaking speed" value={speed === null ? "–" : `${speed} wpm`} detail={`Typing is about ${TYPING_WPM}`} />
          <Stat
            label="Average dictation"
            value={all.dictations ? count(Math.round(all.words / all.dictations), "word", "words") : "–"}
            detail={all.dictations ? `${(all.seconds / all.dictations).toFixed(1)} s of speech` : undefined}
          />
        </div>
      </div>
      <p className="mt-2.5 text-[12.5px] text-muted-foreground">
        Time saved is typing at {TYPING_WPM} words a minute, less speaking time. From{" "}
        <Link to="/history" draggable={false} className="font-medium text-primary hover:underline">
          History
        </Link>
        {historyOff ? ", which isn't saving new dictations." : "."}
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
