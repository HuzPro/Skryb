import { useQuery } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { Link } from "react-router";
import { Bars, Card, dayBars, firstDayLabel, KeyCaps, PAGE_EDGE, PageTitle, Section, Stat, StatusDot } from "../components/parts";
import { useEngineStatus } from "../components/shell";
import { Button } from "../components/ui/button";
import { queries } from "../lib/queries";
import { useStore } from "../lib/store";
import type { BindingMode, HistoryUsage, Usage } from "../lib/types";
import { cn, formatSaved, versusUsual } from "../lib/utils";
import { Detail } from "./meetings";
import { LevelMeter, useMicrophones } from "./microphone";

/** The typing speed time saved is measured against, in words a minute. */
export const TYPING_WPM = 40;

const HOW_IT_FIRES: Record<BindingMode, string> = {
  toggle: "Press to start, press again to stop",
  hold: "Hold while you talk",
  press: "Press once",
};

/** An estimate: typing the words at 40 words a minute, less all the time spent recording, lost recordings included; never below nothing. */
export function timeSaved(usage: Pick<Usage, "words" | "seconds"> & { lost_seconds?: number }): number {
  return Math.max(0, (usage.words / TYPING_WPM) * 60 - usage.seconds - (usage.lost_seconds ?? 0));
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

/** "1 word", "1,912 words", and from ten thousand on "169K words". */
export function count(value: number, one: string, many: string): string {
  const number = value >= 10_000 ? new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(value) : value.toLocaleString();
  return `${number} ${value === 1 ? one : many}`;
}

/** Words a minute of recording, pauses and all; null until there is a second of recording to measure. */
export function dictationPace(usage: Pick<Usage, "words" | "seconds">): number | null {
  return usage.seconds >= 1 ? Math.round(usage.words / (usage.seconds / 60)) : null;
}

/** The line under the streak: the longest run, or how one starts. */
export function streakNote(streak: number, longest: number): string {
  if (!longest) return "Starts with your next dictation";
  return longest > streak ? `Longest ${count(longest, "day", "days")}` : "Your longest yet";
}

/** Recordings that typed something, as a whole percentage; null below ten recordings. */
export function reliability({ typed, empty, failed }: HistoryUsage["reliability"]): number | null {
  const recorded = typed + empty + failed;
  return recorded >= 10 ? Math.floor((typed / recorded) * 100) : null;
}

function UsageSection() {
  const usage = useQuery(queries.usage).data;
  const { settings } = useStore();
  const saved = settings?.saved;
  if (!usage) {
    return (
      <Section title="Your dictation">
        <p className="text-[12.5px] text-muted-foreground">Loading…</p>
      </Section>
    );
  }
  const { all, week, today, month } = usage;
  const pace = dictationPace(month);
  const reliable = reliability(usage.reliability);
  const { empty, failed } = usage.reliability;
  const retention = saved?.history_retention_days ?? 0;
  const words = (value: number) => count(value, "word", "words");
  const losses = [empty ? `${empty} empty` : "", failed ? `${failed} failed` : ""].filter(Boolean).join(", ");
  return (
    <Section title="Your dictation">
      <div className="grid grid-cols-2 gap-3 min-[1100px]:grid-cols-4">
        <Stat
          label="Today so far"
          value={words(today.words)}
          detail={[count(today.dictations, "dictation", "dictations"), usage.usual_day ? `usual day ${count(usage.usual_day, "word", "words")}` : ""].filter(Boolean).join(" · ")}
        />
        <Stat label="This week so far" value={words(week.words)} detail={versusUsual(week.words, usage.usual_week) ?? count(week.dictations, "dictation", "dictations")} />
        <Stat label={retention ? `Last ${retention} days` : "All time"} value={words(all.words)} detail={`≈${formatSaved(timeSaved(all))} saved vs typing`} />
        <Stat label="Streak" value={count(usage.streak, "day", "days")} detail={streakNote(usage.streak, usage.longest_streak)} />
      </div>
      <div className="mt-3 grid gap-3 min-[1100px]:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]">
        <Bars title="Words a day, last 30 days" bars={dayBars(usage.days)} format={words} axis={[firstDayLabel(usage.days.length), "Today"]} usual={usage.usual_day} />
        <div className="grid gap-3 max-[1099px]:grid-cols-3">
          <Stat label="Dictation pace" value={pace === null ? "–" : `${pace} wpm`} detail={`Pauses included · typing ≈${TYPING_WPM}`} />
          <Stat
            label="Typical dictation"
            value={usage.typical_words === null ? "–" : words(usage.typical_words)}
            detail={usage.typical_seconds === null ? undefined : `about ${Math.round(usage.typical_seconds)} s recorded`}
          />
          {reliable === null ? null : (
            <Stat label="Recordings that typed" value={`${reliable}%`} detail={losses ? `Last 30 days · ${losses}` : "None empty or failed"} />
          )}
        </div>
      </div>
      <p className="mt-2.5 text-[12.5px] text-muted-foreground">
        Time saved is an estimate: the same words typed at {TYPING_WPM} a minute, less all time spent recording. From{" "}
        <Link to="/history" draggable={false} className="font-medium text-primary hover:underline">
          History
        </Link>
        {saved?.capture_content === false ? ", which isn't saving new dictations." : "."}
      </p>
    </Section>
  );
}

/** Dictation's Home: whether it is ready, how to start it, and how much it has saved you. */
export function DictationHomePage() {
  return (
    <div className={cn(PAGE_EDGE, "w-full max-w-[1880px] pb-10")}>
      <PageTitle title="Dictation" />
      <div className="mt-4 grid gap-3 min-[900px]:grid-cols-2">
        <EngineCard />
        <DictateCard />
      </div>
      <UsageSection />
    </div>
  );
}
