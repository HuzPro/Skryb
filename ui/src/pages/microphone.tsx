import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowDown, ArrowUp, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Card, InlineError, NumberField, PageBody, PageTitle, Row, Rows, Section } from "../components/parts";
import { Button } from "../components/ui/button";
import { Select } from "../components/ui/select";
import { Switch } from "../components/ui/switch";
import { api, errorText } from "../lib/api";
import { onBeforeClose } from "../lib/platform";
import { onServerEvent } from "../lib/sse";
import { useStore } from "../lib/store";
import type { Duck, Microphone, RecordingSettings, Settings } from "../lib/types";
import { cn } from "../lib/utils";

const DUCK_CHOICES = [
  { value: "off", label: "Leave other audio alone" },
  { value: "lower", label: "Lower other apps" },
  { value: "mute", label: "Mute other apps" },
];
const METER_BARS = 12;
const NO_SIGNAL_AFTER_MS = 2000;

export function useMicrophones(): { microphones: Microphone[]; rescan: () => Promise<void> } {
  const client = useQueryClient();
  const microphones = useQuery({ queryKey: ["microphones"], queryFn: () => api.get<Microphone[]>("/api/microphones") }).data ?? [];
  const rescan = async () => {
    client.setQueryData(["microphones"], await api.get<Microphone[]>("/api/microphones?rescan=1"));
  };
  return { microphones, rescan };
}

type MeterState = "listening" | "failed" | "silent";
const COULD_NOT_OPEN = "This microphone could not be opened.";

let meterRequests: Promise<unknown> = Promise.resolve();

/** One at a time, so switching microphones never lets the old "off" land after the new "on". */
function sendMeter(device: string, on: boolean): Promise<unknown> {
  meterRequests = meterRequests.catch(() => undefined).then(() => api.post("/api/microphones/meter", { device, on }));
  return meterRequests;
}

/** Streams the level of `device` into the bars while mounted; the core only holds the microphone open meanwhile. */
function LevelMeter({ device }: { device: string }) {
  const bars = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<MeterState>("listening");
  const [failure, setFailure] = useState(COULD_NOT_OPEN);

  useEffect(() => {
    let alive = true;
    let heard = false;
    setState("listening");
    const stopListening = onServerEvent((name, data) => {
      if (name !== "meter") return;
      if (!heard) setState("listening");
      heard = true;
      if (!bars.current) return;
      const lit = Math.round(Math.min(Math.max((data as { level: number }).level, 0), 1) * METER_BARS);
      Array.from(bars.current.children).forEach((bar, index) => {
        (bar as HTMLElement).style.opacity = index < lit ? "1" : "0.2";
      });
    });
    const turnOff = () => sendMeter(device, false);
    const forget = onBeforeClose(turnOff);
    sendMeter(device, true).catch((error: unknown) => {
      if (!alive) return;
      setFailure(errorText(error) || COULD_NOT_OPEN);
      setState("failed");
    });
    const watchdog = setTimeout(() => alive && !heard && setState((current) => (current === "listening" ? "silent" : current)), NO_SIGNAL_AFTER_MS);
    return () => {
      alive = false;
      stopListening();
      clearTimeout(watchdog);
      forget();
      void turnOff().catch(() => undefined);
    };
  }, [device]);

  return (
    <div className="flex items-center gap-4">
      <span className={cn("text-[12.5px]", state === "failed" ? "text-destructive" : "text-primary")}>
        {state === "failed" ? failure : "Speak to test the microphone."}
      </span>
      {state === "listening" ? (
        <div ref={bars} aria-hidden className="ml-auto flex h-5 items-center gap-[3px]">
          {Array.from({ length: METER_BARS }, (_, index) => (
            <span key={index} className="h-full w-[5px] rounded-full bg-primary opacity-20 transition-opacity duration-100" />
          ))}
        </div>
      ) : (
        <span className="ml-auto text-[12.5px] text-muted-foreground">No signal</span>
      )}
    </div>
  );
}

interface MicChoice {
  name: string;
  label: string;
  note: string;
}

function microphoneChoices(microphones: Microphone[], chosen: string): MicChoice[] {
  const windowsDefault = microphones.find((microphone) => microphone.is_default);
  const choices: MicChoice[] = [
    {
      name: "",
      label: "Windows default",
      note: windowsDefault ? `Uses the microphone Windows is set to, now ${windowsDefault.label}.` : "Uses the microphone Windows is set to.",
    },
    ...microphones.map((microphone) => ({
      name: microphone.name,
      label: microphone.label,
      note: microphone.connected ? "" : "Not connected right now.",
    })),
  ];
  if (chosen && !microphones.some((microphone) => microphone.name === chosen)) {
    choices.push({ name: chosen, label: `${chosen} (not connected)`, note: "Not connected right now." });
  }
  return choices;
}

export function MicrophonePicker({ settings, onChoose }: { settings: Settings; onChoose: (name: string) => void }) {
  const { microphones, rescan } = useMicrophones();
  const [error, setError] = useState("");
  return (
    <>
      <div role="radiogroup" aria-label="Microphone" className="flex flex-col gap-2">
        {microphoneChoices(microphones, settings.input_device).map((choice) => {
          const selected = choice.name === settings.input_device;
          return (
            <button
              key={choice.name || "default"}
              role="radio"
              aria-checked={selected}
              onClick={() => onChoose(choice.name)}
              className={cn(
                "rounded-xl border bg-card px-4 py-3 text-left shadow-[0_1px_2px_rgb(0_0_0/0.03)] outline-none transition-colors duration-150 hover:bg-accent/40 focus-visible:ring-2 focus-visible:ring-ring/40",
                selected && "border-primary bg-selected hover:bg-selected",
              )}
            >
              <div className="text-[13.5px] font-medium">{choice.label}</div>
              {selected ? (
                <div className="mt-1">
                  <LevelMeter device={choice.name} />
                </div>
              ) : choice.note ? (
                <div className="mt-0.5 text-[12.5px] text-muted-foreground">{choice.note}</div>
              ) : null}
            </button>
          );
        })}
      </div>
      <Button variant="link" className="mt-2" onClick={() => rescan().then(() => setError(""), (failure: unknown) => setError(errorText(failure)))}>
        Look for new microphones
      </Button>
      {error ? <InlineError>{error}</InlineError> : null}
    </>
  );
}

export function MicrophonePage() {
  const { settings, editSettings } = useStore();
  const { microphones } = useMicrophones();
  const draft = settings?.draft;
  if (!draft) return <PageBody><PageTitle title="Microphone & sounds" /></PageBody>;

  const recording = draft.recording;
  const editRecording = (patch: Partial<RecordingSettings>) =>
    editSettings((current) => ({ ...current, recording: { ...current.recording, ...patch } }));
  const preferred = recording.preferred_devices;
  const connected = new Set(microphones.filter((microphone) => microphone.connected).flatMap((microphone) => [microphone.name, microphone.label]));
  const chosen = draft.input_device;
  const chosenFullName = microphones.find((microphone) => microphone.name === chosen)?.label ?? chosen;
  const canAdd = chosen !== "" && !preferred.includes(chosen) && !preferred.includes(chosenFullName);
  const move = (index: number, step: number) => {
    const order = [...preferred];
    const [moved] = order.splice(index, 1);
    if (moved === undefined) return;
    order.splice(index + step, 0, moved);
    editRecording({ preferred_devices: order });
  };

  return (
    <PageBody>
      <PageTitle title="Microphone & sounds" />
      <Section title="Microphone">
        <MicrophonePicker settings={draft} onChoose={(name) => editSettings((current) => ({ ...current, input_device: name }))} />
      </Section>

      <Section title="Preferred microphones" note="When one of these is plugged in, the first one in the list is used.">
        <Card className="divide-y">
          {preferred.length ? (
            preferred.map((name, index) => (
              <div key={name} className="flex items-center gap-3 px-4 py-2">
                <span className="w-5 text-[12.5px] text-muted-foreground tabular-nums">{index + 1}.</span>
                <span className="min-w-0 flex-1 truncate text-[13px]">
                  {name}
                  {connected.has(name) ? null : <span className="ml-2 text-[12px] text-muted-foreground">not connected</span>}
                </span>
                <Button variant="ghost" size="icon" aria-label={`Move ${name} up`} disabled={index === 0} onClick={() => move(index, -1)}>
                  <ArrowUp />
                </Button>
                <Button variant="ghost" size="icon" aria-label={`Move ${name} down`} disabled={index === preferred.length - 1} onClick={() => move(index, 1)}>
                  <ArrowDown />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label={`Remove ${name}`}
                  onClick={() => editRecording({ preferred_devices: preferred.filter((other) => other !== name) })}
                >
                  <X />
                </Button>
              </div>
            ))
          ) : (
            <p className="px-4 py-3 text-[12.5px] text-muted-foreground">None yet. The microphone chosen above is used.</p>
          )}
        </Card>
        <Button variant="link" className="mt-2" disabled={!canAdd} onClick={() => editRecording({ preferred_devices: [...preferred, chosenFullName] })}>
          Add the chosen microphone
        </Button>
      </Section>

      <Section title="Recording">
        <Rows>
          <Row
            title="Boost quiet speech"
            description="Helps when you are far from the microphone. Raises the volume up to 10x without distorting it. Saved recordings stay as heard."
          >
            <Switch aria-label="Boost quiet speech" checked={draft.boost_quiet_speech} onCheckedChange={(on) => editSettings((current) => ({ ...current, boost_quiet_speech: on }))} />
          </Row>
          <Row title="Warn when the microphone goes silent" description="The pill says so when no sound arrives while you record.">
            <Switch aria-label="Warn when the microphone goes silent" checked={recording.warn_when_silent} onCheckedChange={(on) => editRecording({ warn_when_silent: on })} />
          </Row>
        </Rows>
      </Section>

      <Section title="Sounds">
        <Rows>
          <Row title="Play sounds" description="A short tone when recording starts and stops, and when the app is ready.">
            <input
              aria-label="Sound volume"
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={recording.cue_volume}
              disabled={!recording.cues_enabled}
              onChange={(event) => editRecording({ cue_volume: Number(event.target.value) })}
              className="w-28 disabled:opacity-50"
            />
            <Switch aria-label="Play sounds" checked={recording.cues_enabled} onCheckedChange={(on) => editRecording({ cues_enabled: on })} />
          </Row>
          <Row title="While recording" description="What happens to music and calls from other apps.">
            <Select
              label="While recording"
              className="w-52"
              value={recording.duck_other_audio}
              options={DUCK_CHOICES}
              onChange={(value) => editRecording({ duck_other_audio: value as Duck })}
            />
          </Row>
        </Rows>
      </Section>

      <Section title="Recording limit">
        <Rows>
          <Row title="Stop recording after" description="The pill counts down the last seconds before it stops.">
            <NumberField
              label="Minutes"
              value={Math.round((recording.max_duration_seconds / 60) * 100) / 100}
              min={1}
              max={60}
              step={1}
              suffix="min"
              onChange={(minutes) => editRecording({ max_duration_seconds: minutes * 60 })}
            />
            <NumberField
              label="Warning seconds"
              value={recording.warn_before_cap_seconds}
              min={0}
              max={300}
              step={5}
              suffix="s warning"
              onChange={(seconds) => editRecording({ warn_before_cap_seconds: seconds })}
            />
          </Row>
        </Rows>
      </Section>
    </PageBody>
  );
}
