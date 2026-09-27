import { Plus, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { InlineError, KeyCaps, PageBody, PageTitle, Row, Rows, Section } from "../components/parts";
import { Button } from "../components/ui/button";
import { Select } from "../components/ui/select";
import { Switch } from "../components/ui/switch";
import { api, errorText } from "../lib/api";
import { endsOnKey, validateBindings } from "../lib/keys";
import { onBeforeClose } from "../lib/platform";
import { useServerEvent } from "../lib/sse";
import { useStore } from "../lib/store";
import type { Binding, BindingAction, BindingMode, Settings } from "../lib/types";
import { cn } from "../lib/utils";

const MODE_CHOICES = [
  { value: "toggle", label: "Toggle" },
  { value: "hold", label: "Hold" },
];

const OPTIONAL: { action: BindingAction; title: string; description: string; withMode: boolean }[] = [
  { action: "edit_selection", title: "Edit selected text by voice", description: "Select text, press this and say how to change it.", withMode: true },
  { action: "paste_last", title: "Paste last dictation", description: "Paste it again where the cursor is.", withMode: false },
  { action: "copy_last", title: "Copy last dictation", description: "Put it on the clipboard.", withMode: false },
];

const PRESS_ACTIONS: BindingAction[] = ["paste_last", "copy_last"];

export function setBindingKeys(bindings: Binding[], index: number, keys: string): Binding[] {
  return bindings.map((binding, at) => (at === index ? { ...binding, keys } : binding));
}

/** The first shortcut for `action` takes `keys`; empty keys remove it. */
export function setOptionalKeys(bindings: Binding[], action: BindingAction, keys: string): Binding[] {
  const index = bindings.findIndex((binding) => binding.action === action);
  if (!keys) return bindings.filter((binding) => binding.action !== action);
  if (index >= 0) return setBindingKeys(bindings, index, keys);
  return [...bindings, { keys, mode: PRESS_ACTIONS.includes(action) ? "press" : "toggle", action }];
}

interface Capture {
  recordingRow: string | null;
  start: (row: string, apply: (keys: string) => void) => void;
  stop: () => void;
  error: { row: string; message: string } | null;
}

/** Records one shortcut at a time through the core, which pauses the app's own hotkeys meanwhile. */
export function useCapture(): Capture {
  const [recordingRow, setRecordingRow] = useState<string | null>(null);
  const [error, setError] = useState<Capture["error"]>(null);
  const apply = useRef<(keys: string) => void>(() => undefined);
  const rowRef = useRef<string | null>(null);

  const stop = useCallback(() => {
    if (rowRef.current === null) return;
    rowRef.current = null;
    setRecordingRow(null);
    void api.post("/api/shortcuts/capture", { on: false }).catch(() => undefined);
  }, []);

  const start = useCallback((row: string, onKeys: (keys: string) => void) => {
    apply.current = onKeys;
    rowRef.current = row;
    setRecordingRow(row);
    setError(null);
    api.post("/api/shortcuts/capture", { on: true }).catch((failure: unknown) => {
      rowRef.current = null;
      setRecordingRow(null);
      setError({ row, message: errorText(failure) });
    });
  }, []);

  useServerEvent((name, data) => {
    const row = rowRef.current;
    if (name !== "capture" || row === null) return;
    const event = data as { keys?: string; error?: string; cancelled?: boolean };
    if (event.error && !event.cancelled) setError({ row, message: event.error });
    else if (event.keys && event.keys !== "esc") {
      apply.current(event.keys);
      if (!endsOnKey(event.keys)) return;
    }
    stop();
  });

  useEffect(() => {
    if (recordingRow === null) return undefined;
    return onBeforeClose(() => api.post("/api/shortcuts/capture", { on: false }));
  }, [recordingRow]);
  useEffect(() => stop, [stop]);

  return { recordingRow, start, stop, error };
}

export function ShortcutField({
  keys,
  recording,
  onRecord,
  onStop,
  mode,
  onMode,
  onClear,
  clearLabel,
  clearDisabled,
}: {
  keys: string;
  recording: boolean;
  onRecord: () => void;
  onStop: () => void;
  mode?: BindingMode;
  onMode?: (mode: BindingMode) => void;
  onClear?: () => void;
  clearLabel?: string;
  clearDisabled?: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <button
        onClick={recording ? onStop : onRecord}
        aria-label={recording ? "Recording a shortcut" : "Record a shortcut"}
        className={cn(
          "flex h-9 w-[300px] min-w-0 items-center rounded-lg border bg-card px-2 text-left shadow-xs outline-none transition-colors duration-150 hover:bg-accent/40 focus-visible:ring-2 focus-visible:ring-ring/40",
          recording && "border-primary ring-2 ring-primary/20",
        )}
      >
        {recording ? <span className="text-[12.5px] text-primary">Press the keys or mouse button...</span> : <KeyCaps combo={keys} />}
      </button>
      <Button variant="outline" className="w-[76px]" onClick={recording ? onStop : onRecord}>
        {recording ? "Stop" : "Record"}
      </Button>
      {mode && onMode ? (
        <Select label="How it fires" className="w-[92px]" value={mode} options={MODE_CHOICES} onChange={(value) => onMode(value as BindingMode)} />
      ) : onClear ? (
        <span className="w-[92px]" />
      ) : null}
      {onClear ? (
        <Button variant="ghost" size="icon" aria-label={clearLabel} title={clearLabel} disabled={clearDisabled} onClick={onClear}>
          <X />
        </Button>
      ) : null}
    </div>
  );
}

export function ShortcutsPage() {
  const { settings, editSettings } = useStore();
  const capture = useCapture();
  const draft = settings?.draft;
  if (!draft) return <PageBody><PageTitle title="Shortcuts" /></PageBody>;

  const bindings = draft.hotkey_bindings;
  const editBindings = (update: (bindings: Binding[]) => Binding[]) =>
    editSettings((current: Settings) => ({ ...current, hotkey_bindings: update(current.hotkey_bindings) }));
  const filled = bindings.map((binding, index) => ({ binding, index })).filter(({ binding }) => binding.keys.trim());
  const problem = validateBindings(filled.map(({ binding }) => binding));
  const problemAt = problem && problem.index >= 0 ? (filled[problem.index]?.index ?? -1) : -1;
  const dictation = bindings.map((binding, index) => ({ binding, index })).filter(({ binding }) => binding.action === "dictate");
  const errorFor = (row: string, index: number) =>
    (capture.error?.row === row ? capture.error.message : "") || (index === problemAt && problem ? problem.message : "");

  const addDictation = () => {
    const index = bindings.length;
    editBindings((current) => [...current, { keys: "", mode: "toggle", action: "dictate" }]);
    capture.start(`dictate:${index}`, (keys) => editBindings((current) => setBindingKeys(current, index, keys)));
  };

  return (
    <PageBody>
      <PageTitle title="Shortcuts" />
      <Section title="Dictation">
        <Rows>
          <Row
            stacked
            title="Dictate"
            description="Start and stop dictation. Toggle: press to start, press again to stop. Hold: hold to talk, let go to stop."
            error={problem && problem.index === -1 ? problem.message : ""}
          >
            <div className="flex flex-col gap-2">
              {dictation.map(({ binding, index }) => {
                const row = `dictate:${index}`;
                const error = errorFor(row, index);
                return (
                  <div key={row}>
                    <ShortcutField
                      keys={binding.keys}
                      recording={capture.recordingRow === row}
                      onRecord={() => capture.start(row, (keys) => editBindings((current) => setBindingKeys(current, index, keys)))}
                      onStop={capture.stop}
                      mode={binding.mode === "hold" ? "hold" : "toggle"}
                      onMode={(mode) => editBindings((current) => current.map((other, at) => (at === index ? { ...other, mode } : other)))}
                      onClear={() => {
                        capture.stop();
                        editBindings((current) => current.filter((_, at) => at !== index));
                      }}
                      clearLabel="Remove this shortcut"
                      clearDisabled={dictation.length <= 1}
                    />
                    {error ? <InlineError>{error}</InlineError> : null}
                  </div>
                );
              })}
              <Button variant="outline" size="sm" className="mt-1 self-start" onClick={addDictation}>
                <Plus />
                Add another
              </Button>
            </div>
          </Row>
          <Row title="Double-tap to lock" description="For a Hold shortcut. Two quick presses keep it recording; the next press stops it.">
            <Switch
              aria-label="Double-tap to lock"
              checked={draft.double_tap_locks_hold}
              onCheckedChange={(on) => editSettings((current) => ({ ...current, double_tap_locks_hold: on }))}
            />
          </Row>
        </Rows>
      </Section>

      <Section title="More shortcuts">
        <Rows>
          {OPTIONAL.map(({ action, title, description, withMode }) => {
            const index = bindings.findIndex((binding) => binding.action === action);
            const binding = bindings[index];
            const row = `optional:${action}`;
            const error = errorFor(row, index);
            return (
              <Row key={action} stacked title={title} description={description}>
                <div>
                  <ShortcutField
                    keys={binding?.keys ?? ""}
                    recording={capture.recordingRow === row}
                    onRecord={() => capture.start(row, (keys) => editBindings((current) => setOptionalKeys(current, action, keys)))}
                    onStop={capture.stop}
                    {...(withMode
                      ? {
                          mode: binding?.mode === "hold" ? "hold" : "toggle",
                          onMode: (mode: BindingMode) =>
                            editBindings((current) =>
                              current.some((other) => other.action === action)
                                ? current.map((other) => (other.action === action ? { ...other, mode } : other))
                                : current,
                            ),
                        }
                      : {})}
                    onClear={() => {
                      capture.stop();
                      editBindings((current) => setOptionalKeys(current, action, ""));
                    }}
                    clearLabel="Turn this shortcut off"
                    clearDisabled={!binding}
                  />
                  {error ? <InlineError>{error}</InlineError> : null}
                </div>
              </Row>
            );
          })}
        </Rows>
        <p className="mt-2.5 text-[12.5px] text-muted-foreground">
          Click Record, then press the keys or mouse button you want. Mouse buttons 4 and 5 and the middle button work too. Clear a shortcut to turn it off.
        </p>
      </Section>
    </PageBody>
  );
}
