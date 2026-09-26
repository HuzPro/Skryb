import { Plus, X } from "lucide-react";
import { useState } from "react";
import { Card, ChoiceCards, PageBody, PageTitle, Row, Rows, Section } from "../components/parts";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Select } from "../components/ui/select";
import { api, errorText } from "../lib/api";
import { useStore } from "../lib/store";
import type { OutputMode, PasteMethod } from "../lib/types";

const PASTE_METHODS: { value: PasteMethod; label: string }[] = [
  { value: "ctrl_v", label: "Ctrl+V" },
  { value: "ctrl_shift_v", label: "Ctrl+Shift+V" },
  { value: "shift_insert", label: "Shift+Insert" },
  { value: "type", label: "Type the text" },
];

/** Mirrors `output.paste_method.normalized_process_name`: " Mintty " -> "mintty.exe". */
export function normalizedProcessName(name: string): string {
  const process = name.trim().toLowerCase();
  return process && !process.endsWith(".exe") ? `${process}.exe` : process;
}

export function withOverride(overrides: [string, PasteMethod][], process: string, method: PasteMethod): [string, PasteMethod][] {
  return [...overrides.filter(([name]) => name !== process), [process, method] as [string, PasteMethod]].sort(([a], [b]) => a.localeCompare(b));
}

export function OutputPage() {
  const { settings, editSettings } = useStore();
  const [newProcess, setNewProcess] = useState("");
  const [newMethod, setNewMethod] = useState<PasteMethod>("ctrl_v");
  const [pillNote, setPillNote] = useState("");
  const [pillError, setPillError] = useState("");
  const draft = settings?.draft;
  if (!draft) return <PageBody><PageTitle title="Output" /></PageBody>;

  const overrides = draft.paste_method_overrides;
  const setOverrides = (next: [string, PasteMethod][]) => editSettings((current) => ({ ...current, paste_method_overrides: next }));
  const add = () => {
    const process = normalizedProcessName(newProcess);
    if (!process) return;
    setOverrides(withOverride(overrides, process, newMethod));
    setNewProcess("");
  };
  const resetPill = async () => {
    setPillError("");
    try {
      await api.post("/api/pill/reset-position");
      setPillNote("Reset. The pill will appear at the bottom centre next time.");
    } catch (error) {
      setPillError(errorText(error));
    }
  };

  return (
    <PageBody>
      <PageTitle title="Output" />
      <Section title="Paste">
        <ChoiceCards<OutputMode>
          label="Paste"
          value={draft.output_mode}
          onChange={(mode) => editSettings((current) => ({ ...current, output_mode: mode }))}
          choices={[
            { value: "direct_paste", title: "Paste only", description: "Types into the app and leaves your clipboard as it was." },
            { value: "paste_and_clipboard", title: "Paste and copy", description: "Types into the app and keeps the text on the clipboard." },
            { value: "clipboard_only", title: "Clipboard only", description: "Copies the text; you paste it where you want." },
          ]}
        />
      </Section>

      <Section
        title="Paste method per app"
        note="For apps that do not take Ctrl+V: type the app's .exe name, pick how to paste into it and click Add. mintty gets Shift+Insert automatically."
      >
        <Card className="divide-y">
          <div className="grid grid-cols-[1fr_180px_28px] gap-3 px-4 py-2 text-[12px] font-medium text-muted-foreground">
            <span>App</span>
            <span>Paste method</span>
            <span />
          </div>
          {overrides.map(([process, method]) => (
            <div key={process} className="grid grid-cols-[1fr_180px_28px] items-center gap-3 px-4 py-1.5">
              <span className="truncate text-[13px]">{process}</span>
              <Select
                label={`Paste method for ${process}`}
                value={method}
                options={PASTE_METHODS}
                onChange={(value) => setOverrides(withOverride(overrides, process, value as PasteMethod))}
              />
              <Button variant="ghost" size="icon" aria-label={`Remove ${process}`} onClick={() => setOverrides(overrides.filter(([name]) => name !== process))}>
                <X />
              </Button>
            </div>
          ))}
          <form
            className="grid grid-cols-[1fr_180px_auto] items-center gap-3 px-4 py-2.5"
            onSubmit={(event) => {
              event.preventDefault();
              add();
            }}
          >
            <Input aria-label="App name" placeholder="App, e.g. notepad.exe" value={newProcess} onChange={(event) => setNewProcess(event.target.value)} />
            <Select label="New paste method" value={newMethod} options={PASTE_METHODS} onChange={(value) => setNewMethod(value as PasteMethod)} />
            <Button type="submit" variant="outline" disabled={!newProcess.trim()}>
              <Plus />
              Add
            </Button>
          </form>
        </Card>
      </Section>

      <Section title="Recording pill">
        <Rows>
          <Row title="Pill position" description={pillNote || "Drag the pill to move it; it stays there for the next dictation."} error={pillError}>
            <Button variant="outline" onClick={() => void resetPill()}>
              Reset position
            </Button>
          </Row>
        </Rows>
      </Section>
    </PageBody>
  );
}
