import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Pencil, Plus, X } from "lucide-react";
import { useState } from "react";
import { formatGigabytes } from "../components/models";
import { Card, ChoiceCards, Disclosure, InlineError, NumberField, PageBody, PageTitle, ProgressBar, Row, Rows, Section, StatusDot } from "../components/parts";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Input, Textarea } from "../components/ui/input";
import { Select } from "../components/ui/select";
import { Switch } from "../components/ui/switch";
import { api, errorText } from "../lib/api";
import { isDirty } from "../lib/draft";
import { chooseOpenPath } from "../lib/platform";
import { useStore, useSyncedDraft } from "../lib/store";
import type { Cleanup, Transform, WritingSettings, WritingStatus } from "../lib/types";

const BLURB = "Cleans up and formats your dictation on this computer. Nothing leaves it.";
const PREVIEW_CHARS = 90;

type Tone = "success" | "warning" | "destructive" | "muted" | "primary";

function lastLine(text: string): string {
  return text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).at(-1) ?? "";
}

/** The short state beside the model's name and the line under it. */
export function describeWriting(status: WritingStatus): { pill: string; tone: Tone; detail: string } {
  const share = status.bytes_total ? status.bytes_done / status.bytes_total : 0;
  switch (status.state) {
    case "not_installed":
      return { pill: "Not installed", tone: "muted", detail: BLURB };
    case "installing":
      return {
        pill: `Installing ${Math.round(share * 100)}%`,
        tone: "primary",
        detail: `Downloading ${formatGigabytes(status.bytes_done)} of ${formatGigabytes(status.bytes_total)}`,
      };
    case "starting":
      return { pill: "Starting", tone: "primary", detail: BLURB };
    case "ready":
      return {
        pill: "Ready",
        tone: "success",
        detail: status.last_cleanup_seconds === null ? BLURB : `Last cleanup took ${status.last_cleanup_seconds.toFixed(2)} s.`,
      };
    case "failed":
      return { pill: "Failed", tone: "destructive", detail: `Failed: ${lastLine(status.detail) || "no reason was given"}` };
    case "stopped":
      return { pill: "Waiting", tone: "muted", detail: "Starts once dictation is ready." };
  }
}

function WritingModelCard() {
  const status = useQuery({ queryKey: ["writing"], queryFn: () => api.get<WritingStatus>("/api/writing") }).data;
  const [error, setError] = useState("");
  if (!status) return <Card className="h-[74px]">{null}</Card>;
  const { pill, tone, detail } = describeWriting(status);
  const run = async (action: string) => {
    setError("");
    try {
      await api.post(`/api/writing/${action}`);
    } catch (failure) {
      setError(errorText(failure));
    }
  };
  return (
    <Card className="px-4 py-3">
      <div className="flex items-center gap-4">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2.5">
            <span className="text-[13.5px] font-medium">Local writing model</span>
            <span className="flex items-center gap-1.5 text-[12px] text-muted-foreground">
              <StatusDot tone={tone} />
              {pill}
            </span>
          </div>
          <div className="mt-0.5 text-[12.5px] text-muted-foreground">{detail}</div>
        </div>
        <div className="flex shrink-0 gap-2">
          {status.state === "not_installed" ? <Button onClick={() => void run("install")}>Install ({formatGigabytes(status.install_bytes)})</Button> : null}
          {status.state === "installing" ? (
            <Button variant="outline" onClick={() => void run("cancel")}>
              Cancel
            </Button>
          ) : null}
          {status.state === "failed" ? (
            <>
              <Button variant="outline" onClick={() => void run("open-log")}>
                Open log
              </Button>
              <Button onClick={() => void run("retry")}>Retry</Button>
            </>
          ) : null}
          {status.state === "stopped" ? (
            <Button variant="outline" onClick={() => void run("restart")}>
              Start now
            </Button>
          ) : null}
        </div>
      </div>
      {status.state === "installing" ? <ProgressBar className="mt-3" value={status.bytes_total ? status.bytes_done / status.bytes_total : 0} /> : null}
      {error ? <InlineError>{error}</InlineError> : null}
    </Card>
  );
}

function LearnedStyle() {
  const status = useQuery({ queryKey: ["writing"], queryFn: () => api.get<WritingStatus>("/api/writing") }).data;
  const { confirm } = useStore();
  const client = useQueryClient();
  const [error, setError] = useState("");
  const run = async (action: string) => {
    setError("");
    try {
      await api.post(`/api/writing/${action}`);
      await client.invalidateQueries({ queryKey: ["writing"] });
    } catch (failure) {
      setError(errorText(failure));
    }
  };
  return (
    <Row title="Learned so far" description={status?.learned_style_summary || "Nothing yet."} error={error}>
      <Button variant="outline" onClick={() => void run("forget-style")}>
        Forget learned style
      </Button>
      <Button
        variant="destructive"
        onClick={async () => {
          if (await confirm("Reset learning?", "Forget every learned word and the learned style. This cannot be undone.", "Reset")) await run("reset-learning");
        }}
      >
        Reset learning
      </Button>
    </Row>
  );
}

interface Editing {
  index: number;
  name: string;
  instruction: string;
  error: string;
}

function Transformations() {
  const { transforms, dispatchTransforms, saveTransforms } = useStore();
  const client = useQueryClient();
  useSyncedDraft<Transform[]>("transforms", "/api/transforms", dispatchTransforms);
  const [editing, setEditing] = useState<Editing | null>(null);
  const [error, setError] = useState("");
  const list = transforms?.draft ?? [];

  const commit = () => {
    if (!editing) return;
    const name = editing.name.trim();
    const instruction = editing.instruction.trim();
    if (!name || !instruction) return setEditing({ ...editing, error: "A transformation needs a name and an instruction." });
    if (list.some((transform, index) => transform.name === name && index !== editing.index)) {
      return setEditing({ ...editing, error: "Another transformation already has that name." });
    }
    dispatchTransforms({
      type: "edit",
      update: (current) =>
        editing.index < current.length
          ? current.map((transform, index) => (index === editing.index ? { ...transform, name, instruction } : transform))
          : [...current, { name, instruction, builtin: false }],
    });
    setEditing(null);
  };
  const restore = async () => {
    setError("");
    try {
      await api.post("/api/transforms/restore-builtins");
      const fresh = await api.get<Transform[]>("/api/transforms");
      client.setQueryData(["transforms"], fresh);
      dispatchTransforms({ type: "committed", value: fresh });
    } catch (failure) {
      setError(errorText(failure));
    }
  };

  return (
    <Section
      title="Transformations"
      note="Instructions the writing model can apply to a whole dictation."
      action={
        <Button variant="link" onClick={() => void restore()}>
          Restore built-ins
        </Button>
      }
    >
      <Card className="divide-y">
        {list.map((transform, index) =>
          editing?.index === index ? (
            <TransformEditor key={index} editing={editing} onChange={setEditing} onDone={commit} onCancel={() => setEditing(null)} />
          ) : (
            <div key={`${transform.name}-${index}`} className="flex items-center gap-3 px-4 py-2.5">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 text-[13.5px] font-medium">
                  {transform.name}
                  {transform.builtin ? <Badge>Built-in</Badge> : null}
                </div>
                <div className="truncate text-[12.5px] text-muted-foreground">
                  {transform.instruction.length > PREVIEW_CHARS ? `${transform.instruction.slice(0, PREVIEW_CHARS - 1)}…` : transform.instruction}
                </div>
              </div>
              <Button variant="ghost" size="icon" aria-label={`Edit ${transform.name}`} onClick={() => setEditing({ index, name: transform.name, instruction: transform.instruction, error: "" })}>
                <Pencil />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                aria-label={`Remove ${transform.name}`}
                onClick={() => dispatchTransforms({ type: "edit", update: (current) => current.filter((_, at) => at !== index) })}
              >
                <X />
              </Button>
            </div>
          ),
        )}
        {editing && editing.index >= list.length ? (
          <TransformEditor editing={editing} onChange={setEditing} onDone={commit} onCancel={() => setEditing(null)} />
        ) : null}
        {!list.length && !editing ? <p className="px-4 py-3 text-[12.5px] text-muted-foreground">No transformations.</p> : null}
      </Card>
      <div className="mt-2.5 flex items-center gap-2">
        <Button variant="outline" size="sm" disabled={editing !== null} onClick={() => setEditing({ index: list.length, name: "", instruction: "", error: "" })}>
          <Plus />
          Add transformation
        </Button>
        {isDirty(transforms) ? (
          <>
            <span className="ml-auto text-[12.5px] text-muted-foreground">Transformations changed</span>
            <Button variant="ghost" size="sm" onClick={() => dispatchTransforms({ type: "discard" })}>
              Discard
            </Button>
            <Button size="sm" onClick={async () => setError(await saveTransforms())}>
              Save transformations
            </Button>
          </>
        ) : null}
      </div>
      {error ? <InlineError>{error}</InlineError> : null}
    </Section>
  );
}

function TransformEditor({ editing, onChange, onDone, onCancel }: { editing: Editing; onChange: (next: Editing) => void; onDone: () => void; onCancel: () => void }) {
  return (
    <div className="flex flex-col gap-2 bg-muted/40 px-4 py-3">
      <Input aria-label="Name" placeholder="Name" value={editing.name} autoFocus onChange={(event) => onChange({ ...editing, name: event.target.value })} />
      <Textarea
        aria-label="Instruction"
        placeholder="Instruction, e.g. Make it concise. Keep every fact, name and number."
        value={editing.instruction}
        onChange={(event) => onChange({ ...editing, instruction: event.target.value })}
      />
      {editing.error ? <p className="text-[12.5px] text-destructive">{editing.error}</p> : null}
      <div className="flex justify-end gap-2">
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
        <Button size="sm" onClick={onDone}>
          Done
        </Button>
      </div>
    </div>
  );
}

async function browse(title: string, name: string, extension: string, apply: (path: string) => void) {
  const path = await chooseOpenPath(title, name, extension);
  if (path) apply(path);
}

export function WritingPage() {
  const { settings, editSettings, transforms } = useStore();
  const draft = settings?.draft;
  if (!draft) return <PageBody><PageTitle title="Writing" /></PageBody>;
  const writing = draft.writing;
  const edit = (patch: Partial<WritingSettings>) => editSettings((current) => ({ ...current, writing: { ...current.writing, ...patch } }));
  const names = (transforms?.draft ?? []).map((transform) => transform.name);

  return (
    <PageBody>
      <PageTitle title="Writing" />
      <Section title="Writing model">
        <WritingModelCard />
        <Disclosure label="Advanced" defaultOpen={Boolean(writing.runtime_path || writing.model_path)}>
          <Rows>
            <Row title="llama-server.exe" description="Leave empty to use the installed one.">
              <Input aria-label="llama-server.exe path" className="w-72" value={writing.runtime_path} onChange={(event) => edit({ runtime_path: event.target.value })} />
              <Button variant="outline" onClick={() => void browse("Choose llama-server.exe", "Programs", "exe", (path) => edit({ runtime_path: path }))}>
                Browse...
              </Button>
            </Row>
            <Row title="Model (.gguf)" description="Leave empty to use the installed one.">
              <Input aria-label="Model path" className="w-72" value={writing.model_path} onChange={(event) => edit({ model_path: event.target.value })} />
              <Button variant="outline" onClick={() => void browse("Choose a GGUF model", "GGUF models", "gguf", (path) => edit({ model_path: path }))}>
                Browse...
              </Button>
            </Row>
          </Rows>
        </Disclosure>
      </Section>

      <Section title="Cleanup">
        <ChoiceCards<Cleanup>
          label="Cleanup"
          value={writing.cleanup}
          onChange={(cleanup) => edit({ cleanup })}
          choices={[
            { value: "off", title: "Off", description: "Paste exactly what was recognised." },
            { value: "light", title: "Light", description: "Fix fillers, slips and changes of mind, keep my wording." },
            { value: "polished", title: "Polished", description: "Also tidy grammar and repetition." },
          ]}
        />
        <Rows className="mt-2.5">
          <Row title="Turn spoken lists into lists" description="Numbered steps and bullet points when you dictate them.">
            <Switch aria-label="Turn spoken lists into lists" checked={writing.lists} onCheckedChange={(lists) => edit({ lists })} />
          </Row>
          <Row title="Give up after" description="Then the raw dictation is pasted. Long dictations may skip cleanup when this is short.">
            <NumberField label="Seconds" value={writing.deadline_seconds} min={0.5} max={30} step={0.5} suffix="seconds" onChange={(deadline_seconds) => edit({ deadline_seconds })} />
          </Row>
          <Row
            title="Use the text around the cursor"
            description="And the window title, to spell names and terms. Never read from password fields or password managers."
          >
            <Switch aria-label="Use the text around the cursor" checked={writing.context} onCheckedChange={(context) => edit({ context })} />
          </Row>
          <Row title="After each dictation, apply" description="Runs after cleanup, every time.">
            <Select
              label="After each dictation, apply"
              className="w-52"
              value={names.includes(writing.after_dictation_transform) ? writing.after_dictation_transform : ""}
              options={[{ value: "", label: "None" }, ...names.map((name) => ({ value: name, label: name }))]}
              onChange={(after_dictation_transform) => edit({ after_dictation_transform })}
            />
          </Row>
        </Rows>
      </Section>

      <Section title="Learning" note="Learning watches edits you make to text this app pasted, for up to 15 minutes after each dictation. Everything stays on this device.">
        <Rows>
          <Row title="Learn words I correct" description="Names and terms you fix after pasting are spelled right next time.">
            <Switch aria-label="Learn words I correct" checked={writing.learn_words} onCheckedChange={(learn_words) => edit({ learn_words })} />
          </Row>
          <Row title="Learn my writing style" description="How you edit what was pasted shapes the next cleanup.">
            <Switch aria-label="Learn my writing style" checked={writing.learn_style} onCheckedChange={(learn_style) => edit({ learn_style })} />
          </Row>
          <Row stacked title="Style notes" description="Tell the writing model how you like to write.">
            <Textarea
              aria-label="Style notes"
              className="w-full"
              placeholder="e.g. British spelling, no exclamation marks, sign off with Cheers"
              value={writing.style_notes}
              onChange={(event) => edit({ style_notes: event.target.value })}
            />
          </Row>
          <LearnedStyle />
        </Rows>
      </Section>

      <Transformations />
    </PageBody>
  );
}
