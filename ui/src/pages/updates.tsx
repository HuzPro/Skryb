import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, FileDown, FileUp, FolderOpen, Trash2 } from "lucide-react";
import { useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { installDescription, installTitle, useInstall } from "../components/install";
import { PAGE_TITLE, PageBody, ProgressBar, RadioDot, Row, Rows, Section } from "../components/parts";
import { useAppInfo } from "../components/shell";
import { Button } from "../components/ui/button";
import { Switch } from "../components/ui/switch";
import { api, errorText } from "../lib/api";
import { chooseOpenPath, chooseSavePath } from "../lib/platform";
import { queries } from "../lib/queries";
import { useStore } from "../lib/store";
import { deleteUsageData, USAGE_DATA_PROMISE } from "../lib/telemetry";
import type { Colourway, Theme, UpdateStatus } from "../lib/types";
import { cn } from "../lib/utils";

const THEMES: { value: Theme; title: string }[] = [
  { value: "default", title: "Default" },
  { value: "glass", title: "Glass" },
  { value: "material", title: "Material" },
  { value: "paper", title: "Paper" },
];

const events = (count: number) => `${count.toLocaleString()} ${count === 1 ? "event" : "events"}`;

/** The usage data row's title: "Nothing kept yet", "1 event kept", "1,234 events kept". */
export function keptTitle(kept: number): string {
  return kept ? `${events(kept)} kept` : "Nothing kept yet";
}

/** Usage data: the switch that keeps it, how much is kept, and exporting or deleting all of it. */
function UsageData() {
  const { settings, editSettings, confirm } = useStore();
  // Asked again each time Settings opens, and each time the window's own events are kept (telemetry.ts).
  const kept = useQuery({ ...queries.telemetry, staleTime: 0 }).data?.events ?? 0;
  const [note, setNote] = useState({ text: "", error: false });
  const draft = settings?.draft;
  if (!draft) return null;
  const act = async (work: () => Promise<string>) => {
    try {
      setNote({ text: await work(), error: false });
    } catch (error) {
      setNote({ text: errorText(error), error: true });
    }
  };
  const exportData = () =>
    act(async () => {
      const path = await chooseSavePath("Export usage data", `skryb-usage-${new Date().toISOString().slice(0, 10)}.json`, "Usage data", "json");
      if (!path) return "";
      const exported = await api.post<{ events: number }>("/api/telemetry/export", { path });
      return `Exported ${events(exported.events)}.`;
    });
  const deleteData = () =>
    act(async () => {
      if (!(await confirm("Delete usage data", "Everything Skryb has kept about how you use it goes. This can't be undone.", "Delete"))) return "";
      await deleteUsageData();
      return "Deleted.";
    });
  return (
    <Section title="Usage data" note="Off unless you turn it on. It stays on this computer.">
      <Rows>
        <Row title="Keep usage data" description={USAGE_DATA_PROMISE}>
          <Switch aria-label="Keep usage data" checked={draft.telemetry} onCheckedChange={(telemetry) => editSettings((current) => ({ ...current, telemetry }))} />
        </Row>
        <Row
          title={keptTitle(kept)}
          description="Export it as one file to send to Skryb's developers, or to read yourself."
          error={note.error ? note.text : undefined}
        >
          {!note.error && note.text ? <span className="text-[12.5px] text-muted-foreground">{note.text}</span> : null}
          <Button variant="outline" data-track="export-usage" disabled={!kept} onClick={() => void exportData()}>
            <FileDown />
            Export
          </Button>
          <Button variant="outline" data-track="delete-usage" disabled={!kept} onClick={() => void deleteData()}>
            <Trash2 />
            Delete
          </Button>
        </Row>
      </Rows>
    </Section>
  );
}

const COLOURWAYS: { value: Colourway; label: string }[] = [
  { value: "green", label: "Skryb green" },
  { value: "terracotta", label: "Terracotta" },
  { value: "ink", label: "Ink blue" },
  { value: "plum", label: "Plum" },
  { value: "graphite", label: "Graphite" },
];

/** One card per theme; the colour is picked once, below, for every theme. */
function ThemeCards({ theme, onPick }: { theme: Theme; onPick: (theme: Theme) => void }) {
  return (
    <div role="radiogroup" aria-label="Theme" className="grid grid-cols-4 gap-3">
      {THEMES.map(({ value, title }) => {
        const selected = value === theme;
        return (
          <button
            key={value}
            role="radio"
            aria-checked={selected}
            onClick={() => onPick(value)}
            className={cn(
              "surface flex items-center gap-2 rounded-xl border border-edge bg-card px-4 py-3 text-[13.5px] font-medium shadow-card outline-none transition-colors hover:bg-accent/50 focus-visible:ring-2 focus-visible:ring-ring/40",
              selected && "border-primary bg-selected hover:bg-selected",
            )}
          >
            <RadioDot selected={selected} />
            {title}
          </button>
        );
      })}
    </div>
  );
}

/** One swatch per colourway, each drawn as the chosen theme draws it. */
function ColourSwatches({ theme, colourway, onPick }: { theme: Theme; colourway: Colourway; onPick: (colourway: Colourway) => void }) {
  return (
    <div role="radiogroup" aria-label="Colour" className="flex items-center gap-2">
      {COLOURWAYS.map(({ value, label }) => (
        <button
          key={value}
          role="radio"
          aria-checked={value === colourway}
          aria-label={label}
          title={label}
          onClick={() => onPick(value)}
          className={cn(
            "grid size-7 place-items-center rounded-full outline-none ring-offset-2 ring-offset-card transition-shadow focus-visible:ring-2 focus-visible:ring-ring/40",
            value === colourway && "ring-2 ring-primary",
          )}
        >
          <span data-theme={theme} data-colourway={value} className="size-5 rounded-full bg-primary" />
        </button>
      ))}
    </div>
  );
}

/** The same colourway for every theme, as one colour picked once. */
export function everyTheme(colourway: Colourway): Record<Theme, Colourway> {
  return Object.fromEntries(THEMES.map(({ value }) => [value, colourway])) as Record<Theme, Colourway>;
}

/** What the Updates section says about *update*; null means a git checkout, which updates through git. */
export function describeUpdate(update: UpdateStatus | null, running: string): { title: string; description: string } {
  if (update === null) {
    return { title: `Skryb ${running}`, description: "Runs from source and updates through git." };
  }
  const percent = update.bytes_total > 0 ? Math.floor((update.bytes_done * 100) / update.bytes_total) : 0;
  switch (update.state) {
    case "checking":
      return { title: `Skryb ${running}`, description: "Checking GitHub for a newer version..." };
    case "up_to_date":
      return { title: `Skryb ${running}`, description: "You have the latest version." };
    case "downloading":
      // Said the same whether a check or "Restart and update" started it.
      return { title: `Downloading Skryb ${update.version}`, description: `${percent}% downloaded.` };
    case "ready":
      return { title: `Skryb ${update.version} is ready`, description: "Skryb restarts to install it, or anything newer, once any dictation finishes." };
    case "installing":
      return { title: `Installing Skryb ${update.version}`, description: "Skryb will close and open again in a minute." };
    case "updated":
      return { title: `Skryb ${running}`, description: "Updated just now. You have the latest version." };
    case "failed":
      return { title: `Skryb ${running}`, description: "The last check didn't work." };
    case "off":
      return { title: `Skryb ${running}`, description: "Automatic checks are off." };
    default:
      return { title: `Skryb ${running}`, description: "Skryb checks shortly after it starts." };
  }
}

/** What the Updates row offers: a bar while downloading, else the check, and "Restart and update" beside it once a version is ready. */
export function updateControls(update: UpdateStatus | null): ("progress" | "check" | "install")[] {
  if (update === null) return [];
  if (update.state === "downloading") return ["progress"];
  return update.state === "ready" ? ["check", "install"] : ["check"];
}

/**
 * The page title with a way out: back to the page Settings was opened from, or Home when the window opened on it.
 * The arrow fills the inset before the text line, starting at the cards' edge, so the title starts where every other page's does.
 */
function SettingsTitle() {
  const navigate = useNavigate();
  const opened = useLocation().key === "default";
  return (
    <div className="mb-1 flex min-h-10 items-center">
      <Button variant="ghost" size="icon" aria-label="Back" title="Back" className="-ml-[7px] size-6 text-muted-foreground hover:text-foreground" onClick={() => void (opened ? navigate("/home") : navigate(-1))}>
        <ArrowLeft />
      </Button>
      <h1 className={PAGE_TITLE}>Settings</h1>
    </div>
  );
}

/** Skryb's own settings, the same in both modes: updates, the look, the settings file and the logs. */
export function SettingsPage() {
  const app = useAppInfo();
  const install = useInstall();
  const { settings, editSettings, discardSettings, anyDirty, askUnsaved, saveAll, flushDictionary } = useStore();
  const client = useQueryClient();
  const [requestError, setRequestError] = useState("");
  const [fileNote, setFileNote] = useState({ text: "", error: false });
  const draft = settings?.draft;
  // Skryb is closed while setup runs, so this shows without it.
  if (install) {
    return (
      <PageBody>
        <SettingsTitle />
        <Section title="Updates">
          <Rows>
            <Row title={installTitle(install)} description={<span className="selectable">{installDescription(install)}</span>} stacked>
              {install.progress.step === "failed" ? null : <ProgressBar value={install.estimate.fraction} className="w-full" />}
            </Row>
          </Rows>
        </Section>
      </PageBody>
    );
  }
  if (!app || !draft) return <PageBody><SettingsTitle /></PageBody>;

  const update = app.update;
  const colourway = draft.colourways[draft.theme];
  const { title, description } = describeUpdate(update, app.version);
  const busy = update !== null && ["checking", "downloading", "installing"].includes(update.state);
  const send = async (path: string) => {
    setRequestError("");
    try {
      await api.post(path);
    } catch (error) {
      setRequestError(errorText(error));
    }
  };
  // The window closes for the new version's, so unsaved changes are settled first, as on closing it.
  const restartAndUpdate = async () => {
    if (anyDirty()) {
      const choice = await askUnsaved();
      if (choice === "cancel" || (choice !== "discard" && !(await saveAll()))) return;
    }
    await flushDictionary();
    await send("/api/updates/install");
  };
  const fileAction = async (work: () => Promise<string>) => {
    try {
      setFileNote({ text: await work(), error: false });
    } catch (error) {
      setFileNote({ text: errorText(error), error: true });
    }
  };
  const exportSettings = () =>
    fileAction(async () => {
      const path = await chooseSavePath("Export settings", "skryb-settings.json", "Settings", "json");
      if (!path) return "";
      await api.post("/api/settings/export", { path });
      return "Settings exported.";
    });
  const importSettings = () =>
    fileAction(async () => {
      const path = await chooseOpenPath("Import settings", "Settings", "json");
      if (!path) return "";
      await api.post("/api/settings/import", { path });
      discardSettings();
      await client.invalidateQueries({ queryKey: ["settings"] });
      return "Settings imported.";
    });

  return (
    <PageBody>
      <SettingsTitle />
      <Section title="Updates">
        <Rows>
          <Row
            title={title}
            description={description}
            error={requestError || update?.error || ""}
            stacked={update?.state === "downloading"}
          >
            {updateControls(update).map((control) =>
              control === "progress" ? (
                <ProgressBar key={control} value={update && update.bytes_total > 0 ? update.bytes_done / update.bytes_total : 0} className="w-full" />
              ) : control === "check" ? (
                <Button key={control} variant="outline" disabled={busy} onClick={() => void send("/api/updates/check")}>
                  Check for updates
                </Button>
              ) : (
                <Button key={control} onClick={() => void restartAndUpdate()}>Restart and update</Button>
              ),
            )}
          </Row>
          <Row
            title="Check for updates automatically"
            description="Checks hourly and downloads in the background. Nothing installs until you say so."
          >
            <Switch
              aria-label="Check for updates automatically"
              checked={draft.check_for_updates}
              onCheckedChange={(check_for_updates) => editSettings((current) => ({ ...current, check_for_updates }))}
            />
          </Row>
        </Rows>
      </Section>

      <Section title="Theme" note="Follows Windows' light or dark mode.">
        <ThemeCards
          theme={draft.theme}
          onPick={(theme) => editSettings((current) => ({ ...current, theme, colourways: everyTheme(current.colourways[current.theme]) }))}
        />
        <Rows className="mt-3">
          <Row title="Colour" description={COLOURWAYS.find(({ value }) => value === colourway)?.label}>
            <ColourSwatches theme={draft.theme} colourway={colourway} onPick={(picked) => editSettings((current) => ({ ...current, colourways: everyTheme(picked) }))} />
          </Row>
        </Rows>
      </Section>

      <UsageData />

      <Section title="Settings and logs">
        <Rows>
          <Row
            title="Settings file"
            description="Move your settings to another PC."
            error={fileNote.error ? fileNote.text : undefined}
          >
            {!fileNote.error && fileNote.text ? <span className="text-[12.5px] text-muted-foreground">{fileNote.text}</span> : null}
            <Button variant="outline" onClick={() => void exportSettings()}>
              <FileDown />
              Export
            </Button>
            <Button variant="outline" onClick={() => void importSettings()}>
              <FileUp />
              Import
            </Button>
          </Row>
          <Row title="Logs" description={<span className="selectable">{app.diagnostics_directory}</span>}>
            <Button variant="outline" onClick={() => void fileAction(async () => (await api.post("/api/logs/open"), ""))}>
              <FolderOpen />
              Open folder
            </Button>
          </Row>
        </Rows>
      </Section>
    </PageBody>
  );
}
