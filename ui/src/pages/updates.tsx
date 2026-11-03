import { useQueryClient } from "@tanstack/react-query";
import { FileDown, FileUp, FolderOpen } from "lucide-react";
import { useState } from "react";
import { installDescription, installTitle, useInstall } from "../components/install";
import { PageBody, PageTitle, ProgressBar, RadioDot, Row, Rows, Section } from "../components/parts";
import { useAppInfo } from "../components/shell";
import { Button } from "../components/ui/button";
import { Select } from "../components/ui/select";
import { Switch } from "../components/ui/switch";
import { api, errorText } from "../lib/api";
import { chooseOpenPath, chooseSavePath } from "../lib/platform";
import { useStore } from "../lib/store";
import type { Colourway, Theme, UpdateStatus } from "../lib/types";
import { cn } from "../lib/utils";

const THEMES: { value: Theme; title: string }[] = [
  { value: "default", title: "Default" },
  { value: "glass", title: "Glass" },
  { value: "material", title: "Material" },
  { value: "paper", title: "Paper" },
];

const COLOURWAYS: { value: Colourway; label: string }[] = [
  { value: "green", label: "Skryb green" },
  { value: "terracotta", label: "Terracotta" },
  { value: "ink", label: "Ink blue" },
  { value: "plum", label: "Plum" },
  { value: "graphite", label: "Graphite" },
];

/**
 * One card per theme with its own colour. The radio stretches over the card and the colour menu sits above
 * it, so neither takes the other's clicks or keys; opening a card's colour menu picks that card's theme.
 */
function ThemeCards({ theme, colourways, onPick }: { theme: Theme; colourways: Record<Theme, Colourway>; onPick: (theme: Theme, colourway?: Colourway) => void }) {
  return (
    <div role="radiogroup" aria-label="Theme" className="grid grid-cols-4 gap-2.5">
      {THEMES.map(({ value, title }) => {
        const selected = value === theme;
        return (
          <div
            key={value}
            className={cn(
              "relative surface rounded-xl border border-edge bg-card px-3.5 py-3 shadow-card transition-colors hover:bg-accent/50 has-[>button:focus-visible]:ring-2 has-[>button:focus-visible]:ring-ring/40",
              selected && "border-primary bg-selected hover:bg-selected",
            )}
          >
            <button
              role="radio"
              aria-checked={selected}
              onClick={() => onPick(value)}
              className="flex items-center gap-2 text-[13.5px] font-medium outline-none after:absolute after:inset-0 after:rounded-xl"
            >
              <RadioDot selected={selected} />
              {title}
            </button>
            <Select
              label={`${title} colour`}
              className="relative mt-2.5"
              value={colourways[value]}
              options={COLOURWAYS.map((colourway) => ({
                ...colourway,
                icon: <span data-theme={value} data-colourway={colourway.value} className="size-2.5 shrink-0 rounded-full bg-primary" />,
              }))}
              onOpenChange={(open) => open && onPick(value)}
              onChange={(colourway) => onPick(value, colourway as Colourway)}
            />
          </div>
        );
      })}
    </div>
  );
}

/** What the Updates section says about *update*; null means a git checkout, which updates through git. */
export function describeUpdate(update: UpdateStatus | null, running: string): { title: string; description: string } {
  if (update === null) {
    return { title: `Skryb ${running}`, description: "This copy runs from a source folder and updates through git, not from GitHub releases." };
  }
  const percent = update.bytes_total > 0 ? Math.floor((update.bytes_done * 100) / update.bytes_total) : 0;
  switch (update.state) {
    case "checking":
      return { title: `Skryb ${running}`, description: "Checking GitHub for a newer version..." };
    case "up_to_date":
      return { title: `Skryb ${running}`, description: "You have the latest version." };
    case "downloading":
      return { title: `Downloading Skryb ${update.version}`, description: `${percent}% downloaded. You choose when to install it.` };
    case "ready":
      return { title: `Skryb ${update.version} is ready`, description: "Skryb closes, installs the update and opens again. It waits for a dictation to finish first." };
    case "installing":
      return { title: `Installing Skryb ${update.version}`, description: "Skryb will close and open again in a minute." };
    case "updated":
      return { title: `Skryb ${running}`, description: "Updated just now. You have the latest version." };
    case "failed":
      return { title: `Skryb ${running}`, description: "The last check didn't work." };
    case "off":
      return { title: `Skryb ${running}`, description: "Automatic checks are off. Check by hand whenever you like." };
    default:
      return { title: `Skryb ${running}`, description: "Skryb checks for a newer version shortly after it starts." };
  }
}

/** Skryb's own settings, the same in both modes: updates, the look, the settings file and the logs. */
export function SettingsPage() {
  const app = useAppInfo();
  const install = useInstall();
  const { settings, editSettings, discardSettings, anyDirty, askUnsaved, saveAll } = useStore();
  const client = useQueryClient();
  const [requestError, setRequestError] = useState("");
  const [fileNote, setFileNote] = useState({ text: "", error: false });
  const draft = settings?.draft;
  // Skryb is closed while setup runs, so this shows without it.
  if (install) {
    return (
      <PageBody>
        <PageTitle title="Settings" />
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
  if (!app || !draft) return <PageBody><PageTitle title="Settings" /></PageBody>;

  const update = app.update;
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
      <PageTitle title="Settings" />
      <Section title="Updates">
        <Rows>
          <Row
            title={title}
            description={description}
            error={requestError || (update?.state === "failed" ? update.error : "")}
            stacked={update?.state === "downloading"}
          >
            {update === null ? null : update.state === "downloading" ? (
              <ProgressBar value={update.bytes_total > 0 ? update.bytes_done / update.bytes_total : 0} className="w-full" />
            ) : update.state === "ready" ? (
              <Button onClick={() => void restartAndUpdate()}>Restart and update</Button>
            ) : (
              <Button variant="outline" disabled={busy} onClick={() => void send("/api/updates/check")}>
                Check for updates
              </Button>
            )}
          </Row>
          <Row
            title="Check for updates automatically"
            description="Once an hour Skryb asks GitHub for a newer version and downloads it in the background. Nothing installs until you say so."
          >
            <Switch
              aria-label="Check for updates automatically"
              checked={draft.check_for_updates}
              onCheckedChange={(check_for_updates) => editSettings((current) => ({ ...current, check_for_updates }))}
            />
          </Row>
        </Rows>
      </Section>

      <Section title="Theme" note="Each theme follows Windows' light or dark setting. You see it as you pick; Save keeps it.">
        <ThemeCards
          theme={draft.theme}
          colourways={draft.colourways}
          onPick={(theme, colourway) =>
            editSettings((current) => ({ ...current, theme, colourways: colourway ? { ...current.colourways, [theme]: colourway } : current.colourways }))
          }
        />
      </Section>

      <Section title="Settings and logs">
        <Rows>
          <Row
            title="Settings file"
            description="Copy every setting to a file, or load one, to move Skryb to another PC."
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
