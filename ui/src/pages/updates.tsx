import { useQueryClient } from "@tanstack/react-query";
import { FileDown, FileUp, FolderOpen } from "lucide-react";
import { useState } from "react";
import { installDescription, installTitle, useInstall } from "../components/install";
import { ChoiceCards, PageBody, PageTitle, ProgressBar, Row, Rows, Section, type Choice } from "../components/parts";
import { useAppInfo } from "../components/shell";
import { Button } from "../components/ui/button";
import { Switch } from "../components/ui/switch";
import { api, errorText } from "../lib/api";
import { chooseOpenPath, chooseSavePath } from "../lib/platform";
import { useStore } from "../lib/store";
import type { PaperColourway, Theme, UpdateStatus } from "../lib/types";

const THEMES: Choice<Theme>[] = [
  { value: "default", title: "Default", description: "Clean and quiet, the way Skryb has always looked." },
  { value: "glass", title: "Glass", description: "Clear, neutral glass with light along its edges." },
  { value: "material", title: "Material", description: "Rounded, tonal and springy, grown from one indigo." },
  { value: "paper", title: "Paper", description: "Ink on warm paper, with serif headings." },
];

function Swatch({ colourway, name }: { colourway: PaperColourway; name: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span data-colourway={colourway} className="size-2.5 rounded-full bg-primary" />
      {name}
    </span>
  );
}

const COLOURWAYS: Choice<PaperColourway>[] = [
  { value: "green", title: <Swatch colourway="green" name="Skryb green" />, description: "The green of the Skryb icon." },
  { value: "terracotta", title: <Swatch colourway="terracotta" name="Terracotta" />, description: "Warm clay, Paper's first ink." },
  { value: "ink", title: <Swatch colourway="ink" name="Ink blue" />, description: "Fountain-pen blue." },
  { value: "plum", title: <Swatch colourway="plum" name="Plum" />, description: "A deep red-purple." },
  { value: "graphite", title: <Swatch colourway="graphite" name="Graphite" />, description: "Pencil grey, no colour at all." },
];

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
        <ChoiceCards<Theme> label="Theme" value={draft.theme} choices={THEMES} onChange={(theme) => editSettings((current) => ({ ...current, theme }))} />
        {draft.theme === "paper" ? (
          <div className="mt-2.5">
            <ChoiceCards<PaperColourway>
              label="Paper colourway"
              value={draft.paper_colourway}
              choices={COLOURWAYS}
              onChange={(paper_colourway) => editSettings((current) => ({ ...current, paper_colourway }))}
            />
          </div>
        ) : null}
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
