import { useState } from "react";
import { PageBody, PageTitle, Row, Rows, Section } from "../components/parts";
import { useAppInfo } from "../components/shell";
import { Button } from "../components/ui/button";
import { Switch } from "../components/ui/switch";
import { api, errorText } from "../lib/api";
import { useStore } from "../lib/store";
import type { UpdateStatus } from "../lib/types";

/** What the Updates page says about *update*; null means a git checkout, which updates through git. */
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
    case "failed":
      return { title: `Skryb ${running}`, description: "The last check didn't work." };
    case "off":
      return { title: `Skryb ${running}`, description: "Automatic checks are off. Check by hand whenever you like." };
    default:
      return { title: `Skryb ${running}`, description: "Skryb checks for a newer version shortly after it starts." };
  }
}

export function UpdatesPage() {
  const app = useAppInfo();
  const { settings, editSettings } = useStore();
  const [requestError, setRequestError] = useState("");
  const draft = settings?.draft;
  if (!app || !draft) return <PageBody><PageTitle title="Updates" /></PageBody>;

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

  return (
    <PageBody>
      <PageTitle title="Updates" />
      <Section title="Version">
        <Rows>
          <Row title={title} description={description} error={requestError || (update?.state === "failed" ? update.error : "")}>
            {update === null ? null : update.state === "ready" ? (
              <Button onClick={() => void send("/api/updates/install")}>Restart and update</Button>
            ) : (
              <Button variant="outline" disabled={busy} onClick={() => void send("/api/updates/check")}>
                Check for updates
              </Button>
            )}
          </Row>
        </Rows>
      </Section>

      <Section title="Automatic checks">
        <Rows>
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
    </PageBody>
  );
}
