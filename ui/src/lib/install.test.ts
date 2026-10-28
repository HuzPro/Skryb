import { describe, expect, it } from "vitest";
import { estimateInstall, parseInstallProgress, timeLeft, type InstallProgress } from "./install";

const at = (step: InstallProgress["step"], percent = 0): InstallProgress => ({ step, percent, version: "0.4.0", log: "" });

describe("reading setup's progress file", () => {
  it("reads each step setup writes, with its version", () => {
    expect(parseInstallProgress('{"step":"copying","percent":45,"version":"0.4.0"}')).toEqual(at("copying", 45));
    expect(parseInstallProgress('{"step":"closing","percent":0,"version":"0.4.0"}')).toEqual(at("closing"));
  });

  it("keeps the log of a failed update, unicode and backslashes included", () => {
    const text = JSON.stringify({ step: "failed", percent: 0, version: "0.4.0", log: "C:\\Users\\Hüz Ö'Brien\\AppData\\Local\\Skryb\\updates\\install-0.4.0.log" });

    expect(parseInstallProgress(text)?.log).toBe("C:\\Users\\Hüz Ö'Brien\\AppData\\Local\\Skryb\\updates\\install-0.4.0.log");
  });

  it("ignores a missing file, one caught half written, and anything that isn't a step", () => {
    for (const text of [null, undefined, "", "{", '{"step":"copy', "null", "[]", "42", '{"percent":40}', '{"step":"deleting"}', '{"step":7}']) {
      expect(parseInstallProgress(text)).toBeNull();
    }
  });

  it("keeps a percent setup got wrong between 0 and 100", () => {
    expect(parseInstallProgress('{"step":"copying","percent":140}')?.percent).toBe(100);
    expect(parseInstallProgress('{"step":"copying","percent":-3}')?.percent).toBe(0);
    expect(parseInstallProgress('{"step":"copying","percent":"half"}')?.percent).toBe(0);
  });
});

describe("estimating how long the install takes", () => {
  it("starts near the beginning with about 45 seconds to go", () => {
    const estimate = estimateInstall(at("closing"), 0);

    expect(estimate.fraction).toBe(0);
    expect(estimate.secondsLeft).toBe(43);
    expect(estimate.label).toBe("Closing Skryb");
  });

  it("follows setup's own percent while it copies files", () => {
    const half = estimateInstall(at("copying", 50), 999);

    expect(half.secondsLeft).toBe(35);
    expect(estimateInstall(at("copying", 100), 0).fraction).toBeCloseTo(13 / 43);
  });

  it("only ever moves forward through the steps", () => {
    const fractions = [
      estimateInstall(at("closing"), 2.9),
      estimateInstall(at("copying", 0), 0),
      estimateInstall(at("copying", 60), 0),
      estimateInstall(at("runtime"), 0),
      estimateInstall(at("runtime"), 300),
      estimateInstall(at("app"), 0),
      estimateInstall(at("notices"), 1),
      estimateInstall(at("starting"), 0),
    ].map((estimate) => estimate.fraction);

    expect(fractions).toEqual([...fractions].sort((a, b) => a - b));
    expect(Math.max(...fractions)).toBeLessThan(1);
  });

  it("holds near the end of a step that runs long, like a fresh PyTorch download", () => {
    const tenMinutesIn = estimateInstall(at("runtime"), 600);

    expect(tenMinutesIn.fraction).toBeCloseTo(estimateInstall(at("runtime"), 13).fraction, 1);
    expect(tenMinutesIn.fraction).toBeLessThan(estimateInstall(at("app"), 0).fraction);
    expect(tenMinutesIn.secondsLeft).toBeGreaterThanOrEqual(1);
  });

  it("says plainly when the update failed", () => {
    expect(estimateInstall(at("failed"), 10)).toEqual({ fraction: 0, label: "The update didn't finish", secondsLeft: 0 });
  });
});

describe("saying how long is left", () => {
  it("rounds to five seconds under a minute and to minutes above it", () => {
    expect([1, 5, 6, 23, 25, 59, 60, 89, 90, 600].map(timeLeft)).toEqual([
      "A few seconds left",
      "A few seconds left",
      "About 10 seconds left",
      "About 25 seconds left",
      "About 25 seconds left",
      "About 60 seconds left",
      "About a minute left",
      "About a minute left",
      "About 2 minutes left",
      "About 10 minutes left",
    ]);
  });
});
