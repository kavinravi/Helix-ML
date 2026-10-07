// The Mac launcher runs this service directly. No npm installation or UI build.
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { mkdir, appendFile } from "node:fs/promises";
import { createService } from "./server.mjs";

const exec = promisify(execFile);
const root = resolve(process.env.HELIX_DATA_DIR || join(homedir(), "Library", "Application Support", "Helix ML"));
const port = Number(process.env.HELIX_PORT || 4319);
const url = `http://127.0.0.1:${port}`;
const open = async () => {
  if (process.env.HELIX_OPEN_BROWSER === "0") return;
  await exec(process.platform === "darwin" ? "/usr/bin/open" : "xdg-open", [url]);
};
let service;
try {
  const [major, minor] = process.versions.node.split(".").map(Number);
  if (major < 22 || (major === 22 && minor < 13)) throw new Error("Helix needs Node.js 22.13 or newer. Install it, then reopen the app.");
  await mkdir(root, { recursive: true, mode: 0o700 });
  const existing = await fetch(`${url}/health`, { signal: AbortSignal.timeout(1500) })
    .then(response => response.json()).catch(() => null);
  if (existing?.name === "helix-ml") {
    await open();
  } else {
    service = await createService({ root, port, desktop: true });
    await new Promise((ready, reject) => {
      service.server.once("error", reject);
      service.server.listen(port, "127.0.0.1", ready);
    });
    const quit = async () => { await service.close(); process.exit(0); };
    service.server.once("quit", quit);
    for (const signal of ["SIGINT", "SIGTERM"]) process.once(signal, quit);
    console.log(`Helix is ready at ${url}`);
    await open();
  }
} catch (error) {
  if (service) await service.close();
  const message = `Helix could not open. ${error.message}`;
  await appendFile(join(root, "launcher.log"), `${new Date().toISOString()} ${message}\n`, { mode: 0o600 }).catch(() => {});
  if (process.platform === "darwin") await exec("/usr/bin/osascript", ["-e", 'on run argv\ndisplay dialog (item 1 of argv) with title "Helix ML" buttons {"OK"} default button "OK"\nend run', message]).catch(() => {});
  console.error(message);
  process.exitCode = 1;
}
