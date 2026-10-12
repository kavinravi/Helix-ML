#!/usr/bin/env node
import { cp, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { resolve, join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { parseArgs } from "node:util";
import { setupRuntime, runtimeConfig } from "./runtime.mjs";
import { runProcess } from "../local/process.mjs";

const project = fileURLToPath(new URL("..", import.meta.url));
const usage = `Helix ML — local MCP for Codex and Claude Code

helix install codex|claude --allow-local-code  Set up Python, register MCP and install the companion skill
helix setup --allow-local-code                Set up the managed Python environment only
helix serve                                  Start the MCP stdio server (normally launched by your agent)
helix doctor                                 Check the local Python runtime

Requires Node.js 22.13+ and Python 3.11+. No Docker, Helix account or API key.
Local generated code runs with your user permissions, including file/network access.
--home <directory> overrides ~/.helix-ml/mcp (or HELIX_MCP_HOME).
`;

export function registrationArgs(client, entry, home) {
  return ["mcp", "add", ...(client === "claude" ? ["--scope", "user", "--transport", "stdio"] : []), "helix", "--", process.execPath, entry, "serve", "--home", home];
}

async function invokeClient(client, args) {
  if (process.platform === "win32") {
    // PowerShell invokes npm's .cmd shims; literal quoting preserves spaces and metacharacters.
    const literal = value => "'" + value.replaceAll("'", "''") + "'";
    const script = "$ErrorActionPreference='Stop'; & " + [client, ...args].map(literal).join(" ") + "; exit $LASTEXITCODE";
    return runProcess("powershell.exe", ["-NoProfile", "-NonInteractive", "-EncodedCommand", Buffer.from(script, "utf16le").toString("base64")], { timeout: 30_000, onLine: console.error });
  }
  return runProcess(client, args, { timeout: 30_000, onLine: console.error });
}

export async function install(client, home, { runClient = invokeClient, skillRoot = join(homedir(), client === "codex" ? ".agents" : ".claude", "skills") } = {}) {
  if (!["codex", "claude"].includes(client)) throw new Error("Choose helix install codex or helix install claude.");
  // Copy out of npm's temporary cache; a cache purge must not break agent registration.
  const files = [];
  for (const dir of ["local", "mcp", "examples"]) {
    for (const name of await readdir(join(project, dir), { recursive: true })) {
      if (/\.(mjs|py|json|md|txt|csv)$/.test(name) && !name.includes("__pycache__")) files.push(join(dir, name));
    }
  }
  const hash = createHash("sha256");
  for (const file of files.sort()) hash.update(file).update(await readFile(join(project, file)));
  const app = join(home, "app", hash.digest("hex").slice(0, 16));
  await mkdir(app, { recursive: true, mode: 0o700 });
  for (const file of [...files, "LICENSE"]) {
    await mkdir(dirname(join(app, file)), { recursive: true });
    await cp(join(project, file), join(app, file));
  }
  const args = registrationArgs(client, join(app, "mcp", "cli.mjs"), home);
  const receipt = join(home, `${client}-registration.json`);
  const previous = await readFile(receipt, "utf8").then(JSON.parse).catch(error => { if (error.code === "ENOENT") return null; throw error; });
  // Claude refuses replacement entries. Remove only a registration made by this installer.
  if (client === "claude" && previous) await runClient(client, ["mcp", "remove", "--scope", "user", "helix"]);
  try { await runClient(client, args); }
  catch (error) {
    if (client === "claude" && previous) await runClient(client, previous).catch(restore => console.error("Restore the previous MCP entry:", restore.message));
    throw error;
  }
  await writeFile(receipt, JSON.stringify(args), { mode: 0o600 });
  const skill = join(skillRoot, "helix-ml");
  await mkdir(skill, { recursive: true });
  const source = await readFile(join(project, "mcp", "skills", "helix-ml", "SKILL.md"));
  const existing = await readFile(join(skill, "SKILL.md")).catch(error => { if (error.code === "ENOENT") return null; throw error; });
  if (existing && !existing.equals(source)) await writeFile(join(skill, "SKILL.md.backup"), existing);
  await writeFile(join(skill, "SKILL.md"), source);
  console.error(`Helix is installed for ${client}. Restart your agent session, then ask: Use Helix to train a classifier on my CSV and compare F1 and ROC-AUC.\nData and results: ${join(home, "experiments")}`);
}

async function main() {
  const { values, positionals } = parseArgs({ options: { home: { type: "string" }, "allow-local-code": { type: "boolean" }, help: { type: "boolean", short: "h" } }, allowPositionals: true });
  const [command, client] = positionals;
  if (!command || values.help) { console.log(usage); return; }
  const home = resolve(values.home || process.env.HELIX_MCP_HOME || join(homedir(), ".helix-ml", "mcp"));
  if (command === "setup" || command === "install") {
    if (command === "install" && !["codex", "claude"].includes(client)) throw new Error("Specify codex or claude.");
    await setupRuntime(home, { allowLocalCode: values["allow-local-code"] });
    if (command === "install") await install(client, home);
  } else if (command === "doctor") {
    const runtime = await runtimeConfig(home);
    const result = await runProcess(runtime.python, ["-I", "-c", "import sys, numpy, pandas, sklearn; print('Python '+sys.version.split()[0]+'; scikit-learn '+sklearn.__version__)"], { timeout: 30_000 });
    console.log(result.output.trim() + `\nReady for local CPU training. Data: ${join(home, "experiments")}`);
  } else if (command === "serve") {
    const runtime = await runtimeConfig(home);
    const { serve } = await import("./server.mjs");
    await serve(home, runtime);
  } else throw new Error(usage);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  main().catch(error => { console.error("Helix:", error.message); process.exitCode = 1; });
