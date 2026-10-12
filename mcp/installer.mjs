import { chmod, copyFile, mkdir, readFile, rename, realpath } from "node:fs/promises";
import { createHash } from "node:crypto";
import { homedir } from "node:os";
import { delimiter, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createInterface } from "node:readline/promises";
import { parseArgs } from "node:util";
import { install, invokeClient } from "./cli.mjs";
import { setupRuntime } from "./runtime.mjs";
import { lockRunner } from "../local/lock.mjs";

const bundledRuntime = fileURLToPath(new URL("../runtime/", import.meta.url));
const digest = bytes => createHash("sha256").update(bytes).digest("hex");

export async function prepareBundle(home, source = bundledRuntime) {
  const manifest = JSON.parse(await readFile(join(source, "manifest.json"), "utf8"));
  if (manifest.platform !== process.platform || manifest.arch !== process.arch)
    throw new Error(`This download is for ${manifest.platform}/${manifest.arch}; use the ${process.platform}/${process.arch} download.`);
  const nodeName = process.platform === "win32" ? "node.exe" : "node";
  const uvName = process.platform === "win32" ? "uv.exe" : "uv";
  const files = [nodeName, uvName, "NODE-LICENSE.txt", "UV-LICENSE-MIT.txt", "UV-LICENSE-APACHE.txt"];
  const directory = join(home, "runtimes", digest(JSON.stringify(manifest)).slice(0, 16));
  await mkdir(directory, { recursive: true, mode: 0o700 });
  for (const name of files) {
    const expected = manifest.files[name];
    if (!/^[a-f0-9]{64}$/.test(expected) || digest(await readFile(join(source, name))) !== expected)
      throw new Error(`The downloaded ${name} failed its integrity check. Download Helix again.`);
    const target = join(directory, name);
    if (await readFile(target).then(bytes => digest(bytes) === expected).catch(error => { if (error.code === "ENOENT") return false; throw error; })) continue;
    await copyFile(join(source, name), target + ".new");
    await chmod(target + ".new", name === nodeName || name === uvName ? 0o755 : 0o644);
    await rename(target + ".new", target);
  }
  return { node: join(directory, nodeName), uv: join(directory, uvName) };
}

export async function installBundle(client, home, { allowLocalCode, runClient = invokeClient, skillRoot, source = bundledRuntime } = {}) {
  if (!["codex", "claude"].includes(client)) throw new Error("Choose Codex or Claude Code.");
  if (!allowLocalCode) throw new Error("Installation needs --allow-local-code: generated Python runs with your user permissions, including file and network access.");
  const release = await lockRunner(join(home, "installer"));
  const originalPath = process.env.PATH;
  try {
    // Finder and Windows Explorer may not inherit the agent's usual shell PATH.
    const runtime = await prepareBundle(home, source);
    process.env.PATH = [dirname(runtime.node), join(homedir(), ".local", "bin"), ...(process.platform === "win32" ? [join(process.env.APPDATA || join(homedir(), "AppData", "Roaming"), "npm")] : ["/opt/homebrew/bin", "/usr/local/bin"]), process.env.PATH || ""].join(delimiter);
    try { await runClient(client, ["--version"]); }
    catch (error) { throw new Error(`${client === "codex" ? "Codex CLI" : "Claude Code"} couldn't start. Install and sign into it first, then open this installer again. ${client === "codex" ? "https://developers.openai.com/codex/cli" : "https://code.claude.com/docs/en/setup"}\n${error.message}`); }
    await setupRuntime(home, { allowLocalCode, uv: runtime.uv });
    await install(client, home, { node: runtime.node, runClient, skillRoot });
  } finally {
    if (originalPath === undefined) delete process.env.PATH;
    else process.env.PATH = originalPath;
    await release();
  }
}

async function main() {
  const { values, positionals } = parseArgs({ options: { home: { type: "string" }, "allow-local-code": { type: "boolean" } }, allowPositionals: true });
  let [client] = positionals;
  let consent = values["allow-local-code"];
  console.error("\nHelix ML — local setup\n");
  if (!client || !consent) {
    if (!process.stdin.isTTY) throw new Error("Open Install Helix, or specify codex|claude --allow-local-code.");
    const input = createInterface({ input: process.stdin, output: process.stderr });
    try {
      if (!client) {
        const answer = (await input.question("Connect Helix to:\n  1. Codex\n  2. Claude Code\nChoose 1 or 2: ")).trim();
        client = { "1": "codex", "2": "claude" }[answer];
        if (!client) throw new Error("Choose 1 or 2 next time you open the installer.");
      }
      console.error("\nHelix will download Python and ML packages into its own folder.\nExperiments run on your computer with your file and network permissions.");
      consent = /^(y|yes)$/i.test((await input.question("Install and allow local experiments? [y/N] ")).trim());
    } finally { input.close(); }
  }
  if (!consent) { console.error("Installation cancelled."); return; }
  await installBundle(client, resolve(values.home || process.env.HELIX_MCP_HOME || join(homedir(), ".helix-ml", "mcp")), { allowLocalCode: consent });
  console.error("\nYou can delete the downloaded installer folder now.");
}

if (process.argv[1] && await realpath(process.argv[1]).catch(() => null) === fileURLToPath(import.meta.url))
  main().catch(error => { console.error("\nHelix setup:", error.message); process.exitCode = 1; });
