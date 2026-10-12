// Real first-install check: no system Python/Node, no real agent config writes.
import assert from "node:assert/strict";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { installBundle } from "./installer.mjs";
import { runtimeConfig } from "./runtime.mjs";
import { runProcess } from "../local/process.mjs";

const home = resolve(process.env.HELIX_MCP_HOME);
const bundle = fileURLToPath(new URL("..", import.meta.url));
assert.equal(dirname(process.execPath), join(bundle, "runtime"), "Run with the bundled Node executable");
const originalPath = process.env.PATH;
const empty = join(home, "empty-path");
await mkdir(empty, { recursive: true });
// Retain Windows OS utilities (taskkill), excluding user Python/Node installations.
process.env.PATH = process.platform === "win32" ? join(process.env.SystemRoot, "System32") : empty;
process.env.HELIX_PYTHON = join(empty, "python-does-not-exist");
const calls = [];
const options = { allowLocalCode: true, skillRoot: join(home, "test-skills"), runClient: async (client, args) => { calls.push([client, args]); } };
await installBundle("codex", home, options);
const runtime = await runtimeConfig(home);
const base = (await runProcess(runtime.python, ["-I", "-c", "import sys; print(sys.base_prefix)"])).output.trim();
assert.ok(resolve(base).startsWith(join(home, "interpreters")), "Python must belong to Helix, not the system");
const args = JSON.parse(await readFile(join(home, "codex-registration.json"), "utf8"));
const [node, entry] = args.slice(args.indexOf("--") + 1);
assert.ok(node.startsWith(join(home, "runtimes")));
assert.ok(entry.startsWith(join(home, "app")));
assert.deepEqual(calls[0], ["codex", ["--version"]]);
const marker = join(home, "experiments", "keep-on-update.txt");
await mkdir(dirname(marker), { recursive: true });
await writeFile(marker, "preserve existing experiments");
await installBundle("codex", home, options);
assert.equal(await readFile(marker, "utf8"), "preserve existing experiments");

// Windows locks this test's running Node binary; hide the downloaded source instead.
for (const dir of ["mcp", "local", "examples"]) await rename(join(bundle, dir), join(bundle, dir + "-unavailable"));
try {
  const doctor = await runProcess(node, [entry, "doctor", "--home", home], { timeout: 60_000 });
  assert.match(doctor.output, /Ready for local CPU training/);
  process.env.PATH = originalPath;
  await runProcess(node, [join(dirname(entry), "runtime_check.mjs")], { timeout: 600_000, onLine: console.log });
} finally {
  for (const dir of ["mcp", "local", "examples"]) await rename(join(bundle, dir + "-unavailable"), join(bundle, dir));
}
console.log("Bundle installation, private Python, update preservation, download removal, and real training passed.");
