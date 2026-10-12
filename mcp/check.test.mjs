import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, readFile, writeFile, access, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { nativeArguments, nativeEnvironment } from "../local/native.mjs";
import { tools, validateArguments } from "./schema.mjs";
import { setupRuntime } from "./runtime.mjs";
import { registrationArgs, install } from "./cli.mjs";
import { lockRunner } from "../local/lock.mjs";
import { runProcess } from "../local/process.mjs";
import { fileURLToPath } from "node:url";

test("native paths are mapped only as path arguments, preserving spaces and inline code", () => {
  const mounts = [[join(tmpdir(), "space, dir"), "/data"]];
  assert.deepEqual(nativeArguments(["python", "/data/train.py", "--input", "/data/x.csv"], mounts), ["-B", "-s", "-P", join(mounts[0][0], "train.py"), "--input", join(mounts[0][0], "x.csv")]);
  assert.equal(nativeArguments(["python", "-c", 'print("/data/x")'], mounts).at(-1), 'print("/data/x")');
  assert.throws(() => nativeArguments(["sh", "-c", "anything"], mounts), /only executes Python/);
});

test("training gets declared imports and no inherited credentials; scorer gets no candidate imports", () => {
  process.env.HELIX_TEST_SECRET = "must-not-pass";
  const runtime = { python: join(tmpdir(), "venv", "python"), cpus: 2, memoryMiB: 1024 };
  const env = nativeEnvironment(runtime, tmpdir(), [["source", "/code"], ["extras", "/packages"]]);
  assert.equal(env.HELIX_TEST_SECRET, undefined);
  assert.equal(env.HELIX_SOURCE, "source");
  assert.match(env.PYTHONPATH, /source/);
  const scorer = nativeEnvironment(runtime, tmpdir(), [["harness", "/harness"], ["results", "/work"]]);
  assert.equal(scorer.PYTHONPATH, "harness");
  assert.equal(scorer.HELIX_SOURCE, undefined);
  delete process.env.HELIX_TEST_SECRET;
});

test("MCP validates types, required values and unknown nested options", () => {
  const schema = tools.find(t => t.name === "helix_create_experiment").inputSchema;
  const good = { dataset: "data.csv", objective: "classify", metric: "f1", minutes: null, trials: 5, policy: { tuning: false } };
  validateArguments(schema, good);
  for (const bad of [{...good, trials: 1.5}, {...good, policy: { magic: true }}, {...good, minutes: 0}, {...good, target: 1}, {...good, metric: "invented"}, {}]) assert.throws(() => validateArguments(schema, bad));
});

test("setup requires informed local execution opt-in before creating state", async () => {
  await assert.rejects(setupRuntime(join(tmpdir(), "helix-no-consent")), /--allow-local-code/);
});

test("exclusive compute lock prevents two clients writing one experiment and releases", async () => {
  const root = await mkdtemp(join(tmpdir(), "helix-lock-test-"));
  try {
    const release = await lockRunner(root);
    await assert.rejects(lockRunner(root), /already owns/);
    await release();
    await (await lockRunner(root))();
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("agent registration uses argument arrays and a stable installation path", () => {
  const path = join(tmpdir(), "name with spaces", "mcp", "cli.mjs"), home = join(tmpdir(), "my data");
  assert.deepEqual(registrationArgs("codex", path, home), ["mcp", "add", "helix", "--", process.execPath, path, "serve", "--home", home]);
  assert.deepEqual(registrationArgs("claude", path, home).slice(0, 7), ["mcp", "add", "--scope", "user", "--transport", "stdio", "helix"]);
});

test("the CLI starts through npm's executable symlink", async () => {
  const root = await mkdtemp(join(tmpdir(), "helix-bin-test-"));
  try {
    let entry = fileURLToPath(new URL("cli.mjs", import.meta.url));
    if (process.platform !== "win32") { const link = join(root, "helix"); await symlink(entry, link); entry = link; }
    const result = await runProcess(process.execPath, [entry, "--help"]);
    assert.match(result.output, /helix install codex\|claude/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("installer copies a complete runtime, updates its own Claude registration, and preserves edited skills", async () => {
  const root = await mkdtemp(join(tmpdir(), "helix-install-test-")), home = join(root, "state"), skillRoot = join(root, "skills");
  const calls = [];
  const runClient = async (client, args) => { calls.push([client, args]); };
  try {
    await install("claude", home, { runClient, skillRoot });
    const args = calls[0][1], entry = args[args.indexOf("--") + 2];
    assert.ok(entry.startsWith(join(home, "app")));
    await access(entry);
    const app = join(entry, "..", "..");
    for (const file of ["local/metric_catalog.json", "local/native.mjs", "mcp/requirements.txt", "examples/measurements.csv", "LICENSE"]) await access(join(app, file));
    await writeFile(join(skillRoot, "helix-ml", "SKILL.md"), "My edited skill");
    await install("claude", home, { runClient, skillRoot });
    assert.deepEqual(calls[1][1], ["mcp", "remove", "--scope", "user", "helix"]);
    assert.equal(await readFile(join(skillRoot, "helix-ml", "SKILL.md.backup"), "utf8"), "My edited skill");
    const before = await readFile(join(home, "claude-registration.json"), "utf8");
    let failed = false;
    await assert.rejects(install("claude", home, { skillRoot, runClient: async (_, args) => {
      if (args[1] === "add" && !failed) { failed = true; throw new Error("Registration failed"); }
      calls.push(["claude", args]);
    } }), /Registration failed/);
    assert.deepEqual(calls.at(-1)[1], JSON.parse(before));
    assert.equal(await readFile(join(home, "claude-registration.json"), "utf8"), before);
  } finally { await rm(root, { recursive: true, force: true }); }
});
