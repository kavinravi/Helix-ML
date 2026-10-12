import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, readFile, writeFile, access, symlink, mkdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { nativeArguments, nativeEnvironment, nativeLibraryPaths } from "../local/native.mjs";
import { tools, validateArguments } from "./schema.mjs";
import { setupRuntime } from "./runtime.mjs";
import { registrationArgs, install } from "./cli.mjs";
import { lockRunner } from "../local/lock.mjs";
import { runProcess } from "../local/process.mjs";
import { fileURLToPath } from "node:url";
import { prepareBundle, installBundle } from "./installer.mjs";
import { torchPlan, selectDevice, trialDevice } from "./compute.mjs";
import { torchPackageOptions, frameworkPackageOptions } from "../local/tools.mjs";
import { rlTask, validateRlBudget } from "./rl.mjs";
import { dependencyInstallCommand } from "../local/export.mjs";

test("export install commands use the actual candidate Torch variant and RAPIDS index", () => {
  assert.match(dependencyInstallCommand("torch==2.14.1+cpu\n"), /whl\/cpu$/);
  assert.match(dependencyInstallCommand("torch==2.14.1+cu130\ncuml-cu13==26.10.0\n"), /whl\/cu130.*pypi.nvidia.com$/);
  assert.equal(dependencyInstallCommand("torch==2.14.1\nnumpy==2.5.4\n"),"pip install -r requirements.lock");
});

test("native libraries are found in candidate and base wheel locations without unrelated recursion", async () => {
  const temp = await mkdtemp(join(tmpdir(),"helix libraries "));
  try {
    for (const name of ["candidate/libcuml/lib64","candidate/nvidia/cu13/lib","candidate/numpy.libs","base/nvidia/cublas/lib","candidate/unrelated/deep/lib"])
      await mkdir(join(temp,name),{recursive:true});
    const paths = await nativeLibraryPaths([join(temp,"candidate"),join(temp,"base"),join(temp,"missing")]);
    assert.equal(paths.length,4);
    assert.ok(paths.includes(join(temp,"candidate/libcuml/lib64")));
    assert.ok(paths.indexOf(join(temp,"candidate/nvidia/cu13/lib")) < paths.indexOf(join(temp,"base/nvidia/cublas/lib")));
  } finally { await rm(temp,{recursive:true,force:true}); }
});

test("GPU framework packages reject unsupported platforms and devices before download", () => {
  const runtime = {kind:"native",device:"cuda",compute:{cuda:{runtime:"13.0"}}};
  assert.deepEqual(frameworkPackageOptions("cuml-cu13",runtime,"linux"), ["--extra-index-url","https://pypi.nvidia.com"]);
  for (const platform of ["win32","darwin"]) assert.throws(() => frameworkPackageOptions("cuml-cu13",runtime,platform),/Linux or WSL2/);
  assert.throws(() => frameworkPackageOptions("cuml-cu12",{...runtime,device:"cpu"},"linux"),/NVIDIA CUDA/);
  assert.throws(() => frameworkPackageOptions("cuml-cu13",{...runtime,compute:{cuda:{runtime:"12.6"}}},"linux"),/CUDA 12/);
  assert.throws(() => frameworkPackageOptions("cuml-cu12",{...runtime,compute:{cuda:{runtime:"12.6",capability:"6.1"}}},"linux"),/capability 7.0/);
  assert.throws(() => frameworkPackageOptions("tensorflow-cpu",runtime,"linux"),/and-cuda/);
  assert.throws(() => frameworkPackageOptions("tensorflow[and-cuda]",runtime,"win32"),/Linux/);
  assert.throws(() => frameworkPackageOptions("tensorflow",{...runtime,device:"mps"},"darwin"),/MPS/);
  assert.deepEqual(frameworkPackageOptions("tensorflow[and-cuda]",runtime,"linux"),[]);
  assert.deepEqual(frameworkPackageOptions("tensorflow",{...runtime,device:"cpu"},"win32"),[]);
});

test("RL accepts only standard environments and bounded, typed parameters", () => {
  const create = tools.find(t => t.name === "helix_create_rl_experiment").inputSchema;
  const trial = tools.find(t => t.name === "helix_run_rl_trial").inputSchema;
  const good = {envId:"CartPole-v1",objective:"maximize episode return"};
  validateArguments(create,good);
  assert.equal(rlTask(good).metric,"episode_return");
  assert.equal(rlTask(good).evaluationEpisodes,10);
  assert.throws(() => rlTask({...good,envId:"module:untrusted"}),/standard/);
  assert.throws(() => rlTask({...good,objective:" "}),/objective/);
  assert.throws(() => validateRlBudget({minutes:null,trials:null}),/at least one/);
  assert.throws(() => validateRlBudget({minutes:-1,trials:3}));
  for (const bad of [{...good,seed:-1},{...good,seed:2**32},{...good,evaluationEpisodes:1001}]) assert.throws(() => validateArguments(create,bad));
  const candidate = {runId:"id",algorithm:"PPO",totalTimesteps:1000,hyperparameters:{n_steps:64,net_arch:[32,32]}};
  validateArguments(trial,candidate);
  for (const bad of [{...candidate,totalTimesteps:0},{...candidate,hyperparameters:{gamma:1.1}},{...candidate,hyperparameters:{eval_env:"secret"}},{...candidate,algorithm:"invented"}]) assert.throws(() => validateArguments(trial,bad));
});

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

test("GPU setup selects compatible wheels and never silently downgrades an explicit device", () => {
  const nvidia = (driver, capability) => ({ platform: "linux", gpus: [{ driver, capability }] });
  assert.match(torchPlan(nvidia("591.86", 12)).index, /cu130$/);
  assert.match(torchPlan(nvidia("580.65.06", 12)).index, /cu130$/);
  assert.ok(torchPlan(nvidia("580.65.05", 12)).reason);
  assert.match(torchPlan({...nvidia("580.88", 12), platform: "win32"}).index, /cu130$/);
  assert.ok(torchPlan({...nvidia("580.87", 12), platform: "win32"}).reason);
  assert.match(torchPlan(nvidia("591.86", 6.1)).index, /cu126$/);
  assert.match(torchPlan(nvidia("560.35", 8.6)).index, /cu126$/);
  assert.ok(torchPlan(nvidia("550.12", 8.6)).reason);
  assert.ok(torchPlan(nvidia("560.35", 12)).reason);
  assert.equal(torchPlan({platform: "darwin", architecture: "arm64", macos: "15.0", gpus: []}).device, "mps");
  assert.ok(torchPlan({platform: "darwin", architecture: "arm64", macos: "13.0", gpus: []}).reason);
  assert.equal(torchPlan({platform: "darwin", architecture: "x86_64", macos: "15.0", gpus: []}).device, "cpu");
  const compute = { available: ["cpu", "cuda"] };
  assert.equal(selectDevice(compute), "cuda");
  assert.equal(selectDevice({available: ["cpu", "mps"]}), "mps");
  assert.throws(() => selectDevice(compute, "mps"), /unavailable/);
  assert.throws(() => trialDevice(compute, {device: "cuda", requestedDevice: "cuda"}, "cpu"), /requires CUDA/);
  assert.equal(trialDevice(compute, {device: "cuda", requestedDevice: "auto"}, "cpu"), "cpu");
  assert.equal(trialDevice(compute, {}, undefined), "cpu", "old CPU experiments retain their device");
  assert.deepEqual(torchPackageOptions("torch"), ["--index-url", "https://download.pytorch.org/whl/cpu"]);
  assert.deepEqual(torchPackageOptions("torch", {kind:"native",compute:{torchIndex:"https://download.pytorch.org/whl/cu130"}}), process.platform === "darwin" ? [] : ["--index-url", "https://download.pytorch.org/whl/cu130"]);
});

test("native GPU selection reaches child processes without enabling MPS fallback", () => {
  const runtime = {python: "python", cpus: 2, memoryMiB: 1024, device: "cuda", compute:{torchVersion:"2.14.1+cu130"}};
  const env = nativeEnvironment(runtime, tmpdir(), []);
  assert.equal(env.HELIX_DEVICE, "cuda");
  assert.equal(env.CUDA_VISIBLE_DEVICES, process.env.CUDA_VISIBLE_DEVICES ?? "0");
  assert.equal(env.HELIX_TORCH_VERSION, "2.14.1+cu130");
  const cpu = nativeEnvironment({...runtime, device: "cpu"}, tmpdir(), []);
  assert.equal(cpu.CUDA_VISIBLE_DEVICES, "-1");
  const mps = nativeEnvironment({...runtime, device: "mps"}, tmpdir(), []);
  assert.equal(mps.HELIX_DEVICE, "mps");
  assert.equal(mps.PYTORCH_ENABLE_MPS_FALLBACK, "0");
});

test("MCP validates types, required values and unknown nested options", () => {
  const schema = tools.find(t => t.name === "helix_create_experiment").inputSchema;
  const good = { dataset: "data.csv", objective: "classify", metric: "f1", minutes: null, trials: 5, policy: { tuning: false } };
  validateArguments(schema, good);
  for (const bad of [{...good, trials: 1.5}, {...good, policy: { magic: true }}, {...good, minutes: 0}, {...good, target: 1}, {...good, metric: "invented"}, {}]) assert.throws(() => validateArguments(schema, bad));
});

test("setup requires informed local execution opt-in before creating state", async () => {
  await assert.rejects(setupRuntime(join(tmpdir(), "helix-no-consent")), /--allow-local-code/);
  await assert.rejects(installBundle("codex", join(tmpdir(), "helix-no-consent")), /--allow-local-code/);
});

test("bundled runtimes survive download removal, repair damage, and reject wrong or corrupted downloads", async () => {
  const root = await mkdtemp(join(tmpdir(), "helix bundle test "));
  const source = join(root, "download"), home = join(root, "installed");
  const nodeName = process.platform === "win32" ? "node.exe" : "node";
  try {
    await mkdir(source);
    const files = {};
    for (const name of [nodeName, process.platform === "win32" ? "uv.exe" : "uv", "NODE-LICENSE.txt", "UV-LICENSE-MIT.txt", "UV-LICENSE-APACHE.txt"]) {
      await writeFile(join(source, name), name);
      files[name] = createHash("sha256").update(name).digest("hex");
    }
    const manifest = { platform: process.platform, arch: process.arch, files };
    const save = data => writeFile(join(source, "manifest.json"), JSON.stringify(data));
    await save(manifest);
    const runtime = await prepareBundle(home, source);
    assert.ok(runtime.node.startsWith(join(home, "runtimes")));
    await writeFile(runtime.node, "damaged installed binary");
    assert.deepEqual(await prepareBundle(home, source), runtime);
    assert.equal(await readFile(runtime.node, "utf8"), nodeName);
    await save({ ...manifest, arch: "wrong" });
    await assert.rejects(prepareBundle(home, source), /This download is for/);
    await save(manifest);
    await writeFile(join(source, nodeName), "corrupt download");
    await assert.rejects(prepareBundle(home, source), /integrity check/);
    await rm(source, { recursive: true });
    assert.equal(await readFile(runtime.node, "utf8"), nodeName);
  } finally { await rm(root, { recursive: true, force: true }); }
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
