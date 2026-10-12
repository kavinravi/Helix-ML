// Exercises device propagation, real model training and portable export through MCP operations.
import assert from "node:assert/strict";
import { mkdtemp, mkdir, cp, writeFile, readFile, rm } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { runtimeConfig } from "./runtime.mjs";
import { probeCompute, selectDevice, TORCH_VERSION } from "./compute.mjs";
import { createMcpService } from "./service.mjs";
import { runProcess } from "../local/process.mjs";
import { container } from "../local/runtime.mjs";
import { matchingPredictions } from "../local/export.mjs";

const home = resolve(process.env.HELIX_MCP_HOME || join(homedir(), ".helix-ml", "mcp"));
const runtime = await runtimeConfig(home);
let compute = await probeCompute(runtime.python);
if (!compute.torchVersion) {
  if (process.platform === "darwin" && process.arch === "x64") {
    console.log("Intel Mac: CPU harness tested separately; current PyTorch has no Intel Mac wheel. GPU test skipped.");
    process.exit(0);
  }
  // CPU CI workers still exercise the complete neural model/export path.
  await runProcess(runtime.python, ["-m", "pip", "install", `torch==${TORCH_VERSION}`, ...(process.platform === "darwin" ? [] : ["--index-url", "https://download.pytorch.org/whl/cpu"])], { timeout: 900_000 });
  compute = await probeCompute(runtime.python);
}
const device = selectDevice(compute, process.env.HELIX_TEST_DEVICE || "auto");
runtime.compute = compute;
const service = await createMcpService(home, runtime);
const temp = await mkdtemp(join(tmpdir(), "helix gpu check "));
const call = (name, args = {}) => service.call("helix_" + name, args);
async function settled(runId) {
  for (let i = 0; i < 1200; i++) {
    const state = await call("status", { runId, details: true });
    if (state.status !== "running") { assert.ok(["ready", "completed"].includes(state.status), state.error); return state; }
    await delay(250);
  }
  throw new Error("GPU check timed out.");
}
const ids = [];
try {
  const source = join(temp, "source");
  await cp(fileURLToPath(new URL("fixtures/torch", import.meta.url)), source, { recursive: true });
  const policy = Object.fromEntries(["augmentation", "regularization", "features", "tuning", "pretrained", "ensemble"].map(k => [k, false]));
  await writeFile(join(source, "plan.json"), JSON.stringify({ component: "linear classifier", change: "device-aware baseline", rationale: "Verify actual tensor device and portable export", modelFamily: "torch.nn.Linear", strategies: policy }));
  const dataset = join(temp, "data.csv");
  await writeFile(dataset, "x,z,label\n" + Array.from({ length: 160 }, (_, i) => `${(i % 20) / 20},${Math.floor(i / 20) / 8},${i % 20 < 10 ? "short" : "long"}`).join("\n"));
  const info = await call("info");
  assert.ok(info.runtime.compute.available.includes(device));
  const unavailable = ["cuda", "mps"].find(d => !compute.available.includes(d));
  if (unavailable) await assert.rejects(call("create_experiment", { dataset, objective: "reject unavailable GPU", target: "label", metric: "auroc", device: unavailable }), /unavailable/);
  const { runId } = await call("create_experiment", { dataset, objective: "Train a linear classifier on the selected device", target: "label", metric: "auroc", metrics: ["auroc", "f1"], device, validation: "cv", folds: 3, minutes: 15, trials: 2, policy, exportFormat: "pytorch" });
  ids.push(runId);
  await settled(runId);
  if (device !== "cpu") await assert.rejects(call("run_trial", {runId, sourceDir: source, device: "cpu"}), /requires/);
  await call("run_trial", { runId, sourceDir: source, device, packages: [{ name: "torch", version: TORCH_VERSION }] });
  let state = await settled(runId);
  assert.ok(Number.isFinite(state.validationScore));
  assert.equal(state.trials[0].device, device);
  for (const fit of state.trials[0].evaluations) {
    const proof = JSON.parse(await readFile(join(home, "experiments", runId, "fits", "trial-1", fit.id, "output", "device_check.json")));
    assert.ok(proof.device.startsWith(device) && proof.tensorDevice.startsWith(device));
    if (device !== "cpu") assert.ok(proof.gpuBytes > 0, "Training must allocate GPU tensors");
  }
  await call("finalize", {runId}); state = await settled(runId);
  assert.equal(state.status, "completed");
  assert.equal(state.trainingHistory.length, 30);
  const exported = join(temp, "exported");
  await runProcess(runtime.python, ["-I", "-c", "import json,sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); assert z.testzip() is None; e=json.loads(z.read('experiment.json')); assert e['environment']['selectedDevice']==sys.argv[3]; assert json.loads(z.read('config.json'))['device']==sys.argv[3]; z.extractall(sys.argv[2])", state.solutionZip, exported, device]);
  const output = join(temp, "cpu-reload"); await mkdir(output);
  const context = { runtime: {...runtime, device: "cpu"}, deadline: Date.now() + 120_000, runId };
  await container(context, ["python", "-c", "import torch; assert not torch.cuda.is_available()"]);
  await container(context, ["python", "/code/predict.py", "--input", "/data/data.csv", "--metadata", "/export/manifest.json", "--config", "/export/config.json", "--model-dir", "/export/model", "--models", "/export/pretrained", "--output", "/work/predictions.json"], { writable: output, mounts: [[join(exported, "source"), "/code"], [temp, "/data"], [exported, "/export"], [output, "/work"]] });
  const cpuPredictions = JSON.parse(await readFile(join(output, "predictions.json")));
  assert.equal(cpuPredictions.length, 160);
  if (device !== "cpu") {
    await container({...context, runtime: {...runtime, device}}, ["python", "/code/predict.py", "--input", "/data/data.csv", "--metadata", "/export/manifest.json", "--config", "/export/config.json", "--model-dir", "/export/model", "--models", "/export/pretrained", "--output", "/work/gpu.json"], { writable: output, mounts: [[join(exported, "source"), "/code"], [temp, "/data"], [exported, "/export"], [output, "/work"]] });
    assert.ok(matchingPredictions(cpuPredictions, JSON.parse(await readFile(join(output, "gpu.json")))));
  }
  console.log(`${device.toUpperCase()} passed: PyTorch ${compute.torchVersion}, 3-fold training, real tensor allocation, native package reuse, model reload, final test and portable CPU inference. Score ${state.validationScore}.`);
} finally {
  await service.close();
  for (const id of ids) await rm(join(home, "experiments", id), {recursive: true, force: true});
  await rm(temp, {recursive: true, force: true});
}
