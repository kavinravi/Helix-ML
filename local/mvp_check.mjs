// Opt-in acceptance check: consumes the selected native CLI subscription.
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout } from "node:timers/promises";
import { createService } from "./server.mjs";
import { cleanupContainers } from "./runtime.mjs";

const root = await mkdtemp(join(tmpdir(), "helix-mvp-"));
let service;
try {
  const dataset = join(root, "dataset");
  await mkdir(dataset);
  await writeFile(join(dataset, "train.csv"), "length,width,label\n" + Array.from({ length: 80 }, (_, i) => `${i / 10},${(i * 17 % 31) / 10},${i < 40 ? "short" : "long"}`).join("\n"));
  service = await createService({ root, port: 0 });
  await new Promise((resolve) => service.server.listen(0, "127.0.0.1", resolve));
  const url = `http://127.0.0.1:${service.server.address().port}`;
  const request = async (path, body) => {
    const response = await fetch(url + "/api" + path, { method: body ? "POST" : "GET", headers: { Authorization: `Bearer ${service.token}`, "Content-Type": "application/json" }, ...(body ? { body: JSON.stringify(body) } : {}) });
    return { status: response.status, body: await response.json() };
  };
  const task = { agent: process.env.HELIX_TEST_AGENT || "claude", dataset, target: "label", objective: "Predict short or long from the numeric measurements. Use a plain unpruned sklearn.tree.DecisionTreeClassifier with all default parameters except random_state; no package installs are needed.", metric: "accuracy", minutes: 15, trials: 3, searchModels: false, model: "sklearn.tree.DecisionTreeClassifier", output: "ipynb", exportModel: true, exportFormat: "joblib", validation: "holdout", folds: 3, seeds: [42], splitStrategy: "independent", groupColumn: "", timeColumn: "", assetColumns: [], policy: { augmentation: false, regularization: false, features: false, tuning: false, pretrained: false, ensemble: false } };
  task.output = process.env.HELIX_TEST_OUTPUT || task.output;
  task.exportFormat = process.env.HELIX_TEST_FORMAT || task.exportFormat;
  if (process.env.HELIX_TEST_SEARCH === "1") {
    task.searchModels = true; task.model = "";
    task.objective = "Predict short or long from the numeric measurements. Start with an economical model using preinstalled packages, then compare distinct approaches.";
  }
  if (process.env.HELIX_TEST_FOLLOWUP === "1") {
    task.trials = 1; task.policy.tuning = true; task.minutes = 5;
    task.objective = "Predict short or long from the numeric measurements using a DecisionTreeClassifier. Start with the defaults; tuning is allowed in later trials.";
  }
  const requestedAt = Date.now();
  const started = await request("/runs", task);
  const readinessSeconds = (Date.now() - requestedAt) / 1000;
  console.log(`Readiness: ${readinessSeconds.toFixed(1)} seconds.`);
  assert.equal(started.status, 201, JSON.stringify(started.body));
  const id = started.body.id;
  assert.equal((await request("/runs", task)).status, 409);
  assert.equal((await request(`/runs/${id}/action`, { action: "pause" })).body.status, "paused");
  // Reload persisted state before resuming; browser/runner restarts retain the task.
  await service.close();
  service = await createService({ root, port: 0, token: service.token });
  const oldPort = Number(new URL(url).port);
  await new Promise((resolve) => service.server.listen(oldPort, "127.0.0.1", resolve));
  assert.equal((await request(`/runs/${id}`)).body.status, "paused");
  const resumedAt = Date.now();
  assert.equal((await request(`/runs/${id}/action`, { action: "resume" })).status, 200);
  const cachedReadinessSeconds = (Date.now() - resumedAt) / 1000;
  console.log(`Readiness after restart: ${cachedReadinessSeconds.toFixed(1)} seconds.`);
  let run, previous = "";
  let firstTrialSeconds;
  const deadline = Date.now() + 16 * 60_000;
  while (Date.now() < deadline) {
    run = (await request(`/runs/${id}`)).body;
    if (firstTrialSeconds === undefined && run.trials.some(trial => trial.score !== null)) {
      firstTrialSeconds = (Date.now() - resumedAt) / 1000;
      console.log(`First measured trial after resume: ${firstTrialSeconds.toFixed(1)} seconds.`);
    }
    const state = `${run.status}: ${run.phase}; ${run.trials.length} trials; ${run.logs.at(-1)?.message || ""}`;
    if (state !== previous) { console.log(state.slice(0, 800)); previous = state; }
    if (!["running", "queued"].includes(run.status)) break;
    await setTimeout(2000);
  }
  assert.equal(run.status, "completed", run.error || "Run did not finish");
  assert.ok(Number.isFinite(run.score) && Number.isFinite(run.testScore));
  assert.equal(run.trials.length, task.searchModels ? 3 : 1);
  assert.ok(run.trials.every(trial => Number.isFinite(trial.score)), "Each scheduled candidate must produce a measured score");
  await writeFile(join(root, "timing.json"), JSON.stringify({ agent: task.agent, searchModels: task.searchModels, readinessSeconds, cachedReadinessSeconds, firstTrialSeconds, totalSeconds: (Date.now() - requestedAt) / 1000 }, null, 2));
  const artifacts = (await request(`/runs/${id}/artifacts`)).body;
  for (const name of ["final/helix-solution.zip", `final/source/${task.output === "ipynb" ? "solution.ipynb" : "train.py"}`, "final/model/model_manifest.json", "final/checksums.json", "final/requirements.lock"]) assert.ok(artifacts.some((file) => file.path === name), `Missing ${name}`);
  const manifest = JSON.parse(await readFile(join(root, "runs", id, "final", "experiment.json"), "utf8"));
  assert.equal(manifest.exportVerified, true);
  const download = await fetch(url + `/api/runs/${id}/file?path=final/helix-solution.zip`, { headers: { Authorization: `Bearer ${service.token}` } });
  assert.equal(download.status, 200);
  assert.ok((await download.arrayBuffer()).byteLength > 1000);
  console.log(`PASS: ${task.agent} → ${task.output} training → ${task.exportFormat} reload → reserved test → downloadable bundle. Validation ${run.score}; test ${run.testScore}.`);
  if (process.env.HELIX_TEST_FOLLOWUP === "1") {
    const original = await readFile(join(root, "runs", id, "final", "helix-solution.zip"));
    const waitMessage = async () => {
      const until = Date.now() + 240_000;
      while (Date.now() < until) {
        const result = (await request(`/runs/${id}`)).body.messages.at(-1);
        if (result.status !== "pending") { assert.equal(result.status, "completed", result.error); return result; }
        await setTimeout(1000);
      }
      throw new Error("Follow-up timed out");
    };
    assert.equal((await request(`/runs/${id}/messages`, { mode: "chat", message: "What validation accuracy was measured, and how many trials ran? Answer in one sentence." })).status, 202);
    const reply = await waitMessage();
    assert.ok(reply.reply.length > 10);
    console.log("Native follow-up reply: " + reply.reply);
    assert.equal((await request(`/runs/${id}/messages`, { mode: "trials", message: "Try max_depth=2 for the existing DecisionTreeClassifier. Keep everything else the same.", minutes: 5, trials: 1 })).status, 202);
    const continued = await waitMessage();
    assert.ok(continued.childRunId);
    const until = Date.now() + 360_000;
    let child, last = "";
    while (Date.now() < until) {
      child = (await request(`/runs/${continued.childRunId}`)).body;
      const progress = `${child.status}: ${child.phase}; ${child.logs.at(-1)?.message || ""}`;
      if (progress !== last) { console.log("Follow-up " + progress.slice(0, 400)); last = progress; }
      if (!["running", "queued"].includes(child.status)) break;
      await setTimeout(1000);
    }
    assert.equal(child.status, "completed", child.error);
    assert.equal(child.trials.length, 2, "One inherited model plus exactly one additional trial");
    assert.ok(Number.isFinite(child.trials[1].score), child.trials[1].detail);
    assert.equal(child.testScore, null);
    assert.equal(child.protocolHash, run.protocolHash);
    assert.deepEqual(await readFile(join(root, "runs", id, "final", "helix-solution.zip")), original);
    assert.equal((await request(`/runs/${id}`)).body.testScore, run.testScore);
    console.log(`PASS: native discussion → one additional measured trial → development-only refit → original bundle preserved. Additional validation ${child.trials[1].score}.`);
  }
  if (process.env.HELIX_KEEP_TEST_RUN) console.log(`Artifacts retained at ${root}`);
} finally {
  await service?.close();
  await cleanupContainers(join(root, "runs")).catch(() => {});
  if (!process.env.HELIX_KEEP_TEST_RUN) await rm(root, { recursive: true, force: true });
}
