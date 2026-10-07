// Opt-in: real public data, real native agents and Docker. Retains evidence for inspection.
import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import { resolve, join } from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout } from "node:timers/promises";
import { createService } from "./server.mjs";
import { container, cleanupContainers } from "./runtime.mjs";
import { runProcess } from "./process.mjs";
import { isBetter } from "./metrics.mjs";
import { matchingPredictions } from "./export.mjs";

const project = fileURLToPath(new URL("..", import.meta.url));
const python = process.env.HELIX_PYTHON || "python3";
const evidence = resolve(project, "output", "acceptance");
await mkdir(evidence, { recursive: true });
const root = process.env.HELIX_ACCEPTANCE_ROOT ? resolve(process.env.HELIX_ACCEPTANCE_ROOT) : await mkdtemp(join(evidence, new Date().toISOString().replaceAll(":", "-") + "-"));
const cases = process.argv.slice(2).length ? process.argv.slice(2) : ["bank", "bike"];
assert.ok(cases.every(name => ["bank", "bike"].includes(name)), "Choose bank or bike");
await mkdir(root, { recursive: true });
const report = { startedAt: new Date().toISOString(), platform: `${process.platform}/${process.arch}`, results: [] };
const previousReport = JSON.parse(await readFile(join(root, "report.json"), "utf8").catch(() => "{}"));
let service, url;
let interrupted = false;
for (const signal of ["SIGINT", "SIGTERM"]) process.once(signal, () => { interrupted = true; void service?.close(); });
const open = async () => {
  service = await createService({ root, port: 0 });
  await new Promise(resolve => service.server.listen(0, "127.0.0.1", resolve));
  url = `http://127.0.0.1:${service.server.address().port}/api`;
};
const request = async (path, body) => {
  const response = await fetch(url + path, { method: body ? "POST" : "GET", headers: { Authorization: `Bearer ${service.token}`, "Content-Type": "application/json" }, ...(body ? { body: JSON.stringify(body) } : {}), signal: AbortSignal.timeout(360_000) });
  const value = await response.json();
  assert.ok(response.ok, value.error || `${path}: ${response.status}`);
  return value;
};
try {
  await open();
  for (const name of cases) {
    const record = { dataset: name, status: "running" };
    report.results.push(record);
    const data = join(root, "data", name);
    record.provenance = JSON.parse((await runProcess(python, [join(project, "local", "acceptance_data.py"), name, data], { timeout: 60_000 })).output);
    const task = {
      agent: process.env.HELIX_TEST_AGENT || (name === "bank" ? "codex" : "claude"), dataset: join(data, "train.csv"),
      target: name === "bank" ? "y" : "cnt", metric: name === "bank" ? "auroc" : "rmse",
      objective: name === "bank" ? "Estimate term-deposit subscription probability before a marketing call. Handle missing categories and imbalanced labels. Compare suitable scikit-learn CPU models using the installed packages; no external downloads are needed. Use Helix's measured validation feedback for improvements." : "Estimate daily bike rentals from observed weather and calendar inputs. Validation and test dates occur after training dates. Compare suitable scikit-learn CPU models using installed packages; no external downloads are needed. Use Helix's measured validation feedback for improvements.",
      minutes: 30, trials: 6, searchModels: true, model: "", output: "py", exportModel: true, exportFormat: "joblib",
      validation: "cv", folds: 3, seeds: [42, 17], testFraction: .2, holdoutFraction: .2,
      splitStrategy: name === "bike" ? "time" : "independent", timeColumn: name === "bike" ? "dteday" : "", groupColumn: "", assetColumns: [],
      policy: { augmentation: false, regularization: true, features: true, tuning: true, pretrained: false, ensemble: false },
    };
    record.agent = task.agent;
    const inspection = await request("/datasets/inspect", task);
    assert.equal(inspection.fitsPerTrial, 6);
    if (name === "bank") assert.ok(inspection.schema.some(column => column.missing > 0));
    console.log(`${name}: ${inspection.rows} rows, ${task.agent}, ${task.metric}, 6 trials, 6 fits per trial. Evidence: ${root}`);
    let run = (await request("/runs")).find(saved => saved.task.dataset === task.dataset);
    if (run && ["paused", "failed"].includes(run.status)) run = await request(`/runs/${run.id}/action`, { action: "resume" });
    else if (!run) run = await request("/runs", task);
    record.runId = run.id;
    let previous = "", restarted = previousReport.results?.some(result => result.runId === run.id && result.restartVerified) || false;
    if (restarted) record.restartVerified = true;
    const deadline = Date.now() + (task.minutes + 4) * 60_000;
    while (Date.now() < deadline) {
      if (interrupted) throw new Error("Acceptance check interrupted; completed artifacts are preserved.");
      run = await request(`/runs/${run.id}`);
      const message = `${name}: ${run.status}, ${run.phase}, ${run.trials.length} trials; ${run.logs.at(-1)?.message || ""}`;
      if (message !== previous) { console.log(message.slice(0, 500)); previous = message; }
      if (!restarted && name === "bank" && run.trials.some(trial => trial.score != null) && run.status === "running" && run.phase !== "finalizing") {
        const completed = run.trials.map(trial => ({ id: trial.id, score: trial.score, sourceHash: trial.sourceHash }));
        await request(`/runs/${run.id}/action`, { action: "pause" });
        await service.close();
        await open();
        const saved = await request(`/runs/${run.id}`);
        assert.equal(saved.status, "paused");
        assert.deepEqual(saved.trials.slice(0, completed.length).map(trial => ({ id: trial.id, score: trial.score, sourceHash: trial.sourceHash })), completed);
        await request(`/runs/${run.id}/action`, { action: "resume" });
        restarted = true;
        record.restartVerified = true;
      }
      if (!["running", "queued"].includes(run.status)) break;
      await setTimeout(2000);
    }
    record.status = run.status;
    record.metric = task.metric;
    record.validation = run.score;
    record.baseline = run.baseline?.score;
    record.test = run.testScore;
    record.testBaseline = run.testBaseline?.score;
    record.elapsed = run.elapsed;
    record.trials = run.trials.map(({ name, phase, score, status, sourceHash }) => ({ name, phase, score, status, sourceHash }));
    assert.equal(run.status, "completed", run.error || "Experiment did not finish");
    assert.ok(run.elapsed <= task.minutes * 60 + 10, "Experiment exceeded its budget");
    assert.ok(run.trials.filter(trial => trial.phase === "baseline" && trial.score != null).length >= 2, "Need at least two real candidate comparisons");
    assert.ok(run.trials.some(trial => trial.phase === "refinement" && trial.score != null), "Need a measured refinement");
    for (const trial of run.trials.filter(trial => trial.score != null)) assert.equal(trial.evaluations.length, 6);
    assert.ok(isBetter(task.metric, run.score, run.baseline.score), "Selected model did not beat the training-only baseline");
    assert.ok(isBetter(task.metric, run.testScore, run.testBaseline.score), "Model did not beat baseline on the held-out test");
    const folder = join(root, "runs", run.id);
    const download = await fetch(`${url}/runs/${run.id}/file?path=final/helix-solution.zip`, { headers: { Authorization: `Bearer ${service.token}` } });
    assert.equal(download.status, 200);
    const archive = join(root, `${name}-solution.zip`), extracted = join(root, `${name}-solution`);
    await writeFile(archive, Buffer.from(await download.arrayBuffer()));
    await runProcess(python, ["-m", "zipfile", "-e", archive, extracted]);
    const manifest = JSON.parse(await readFile(join(extracted, "experiment.json"), "utf8"));
    assert.equal(manifest.exportVerified, true);
    assert.equal(manifest.baseline.score, run.baseline.score);
    const input = join(root, `${name}-inference`), output = join(root, `${name}-predictions`);
    await runProcess(python, [join(project, "local", "prepare.py"), "--evaluation", join(folder, "evaluation"), "--output", input, "--fold", "final", "--truth", join(root, `${name}-truth.json`)]);
    await rm(join(input, "train.csv"));
    await rm(join(root, `${name}-truth.json`));
    await mkdir(output);
    // Only the downloaded bundle and unlabeled input are mounted. No run workspace is available.
    await container({ root, runId: run.id, image: run.environment.image, deadline: Date.now() + 60_000, user: `${process.getuid()}:${process.getgid()}` },
      ["python", "/code/predict.py", "--input", "/input/validation.csv", "--metadata", "/solution/manifest.json", "--config", "/solution/config.json", "--model-dir", "/solution/model", "--models", "/solution/pretrained", "--output", "/work/predictions.json"],
      { writable: output, mounts: [[join(extracted, "source"), "/code"], [extracted, "/solution"], [input, "/input"], [output, "/work", "rw"]] });
    assert.ok(matchingPredictions(JSON.parse(await readFile(join(folder, "fits", "selected", "final", "output", "predictions.json"), "utf8")), JSON.parse(await readFile(join(output, "predictions.json"), "utf8"))), "Downloaded solution predictions changed");
    record.downloadVerified = true;
    record.status = "passed";
    console.log(`PASS ${name}: validation ${run.score} vs baseline ${run.baseline.score}; test ${run.testScore} vs ${run.testBaseline.score}; downloaded model predicts independently.`);
    await writeFile(join(root, "report.json"), JSON.stringify(report, null, 2));
  }
} catch (error) {
  const record = report.results.at(-1);
  if (record) { record.status = "failed"; record.error = error.message; }
  console.error(error.message);
  process.exitCode = 1;
} finally {
  await service?.close();
  await cleanupContainers(join(root, "runs")).catch(() => {});
  report.finishedAt = new Date().toISOString();
  await writeFile(join(root, "report.json"), JSON.stringify(report, null, 2));
  console.log(`Acceptance report: ${join(root, "report.json")}`);
}
