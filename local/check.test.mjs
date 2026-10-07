import test from "node:test";
import assert from "node:assert/strict";
import {
  mkdtemp,
  mkdir,
  readFile,
  readdir,
  writeFile,
  symlink,
  rm,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { request } from "node:http";
import { spawn } from "node:child_process";
import { createService } from "./server.mjs";
import { validateTask, inside } from "./validate.mjs";
import { scorePredictions, isBetter } from "./metrics.mjs";
import { runProcess } from "./process.mjs";
import { agentStream, agentArguments, subscriptionEnvironment } from "./agents.mjs";
import { availableTools, callTool } from "./tools.mjs";
import { steps, createRun, loadRuns, saveRun } from "./engine.mjs";
import { aggregateScores } from "./evaluation.mjs";
import { matchingPredictions } from "./export.mjs";
import { readinessKey } from "./readiness.mjs";

test("runner validates configuration, isolates files, and enforces lifecycle boundaries", async () => {
  const root = await mkdtemp(join(tmpdir(), "helix-check-"));
  let service;
  try {
    const task = {
      agent: "codex",
      dataset: "/data/train.csv",
      target: "label",
      objective: "Predict labels",
      metric: "accuracy",
      minutes: 30,
      trials: 12,
      searchModels: false,
      model: "tree-based models only",
      output: "ipynb",
      exportModel: true,
      exportFormat: "joblib",
      validation: "cv",
      folds: 5,
      testFraction: 0.2,
      holdoutFraction: 0.2,
      seeds: [42, 43],
      splitStrategy: "independent",
      groupColumn: "",
      timeColumn: "",
      assetColumns: [],
      policy: {
        augmentation: false,
        regularization: true,
        features: true,
        tuning: true,
        pretrained: false,
        ensemble: false,
      },
    };
    assert.deepEqual(validateTask(task), task);
    assert.equal(validateTask({ ...task, testFraction: 0 }).testFraction, 0);
    for (const value of [-0.1, 1, NaN, "0.2"]) assert.throws(() => validateTask({ ...task, testFraction: value }), /test split/);
    assert.throws(() => validateTask({ ...task, holdoutFraction: 0 }), /validation split/);
    assert.equal(matchingPredictions([[0.2, 0.8]], [[0.20000001, 0.79999999]]), true);
    assert.equal(matchingPredictions(["a"], ["b"]), false);
    assert.equal(matchingPredictions([1], [null]), false);
    await assert.rejects(runProcess(process.execPath, ["-e", "process.stdout.write('x'.repeat(2100000))"]), /line larger than 2 MB/);
    assert.equal(steps(12, Object.fromEntries(Object.keys(task.policy).map(key => [key, false])), 1).length, 1);
    assert.equal(steps(12, task.policy).some((step) => ["merge", "ensemble"].includes(step.phase)), false);
    assert.equal(availableTools(task).some((tool) => tool.name === "search_models" || tool.name === "cache_model"), false);
    assert.equal(aggregateScores([{ score: .2 }, { score: .4 }], 2).score, .30000000000000004);
    assert.throws(() => aggregateScores([{ score: 1 }], 2), /Every requested/);
    for (const provider of ["codex", "claude"]) {
      const success = agentStream(provider);
      success.line(JSON.stringify(provider === "codex" ? { type: "turn.completed" } : { type: "result", subtype: "success", is_error: false }));
      success.finish();
      const failure = agentStream(provider);
      failure.line(JSON.stringify(provider === "codex" ? { type: "turn.failed", error: { message: "quota exhausted" } } : { type: "result", subtype: "error_max_turns", is_error: true, result: "quota exhausted" }));
      assert.throws(() => failure.finish(), /quota exhausted/);
      assert.throws(() => agentStream(provider).finish(), /without a successful/);
      const malformed = agentStream(provider);
      malformed.line("truncated JSON {");
      assert.throws(() => malformed.finish(), /malformed/);
    }
    assert.ok(agentArguments("codex", root, join(root, "context.json"), "test").includes("--ignore-user-config"));
    assert.ok(agentArguments("claude", root, join(root, "context.json"), "test").includes("--restricted"));
    assert.equal(subscriptionEnvironment().OPENAI_API_KEY, undefined);
    await writeFile(join(root, "run.json"), JSON.stringify({ task, trials: [] }));
    const history = { task, best: "trial-001", score: .7, baseline: { score: .5 }, trials: [
      { id: "trial-001", name: "Selected model", phase: "baseline", status: "accepted", score: .7 },
      { id: "trial-002", name: "Diagnostic ablation", phase: "ablation", status: "rejected", score: .8, baseTrial: "trial-001", baseScore: .7, component: "scaling", impact: -.1 },
    ] };
    await writeFile(join(root, "run.json"), JSON.stringify(history));
    const feedback = await callTool("previous_experiments", {}, { runFile: join(root, "run.json"), deadline: Date.now() + 1000 });
    assert.equal(feedback.selectedTrial, "trial-001", "The diagnostic raw winner must not replace the selected model in feedback");
    assert.equal(feedback.trials[1].baseTrial, "trial-001");
    assert.equal(feedback.trials[1].status, "rejected");
    assert.equal(feedback.trials[1].impact, -.1);
    assert.equal(feedback.baseline.score, .5);
    await writeFile(join(root, "run.json"), JSON.stringify({ task, trials: [] }));
    await assert.rejects(callTool("cache_model", { repository: "example/model" }, { runFile: join(root, "run.json"), deadline: Date.now() + 1000 }), /disabled/);
    await assert.rejects(callTool("write_source", { name: "../escape.py", content: "bad" }, { runFile: join(root, "run.json"), deadline: Date.now() + 1000, workspace: root }), /Write a Python/);
    const toolContext = { runFile: join(root, "run.json"), deadline: Date.now() + 1000, workspace: join(root, "candidate") };
    await mkdir(toolContext.workspace);
    await callTool("write_source", { name: "train.py", content: "print('source')" }, toolContext);
    await callTool("write_source", { name: "candidate.json", content: '{"name":"Requested model","approach":"Use the chosen estimator","sources":[]}' }, toolContext);
    assert.equal(JSON.parse(await callTool("read_source", { name: "candidate.json" }, toolContext)).name, "Requested model");
    assert.equal(await callTool("read_source", { name: "train.py" }, toolContext), "print('source')");
    await assert.rejects(callTool("read_source", { name: "../run.json" }, toolContext), /Invalid/);
    assert.throws(
      () => validateTask({ ...task, model: " " }),
      /Specify a model/,
    );
    assert.throws(() => validateTask({ ...task, seeds: [42, 42] }), /unique/);
    assert.throws(
      () => validateTask({ ...task, exportFormat: "made-up" }),
      /export format/,
    );
    assert.throws(() => validateTask({ ...task, policy: {} }), /permissions/);
    assert.equal(scorePredictions("accuracy", ["a", "b"], ["a", "a"]), 0.5);
    assert.equal(
      scorePredictions(
        "auroc",
        ["n", "p", "n", "p"],
        [0.1, 0.8, 0.4, 0.9],
        ["n", "p"],
      ),
      1,
    );
    assert.equal(
      scorePredictions("auroc", ["n", "p"], [0.5, 0.5], ["n", "p"]),
      0.5,
    );
    assert.equal(scorePredictions("rmse", [1, 3], [1, 5]), Math.sqrt(2));
    assert.equal(isBetter("log_loss", 0.2, 0.4), true);
    assert.throws(() => scorePredictions("accuracy", ["a"], []), /Expected/);
    assert.throws(() => scorePredictions("rmse", ["NaN"], [1]), /targets/);
    assert.throws(() => scorePredictions("auroc", ["n", "unknown"], [.1, .9], ["n", "p"]), /Unknown/);
    assert.throws(
      () => scorePredictions("log_loss", ["a"], [[0.8, 0.8]], ["a", "b"]),
      /summing/,
    );

    const source = join(root, "source");
    await mkdir(join(source, "images"), { recursive: true });
    await writeFile(
      join(source, "train.csv"),
      "id,feature,label\n" +
        Array.from(
          { length: 20 },
          (_, i) => `${i},${i * 2},${i % 2 ? "yes" : "no"}`,
        ).join("\n"),
    );
    await writeFile(join(root, "private.txt"), "must not enter candidate data");
    await symlink(
      join(root, "private.txt"),
      join(source, "images", "outside.txt"),
    );
    await assert.rejects(inside(source, "../private.txt"), /Invalid/);
    await assert.rejects(inside(source, "images/outside.txt"), /outside/);
    await runProcess(process.env.HELIX_PYTHON || "python3", [
      fileURLToPath(new URL("./protocol_check.py", import.meta.url)),
    ]);

    service = await createService({
      root: join(root, "runner"),
      token: "test-pairing-code-0123456789012345",
      port: 0,
    });
    await assert.rejects(createService({ root: join(root, "runner"), port: 0 }), /already owns/);
    await new Promise((resolve) =>
      service.server.listen(0, "127.0.0.1", resolve),
    );
    const url = `http://127.0.0.1:${service.server.address().port}`;
    const headers = {
      Authorization: `Bearer ${service.token}`,
      Origin: "http://127.0.0.1:5173",
    };
    const sessionHeaders = { Origin: url, "X-Helix-Local": "1", "Sec-Fetch-Site": "same-origin" };
    const session = await fetch(url + "/api/session", { method: "POST", headers: sessionHeaders });
    assert.equal(session.status, 200);
    assert.equal((await session.json()).token, service.token);
    assert.equal(session.headers.get("cache-control"), "no-store");
    for (const unsafe of [
      {}, { Origin: url }, { ...sessionHeaders, Origin: "https://untrusted.example" },
      { ...sessionHeaders, Origin: "http://127.0.0.1:5173" },
      { ...sessionHeaders, "Sec-Fetch-Site": "cross-site" },
    ]) {
      const result = await fetch(url + "/api/session", { method: "POST", headers: unsafe });
      assert.equal(result.status, 403, "Only the runner's own UI can bootstrap a session");
      assert.equal(result.headers.get("access-control-allow-origin"), null);
    }
    assert.equal((await fetch(url + "/api/session")).status, 405);
    assert.equal((await fetch(url + "/api/session", { method: "OPTIONS", headers })).status, 405);
    assert.equal((await fetch(url + "/api/runs")).status, 401);
    assert.equal(
      (
        await fetch(url + "/api/runs", {
          headers: { ...headers, Origin: "https://untrusted.example" },
        })
      ).status,
      403,
    );
    const badHostStatus = await new Promise((resolve, reject) => {
      const check = request(
        url + "/health",
        { headers: { Host: "untrusted.example" } },
        (response) => {
          response.resume();
          resolve(response.statusCode);
        },
      );
      check.on("error", reject);
      check.end();
    });
    assert.equal(badHostStatus, 403);
    assert.equal((await fetch(url + "/api/runs", { headers })).status, 200);
    const artifactRun = "00000000-0000-4000-8000-000000000001";
    const artifactRoot = join(root, "runner", "runs", artifactRun);
    await mkdir(join(artifactRoot, "trials"), { recursive: true });
    await mkdir(join(artifactRoot, "evaluation"));
    await writeFile(join(artifactRoot, "evaluation", "truth.json"), "private-targets");
    await symlink(join(artifactRoot, "evaluation", "truth.json"), join(artifactRoot, "trials", "leak.json"));
    service.runs.set(artifactRun, { id: artifactRun });
    for (const path of ["trials/../evaluation/truth.json", "trials/leak.json"]) {
      assert.equal((await fetch(`${url}/api/runs/${artifactRun}/file?path=${encodeURIComponent(path)}`, { headers })).status, 400);
    }
    service.runs.delete(artifactRun);
    const inspected = await fetch(url + "/api/datasets/inspect", {
      method: "POST", headers, body: JSON.stringify({ ...task, dataset: source }),
    });
    assert.equal(inspected.status, 200);
    const summary = await inspected.json();
    assert.equal(summary.fitsPerTrial, 10);
    assert.equal(summary.testRows, 4);
    assert.equal(summary.developmentRows, 16);
    assert.equal(summary.fingerprint.length, 64);
    assert.equal(
      (
        await fetch(url + "/api/runs", {
          method: "POST",
          headers,
          body: JSON.stringify({ ...task, agent: "invalid" }),
        })
      ).status,
      400,
    );
    const dataset = await (
      await fetch(url + "/api/datasets", {
        method: "POST",
        headers,
        body: "{}",
      })
    ).json();
    const fileUrl = `${url}/api/datasets/${dataset.id}/file?name=train.csv`;
    assert.equal(
      (await fetch(fileUrl, { method: "PUT", headers, body: "original" }))
        .status,
      201,
    );
    assert.equal(
      (await fetch(fileUrl, { method: "PUT", headers, body: "replacement" }))
        .status,
      400,
    );
    assert.equal(
      await readFile(join(dataset.path, "train.csv"), "utf8"),
      "original",
    );
    assert.equal(
      (
        await fetch(`${url}/api/datasets/${dataset.id}/file?name=../escape`, {
          method: "PUT",
          headers,
          body: "bad",
        })
      ).status,
      400,
    );
    const lifecycle = await createRun(join(root, "runner", "runs"), task);
    service.runs.set(lifecycle.id, lifecycle);
    lifecycle.status = "paused";
    lifecycle.elapsed = task.minutes * 60;
    const act = (action) => fetch(`${url}/api/runs/${lifecycle.id}/action`, { method: "POST", headers, body: JSON.stringify({ action }) });
    assert.match((await (await act("resume")).json()).error, /time budget/);
    lifecycle.elapsed = 1;
    lifecycle.selectionFrozen = true;
    assert.match((await (await act("resume")).json()).error, /Final evaluation/);
    assert.equal((await (await act("stop")).json()).status, "stopped");
    assert.match((await (await act("resume")).json()).error, /Only paused/);
    assert.equal((await act("invalid")).status, 400);
    assert.equal(service.runs.size, 1);
  } finally {
    await service?.close();
    await rm(root, { recursive: true, force: true });
  }
});

test("agent readiness requires actual tool use and blocks experiments on failure", async () => {
  const root = await mkdtemp(join(tmpdir(), "helix-readiness-check-"));
  const previousPath = process.env.PATH;
  const previousMode = process.env.HELIX_TEST_CLI_MODE;
  let service;
  try {
    const bin = join(root, "bin");
    await mkdir(bin);
    const stub = `#!${process.execPath}
const fs = require('node:fs');
const args = process.argv.slice(2);
const name = process.argv[1].split('/').pop();
if (name === 'npm') {
  fs.appendFileSync(process.env.HELIX_TEST_SETUP_LOG, args.join(' ') + '\\n');
  if (process.env.HELIX_TEST_CLI_MODE === 'setup-wait') {
    process.on('SIGINT', () => { fs.appendFileSync(process.env.HELIX_TEST_SETUP_LOG, 'stopped\\n'); process.exit(0); });
    setTimeout(() => process.exit(1), 10000);
  } else if (args.includes('build')) { console.error('intentional build failure'); process.exit(1); }
} else if (name === 'docker') {
  if (process.env.HELIX_TEST_CLI_MODE === 'missing-docker') process.exit(1);
  if (args[0] === 'image') console.log('2 sha256:${"a".repeat(64)}');
} else if (args[0] === '--version') console.log(name + ' test-version');
else if (args[0] === 'login') console.log('Logged in using ChatGPT');
else if (args[0] === 'auth') console.log(JSON.stringify({loggedIn:true,authMethod:'claude.ai'}));
else {
  // A successful CLI completion alone must never count as a working harness.
  setTimeout(() => console.log(JSON.stringify(name === 'codex' ? {type:'turn.completed'} : {type:'result',subtype:'success',is_error:false})), process.env.HELIX_TEST_CLI_MODE === 'stall' ? 30000 : 700);
}
`;
    for (const name of ["codex", "claude", "docker", "npm"]) await writeFile(join(bin, name), stub, { mode: 0o700 });
    process.env.PATH = bin + ":" + previousPath;
    const setupLog = join(root, "setup.log");
    const launcher = fileURLToPath(new URL("./start.mjs", import.meta.url));
    await assert.rejects(runProcess(process.execPath, [launcher], { env: { ...process.env, HELIX_TEST_CLI_MODE: "missing-docker", HELIX_TEST_SETUP_LOG: setupLog } }), /Docker is unavailable/);
    await assert.rejects(readFile(setupLog), { code: "ENOENT" });
    await assert.rejects(runProcess(process.execPath, [launcher], { env: { ...process.env, HELIX_TEST_SETUP_LOG: setupLog } }), /intentional build failure/);
    assert.equal(await readFile(setupLog, "utf8"), "ci\nrun build\n");
    await writeFile(setupLog, "");
    const interruptedSetup = spawn(process.execPath, [launcher], { env: { ...process.env, HELIX_TEST_SETUP_LOG: setupLog, HELIX_TEST_CLI_MODE: "setup-wait" }, stdio: "ignore" });
    const setupExit = new Promise(resolve => interruptedSetup.once("exit", resolve));
    try {
      for (let i = 0; i < 100 && !(await readFile(setupLog, "utf8")); i++) await new Promise(resolve => setTimeout(resolve, 20));
      assert.equal(await readFile(setupLog, "utf8"), "ci\n");
      interruptedSetup.kill("SIGINT");
      assert.equal(await setupExit, 130);
      assert.equal(await readFile(setupLog, "utf8"), "ci\nstopped\n", "Interrupting setup must stop before building or launching a server");
    } finally { interruptedSetup.kill("SIGTERM"); }
    const deadPid = Number((await runProcess(process.execPath, ["-e", "console.log(process.pid)"])).output.trim());
    const staleRoot = join(root, "stale-runner");
    await mkdir(join(staleRoot, ".runner-lock"), { recursive: true });
    await writeFile(join(staleRoot, ".runner-lock", `${deadPid}-00000000-0000-4000-8000-000000000001`), "");
    const recovered = await createService({ root: staleRoot, port: 0 });
    await recovered.close();
    assert.ok(!(await readdir(staleRoot)).includes(".runner-lock"));
    service = await createService({ root: join(root, "runner"), port: 0 });
    await new Promise(resolve => service.server.listen(0, "127.0.0.1", resolve));
    const url = `http://127.0.0.1:${service.server.address().port}/api`;
    const headers = { Authorization: `Bearer ${service.token}`, "Content-Type": "application/json" };
    const post = (path, body = {}) => fetch(url + path, { method: "POST", headers, body: JSON.stringify(body) });
    const status = async () => (await (await fetch(url + "/providers", { headers })).json()).find(p => p.id === "codex").capability.status;
    assert.equal((await fetch(url + "/providers/codex/verify", { method: "POST", body: "{}" })).status, 401);
    assert.equal((await post("/providers/invalid/verify")).status, 404);
    assert.equal((await post("/providers/codex/verify", { force: "yes" })).status, 400);
    assert.equal(await status(), "unchecked");
    const pending = post("/providers/codex/verify");
    for (let i = 0; i < 50 && await status() !== "checking"; i++) await new Promise(resolve => setTimeout(resolve, 20));
    assert.equal(await status(), "checking");
    assert.equal((await post("/runs")).status, 409);
    const second = post("/providers/codex/verify");
    assert.match((await (await post("/providers/claude/verify")).json()).error, /Another agent check/);
    for (const response of await Promise.all([pending, second])) {
      assert.equal(response.status, 400);
      assert.match((await response.json()).error, /did not demonstrate access/);
    }
    assert.equal(await status(), "failed");
    assert.equal(service.runs.size, 0);
    const dataset = join(root, "train.csv");
    await writeFile(dataset, "x,label\n" + Array.from({ length: 40 }, (_, i) => `${i},${i % 2 ? 'yes' : 'no'}`).join("\n"));
    const task = { agent: "codex", dataset, target: "label", objective: "Test failed agent gate", metric: "accuracy", minutes: 1, trials: 3, searchModels: false, model: "tree-based models only", output: "py", exportModel: false, exportFormat: "native", validation: "holdout", folds: 3, seeds: [42], policy: { augmentation: false, regularization: false, features: false, tuning: false, pretrained: false, ensemble: false } };
    const invalidDataset = await post("/runs", { ...task, target: "absent" });
    assert.equal(invalidDataset.status, 400);
    assert.match((await invalidDataset.json()).error, /Target column.*absent/);
    assert.equal(await status(), "failed", "Invalid data must be rejected before another subscription check");
    const start = await post("/runs", task);
    assert.equal(start.status, 400);
    assert.match((await start.json()).error, /Agent verification failed/);
    assert.equal(service.runs.size, 0, "A failed verification must not create a user experiment");
    assert.deepEqual(await readdir(join(root, "runner", "checks")), []);
    const persisted = await createRun(join(root, "recovery"), task);
    Object.assign(persisted, { status: "running", startedAt: new Date(Date.now() - 10_000).toISOString(), elapsed: 15,
      next: 1, best: "trial-001", score: .75, baseline: { name: "Most frequent class", score: .5 }, trials: [{ id: "trial-001", score: .75 }] });
    await saveRun(join(root, "recovery"), persisted);
    const restored = (await loadRuns(join(root, "recovery"))).get(persisted.id);
    assert.equal(restored.status, "paused");
    assert.equal(restored.startedAt, null);
    assert.ok(restored.elapsed >= 25 && restored.elapsed < 27);
    assert.equal(restored.next, 1);
    assert.deepEqual(restored.trials, persisted.trials);
    assert.deepEqual(restored.baseline, persisted.baseline);
    process.env.HELIX_TEST_CLI_MODE = "stall";
    const cancelled = post("/providers/codex/verify").catch(() => null);
    for (let i = 0; i < 50 && await status() !== "checking"; i++) await new Promise(resolve => setTimeout(resolve, 20));
    assert.equal(await status(), "checking");
    await service.close();
    service = null;
    await cancelled;
    assert.deepEqual(await readdir(join(root, "runner", "checks")), []);
    process.env.HELIX_TEST_CLI_MODE = "";
    const cacheRoot = join(root, "cached-runner"), cache = join(cacheRoot, "verified-codex.json");
    await mkdir(cacheRoot);
    const evidence = { status: "verified", key: await readinessKey("codex test-version", `sha256:${"a".repeat(64)}`),
      version: "codex test-version", image: `sha256:${"a".repeat(64)}`, verifiedAt: new Date().toISOString(),
      checks: ["mcp", "write", "train", "predict", "reload"], score: 1, sourceHash: "b".repeat(64) };
    const checkCache = async (force = false) => {
      service = await createService({ root: cacheRoot, port: 0 });
      await new Promise(resolve => service.server.listen(0, "127.0.0.1", resolve));
      try {
        const response = await fetch(`http://127.0.0.1:${service.server.address().port}/api/providers/codex/verify`, {
          method: "POST", headers: { Authorization: `Bearer ${service.token}`, "Content-Type": "application/json" }, body: JSON.stringify({ force }),
        });
        return { status: response.status, body: await response.json() };
      } finally { await service.close(); service = null; }
    };
    // Fixture evidence tests reuse/invalidation. Actual successful writing, fits
    // and reloads are covered by the opt-in native-agent acceptance checks.
    await writeFile(cache, JSON.stringify(evidence));
    assert.equal((await checkCache()).body.verifiedAt, evidence.verifiedAt);
    assert.equal((await checkCache()).status, 200, "A runner restart must reuse valid evidence");
    assert.equal((await checkCache(true)).status, 400, "Force must invoke the agent even with cached evidence");
    await assert.rejects(readFile(cache), { code: "ENOENT" });
    for (const invalid of [{ ...evidence, key: "old-code-or-runtime" }, { ...evidence, verifiedAt: "2000-01-01T00:00:00Z" }, { ...evidence, checks: ["mcp"] }]) {
      await writeFile(cache, JSON.stringify(invalid));
      assert.equal((await checkCache()).status, 400, "Stale or incomplete evidence must run a fresh check");
    }
  } finally {
    await service?.close();
    process.env.PATH = previousPath;
    if (previousMode === undefined) delete process.env.HELIX_TEST_CLI_MODE;
    else process.env.HELIX_TEST_CLI_MODE = previousMode;
    await rm(root, { recursive: true, force: true });
  }
});
