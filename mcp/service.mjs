import { mkdir, readFile, writeFile, readdir, cp, realpath, lstat, rm } from "node:fs/promises";
import { join, resolve, relative, isAbsolute } from "node:path";
import { createRun, saveRun, log, contract } from "../local/engine.mjs";
import { validateTask, redact } from "../local/validate.mjs";
import { prepareEvaluation, evaluateCandidate, evaluateFinal, sourceFingerprint, publicDatasetInfo } from "../local/evaluation.mjs";
import { finishBundle, harness, makeNotebook } from "../local/export.mjs";
import { container, directoryBytes } from "../local/runtime.mjs";
import { callTool } from "../local/tools.mjs";
import { CATALOG, isBetter } from "../local/metrics.mjs";
import { expired, runProcess, remainingTime } from "../local/process.mjs";
import { lockRunner } from "../local/lock.mjs";
import { tools, validateArguments } from "./schema.mjs";

const json = async path => JSON.parse(await readFile(path, "utf8"));
const within = (root, path) => { const name = relative(root, path); return !name.startsWith("..") && !isAbsolute(name); };
const defaults = { agent: "codex", learning: "supervised", target: "", model: "", minutes: 30, trials: 12, validation: "holdout", folds: 3, seeds: [42], output: "py", exportModel: true, exportFormat: "joblib", testFraction: .2, holdoutFraction: .2 };

export async function createMcpService(home, runtime) {
  const root = join(home, "experiments");
  await mkdir(root, { recursive: true, mode: 0o700 });
  // Only this dedicated MCP process uses native execution; the web runner retains Docker.
  process.env.HELIX_PYTHON = runtime.python;
  process.env.HELIX_TRAIN_MEMORY_MB = String(runtime.memoryMiB);
  process.env.HELIX_PREP_MEMORY_MB = String(runtime.memoryMiB);
  const jobs = new Map();
  let closing = false;
  const folder = id => {
    if (typeof id !== "string" || !/^[a-f0-9-]{36}$/.test(id)) throw new Error("Invalid experiment ID.");
    return join(root, id);
  };
  const load = async id => {
    const run = jobs.get(id)?.run || await json(join(folder(id), "run.json"));
    if (run.environment?.runtime !== "native" || run.id !== id) throw new Error("This is not a local MCP experiment.");
    return run;
  };
  const context = run => ({ root, runId: run.id, runtime, deadline: run.deadline, models: join(folder(run.id), "models"), runFile: join(folder(run.id), "run.json") });
  const view = async (run, details = false) => {
    let status = run.status;
    if (status === "running" && run.ownerPid) {
      try { process.kill(run.ownerPid, 0); } catch (error) { if (error.code === "ESRCH") status = "interrupted"; }
    }
    return {
      runId: run.id, status, phase: run.phase, objective: run.task.objective, task: run.task,
      deadline: run.deadline === null ? null : new Date(run.deadline).toISOString(),
      remainingSeconds: run.deadline === null ? null : Math.max(0, Math.floor((run.deadline - Date.now()) / 1000)),
      trials: details ? run.trials : run.trials.slice(-5).map(({ id, status, score, metricScores, error }) => ({ id, status, score, metricScores, error })),
      trialCount: run.trials.length, best: run.best, validationScore: run.score, metricScores: run.metricScores,
      testScore: run.testScore, testMetricScores: run.testMetricScores,
      baseline: run.baseline, trainingHistory: run.trainingHistory, projection: run.projection, cachedModels: run.cachedModels,
      error: run.error, logs: run.logs.slice(-10),
      ...(run.prepared && details ? { dataset: await json(join(folder(run.id), "data", "manifest.json")), contract: contract(run) } : {}),
      ...(run.bundle ? { solutionZip: run.bundle, sourceDir: join(folder(run.id), "final", "source") } : {}),
    };
  };

  async function start(id, phase, check, action) {
    if (closing) throw new Error("The MCP server is shutting down.");
    if (jobs.size) throw new Error("A local job is already running. Wait or cancel it first.");
    // Shared by all Codex/Claude sessions: never race a final test or oversubscribe this runtime.
    const release = await lockRunner(join(home, "compute"));
    let run;
    try {
      run = await load(id);
      if (expired(run.deadline)) throw new Error("The time budget has ended. Use helix_extend_budget if the user wants more time; completed trials are preserved.");
      await check(run);
      run.status = "running"; run.phase = phase; run.ownerPid = process.pid; delete run.error;
      await saveRun(root, run);
    } catch (error) { await release(); throw error; }
    const controller = new AbortController();
    const job = { run, controller };
    jobs.set(id, job);
    const heartbeat = setInterval(() => { void saveRun(root, run).catch(() => {}); }, 2000);
    job.promise = (async () => {
      try { await action(run, controller.signal); run.status = run.bundle ? "completed" : "ready"; run.phase = run.bundle ? "complete" : "ready"; }
      catch (error) { run.status = controller.signal.aborted ? "paused" : "error"; run.error = redact(error.message); log(run, "error", run.error); }
      finally {
        clearInterval(heartbeat); delete run.ownerPid;
        try { await saveRun(root, run); } finally { jobs.delete(id); await release(); }
      }
    })();
    // Persisting a job failure must not become an unhandled promise rejection.
    job.promise.catch(error => console.error("Helix could not save its job:", redact(error.message)));
    return { runId: id, status: "running", phase, next: "Poll helix_status in a few seconds." };
  }

  const ready = run => {
    if (!run.prepared) throw new Error("Dataset preparation has not completed. Check helix_status.");
    if (run.selectionFrozen) throw new Error("Selection is frozen. Further validation trials cannot use an experiment whose test was revealed. Create a new experiment for a new study.");
  };

  async function call(name, args = {}) {
    const tool = tools.find(tool => tool.name === name);
    if (!tool) throw new Error("Unknown Helix tool.");
    validateArguments(tool.inputSchema, args);
    if (name === "helix_info") return {
      runtime: { ...runtime, platform: process.platform, architecture: process.arch },
      execution: "Trusted local Python with normal user permissions. No Docker, host sandbox, or hard RAM quota. Training data remains local unless your agent sends it to its provider. No Helix cloud sync.",
      metrics: CATALOG,
      workflow: "Show a proposed task, metrics, target, restrictions, splits and budgets. Create experiment; poll status for schema and code contract. Write source using your own editor tools. Submit run_trial; poll status and improve from measured validation results. Finalize the best candidate once. All long operations return immediately. Closing the client cancels active work and preserves completed trials. No nested agent process is launched.",
      examples: join(harness, "..", "examples", "measurements.csv"),
    };
    if (name === "helix_list_experiments") {
      const runs = [];
      for (const dir of await readdir(root, { withFileTypes: true })) {
        if (!dir.isDirectory() || !/^[a-f0-9-]{36}$/.test(dir.name)) continue;
        const run = await load(dir.name);
        runs.push({ runId: run.id, objective: run.task.objective, status: run.status, createdAt: run.createdAt, trials: run.trials.length, validationScore: run.score, solutionZip: run.bundle });
      }
      return runs.sort((a, b) => b.createdAt.localeCompare(a.createdAt)).slice(0, 100);
    }
    if (name === "helix_create_experiment") {
      if (!isAbsolute(args.dataset)) throw new Error("Use an absolute dataset path.");
      const dataset = await realpath(args.dataset);
      if (!(await lstat(dataset)).isFile() || !dataset.toLowerCase().endsWith(".csv")) throw new Error("Choose a local CSV file.");
      if (within(await realpath(root), dataset)) throw new Error("Choose the original dataset outside Helix's private experiment directory.");
      const policy = { augmentation: false, regularization: true, features: true, tuning: true, pretrained: false, ensemble: false, ...args.policy };
      const task = validateTask({ ...defaults, ...args, dataset, policy, searchModels: !args.model });
      const run = await createRun(root, task);
      run.environment = { runtime: "native", python: runtime.pythonVersion, platform: process.platform, architecture: process.arch, cpus: runtime.cpus, memoryMiB: runtime.memoryMiB, isolation: "trusted-user-process" };
      run.deadline = task.minutes === null ? null : Date.now() + task.minutes * 60_000;
      await saveRun(root, run);
      return start(run.id, "preparing", () => {}, async (run, signal) => {
        const manifest = publicDatasetInfo(await prepareEvaluation(root, run, { signal, deadline: run.deadline }));
        await mkdir(join(folder(run.id), "data"));
        await mkdir(join(folder(run.id), "models"));
        manifest.installedPackages = JSON.parse((await container(context(run), ["python", "-c", "import json, importlib.metadata as m; print(json.dumps({d.metadata['Name']:d.version for d in m.distributions() if d.metadata['Name']}))"], { signal })).output);
        await writeFile(join(folder(run.id), "data", "manifest.json"), JSON.stringify(manifest));
        run.protocolHash = (await json(join(folder(run.id), "evaluation", "protocol.json"))).sha256;
        run.prepared = true;
        log(run, "system", "Dataset prepared. Write a candidate using the returned schema and contract.");
      }).catch(async error => { await rm(folder(run.id), { recursive: true, force: true }); throw error; });
    }
    if (name === "helix_status") return view(await load(args.runId), args.details);
    if (name === "helix_cache_model") return start(args.runId, "caching model", run => {
      ready(run);
      if (!run.task.policy.pretrained) throw new Error("Pretrained models are disabled for this experiment.");
    }, async (run, signal) => {
      const record = await callTool("cache_model", { repository: args.repository, revision: args.revision }, { ...context(run), signal });
      run.cachedModels = [...(run.cachedModels || []).filter(model => model.repository !== record.repository || model.revision !== record.revision), record];
    });
    if (name === "helix_cancel") {
      const job = jobs.get(args.runId);
      if (!job) { const run = await load(args.runId); if (run.status === "running") throw new Error("Cancel from the Codex/Claude session running this job, or close that session."); return { runId: run.id, status: run.status }; }
      job.controller.abort();
      return { runId: args.runId, status: "cancelling", next: "Poll helix_status until the job stops." };
    }
    if (name === "helix_extend_budget") {
      const release = await lockRunner(join(home, "compute"));
      try {
        const run = await load(args.runId); ready(run);
        run.task = validateTask({ ...run.task, minutes: args.minutes, trials: args.trials });
        if (run.task.trials !== null && run.task.trials < run.trials.length) throw new Error("The new total must include all trials already attempted.");
        run.deadline = args.minutes === null ? null : Date.parse(run.createdAt) + args.minutes * 60_000;
        await saveRun(root, run);
        return view(run);
      } finally { await release(); }
    }
    if (name === "helix_run_trial") {
      if (!isAbsolute(args.sourceDir)) throw new Error("Use an absolute source directory.");
      const source = await realpath(args.sourceDir), state = await realpath(home);
      if (within(state, source) || within(source, state)) throw new Error("Keep candidate source outside Helix's private state directory, in a separate project folder.");
      return start(args.runId, "training", run => {
        ready(run);
        if (run.task.trials !== null && run.trials.length >= run.task.trials) throw new Error("The trial budget has ended. Finalize, or extend the budget at the user's request.");
      }, async (run, signal) => {
        const id = `trial-${run.trials.length + 1}`;
        const trial = { id, status: "running", createdAt: new Date().toISOString(), score: null };
        run.trials.push(trial);
        const workspace = join(folder(run.id), "trials", id), packages = join(folder(run.id), "dependencies", id);
        try {
          const sourceInfo = await sourceFingerprint(source);
          if (!sourceInfo.files.includes("train.py") || !sourceInfo.files.includes("plan.json")) throw new Error("Provide train.py and plan.json.");
          if (sourceInfo.files.length > 200 || sourceInfo.files.some(name => !/\.(py|json|md|txt|ipynb)$/.test(name)) || await directoryBytes(source) > 10_000_000)
            throw new Error("Submit a source-only directory: at most 200 Python/JSON/Markdown/text/notebook files, 10 MB total. Keep data, environments and credentials elsewhere.");
          await cp(source, workspace, { recursive: true });
          await mkdir(packages, { recursive: true });
          const plan = await json(join(workspace, "plan.json"));
          for (const field of ["component", "change", "rationale", "modelFamily"]) if (typeof plan[field] !== "string" || !plan[field].trim()) throw new Error(`plan.json needs ${field}.`);
          for (const [key, allowed] of Object.entries(run.task.policy)) {
            if (typeof plan.strategies?.[key] !== "boolean") throw new Error(`Declare strategy ${key} in plan.json.`);
            if (!allowed && plan.strategies[key]) throw new Error(`Candidate uses forbidden strategy: ${key}.`);
          }
          trial.plan = plan;
          await runProcess(runtime.python, [join(harness, "audit.py"), workspace, join(folder(run.id), "task.json")], { signal, timeout: remainingTime(run.deadline) });
          for (const pkg of args.packages || []) {
            log(run, "system", `Installing ${pkg.name}==${pkg.version} for ${id}.`);
            await callTool("install_package", pkg, { ...context(run), signal, workspace, packages });
          }
          if (run.task.output === "ipynb") await makeNotebook(workspace, run.task.seeds[0]);
          const measured = await evaluateCandidate(root, run, workspace, { signal, deadline: run.deadline, candidateId: id, runtime, packages, onEvent: (type, message) => log(run, type, message) });
          Object.assign(trial, measured, { status: "completed" });
          if (isBetter(run.task.metric, measured.score, run.score)) {
            run.best = id; run.score = measured.score; run.metricScores = measured.metricScores; run.baseline = measured.baseline;
          }
        } catch (error) { trial.status = signal.aborted ? "cancelled" : "failed"; trial.error = redact(error.message); throw error; }
      });
    }
    if (name === "helix_finalize") return start(args.runId, "finalizing", run => {
      if (!run.prepared || !run.best) throw new Error("Complete at least one successful validation trial first.");
      if (run.bundle) throw new Error("The experiment is already exported. Read helix_status for its solution ZIP.");
      // A frozen run may retry packaging, but evaluateFinal never repeats a started test.
    }, async (run, signal) => {
      run.selectionFrozen = true;
      await saveRun(root, run);
      const final = join(folder(run.id), "final", "source");
      if (!await lstat(final).catch(error => { if (error.code === "ENOENT") return null; throw error; }))
        await cp(join(folder(run.id), "trials", run.best), final, { recursive: true });
      const result = await evaluateFinal(root, run, final, { signal, deadline: run.deadline, runtime, packages: join(folder(run.id), "dependencies", run.best), onEvent: (type, message) => log(run, type, message) });
      run.testScore = result.score; run.testMetricScores = result.metricScores; run.testBaseline = result.baseline;
      run.trainingHistory = result.trainingHistory; run.projection = result.projection;
      await finishBundle(root, run, context(run), signal);
      run.bundle = join(folder(run.id), "final", "helix-solution.zip");
    });
  }
  return { call, async close() { closing = true; for (const job of jobs.values()) job.controller.abort(); await Promise.allSettled([...jobs.values()].map(job => job.promise)); } };
}
