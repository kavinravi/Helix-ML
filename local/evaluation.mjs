import { mkdir, mkdtemp, readFile, writeFile, rm, readdir, lstat, rename } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { runProcess } from "./process.mjs";
import { container, IMAGE } from "./runtime.mjs";
import { scorePredictions } from "./metrics.mjs";
import { harness, verifyExport } from "./export.mjs";

const preparer = fileURLToPath(new URL("./prepare.py", import.meta.url));
const python = () => process.env.HELIX_PYTHON || "python3";
const json = async (path) => JSON.parse(await readFile(path, "utf8"));

export async function inspectDataset(task, { signal } = {}) {
  const temporary = await mkdtemp(join(tmpdir(), "helix-inspect-"));
  try {
    const config = join(temporary, "task.json");
    await writeFile(config, JSON.stringify(task), { mode: 0o600 });
    const result = await runProcess(python(), [preparer, "--task", config, "--inspect"], { timeout: 120_000, signal });
    return JSON.parse(result.output);
  } finally { await rm(temporary, { recursive: true, force: true }); }
}

export async function prepareEvaluation(root, run, { signal, deadline }) {
  const folder = join(root, run.id);
  const taskFile = join(folder, "task.json");
  await writeFile(taskFile, JSON.stringify(run.task), { mode: 0o600 });
  await runProcess(python(), [preparer, "--task", taskFile, "--output", join(folder, "evaluation")], {
    signal, timeout: Math.max(1, deadline - Date.now()),
  });
  return json(join(folder, "evaluation", "manifest.json"));
}

export async function sourceFingerprint(folder) {
  const hash = createHash("sha256");
  const files = [];
  async function visit(path, prefix = "") {
    for (const name of (await readdir(path)).sort()) {
      const stat = await lstat(join(path, name));
      if (stat.isSymbolicLink()) throw new Error("Candidate source cannot contain symlinks.");
      if (stat.isDirectory()) await visit(join(path, name), prefix + name + "/");
      else if (stat.isFile()) {
        if (stat.size > 10_000_000) throw new Error("Each candidate source file must be smaller than 10 MB.");
        const data = await readFile(join(path, name));
        files.push(prefix + name);
        hash.update(JSON.stringify([prefix + name, data.length])).update(data);
      } else throw new Error("Candidate source must contain only regular files.");
    }
  }
  await visit(folder);
  return { sha256: hash.digest("hex"), files };
}

export function aggregateScores(results, expected) {
  if (results.length !== expected || results.some((r) => !Number.isFinite(r.score)))
    throw new Error("Every requested fold and seed must finish before a candidate can be compared.");
  const score = results.reduce((sum, r) => sum + r.score / expected, 0);
  const deviation = Math.sqrt(results.reduce((sum, r) => sum + (r.score - score) ** 2 / expected, 0));
  return { score, deviation, foldScores: results.map((r) => r.score), evaluations: results };
}

export async function evaluateCandidate(root, run, workspace, { signal, deadline, candidateId, onEvent = () => {}, final = false, image, packages: packagePath }) {
  const folder = join(root, run.id), evaluation = join(folder, "evaluation");
  const protocol = await json(join(evaluation, "protocol.json"));
  const metadata = await json(join(evaluation, "manifest.json"));
  const source = await sourceFingerprint(workspace);
  if (!source.files.includes("train.py")) throw new Error("The candidate must provide train.py.");
  const context = {
    root, runId: run.id, deadline, image: image || IMAGE,
    user: typeof process.getuid === "function" ? `${process.getuid()}:${process.getgid()}` : "65534:65534",
  };
  const models = join(folder, "models"), packages = packagePath || join(folder, "packages");
  await mkdir(models, { recursive: true });
  await mkdir(packages, { recursive: true });
  const unscoredRefit = final && !protocol.test.length;
  const folds = final ? [{ id: "final", seed: protocol.seeds[0], fold: 1, train: protocol.development, validation: protocol.test.length ? protocol.test : protocol.evaluations[0].validation }] : protocol.evaluations;
  const results = [], baselineResults = [];
  let baselineName;
  for (const fold of folds) {
    if (signal?.aborted) throw new Error("Interrupted");
    if (Date.now() >= deadline) throw new Error("Time limit reached before the evaluation protocol finished.");
    const scratch = join(folder, "fits", candidateId, fold.id);
    await rm(scratch, { recursive: true, force: true });
    await mkdir(scratch, { recursive: true, mode: 0o700 });
    const data = join(scratch, "data"), output = join(scratch, "output"), truthPath = join(scratch, "truth.json");
    await mkdir(output);
    const started = Date.now();
    try {
      await runProcess(python(), [preparer, "--evaluation", evaluation, "--output", data, "--fold", fold.id, "--truth", truthPath], {
        signal, timeout: Math.max(1, deadline - Date.now()),
      });
      const config = {
        metric: run.task.metric, seed: fold.seed, policy: run.task.policy,
        model: run.task.model, searchModels: run.task.searchModels,
        exportModel: run.task.exportModel, exportFormat: run.task.exportFormat,
      };
      await writeFile(join(data, "config.json"), JSON.stringify(config), { mode: 0o600 });
      onEvent("training", `Evaluating ${fold.id}: ${fold.train.length} training rows, ${fold.validation.length} evaluation rows.`);
      const command = run.task.output === "ipynb" ? ["python", "/harness/notebook.py", String(fold.seed)] : ["python", "/code/train.py", "--train", "/data/train.csv", "--validation", "/data/validation.csv",
        "--metadata", "/data/manifest.json", "--models", "/models", "--output", "/work/predictions.json",
        "--seed", String(fold.seed), "--config", "/data/config.json"];
      await container(context, command, {
        signal, writable: output,
        mounts: [[workspace, "/code"], [data, "/data"], [models, "/models"], [packages, "/packages"], [output, "/work", "rw"], [harness, "/harness"]],
      });
      if ((await lstat(join(output, "predictions.json"))).size > 64_000_000) throw new Error("Predictions exceed the 64 MB limit.");
      const predictions = await json(join(output, "predictions.json"));
      if (run.task.exportModel) {
        // The reload process receives no training rows, labels, or in-memory model.
        await rm(join(data, "train.csv"));
        const format = await verifyExport(context, workspace, data, output, models, packages, predictions, run.task, signal);
        onEvent("export", `${fold.id}: ${format} export reloaded; predictions match.`);
      }
      const { targets: truth, baseline } = await json(truthPath);
      baselineName = baseline.name;
      const baselineScore = scorePredictions(run.task.metric, truth, Array(truth.length).fill(baseline.prediction), metadata.classes);
      baselineResults.push({ score: baselineScore });
      // Validate prediction shape/values even for an unscored refit. Never report its in-sample score.
      const checkedScore = scorePredictions(run.task.metric, truth, predictions, metadata.classes);
      const score = unscoredRefit ? null : checkedScore;
      const result = { id: fold.id, seed: fold.seed, fold: fold.fold, trainingRows: fold.train.length,
        validationRows: fold.validation.length, score, baselineScore: unscoredRefit ? null : baselineScore, duration: (Date.now() - started) / 1000 };
      results.push(result);
      await writeFile(join(scratch, "result.json"), JSON.stringify(result), { mode: 0o600 });
      onEvent("evaluation", score === null ? "Final refit complete. No test score requested." : `${fold.id} scored ${score.toFixed(6)}.`);
    } finally {
      // Each fit sees only its own rows, and no data or output from earlier fits.
      await rm(data, { recursive: true, force: true });
      await rm(truthPath, { force: true });
    }
  }
  if ((await sourceFingerprint(workspace)).sha256 !== source.sha256)
    throw new Error("Candidate source changed during evaluation; its scores cannot be accepted.");
  const baseline = unscoredRefit ? null : { name: baselineName, ...aggregateScores(baselineResults, folds.length) };
  return { ...(unscoredRefit ? { score: null, deviation: null, foldScores: [], evaluations: results } : aggregateScores(results, folds.length)), baseline, protocolHash: protocol.sha256, sourceHash: source.sha256 };
}

export async function evaluateFinal(root, run, workspace, options) {
  const file = join(root, run.id, "final-test.json");
  const source = await sourceFingerprint(workspace);
  const protocol = await json(join(root, run.id, "evaluation", "protocol.json"));
  const frozen = { sourceHash: source.sha256, protocolHash: protocol.sha256 };
  try {
    await writeFile(file, JSON.stringify({ ...frozen, status: "started" }), { flag: "wx", mode: 0o600 });
  } catch (error) {
    if (error.code !== "EEXIST") throw error;
    const previous = await json(file);
    if (previous.sourceHash !== frozen.sourceHash || previous.protocolHash !== frozen.protocolHash)
      throw new Error("Final test selection is frozen. Its source and protocol cannot change.");
    if (previous.status !== "completed")
      throw new Error("Final evaluation was interrupted or failed. It cannot be repeated in this run.");
    return previous.result;
  }
  try {
    const result = await evaluateCandidate(root, run, workspace, { ...options, candidateId: "selected", final: true });
    if (result.sourceHash !== frozen.sourceHash || result.protocolHash !== frozen.protocolHash)
      throw new Error("The frozen final-test source or protocol changed during evaluation.");
    await writeFile(file + ".tmp", JSON.stringify({ ...frozen, status: "completed", result }), { mode: 0o600 });
    await rename(file + ".tmp", file);
    return result;
  } catch (error) {
    await writeFile(file + ".tmp", JSON.stringify({ ...frozen, status: "failed" }), { mode: 0o600 });
    await rename(file + ".tmp", file);
    throw error;
  }
}
