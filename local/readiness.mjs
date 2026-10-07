import { mkdir, readFile, writeFile, rm, readdir } from "node:fs/promises";
import { join } from "node:path";
import { randomUUID, createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { invokeAgent } from "./agents.mjs";
import { contract } from "./engine.mjs";
import { prepareEvaluation, evaluateCandidate } from "./evaluation.mjs";
import { cleanupContainers } from "./runtime.mjs";
import { validateTask } from "./validate.mjs";

export async function readinessKey(version, image) {
  const folder = fileURLToPath(new URL(".", import.meta.url));
  const hash = createHash("sha256").update(JSON.stringify([version, image, process.version, process.platform, process.arch]));
  for (const name of (await readdir(folder)).sort().filter(name => /\.(mjs|py)$/.test(name))) {
    hash.update(name).update(await readFile(join(folder, name)));
  }
  return hash.digest("hex");
}

// Exercise the real source-writing and isolated evaluation path using only synthetic data.
export async function checkAgentCapability(root, agent, { image, signal, onProgress = () => {} } = {}) {
  const id = randomUUID(), folder = join(root, id), workspace = join(folder, "source");
  const deadline = Date.now() + 180_000;
  const dataset = join(folder, "sample.csv");
  const task = validateTask({
    agent, dataset, target: "label", objective: "Classify the synthetic measurement as low or high using a DecisionTreeClassifier.",
    metric: "accuracy", minutes: 3, trials: 3, searchModels: false, model: "sklearn.tree.DecisionTreeClassifier",
    output: "py", exportModel: true, exportFormat: "joblib", validation: "holdout", folds: 3,
    seeds: [42], testFraction: 0, holdoutFraction: .25, splitStrategy: "independent",
    policy: { augmentation: false, regularization: false, features: false, tuning: false, pretrained: false, ensemble: false },
  });
  const run = { id, task, trials: [] };
  const context = { root, runId: id, workspace, deadline, image, runFile: join(folder, "run.json"),
    user: typeof process.getuid === "function" ? `${process.getuid()}:${process.getgid()}` : "65534:65534",
    data: join(folder, "data"), packages: join(folder, "packages"), models: join(folder, "models") };
  try {
    for (const path of [workspace, context.data, context.packages, context.models]) await mkdir(path, { recursive: true, mode: 0o700 });
    await writeFile(dataset, "measurement,label\n" + Array.from({ length: 80 }, (_, i) => `${i},${i < 40 ? "low" : "high"}`).join("\n"), { mode: 0o600 });
    await writeFile(context.runFile, JSON.stringify(run), { mode: 0o600 });
    const metadata = await prepareEvaluation(root, run, { signal, deadline });
    const publicMetadata = Object.fromEntries(["target", "metric", "taskType", "features", "classes", "assetColumns", "assetTypes", "classCounts", "fitsPerTrial"].map(key => [key, metadata[key]]));
    publicMetadata.schema = metadata.developmentSchema.map(({ name, type }) => ({ name, type }));
    await writeFile(join(context.data, "manifest.json"), JSON.stringify(publicMetadata), { mode: 0o600 });
    const contextFile = join(folder, "context.json");
    await writeFile(contextFile, JSON.stringify(context), { mode: 0o600 });
    let wroteSource = false, readSchema = false;
    onProgress("Checking agent code writing…");
    await invokeAgent(agent, workspace, contextFile, `${contract(run)}
This is a small Helix readiness check, not a research task. Do not browse or install packages.
Use Helix dataset_info to read the schema, then Helix write_source to write train.py and predict.py. Use the preinstalled pandas, scikit-learn and joblib. Fit a default DecisionTreeClassifier with random_state from --seed on the numeric measurement column. Save model.joblib and its model_manifest.json, and make predict.py reload that model without fitting. Do not write fixed predictions. Do not run code yourself; the Helix runner executes and independently validates the files after you finish.`, {
      signal, timeout: Math.max(1, deadline - Date.now() - 30_000),
      onEvent(type, message) {
        if (type !== "tool") return;
        if (message.includes("write_source")) wroteSource = true;
        if (message.includes("dataset_info")) readSchema = true;
      },
    });
    if (!readSchema || !wroteSource) throw new Error("The agent did not demonstrate access to Helix dataset_info and write_source tools.");
    for (const name of ["train.py", "predict.py"]) {
      if (!(await readFile(join(workspace, name), "utf8")).trim()) throw new Error(`The agent did not write ${name}.`);
    }
    onProgress("Checking CPU training and model reload…");
    const result = await evaluateCandidate(root, run, workspace, { signal, deadline, candidateId: "readiness", image });
    if (!Number.isFinite(result.score) || result.score < .8) throw new Error("The test experiment did not learn the simple synthetic task.");
    return { score: result.score, sourceHash: result.sourceHash, checks: ["mcp", "write", "train", "predict", "reload"] };
  } finally {
    await cleanupContainers(root, id).catch(() => {});
    await rm(folder, { recursive: true, force: true });
  }
}
