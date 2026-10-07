import { mkdir, readFile, writeFile, cp, rm, lstat, readdir } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { container, directoryBytes } from "./runtime.mjs";
import { inside } from "./validate.mjs";
import { runProcess } from "./process.mjs";

export const harness = fileURLToPath(new URL(".", import.meta.url));

export async function makeNotebook(workspace, seed = 42) {
  const cells = [
    { id: "overview", cell_type: "markdown", metadata: {}, source: ["# Helix ML solution\n", "Run alongside the bundled Python source. Set the paths below to your labeled training CSV, unlabeled evaluation CSV, metadata and configuration. Learned preprocessing is fitted on training rows only. See README.md for the recorded environment and prediction contract."] },
    { id: "reproduce", cell_type: "code", metadata: {}, execution_count: null, outputs: [], source: [
      "import os, sys, subprocess\n",
      "from pathlib import Path\n",
      "source = Path(os.environ.get('HELIX_SOURCE', '.')).resolve()\n",
      "data = Path(os.environ.get('HELIX_DATA', '../data')).resolve()\n",
      "output = Path(os.environ.get('HELIX_OUTPUT', '../output')).resolve()\n",
      "models = Path(os.environ.get('HELIX_MODELS', '../pretrained')).resolve()\n",
      "output.mkdir(parents=True, exist_ok=True)\n",
      `seed = os.environ.get('HELIX_SEED', '${seed}')\n`,
      "subprocess.run([sys.executable, str(source / 'train.py'), '--train', str(data / 'train.csv'), '--validation', str(data / 'validation.csv'), '--metadata', str(data / 'manifest.json'), '--config', str(data / 'config.json'), '--models', str(models), '--output', str(output / 'predictions.json'), '--seed', seed], check=True)\n",
    ] },
  ];
  await writeFile(join(workspace, "solution.ipynb"), JSON.stringify({ cells, metadata: { kernelspec: { display_name: "Python 3", language: "python", name: "python3" } }, nbformat: 4, nbformat_minor: 5 }, null, 2));
}

export function matchingPredictions(expected, actual) {
  if (Array.isArray(expected)) return Array.isArray(actual) && expected.length === actual.length && expected.every((v, i) => matchingPredictions(v, actual[i]));
  return typeof expected === "number" ? Number.isFinite(actual) && Math.abs(expected - actual) <= 1e-6 * Math.max(1, Math.abs(expected)) : expected === actual;
}

export async function verifyExport(context, workspace, data, output, models, packages, predictions, task, signal) {
  const model = join(output, "model");
  await directoryBytes(model); // Reject symlinks and special files, including undeclared files.
  const manifest = JSON.parse(await readFile(await inside(model, "model_manifest.json"), "utf8"));
  const extensions = { joblib: ".joblib", pickle: ".pkl", pytorch: ".pt", torchscript: ".pt", keras: ".keras", savedmodel: "saved_model.pb", onnx: ".onnx" };
  if (!extensions[manifest.format] || (task.exportFormat !== "native" && manifest.format !== task.exportFormat)) throw new Error("Export format does not match the requested format.");
  if (!Array.isArray(manifest.files) || !manifest.files.length || manifest.files.length > 1000) throw new Error("model_manifest.json must list the serialized model files.");
  for (const name of manifest.files) {
    const file = await inside(model, name);
    if (!(await lstat(file)).size) throw new Error("Model artifacts must not be empty.");
  }
  if (!manifest.files.some((file) => file.endsWith(extensions[manifest.format]))) throw new Error("No model file has the selected format's extension.");
  await container(context, ["python", "/harness/verify_model.py", "/model/model_manifest.json"], {
    signal, mounts: [[harness, "/harness"], [workspace, "/code"], [model, "/model"], [packages, "/packages"]],
  });
  await inside(workspace, "predict.py");
  const reloaded = join(output, "reload");
  await mkdir(reloaded);
  await container(context, ["python", "/code/predict.py", "--input", "/data/validation.csv", "--metadata", "/data/manifest.json", "--config", "/data/config.json", "--model-dir", "/model", "--models", "/models", "--output", "/work/predictions.json"], {
    signal, writable: reloaded,
    mounts: [[workspace, "/code"], [data, "/data"], [model, "/model"], [models, "/models"], [packages, "/packages"], [reloaded, "/work", "rw"]],
  });
  const actualFile = join(reloaded, "predictions.json");
  if ((await lstat(actualFile)).size > 64_000_000) throw new Error("Reload predictions exceed the size limit.");
  if (!matchingPredictions(predictions, JSON.parse(await readFile(actualFile, "utf8")))) throw new Error("Reloaded model predictions differ from the trained model. Serialize all learned preprocessing and preserve class order.");
  await rm(reloaded, { recursive: true });
  return manifest.format;
}

export async function finishBundle(root, run, context, signal) {
  const folder = join(root, run.id), final = join(folder, "final");
  if (run.task.exportModel) {
    await directoryBytes(join(folder, "fits", "selected", "final", "output", "model"));
    await cp(join(folder, "fits", "selected", "final", "output", "model"), join(final, "model"), { recursive: true });
  }
  const packages = join(folder, "dependencies", run.best);
  const versions = await container(context, ["python", "-c", "import importlib.metadata as m; found={}; [(found.setdefault(d.metadata['Name'].lower(), d.version)) for d in m.distributions() if d.metadata['Name']]; print('\\n'.join(k+'=='+v for k,v in sorted(found.items())))"], {
    signal, mounts: [[packages, "/packages"]],
  });
  await writeFile(join(final, "requirements.lock"), versions.output);
  await cp(join(folder, "evaluation", "protocol.json"), join(final, "protocol.json"));
  await cp(join(folder, "data", "manifest.json"), join(final, "manifest.json"));
  await writeFile(join(final, "config.json"), JSON.stringify({ metric: run.task.metric, seed: run.task.seeds[0], policy: run.task.policy, model: run.task.model, searchModels: run.task.searchModels, exportModel: run.task.exportModel, exportFormat: run.task.exportFormat }, null, 2));
  const records = [];
  for (const name of await readdir(context.models)) if (name.endsWith(".json")) records.push(JSON.parse(await readFile(join(context.models, name), "utf8")));
  await writeFile(join(final, "pretrained.json"), JSON.stringify(records, null, 2));
  await writeFile(join(final, "experiment.json"), JSON.stringify({ task: { ...run.task, dataset: "<your local dataset>" }, constraints: run.constraints, protocolHash: run.protocolHash, environment: run.environment, trials: run.trials, validationScore: run.score, testScore: run.testScore, baseline: run.baseline, testBaseline: run.testBaseline, exportVerified: run.task.exportModel, reloadTolerance: 1e-6, selected: run.best, followup: run.followup, trainingHistory: run.trainingHistory }, null, 2));
  const baseline = run.baseline ? `\nValidation baseline (${run.baseline.name}): ${run.baseline.score}. ${run.testBaseline ? `Test baseline: ${run.testBaseline.score}.` : ""} Each baseline uses only its own training labels and the same evaluation rows as the model.\n` : "";
  const inference = run.task.exportModel ? "To predict on an unlabeled CSV (including referenced assets relative to the CSV):\n\n```sh\npython source/predict.py --input data/validation.csv --metadata manifest.json --config config.json --model-dir model --models pretrained --output predictions.json\n```\n\nThe serialized model and preprocessing passed a fresh-process reload comparison on the evaluation inputs. Load artifacts only from runs you trust.\n" : "Model export was disabled. Use train.py to fit on your labeled training rows and predict on an unlabeled CSV in one invocation.\n";
  await writeFile(join(final, "README.md"), `# Helix ML experiment\n\nValidation ${run.task.metric}: ${run.score}. ${run.testScore == null ? run.followup ? "Follow-up run: the original test partition was not evaluated again." : "No test partition was requested." : `Reserved test: ${run.testScore}.`}\n${baseline}\nModel selection used validation scores only. The final model was fitted on ${run.task.testFraction === 0 ? "all" : "training"} rows using seed ${run.task.seeds[0]}. protocol.json records exact row indices and seeds against the original CSV order. Raw data is not included.\n\n## Reproduce\n\nUse Python 3.12 in a fresh virtual environment and install \`pip install -r requirements.lock\`. The recorded Docker image ID is in experiment.json; platform differences can affect numerical results. Restore any pretrained revisions listed in pretrained.json into a local pretrained folder.\n\n${inference}\nTo refit on your own split, place train.csv (labeled) and validation.csv (unlabeled) with assets in data/, then run:\n\n\`\`\`sh\npython source/train.py --train data/train.csv --validation data/validation.csv --metadata manifest.json --config config.json --models pretrained --output output/predictions.json --seed ${run.task.seeds[0]}\n\`\`\`\n\nCreate output/ first. For notebook output, copy manifest.json and config.json into data/ and open source/solution.ipynb. It runs the same checked Python source; edit paths in its first code cell. Scores cannot be reproduced without the original dataset and recorded partitions. Source strategy checks are advisory syntax checks plus agent review, not a formal proof of policy compliance.\n`);
  await runProcess(process.env.HELIX_PYTHON || "python3", [join(harness, "bundle.py"), final], { signal, timeout: Math.max(1, context.deadline - Date.now()) });
}
