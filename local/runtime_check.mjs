// Opt-in check: real CPU fits in Docker; never calls an agent or downloads data.
import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, readFile, rm, readdir, cp } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";
import { prepareEvaluation, evaluateCandidate, evaluateFinal, aggregateScores } from "./evaluation.mjs";
import { cleanupContainers, container, IMAGE, runtimeMemoryMb } from "./runtime.mjs";
import { runProcess } from "./process.mjs";
import { makeNotebook, finishBundle } from "./export.mjs";
import { scorePredictions } from "./metrics.mjs";

const root = await mkdtemp(join(tmpdir(), "helix-runtime-check-"));
const controller = new AbortController();
const deadline = Date.now() + 180_000;
const source = `import argparse, csv, json, os
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
p = argparse.ArgumentParser()
for flag in ('train', 'validation', 'metadata', 'models', 'output', 'seed', 'config'):
    p.add_argument('--' + flag, required=True)
a = p.parse_args()
assert not Path('/data/truth.json').exists()
assert not Path('/evaluation').exists()
assert not Path('/data/dataset.csv').exists()
with open(a.train) as f: train = list(csv.DictReader(f))
with open(a.validation) as f: validation = list(csv.DictReader(f))
assert all('label' not in row for row in validation)
def features(rows):
    result = []
    for row in rows:
        values = [float(row['x'])]
        if 'asset' in row:
            path = Path(a.train).parent / row['asset']
            if path.suffix == '.ppm':
                from PIL import Image
                values.append(float(np.asarray(Image.open(path)).mean()))
            elif path.suffix == '.wav':
                from scipy.io import wavfile
                _, samples = wavfile.read(path)
                values.append(float(np.abs(samples.astype(float)).mean()))
            else:
                values.append(float('positive' in path.read_text()))
        result.append(values)
    return result
model = make_pipeline(SimpleImputer(), DecisionTreeClassifier(random_state=int(a.seed), max_depth=3))
model.fit(features(train), [row['label'] for row in train])
Path(a.output).write_text(json.dumps(model.predict(features(validation)).tolist()))
if json.loads(Path(a.config).read_text()).get('exportModel'):
    import joblib
    output = Path(a.output).parent / 'model'
    output.mkdir()
    joblib.dump(model, output / 'model.joblib')
    (output / 'model_manifest.json').write_text(json.dumps({'format': 'joblib', 'files': ['model.joblib']}))
`;
try {
  assert.throws(() => aggregateScores([{ score: 1 }], 2), /Every requested/);
  const { output: image } = await runProcess("docker", ["image", "inspect", IMAGE, "--format", "{{.Id}}"]);
  for (const modality of ["tabular", "text", "image", "audio"]) {
    // Exercise preparation, every fold, final fitting and export without a time cap.
    const deadline = modality === "tabular" ? null : Date.now() + 180_000;
    const run = { id: randomUUID(), task: { dataset: join(root, modality), target: "label", metric: "accuracy", validation: "cv", folds: 3,
      seeds: [42, 17], splitStrategy: "independent", assetColumns: modality === "tabular" ? [] : ["asset"],
      searchModels: false, model: "decision tree", exportModel: modality === "tabular", exportFormat: "joblib", output: modality === "tabular" ? "ipynb" : "py", policy: { features: true, augmentation: false, pretrained: false, ensemble: false, tuning: false, regularization: false } } };
    await mkdir(run.task.dataset);
    await mkdir(join(root, run.id));
    const lines = ["x,label" + (modality === "tabular" ? "" : ",asset")];
    for (let i = 0; i < 60; i++) {
      const label = i >= 30 ? "yes" : "no";
      let asset = "";
      if (modality !== "tabular") {
        asset = `${i}.${modality === "image" ? "ppm" : modality === "audio" ? "wav" : "txt"}`;
        let content;
        if (modality === "text") content = i >= 30 ? `positive example ${i}` : `negative example ${i}`;
        if (modality === "image") content = `P3\n1 1\n255\n${i * 4} ${i * 4} ${i * 4}\n`;
        if (modality === "audio") {
          content = Buffer.alloc(46);
          content.write("RIFF", 0); content.writeUInt32LE(38, 4); content.write("WAVEfmt ", 8);
          content.writeUInt32LE(16, 16); content.writeUInt16LE(1, 20); content.writeUInt16LE(1, 22);
          content.writeUInt32LE(8000, 24); content.writeUInt32LE(16000, 28); content.writeUInt16LE(2, 32);
          content.writeUInt16LE(16, 34); content.write("data", 36); content.writeUInt32LE(2, 40); content.writeInt16LE(i * 100, 44);
        }
        await writeFile(join(run.task.dataset, asset), content);
      }
      lines.push(`${i},${label}${asset ? "," + asset : ""}`);
    }
    await writeFile(join(run.task.dataset, "train.csv"), lines.join("\n"));
    const workspace = join(root, run.id, "candidate");
    await mkdir(workspace);
    await writeFile(join(workspace, "train.py"), source);
    if (run.task.output === "ipynb") await makeNotebook(workspace);
    const predictor = `import argparse, csv, json, joblib
from pathlib import Path
p = argparse.ArgumentParser()
for flag in ('input', 'metadata', 'config', 'model-dir', 'models', 'output'): p.add_argument('--' + flag, required=True)
a = p.parse_args()
assert not Path('/data/train.csv').exists()
with open(a.input) as f: rows = list(csv.DictReader(f))
model = joblib.load(Path(a.model_dir) / 'model.joblib')
Path(a.output).write_text(json.dumps(model.predict([[float(r['x'])] for r in rows]).tolist()))
`;
    await writeFile(join(workspace, "predict.py"), predictor);
    const manifest = await prepareEvaluation(root, run, { signal: controller.signal, deadline });
    assert.equal(manifest.fitsPerTrial, 6);
    const result = await evaluateCandidate(root, run, workspace, { candidateId: "baseline", deadline, signal: controller.signal, image: image.trim() });
    assert.equal(result.evaluations.length, 6);
    assert.ok(result.score >= .8 && result.score <= 1);
    assert.ok(result.score > result.baseline.score);
    assert.equal(result.baseline.foldScores.length, 6);
    assert.equal(result.baseline.name, "Most frequent class");
    assert.equal(result.sourceHash.length, 64);
    for (const fold of result.evaluations) {
      const contents = await readdir(join(root, run.id, "fits", "baseline", fold.id));
      assert.ok(!contents.includes("data") && !contents.includes("truth.json"));
    }
    console.log(`${modality}: ${result.evaluations.length} isolated CPU fits, mean accuracy ${result.score.toFixed(4)}`);
    if (modality === "text") {
      run.best = "baseline"; run.score = result.score; run.trials = [];
      run.testScore = (await evaluateFinal(root, run, workspace, { deadline, signal: controller.signal, image: image.trim() })).score;
      await mkdir(join(root, run.id, "data"));
      await cp(join(root, run.id, "evaluation", "manifest.json"), join(root, run.id, "data", "manifest.json"));
      await mkdir(join(root, run.id, "dependencies", "baseline"), { recursive: true });
      await mkdir(join(root, run.id, "final", "source"), { recursive: true });
      await cp(workspace, join(root, run.id, "final", "source"), { recursive: true });
      await finishBundle(root, run, { root, runId: run.id, deadline, image: image.trim(), user: `${process.getuid()}:${process.getgid()}`, models: join(root, run.id, "models") }, controller.signal);
      const finalFiles = await readdir(join(root, run.id, "final"));
      assert.ok(finalFiles.includes("helix-solution.zip") && !finalFiles.includes("model"));
      const report = JSON.parse(await readFile(join(root, run.id, "final", "experiment.json"), "utf8"));
      assert.equal(report.exportVerified, false);
      assert.ok(!(await readFile(join(root, run.id, "final", "README.md"), "utf8")).includes("python source/predict.py"));
    }
    if (modality === "tabular") {
      const final = await evaluateFinal(root, run, workspace, { deadline, signal: controller.signal, image: image.trim() });
      assert.equal(final.evaluations.length, 1);
      assert.equal(final.evaluations[0].trainingRows, 48);
      assert.equal(final.evaluations[0].validationRows, 12);
      assert.deepEqual(await evaluateFinal(root, run, workspace, { deadline, signal: controller.signal, image: image.trim() }), final);
      const followup = { id: randomUUID(), task: run.task, followup: { parentId: run.id } };
      await mkdir(join(root, followup.id));
      const continuedSource = join(root, followup.id, "candidate");
      await cp(workspace, continuedSource, { recursive: true });
      // Real iterative training: telemetry is measured, never fabricated for the chart.
      const iterative = source.replace("from sklearn.tree import DecisionTreeClassifier", "from sklearn.linear_model import SGDClassifier\nfrom sklearn.metrics import log_loss, accuracy_score")
        .replace("model = make_pipeline(SimpleImputer(), DecisionTreeClassifier(random_state=int(a.seed), max_depth=3))", "model = SGDClassifier(loss='log_loss', random_state=int(a.seed))")
        .replace("model.fit(features(train), [row['label'] for row in train])", `history = []
labels = [row['label'] for row in train]
for epoch in range(5):
    model.partial_fit(features(train), labels, classes=np.unique(labels))
    history.append({'step': epoch + 1, 'loss': log_loss(labels, model.predict_proba(features(train)), labels=model.classes_), 'accuracy': accuracy_score(labels, model.predict(features(train)))})
(Path(a.output).parent / 'training_history.json').write_text(json.dumps(history))`);
      await writeFile(join(continuedSource, "train.py"), iterative);
      const continuedManifest = await prepareEvaluation(root, followup, { deadline, signal: controller.signal });
      assert.equal(continuedManifest.protocolHash, manifest.protocolHash);
      const continued = await evaluateFinal(root, followup, continuedSource, { deadline, signal: controller.signal, image: image.trim() });
      assert.equal(continued.score, null, "Follow-ups must not reuse the held-out test score");
      assert.equal(continued.baseline, null);
      assert.equal(continued.evaluations[0].trainingRows, 48);
      assert.equal(continued.evaluations[0].validationRows, 16, "Refit checks reload on development rows, not the 12 test rows");
      assert.equal(continued.trainingHistory.length, 5);
      assert.ok(continued.trainingHistory.every(point => Number.isFinite(point.loss) && point.accuracy >= 0 && point.accuracy <= 1));
      assert.deepEqual(await evaluateFinal(root, run, workspace, { deadline, signal: controller.signal, image: image.trim() }), final, "The parent's frozen result is unchanged");
      console.log("follow-up: development-only refit, verified model reload, five measured loss/accuracy points; original test result unchanged");
      const refit = { id: randomUUID(), task: { ...run.task, testFraction: 0, holdoutFraction: .3, validation: "holdout", seeds: [42], output: "py" } };
      await mkdir(join(root, refit.id));
      const refitSource = join(root, refit.id, "candidate");
      await cp(workspace, refitSource, { recursive: true });
      const refitManifest = await prepareEvaluation(root, refit, { deadline, signal: controller.signal });
      assert.equal(refitManifest.testRows, 0);
      const refitResult = await evaluateFinal(root, refit, refitSource, { deadline, signal: controller.signal, image: image.trim() });
      assert.equal(refitResult.score, null);
      assert.equal(refitResult.baseline, null);
      assert.equal(refitResult.evaluations[0].trainingRows, 60);
      assert.equal(refitResult.evaluations[0].validationRows, 18);
      assert.equal(refitResult.evaluations[0].score, null);
      assert.deepEqual(await evaluateFinal(root, refit, refitSource, { deadline, signal: controller.signal, image: image.trim() }), refitResult);
      console.log("0% test: refit on all 60 rows, verified export, no fabricated test score.");
      await writeFile(join(workspace, "predict.py"), predictor.replace("model.predict([[float(r['x'])] for r in rows]).tolist()", "['no'] * len(rows)"));
      await assert.rejects(evaluateCandidate(root, run, workspace, { candidateId: "bad-reload", deadline, signal: controller.signal, image: image.trim() }), /Reloaded model predictions differ/);
      await writeFile(join(workspace, "predict.py"), predictor);
      await writeFile(join(workspace, "train.py"), source.replace("model.fit(features(train)", "raise RuntimeError('intentional fit failure')\nmodel.fit(features(train)"));
      await assert.rejects(evaluateFinal(root, run, workspace, { deadline, signal: controller.signal }), /frozen/);
      await assert.rejects(evaluateCandidate(root, run, workspace, { candidateId: "failure", deadline, signal: controller.signal, image: image.trim() }), /intentional fit failure/);
    }
  }
  const context = { root, runId: randomUUID(), deadline, user: `${process.getuid()}:${process.getgid()}` };
  const reference = JSON.parse((await container(context, ["python", "-c", "import json; from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, root_mean_squared_error, mean_absolute_error; print(json.dumps([accuracy_score([0,1,0,1],[0,1,1,1]),roc_auc_score([0,1,0,1],[.1,.5,.5,.9]),log_loss([0,1],[[0.,1.],[0.,1.]],labels=[0,1]),root_mean_squared_error([1,3],[1,5]),mean_absolute_error([1,3],[1,5])]))"])).output);
  const measured = [scorePredictions("accuracy", [0,1,0,1], [0,1,1,1], [0,1]), scorePredictions("auroc", [0,1,0,1], [.1,.5,.5,.9], [0,1]), scorePredictions("log_loss", [0,1], [[0,1],[0,1]], [0,1]), scorePredictions("rmse", [1,3], [1,5]), scorePredictions("mae", [1,3], [1,5])];
  reference.forEach((value, i) => assert.ok(Math.abs(value - measured[i]) < 1e-12, `Metric ${i} differs from scikit-learn`));
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), 1000);
  try {
    await assert.rejects(container({ ...context, deadline: null }, ["python", "-c", "import time; time.sleep(60)"], { signal: abort.signal }), /Interrupted/);
  } finally { clearTimeout(timer); }
  const previousMemory = process.env.HELIX_TRAIN_MEMORY_MB;
  try {
    process.env.HELIX_TRAIN_MEMORY_MB = "invalid";
    assert.throws(runtimeMemoryMb, /integer from 128/);
    process.env.HELIX_TRAIN_MEMORY_MB = "128";
    assert.equal(runtimeMemoryMb(), 128);
    await assert.rejects(container({ ...context, deadline: null }, ["python", "-c", "x = bytearray(512 * 1024**2)" ]), /128 MiB RAM limit/);
  } finally {
    if (previousMemory === undefined) delete process.env.HELIX_TRAIN_MEMORY_MB;
    else process.env.HELIX_TRAIN_MEMORY_MB = previousMemory;
  }
  const remaining = await runProcess("docker", ["ps", "-aq", "--filter", `label=helix.run=${context.runId}`]);
  assert.equal(remaining.output.trim(), "");
  console.log("Metrics match scikit-learn; final test reused; changed source and failed candidates rejected; interruption and out-of-memory failure cleaned up.");
} finally {
  controller.abort();
  await cleanupContainers(root).catch(() => {});
  await rm(root, { recursive: true, force: true });
}
