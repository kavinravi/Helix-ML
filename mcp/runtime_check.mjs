// Real stdio requests, local fits, independent scoring, cancellation and exported artifacts.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { mkdtemp, mkdir, writeFile, readFile, rm, access } from "node:fs/promises";
import { once } from "node:events";
import { tmpdir, homedir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";
import { setTimeout as delay } from "node:timers/promises";
import { runProcess } from "../local/process.mjs";
import { runtimeConfig } from "./runtime.mjs";
import { container } from "../local/runtime.mjs";

const home = resolve(process.env.HELIX_MCP_HOME || join(homedir(), ".helix-ml", "mcp"));
const runtime = await runtimeConfig(home);
const temp = await mkdtemp(join(tmpdir(), "helix mcp check,"));
const packagedExample = join(home, "app", "test-" + randomUUID(), "examples", "data.csv");
const entry = fileURLToPath(new URL("cli.mjs", import.meta.url));
let child, sequence = 0, pending, errors;
function connect() {
  pending = new Map(); errors = "";
  child = spawn(process.execPath, [entry, "serve", "--home", home], { stdio: ["pipe", "pipe", "pipe"] });
  child.stderr.on("data", chunk => { errors += chunk; });
  createInterface({ input: child.stdout }).on("line", line => {
    const message = JSON.parse(line);
    const callback = pending.get(message.id);
    if (callback) { pending.delete(message.id); callback.resolve(message); }
  });
  child.on("exit", code => { for (const callback of pending.values()) callback.reject(new Error(`MCP exited ${code}: ${errors}`)); });
}
async function rpc(method, params = {}) {
  const id = ++sequence;
  const result = await Promise.race([
    new Promise((resolve, reject) => { pending.set(id, { resolve, reject }); child.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n"); }),
    delay(15_000, null, { ref: false }).then(() => { throw new Error(`MCP request timed out: ${method}`); }),
  ]);
  assert.ok(!result.error, JSON.stringify(result));
  return result.result;
}
async function call(name, args = {}) {
  const result = await rpc("tools/call", { name: "helix_" + name, arguments: args });
  if (result.isError) throw new Error(result.content[0].text);
  return JSON.parse(result.content[0].text);
}
async function settled(runId) {
  for (let i = 0; i < 600; i++) {
    const status = await call("status", { runId });
    if (status.status !== "running") return status;
    await delay(200);
  }
  throw new Error("Training exceeded the test timeout.");
}
async function disconnect() { const exited = once(child, "exit"); child.stdin.end(); await exited; }

const train = `import argparse, json, joblib, os
from pathlib import Path
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
p=argparse.ArgumentParser()
for key in ('train','validation','metadata','models','output','seed','config'): p.add_argument('--'+key)
a=p.parse_args(); meta=json.loads(Path(a.metadata).read_text()); c=json.loads(Path(a.config).read_text())
assert not os.environ.get('HELIX_SECRET_TEST')
frame=pd.read_csv(a.train, dtype=str); valid=pd.read_csv(a.validation, dtype=str)
assert not meta['target'] or meta['target'] not in valid.columns
if c['learning']=='supervised':
    x=frame[meta['features']].astype(float); v=valid[meta['features']].astype(float)
    clf=meta['taskType']=='classification'
    model=(DecisionTreeClassifier(random_state=int(a.seed)) if clf else DecisionTreeRegressor(random_state=int(a.seed)))
    y=frame[meta['target']] if clf else frame[meta['target']].astype(float)
    model.fit(x,y)
    pred=model.predict_proba(v)[:,[list(model.classes_).index(k) for k in meta['classes']]] if clf else model.predict(v)
else:
    from helix_features import preprocessor, read_features, dense_features
    x=read_features(a.train,meta['features']); v=read_features(a.validation,meta['features'])
    model=make_pipeline(preprocessor(x),KMeans(n_clusters=3,n_init=2,random_state=int(a.seed))) if c['learning']=='clustering' else make_pipeline(preprocessor(x),FunctionTransformer(dense_features),PCA(n_components=c['varianceTarget'] if c['reductionMode']=='variance' else c['dimensions'],svd_solver='full'))
    model.fit(x); pred=model.predict(v) if c['learning']=='clustering' else model.transform(v)
Path(a.output).write_text(json.dumps(pred.tolist()))
if c['exportModel']:
    modeldir=Path(a.output).parent/'model'; modeldir.mkdir()
    joblib.dump(model,modeldir/'model.joblib')
    (modeldir/'model_manifest.json').write_text(json.dumps({'format':'joblib','files':['model.joblib']}))
`;
const predict = `import argparse,json,joblib
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser()
for key in ('input','metadata','config','model-dir','models','output'): p.add_argument('--'+key)
a=p.parse_args(); meta=json.loads(Path(a.metadata).read_text()); c=json.loads(Path(a.config).read_text())
assert not (Path(a.input).parent/'train.csv').exists()
model=joblib.load(Path(a.model_dir)/'model.joblib')
if c['learning']=='supervised':
    v=pd.read_csv(a.input)[meta['features']]
    pred=model.predict_proba(v)[:,[list(model.classes_).index(k) for k in meta['classes']]] if meta['taskType']=='classification' else model.predict(v)
else:
    from helix_features import read_features
    v=read_features(a.input,meta['features'])
    pred=model.predict(v) if c['learning']=='clustering' else model.transform(v)
Path(a.output).write_text(json.dumps(pred.tolist()))
`;
const ids = [];
try {
  const hooks = join(temp, "wheel-hooks");
  await mkdir(hooks);
  await writeFile(join(hooks, "startup.pth"), "import os; os.environ['HELIX_WHEEL_HOOK']='active'\n");
  const hookContext = {runtime: {...runtime,device:"cpu"},deadline:Date.now()+30000};
  assert.equal((await container(hookContext,["python","-c","import os,sys; assert sys.argv[1]=='argument with spaces'; print(os.environ['HELIX_WHEEL_HOOK'])","argument with spaces"],{mounts:[[hooks,"/packages"]]})).output.trim(),"active");
  assert.equal((await container(hookContext,["python","-c","import os; print(os.environ.get('HELIX_WHEEL_HOOK','absent'))"])).output.trim(),"absent");
  await writeFile(join(hooks,"hook_probe.py"),"import os; print(os.environ['HELIX_WHEEL_HOOK'])\n");
  assert.equal((await container(hookContext,["python","-m","hook_probe"],{mounts:[[hooks,"/packages"]]})).output.trim(),"active");
  const source = join(temp, "source"), dataset = join(temp, "data.csv");
  await mkdir(source);
  await writeFile(dataset, "a,b,c,d,label,y\n" + Array.from({ length: 150 }, (_, i) => `${i%3*8+Math.sin(i)},${i%3*6+Math.cos(i)},${Math.sin(i*3)},${i%3*5+Math.cos(i*2)},${i%2?'yes':'no'},${i%3*4+Math.sin(i)}`).join("\n"));
  await mkdir(join(packagedExample, ".."), { recursive: true });
  await writeFile(packagedExample, await readFile(dataset));
  await writeFile(join(source, "train.py"), train); await writeFile(join(source, "predict.py"), predict);
  await writeFile(join(source, "plan.json"), JSON.stringify({ component: "baseline", change: "fit one estimator", rationale: "integration test", modelFamily: "test fixture", strategies: { augmentation: false, regularization: false, features: false, tuning: false, pretrained: false, ensemble: false } }));
  await runProcess(runtime.python, [fileURLToPath(new URL('../local/metrics_check.py', import.meta.url))], { timeout: 120_000 });
  console.log('All 61 metric definitions passed.');
  process.env.HELIX_SECRET_TEST = "not-for-training";
  connect();
  assert.equal((await rpc("initialize", { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "helix-test", version: "1" } })).serverInfo.name, "helix-ml-local");
  assert.ok((await rpc("tools/list")).tools.length >= 8);
  assert.ok((await call("info")).metrics.f1);
  await assert.rejects(call("status", { runId: "../../escape" }), /Invalid experiment ID/);
  const scenarios = [
    { metric: "auroc", metrics: ["auroc", "f1"], target: "label", excludedColumns: ["y"] },
    { metric: "rmse", target: "y", excludedColumns: ["label"], validation: "cv", folds: 3 },
    { learning: "clustering", metric: "davies_bouldin", excludedColumns: ["label", "y"] },
    { learning: "reduction", metric: "trustworthiness", dimensions: 2, excludedColumns: ["label", "y"], output: "ipynb" },
    { learning: "reduction", metric: "continuity", reductionMode: "variance", varianceTarget: .9, model: "PCA", excludedColumns: ["label", "y"] },
  ];
  for (const scenario of scenarios) {
    // AST restrictions apply to all source branches, so omit model-specific unused branches for PCA.
    if (scenario.model === "PCA") await writeFile(join(source, "train.py"), train.replace("KMeans(n_clusters=3,n_init=2,random_state=int(a.seed))", "PCA(n_components=2)").replace("DecisionTreeClassifier(random_state=int(a.seed))", "PCA(n_components=2)").replace("DecisionTreeRegressor(random_state=int(a.seed))", "PCA(n_components=2)"));
    const { runId } = await call("create_experiment", { dataset: scenario.metrics ? packagedExample : dataset, objective: `Test ${scenario.metric}`, minutes: 10, trials: 2, ...scenario }); ids.push(runId);
    let state = await settled(runId); assert.equal(state.status, "ready", state.error);
    state = await call("status", { runId, details: true });
    assert.match(state.contract, /trusted native/);
    assert.equal(state.dataset.sourceSha256, undefined); assert.equal(state.dataset.assets, undefined);
    const before = Date.now(); await call("run_trial", { runId, sourceDir: source, device: "cpu" }); assert.ok(Date.now()-before < 5000);
    state = await settled(runId); assert.equal(state.status, "ready", state.error); assert.ok(Number.isFinite(state.validationScore));
    if (scenario.metrics) { assert.ok(Number.isFinite(state.metricScores.f1)); assert.ok(Number.isFinite(state.metricScores.auroc)); }
    await call("finalize", { runId }); state = await settled(runId); assert.equal(state.status, "completed", state.error);
    await access(state.solutionZip);
    const inspected = await runProcess(runtime.python, ["-c", "import zipfile,sys,json; z=zipfile.ZipFile(sys.argv[1]); assert z.testzip() is None; assert 'model/model.joblib' in z.namelist(); assert not any(n.endswith('train.csv') or n.endswith('truth.json') for n in z.namelist()); print(json.loads(z.read('experiment.json'))['validationScore'])", state.solutionZip]);
    assert.equal(Number(inspected.output.trim()), state.validationScore);
    await assert.rejects(call("run_trial", { runId, sourceDir: source, device: "cpu" }), /frozen/);
    console.log(`${scenario.metric}: validation ${state.validationScore}; test ${state.testScore}; verified ZIP/model reload`);
  }
  await disconnect(); connect();
  assert.equal((await call("status", { runId: ids[0] })).status, "completed");
  assert.ok((await call("list_experiments")).some(run => run.runId === ids[0]));
  const { runId } = await call("create_experiment", { dataset, objective: "Cancellation", target: "y", metric: "rmse", excludedColumns: ["label"], minutes: null, trials: 1, exportModel: false }); ids.push(runId);
  assert.equal((await settled(runId)).status, "ready");
  await writeFile(join(source, "train.py"), "import time\ntime.sleep(120)\n");
  await call("run_trial", { runId, sourceDir: source, device: "cpu" }); await delay(1000);
  const other = spawn(process.execPath, [entry, "doctor", "--home", home]); await once(other, "exit");
  await call("cancel", { runId }); let state = await settled(runId);
  assert.equal(state.status, "paused"); assert.equal(state.trials[0].status, "cancelled");
  await assert.rejects(call("run_trial", { runId, sourceDir: source, device: "cpu" }), /trial budget/);
  await call("extend_budget", { runId, minutes: null, trials: 2 });
  await call("run_trial", { runId, sourceDir: source, device: "cpu" }); await delay(700); await disconnect(); connect();
  state = await call("status", { runId }); assert.equal(state.status, "paused"); assert.equal(state.trials.length, 2);
  console.log("Cancellation, disconnect recovery, budgets, persistence and protocol checks passed.");
} finally {
  if (child && child.exitCode === null) await disconnect();
  delete process.env.HELIX_SECRET_TEST;
  for (const id of ids) await rm(join(home, "experiments", id), { recursive: true, force: true });
  await rm(temp, { recursive: true, force: true });
  await rm(join(packagedExample, "..", ".."), { recursive: true, force: true });
}
