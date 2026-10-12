// Real fits, independent scoring and native export/reload; run optional heavyweight stacks explicitly.
import assert from 'node:assert/strict';
import { mkdtemp, cp, writeFile, readFile, rm } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import { runtimeConfig } from './runtime.mjs';
import { createMcpService } from './service.mjs';
import { runProcess } from '../local/process.mjs';

const framework = process.env.HELIX_TEST_FRAMEWORK || 'keras';
assert.ok(['keras', 'cuml'].includes(framework));
const device = process.env.HELIX_TEST_DEVICE || (framework === 'cuml' ? 'cuda' : 'cpu');
const home = resolve(process.env.HELIX_MCP_HOME || join(homedir(), '.helix-ml', 'mcp'));
const runtime = await runtimeConfig(home);
const service = await createMcpService(home, runtime);
const temp = await mkdtemp(join(tmpdir(), 'helix frameworks '));
const call = (name, args = {}) => service.call('helix_' + name, args);
async function settled(runId) {
  for (let i = 0; i < 7200; i++) {
    const state = await call('status', {runId, details: true});
    if (state.status !== 'running') { assert.ok(['ready','completed'].includes(state.status), state.error); return state; }
    await delay(500);
  }
  throw new Error('Framework check timed out');
}
const ids = [];
let passed = false;
try {
  const source = join(temp, 'source');
  await cp(fileURLToPath(new URL(`fixtures/${framework}`, import.meta.url)), source, {recursive:true});
  const policy = Object.fromEntries(['augmentation','regularization','features','tuning','pretrained','ensemble'].map(k => [k,k === 'features' && framework === 'cuml']));
  await writeFile(join(source, 'plan.json'), JSON.stringify({component:framework, change:'verified baseline', rationale:'Test real training and export', modelFamily:framework, strategies:policy}));
  const dataset = join(temp, 'data.csv');
  await writeFile(dataset, 'x,z,w,label\n' + Array.from({length:160}, (_,i) => `${i%20 + (i%2)*30},${(i%13)/13 + (i%2)*30},${(i%7)/7},${i%2 ? 'long':'short'}`).join('\n'));
  for (const variant of framework === 'cuml' ? ['clustering','reduction'] : ['keras','savedmodel']) {
    const args = framework === 'cuml' ? {learning:variant, excludedColumns:['label'], metric:variant==='clustering'?'silhouette':'trustworthiness', dimensions:2, exportFormat:'joblib'} : {target:'label',metric:'auroc',metrics:['auroc','f1'],exportFormat:variant};
    const {runId} = await call('create_experiment', {dataset, objective:`Test ${framework} ${variant}`, ...args, device, minutes:60,trials:1, policy});
    ids.push(runId); console.log(`Checking ${framework} ${variant} on ${device}; run ${runId}`);
    await settled(runId);
    const name = framework === 'cuml' ? 'cuml-cu13' : device === 'cuda' ? 'tensorflow[and-cuda]' : 'tensorflow';
    await call('run_trial', {runId,sourceDir:source, device, packages:[{name, version: framework === 'cuml' ? '26.10.0':'2.21.0'}]});
    let state = await settled(runId);
    assert.ok(Number.isFinite(state.validationScore));
    for (const fit of state.trials[0].evaluations) {
      const proof = JSON.parse(await readFile(join(home,'experiments',runId,'fits','trial-1',fit.id,'output','device_check.json')));
      if (framework === 'cuml') assert.ok(proof.gpuBytes > 0);
      else assert.match(proof.weightDevice, new RegExp(device === 'cuda' ? 'GPU:0':'CPU:0'));
    }
    await call('finalize', {runId}); state = await settled(runId);
    assert.ok(Number.isFinite(state.testScore));
    await runProcess(runtime.python, ['-I','-c', "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); assert z.testzip() is None; assert 'experiment.json' in z.namelist()",state.solutionZip]);
    console.log(`${framework} ${variant} ${device}: training, validation, reserved test, native reload and ZIP passed; validation ${state.validationScore}.`);
  }
  passed = true;
} finally {
  await service.close();
  if (passed) { for (const id of ids) await rm(join(home,'experiments',id), {recursive:true,force:true}); await rm(temp,{recursive:true,force:true}); }
  else console.error('Failed test artifacts retained:', temp, ids);
}
