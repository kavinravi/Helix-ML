// Run with node --test local/followup.test.mjs. No subscription or Docker required.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createService } from './server.mjs';
import { createRun, continueRun, loadRuns, saveRun, steps as iterateSteps } from './engine.mjs';
import { availableTools, callTool } from './tools.mjs';
import { validateTrainingHistory } from './evaluation.mjs';

const task = { agent: 'codex', dataset: '/example.csv', objective: 'Predict labels', target: 'label', metric: 'accuracy',
  minutes: 5, trials: 3, searchModels: false, model: 'DecisionTreeClassifier', output: 'py', exportModel: false,
  exportFormat: 'native', validation: 'holdout', folds: 3, testFraction: .2, holdoutFraction: .2, seeds: [42],
  splitStrategy: 'independent', groupColumn: '', timeColumn: '', assetColumns: [],
  policy: { augmentation: false, regularization: false, features: false, tuning: true, pretrained: false, ensemble: false } };

const steps = (...args) => Array.from(iterateSteps(...args));

test('follow-ups persist, isolate source, and keep the selected result and test partition frozen', async () => {
  const root = await mkdtemp(join(tmpdir(), 'helix-followups-')), runRoot = join(root, 'runs');
  const previousPath = process.env.PATH;
  let service;
  try {
    const curve = [{ step: 0, loss: .8, accuracy: .5 }, { step: 1, loss: .3, accuracy: .9 }];
    assert.deepEqual(validateTrainingHistory(curve), curve);
    for (const invalid of [[], [{ step: 1, loss: .1 }], [{ step: 0, loss: NaN }, { step: 1, loss: .1 }], [{ step: 1, loss: .2 }, { step: 1, loss: .1 }], [{ step: 0, accuracy: 2 }, { step: 1, loss: .1 }], Array(501).fill({ step: 1, loss: .1 })])
      assert.throws(() => validateTrainingHistory(invalid));
    assert.equal(steps(1, task.policy, 3).length, 1, 'Never exceed a one-trial budget');
    assert.deepEqual(availableTools(task, 'discussion').map(tool => tool.name).sort(), ['dataset_info', 'list_artifacts', 'previous_experiments', 'read_source']);
    const parent = await createRun(runRoot, task);
    Object.assign(parent, { status: 'completed', phase: 'complete', best: 'trial-001', score: .8, testScore: .75, protocolHash: 'frozen-protocol',
      trials: [{ id: 'trial-001', name: 'Tree', phase: 'baseline', score: .8, status: 'accepted', artifact: 'trial-001/train.py', detail: 'Measured tree' }] });
    for (const path of ['trials/trial-001', 'dependencies/trial-001', 'models', 'data', 'final']) await mkdir(join(runRoot, parent.id, path), { recursive: true });
    await writeFile(join(runRoot, parent.id, 'trials/trial-001/train.py'), '# original selected source');
    await writeFile(join(runRoot, parent.id, 'trials/trial-001/AGENTS.md'), 'Old agent instructions');
    await writeFile(join(runRoot, parent.id, 'data/manifest.json'), '{}');
    await writeFile(join(runRoot, parent.id, 'final/helix-solution.zip'), 'original bundle');
    await saveRun(runRoot, parent);
    const context = { runFile: join(runRoot, parent.id, 'run.json'), workspace: join(runRoot, parent.id, 'trials/trial-001'), mode: 'discussion', deadline: Date.now() + 10000 };
    for (const name of ['write_source', 'install_package', 'cache_model']) await assert.rejects(callTool(name, {}, context), /disabled/);
    const child = await continueRun(runRoot, parent, 'Try a shallower tree', null, 1);
    assert.equal(child.followup.parentId, parent.id);
    assert.equal(child.followup.protocolHash, parent.protocolHash);
    assert.equal(child.best, 'inherited'); assert.equal(child.score, .8); assert.equal(child.testScore, undefined);
    assert.equal(child.task.minutes, null); assert.equal(child.task.trials, 1); assert.equal(child.next, 0);
    assert.deepEqual(child.task.policy, parent.task.policy);
    await assert.rejects(readFile(join(runRoot, child.id, 'trials/inherited/AGENTS.md')), /ENOENT/);
    await writeFile(join(runRoot, child.id, 'trials/inherited/train.py'), '# child only');
    assert.equal(await readFile(join(runRoot, parent.id, 'trials/trial-001/train.py'), 'utf8'), '# original selected source');
    // Simulate completion for the API checks without launching paid agent training.
    child.status = 'completed'; await saveRun(runRoot, child);
    const bin = join(root, 'bin'); await mkdir(bin);
    const stub = `#!${process.execPath}
const fs = require('node:fs');
const args = process.argv.slice(2);
if(args[0] === '--version') console.log('test-version');
else if(args[0] === 'login' || args[0] === 'auth') console.log('Logged in using ChatGPT');
else {
 fs.writeFileSync('train.py', '# disposable discussion copy');
 setTimeout(() => { console.log(JSON.stringify({type:'item.completed',item:{type:'agent_message',text:'The measured validation accuracy is 0.8.'}})); console.log(JSON.stringify({type:'turn.completed'})); }, 500);
}
`;
    await writeFile(join(bin, 'codex'), stub, { mode: 0o700 });
    process.env.PATH = bin + ':' + previousPath;
    service = await createService({ root, port: 0 });
    await new Promise(resolve => service.server.listen(0, '127.0.0.1', resolve));
    const url = `http://127.0.0.1:${service.server.address().port}/api`;
    const headers = { Authorization: `Bearer ${service.token}`, 'Content-Type': 'application/json' };
    const post = (path, body) => fetch(url + path, { method: 'POST', headers, body: JSON.stringify(body) });
    const path = `/runs/${parent.id}/messages`;
    assert.equal((await post(path, { mode: 'chat', message: '' })).status, 400);
    assert.equal((await post(path, { mode: 'trials', message: 'More', trials: 0, minutes: 2 })).status, 400);
    const reply = await post(path, { mode: 'chat', message: 'Explain the result' });
    assert.equal(reply.status, 202);
    assert.equal((await reply.json()).messages[0].status, 'pending');
    assert.equal((await post(path, { mode: 'chat', message: 'Duplicate pending' })).status, 409);
    assert.equal((await fetch(url + `/runs/${parent.id}`, { method: 'DELETE', headers })).status, 409);
    assert.equal((await post('/runs', task)).status, 409);
    const live = service.runs.get(parent.id);
    for (let attempt = 0; live.messages[0].status === 'pending' && attempt < 80; attempt++) await new Promise(resolve => setTimeout(resolve, 50));
    assert.equal(live.messages[0].status, 'completed', live.messages[0].error);
    assert.equal(live.messages[0].reply, 'The measured validation accuracy is 0.8.');
    assert.equal(live.testScore, .75);
    assert.equal(await readFile(join(runRoot, parent.id, 'trials/trial-001/train.py'), 'utf8'), '# original selected source');
    assert.equal(await readFile(join(runRoot, parent.id, 'final/helix-solution.zip'), 'utf8'), 'original bundle');
    await service.close(); service = null;
    let loaded = await loadRuns(runRoot);
    assert.equal(loaded.get(parent.id).messages[0].reply, live.messages[0].reply);
    assert.equal(loaded.get(child.id).task.minutes, null, 'Disabled time limits survive a restart');
    live.messages.push({ id: 'interrupted', status: 'pending', message: 'Interrupted turn' });
    await saveRun(runRoot, live);
    loaded = await loadRuns(runRoot);
    assert.equal(loaded.get(parent.id).messages[1].status, 'failed');
    assert.equal(loaded.get(parent.id).status, 'completed');
  } finally {
    await service?.close(); process.env.PATH = previousPath;
    await rm(root, { recursive: true, force: true });
  }
});
