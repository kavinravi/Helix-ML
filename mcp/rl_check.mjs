import assert from 'node:assert/strict';
import { readFile, rm, mkdtemp } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { runtimeConfig } from './runtime.mjs';
import { createMcpService } from './service.mjs';
import { runProcess } from '../local/process.mjs';
import { container } from '../local/runtime.mjs';

const home = resolve(process.env.HELIX_MCP_HOME || join(homedir(), '.helix-ml', 'mcp'));
const runtime = await runtimeConfig(home);
if (process.platform === 'darwin' && process.arch === 'x64') { console.log('Intel Mac: current SB3 requires a modern PyTorch wheel unavailable on this platform. RL integration check skipped.'); process.exit(0); }
const service = await createMcpService(home, runtime);
const call = (name, args = {}) => service.call('helix_' + name, args);
const ids = [], temp = await mkdtemp(join(tmpdir(), 'helix rl check '));
async function settled(runId, expected = ['ready','completed']) {
  for (let i = 0; i < 3600; i++) {
    const state = await call('status', {runId, details:true});
    if (state.status !== 'running') { assert.ok(expected.includes(state.status), state.error); return state; }
    await delay(250);
  }
  throw new Error('RL test timed out');
}
let passed = false;
try {
  await assert.rejects(call('create_rl_experiment', {envId:'evil:Env-v0',objective:'bad'}));
  await assert.rejects(call('create_rl_experiment', {envId:'CartPole-v1',objective:'bad', minutes:null,trials:null}), /at least one/);
  for (const envId of ['CartPole-v1','Pendulum-v1']) {
    const {runId} = await call('create_rl_experiment', {envId,objective:'Check actual SB3 training and evaluation',device:process.env.HELIX_TEST_DEVICE || 'cpu',minutes:15,trials:5,evaluationEpisodes:2,maxEpisodeSteps:100});
    ids.push(runId); console.log('Checking RL:', envId, runId);
    let state = await settled(runId);
    assert.ok(state.contract.includes('helix_run_rl_trial'));
    assert.equal(state.environment.maxEpisodeSteps,100);
    assert.equal(state.testScore, undefined);
    const protocol = JSON.parse(await readFile(join(home,'experiments',runId,'rl-protocol.json')));
    assert.equal(new Set([protocol.trainingSeed,...protocol.validationSeeds,...protocol.testSeeds]).size,5);
    const algorithms = envId === 'CartPole-v1' ? ['PPO','A2C','DQN'] : ['SAC','TD3','DDPG'];
    for (const algorithm of algorithms) {
      const hyperparameters = ['PPO','A2C'].includes(algorithm) ? {n_steps:32, ...(algorithm === 'PPO' ? {batch_size:32,n_epochs:1}:{}),net_arch:[16]} : {learning_starts:16,buffer_size:1000,batch_size:16,net_arch:[16]};
      await call('run_rl_trial',{runId, algorithm,totalTimesteps:128,hyperparameters});
      state = await settled(runId);
      const trial = state.trials.at(-1);
      assert.ok(Number.isFinite(trial.score));
      assert.equal(trial.reloadVerified,true);
      assert.ok(trial.training.timesteps >= 128);
      assert.equal(trial.evaluationEpisodes,2);
      assert.ok(trial.episodeLengths.every(n => n<=100));
      console.log(algorithm,trial.score,trial.training.device);
    }
    await call('run_rl_trial',{runId,algorithm:envId==='CartPole-v1'?'SAC':'DQN',totalTimesteps:128});
    state = await settled(runId,['error']);
    assert.match(state.error,/requires (continuous|discrete)/);
    assert.equal(state.trials.at(-1).status,'failed');
    assert.ok(state.best);
    if (envId === 'CartPole-v1') {
      await call('run_rl_trial',{runId,algorithm:'PPO',totalTimesteps:100000000});
      await delay(1000);
      await call('cancel',{runId}); state=await settled(runId,['paused']);
      assert.equal(state.trials.at(-1).status,'cancelled');
      await call('extend_budget',{runId,minutes:15,trials:6});
    }
    await call('finalize',{runId}); state=await settled(runId);
    assert.equal(state.status,'completed');
    assert.ok(Number.isFinite(state.testScore));
    await assert.rejects(call('run_rl_trial',{runId,algorithm:'PPO',totalTimesteps:128}),/frozen/);
    await assert.rejects(call('finalize',{runId}),/already exported/);
    const exported=join(temp,envId);
    await runProcess(runtime.python,['-I','-c',"import sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); assert z.testzip() is None; assert 'policy.zip' in z.namelist(); z.extractall(sys.argv[2])",state.solutionZip,exported]);
    await container({runtime:{...runtime,device:'cpu'},deadline:Date.now()+120000},['python',join(exported,'rl.py'),'evaluate','--config',join(exported,'test-config.json'),'--model',join(exported,'policy.zip'),'--output',join(exported,'reloaded.json')],{mounts:[[join(home,'experiments',runId,'dependencies','rl'),'/packages']]});
    const reproduced=JSON.parse(await readFile(join(exported,'reloaded.json')));
    assert.ok(Math.abs(reproduced.score-state.testScore) <= 1e-5*Math.max(1,Math.abs(state.testScore)), 'CPU reload return differs from the exported policy');
    console.log(envId,'passed: training, independent evaluation, final held-out rewards, export and CPU reload.');
  }
  passed=true;
} finally {
  await service.close();
  if (passed) { for (const id of ids) await rm(join(home,'experiments',id),{recursive:true,force:true}); await rm(temp,{recursive:true,force:true}); }
  else console.error('Failed RL runs retained:',ids);
}
