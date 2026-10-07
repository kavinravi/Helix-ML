import { Sandbox, SandboxNotFoundError } from 'railway';
import { randomBytes } from 'node:crypto';
import { readdir, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const project = fileURLToPath(new URL('..', import.meta.url));
export function workspaces(store) {
  const pending = new Map(), handles = new Map(), seen = new Map(), busy = new Map();
  let creating = 0;
  const request = (value, path, method = 'GET') => fetch(value.url + path, {
    method, headers: {Authorization: `Bearer ${value.token}`}, signal: AbortSignal.timeout(30_000),
  });
  async function start(sandbox, value, fresh = false) {
    if (fresh) {
      // A checkpoint restores files, not processes. Its old PID lock is no longer valid.
      const result = await sandbox.exec('rm -rf /var/lib/helix/.runner-lock');
      if (result.exitCode !== 0) throw new Error('Could not prepare the saved workspace.');
    }
    for (const folder of ['local', 'hosted']) {
      const names = (await readdir(join(project, folder))).filter(name => folder === 'local' ? /\.(mjs|py)$/.test(name) : name === 'worker.mjs');
      await Promise.all(names.map(async name => sandbox.files.write(`/opt/helix/${folder}/${name}`, await readFile(join(project, folder, name)))));
    }
    const command = sandbox.exec('node /opt/helix/hosted/worker.mjs', {
      env: {HELIX_WORKER_TOKEN: value.token, HELIX_HOSTED_WORKER: '1'}, maxOutputBytes: 4096,
    });
    await command.detach();
    for (let attempt = 0; attempt < 30; attempt++) {
      try { if ((await request(value, '/api/runs')).ok) return; } catch {}
      await delay(1000);
    }
    throw new Error('Your workspace is taking longer to start. Please try again.');
  }
  async function ensure(id) {
    if (!process.env.RAILWAY_TOKEN && !process.env.RAILWAY_API_TOKEN) throw new Error('Cloud workspaces are being configured. Please try again later.');
    seen.set(id, Date.now());
    // Waiting may finish a save rather than a boot, so always re-read the record.
    if (pending.has(id)) { await pending.get(id); return ensure(id); }
    const saved = store.workspace(id);
    if (saved?.url && handles.has(id)) return saved;
    const operation = (async () => {
      if (saved?.sandboxId) {
        try {
          const sandbox = await Sandbox.connect(saved.sandboxId);
          if (sandbox.status === 'RUNNING') {
            handles.set(id, sandbox);
            const alive = await request(saved, '/api/runs').then(r => r.ok).catch(() => false);
            if (!alive) await start(sandbox, saved);
            return saved;
          }
        } catch (error) { if (!(error instanceof SandboxNotFoundError)) { handles.delete(id); throw error; } }
      }
      if (saved?.sandboxId) store.saveWorkspace(id, {checkpoint: saved.checkpoint});
      // ponytail: a single gateway admits three workspaces. Use database admission before adding replicas.
      if (store.users().filter(user => user.workspace?.sandboxId).length + creating >= 3) throw new Error('All workspaces are busy. Please try again shortly.');
      const base = saved?.checkpoint || process.env.HELIX_BASE_CHECKPOINT;
      if (!base) throw new Error('The cloud training runtime is not configured.');
      creating++;
      let sandbox;
      try { sandbox = await Sandbox.create(base, {networkIsolation: 'PRIVATE', domains: [{port: 8080}], idleTimeoutMinutes: 5}); }
      finally { creating--; }
      const domain = sandbox.domains[0]?.domain;
      if (!domain) { await sandbox.destroy(); throw new Error('The workspace address is not available yet.'); }
      const value = {...saved, sandboxId: sandbox.id, token: randomBytes(32).toString('base64url'), url: `https://${domain}`};
      // Keep the VM even if startup fails; retries must not erase uploaded data or native logins.
      store.saveWorkspace(id, value);
      try { await start(sandbox, value, true); handles.set(id, sandbox); return value; }
      catch (error) { handles.delete(id); throw error; }
    })().finally(() => pending.delete(id));
    pending.set(id, operation);
    return operation;
  }
  async function sleep(id) {
    if (pending.has(id) || busy.get(id)) return;
    const value = store.workspace(id);
    if (!value?.sandboxId) { seen.delete(id); return; }
    const operation = (async () => {
      const sandbox = handles.get(id) || await Sandbox.connect(value.sandboxId);
      if (sandbox.status !== 'RUNNING') { store.saveWorkspace(id, {checkpoint: value.checkpoint}); handles.delete(id); seen.delete(id); return; }
      const runs = await request(value, '/api/runs');
      if (!runs.ok) throw new Error('Could not inspect workspace before saving.');
      if ((await runs.json()).some(run => ['running', 'queued'].includes(run.status))) return;
      const auth = await request(value, '/native-login');
      if (!auth.ok || (await auth.json())?.status === 'waiting') return;
      const providers = await request(value, '/api/providers');
      if (!providers.ok || (await providers.json()).some(p => p.capability?.status === 'checking')) return;
      // Leave the VM running until its files have been saved. No active fits or sign-in writes remain.
      const checkpoint = `helix-user-${id}`;
      await sandbox.checkpoint(checkpoint);
      store.saveWorkspace(id, {...value, checkpoint});
      await sandbox.destroy();
      store.saveWorkspace(id, {checkpoint});
      handles.delete(id); seen.delete(id);
    })().catch(error => {
      if (error instanceof SandboxNotFoundError) { store.saveWorkspace(id, {checkpoint: value.checkpoint}); handles.delete(id); seen.delete(id); }
      else console.error('Workspace save failed:', id, error.message);
    }).finally(() => pending.delete(id));
    pending.set(id, operation);
    await operation;
  }
  // Adopt workspaces after a gateway restart, including users who have closed their browser.
  for (const {id, workspace} of store.users()) if (workspace?.sandboxId) seen.set(id, 0);
  const timer = setInterval(() => {
    for (const [id, time] of seen) if (Date.now() - time > 5 * 60_000) void sleep(id);
  }, 60_000);
  timer.unref();
  return {ensure, sleep, invalidate: id => handles.delete(id),
    retain(id) {
      busy.set(id, (busy.get(id) || 0) + 1);
      return () => { busy.set(id, busy.get(id) - 1); seen.set(id, Date.now()); };
    },
    async close() { clearInterval(timer); await Promise.allSettled([...pending.values()]); },
  };
}
