import assert from 'node:assert/strict';
import {mkdtemp,rm,readFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createServer} from 'node:http';
import {accounts} from './accounts.mjs';
import {hostedServer} from './server.mjs';
import {accountOAuth} from './oauth.mjs';

const root=await mkdtemp(join(tmpdir(),'helix-accounts-'));
const worker=createServer((req,res)=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify({user:req.headers.authorization}));});
await new Promise(resolve=>worker.listen(0,'127.0.0.1',resolve));
let app;
try {
  let db=accounts(root);
  const a=await db.signIn('alice','a long password for alice',true);
  const b=await db.signIn('bob','a long password for bob',true);
  await assert.rejects(db.signIn('alice','incorrect password here',false),/Incorrect/);
  await assert.rejects(db.signIn('ALICE','another long password',true),/taken/);
  await assert.rejects(db.signIn('eve','short',true),/12 characters/);
  const linked=db.identify('github','123','alice',a.id);
  assert.equal(linked.id,a.id);assert.equal(db.identify('github','123','renamed').id,a.id);
  assert.throws(()=>db.identify('github','123','alice',b.id),/another Helix account/);
  const google=db.identify('google','123','alice');assert.notEqual(google.id,a.id);
  await assert.rejects(db.signIn(google.username,'a long random password',false),/Incorrect/);
  const browser='a'.repeat(43), flow=db.startOAuth('github',browser,a.id);
  assert.throws(()=>db.takeOAuth('github',flow.state,'b'.repeat(43),a.id),/expired/);
  assert.throws(()=>db.takeOAuth('google',flow.state,browser,a.id),/expired/);
  assert.ok(db.takeOAuth('github',flow.state,browser,a.id).verifier);
  assert.throws(()=>db.takeOAuth('github',flow.state,browser,a.id),/expired/);
  const before=process.env.HELIX_GITHUB_CLIENT_ID, secret=process.env.HELIX_GITHUB_CLIENT_SECRET;
  process.env.HELIX_GITHUB_CLIENT_ID='test-client';process.env.HELIX_GITHUB_CLIENT_SECRET='test-secret';
  const oauth=accountOAuth(db,'https://helix.example');const start=oauth.begin('github',a.id);const target=new URL(start.url);
  assert.equal(target.origin,'https://github.com');assert.equal(target.searchParams.get('code_challenge_method'),'S256');
  assert.equal(target.searchParams.get('redirect_uri'),'https://helix.example/account/oauth/github/callback');assert.match(start.cookie,/HttpOnly.*SameSite=Lax.*Secure/);
  const realFetch=globalThis.fetch;
  globalThis.fetch=async (url,options)=>{
    if(url==='https://github.com/login/oauth/access_token') {
      assert.equal(options.body.get('client_secret'),'test-secret');assert.ok(options.body.get('code_verifier'));
      return Response.json({access_token:'temporary-test-token'});
    }
    assert.equal(url,'https://api.github.com/user');assert.equal(options.headers.Authorization,'Bearer temporary-test-token');
    return Response.json({id:123,login:'renamed'});
  };
  try {
    target.searchParams.set('code','one-time-code');
    assert.equal((await oauth.finish('github',target.searchParams,start.cookie.match(/helix_oauth=([^;]+)/)[1],a.id)).id,a.id);
  } finally {
    globalThis.fetch=realFetch;
    if(before===undefined)delete process.env.HELIX_GITHUB_CLIENT_ID;else process.env.HELIX_GITHUB_CLIENT_ID=before;
    if(secret===undefined)delete process.env.HELIX_GITHUB_CLIENT_SECRET;else process.env.HELIX_GITHUB_CLIENT_SECRET=secret;
  }
  const token=db.session(a.id);db.saveWorkspace(a.id,{checkpoint:'alice'});db.close();
  db=accounts(root);assert.equal(db.authenticate(token.token).id,a.id);assert.equal(db.workspace(a.id).checkpoint,'alice');assert.equal(db.workspace(b.id),null);assert.deepEqual(db.identities(a.id),['github']);assert.equal(db.identify('google','123','alice').id,google.id);
  db.logout(token.token);assert.equal(db.authenticate(token.token),null);db.close();
  assert.ok(!(await readFile(join(root,'accounts.sqlite'))).includes(Buffer.from('a long password')));
  const origin='http://127.0.0.1:4328';
  app=hostedServer({root,origin,makeWorkspaces:()=>({ensure:async id=>({url:`http://127.0.0.1:${worker.address().port}`,token:id}),retain:()=>()=>{},invalidate:()=>{},close:()=>{}})});
  await new Promise(resolve=>app.server.listen(4328,'127.0.0.1',resolve));
  const login=async username=>{
    const r=await fetch(origin+'/account/login',{method:'POST',headers:{Origin:origin},body:JSON.stringify({username,password:`a long password for ${username}`})});
    assert.equal(r.status,200); const cookie=r.headers.get('set-cookie');assert.match(cookie,/HttpOnly/);assert.match(cookie,/SameSite=Lax/);
    return {cookie,Authorization:`Bearer ${(await r.json()).csrf}`};
  };
  const alice=await login('alice'),bob=await login('bob');
  assert.equal((await fetch(origin+'/api/runs')).status,401);
  assert.equal((await fetch(origin+'/api/runs',{headers:{cookie:alice.cookie}})).status,401);
  assert.equal((await fetch(origin+'/account/workspace',{method:'POST',headers:{...alice,Origin:'https://evil.example'}})).status,403);
  assert.equal((await fetch(origin+'/api/runs',{headers:alice}).then(r=>r.json())).user,`Bearer ${a.id}`);
  assert.equal((await fetch(origin+'/api/runs',{headers:bob}).then(r=>r.json())).user,`Bearer ${b.id}`);
  const deletePath='/api/runs/00000000-0000-4000-8000-000000000001';
  assert.equal((await fetch(origin+deletePath,{method:'DELETE',headers:{cookie:bob.cookie,Origin:origin}})).status,401);
  assert.equal((await fetch(origin+deletePath,{method:'DELETE',headers:{...bob,Origin:'https://evil.example'}})).status,403);
  assert.equal((await fetch(origin+deletePath,{method:'DELETE',headers:{...bob,Origin:origin}}).then(r=>r.json())).user,`Bearer ${b.id}`);
  assert.equal((await fetch(origin+'/api/session',{method:'POST',headers:{...alice,Origin:origin}})).status,404);
  assert.equal((await fetch(origin+'/account/logout',{method:'POST',headers:{...alice,Origin:origin}})).status,200);
  assert.equal((await fetch(origin+'/api/runs',{headers:alice})).status,401);
  console.log('Hosted checks passed: password hashing, persistence, account separation, CSRF, OAuth PKCE/state/replay protection, explicit linking and worker proxy.');
} finally {if(app)await app.close();await new Promise(resolve=>worker.close(resolve));await rm(root,{recursive:true,force:true});}
