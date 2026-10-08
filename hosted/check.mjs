import assert from 'node:assert/strict';
import {mkdtemp,rm,readFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createServer} from 'node:http';
import {DatabaseSync} from 'node:sqlite';
import {accounts} from './accounts.mjs';
import {hostedServer} from './server.mjs';
import {accountOAuth} from './oauth.mjs';

const root=await mkdtemp(join(tmpdir(),'helix-accounts-'));
const worker=createServer((req,res)=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify({user:req.headers.authorization}));});
await new Promise(resolve=>worker.listen(0,'127.0.0.1',resolve));
let app;
try {
  const legacy=new DatabaseSync(join(root,'accounts.sqlite'));
  legacy.exec("CREATE TABLE users (id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, salt TEXT NOT NULL, password TEXT NOT NULL, workspace TEXT); INSERT INTO users VALUES('legacy','legacy','','',NULL)");
  legacy.close();
  let db=accounts(root);
  const a=await db.signIn('alice','a long password for alice',true);
  const b=await db.signIn('bob','a long password for bob',true);
  await assert.rejects(db.signIn('alice','incorrect password here',false),/Incorrect/);
  await assert.rejects(db.signIn('ALICE','another long password',true),/taken/);
  await assert.rejects(db.signIn('eve','short',true),/12 characters/);
  const linked=db.identify('github','123','alice',a.id);
  assert.equal(linked.id,a.id);assert.equal(db.identify('github','123','renamed').id,a.id);
  assert.equal(db.isAdmin(a.id,'123'),true);
  assert.equal(db.isAdmin(a.id,''),false);
  assert.equal(db.isAdmin(b.id,'123'),false);
  const lookalike=db.identify('github','unrelated-id','alice');
  assert.equal(db.isAdmin(lookalike.id,'123'),false,'Display names must never grant admin access');
  assert.throws(()=>db.identify('github','123','alice',b.id),/another Helix account/);
  const google=db.identify('google','123','alice');assert.notEqual(google.id,a.id);
  await assert.rejects(db.signIn(google.username,'a long random password',false),/Incorrect/);
  const browser='a'.repeat(43), flow=db.startOAuth('github',browser,a.id);
  assert.throws(()=>db.takeOAuth('github',flow.state,'b'.repeat(43),a.id),/expired/);
  assert.throws(()=>db.takeOAuth('google',flow.state,browser,a.id),/expired/);
  assert.ok(db.takeOAuth('github',flow.state,browser,a.id).verifier);
  assert.throws(()=>db.takeOAuth('github',flow.state,browser,a.id),/expired/);
  const before=process.env.HELIX_GITHUB_CLIENT_ID, secret=process.env.HELIX_GITHUB_CLIENT_SECRET;
  const googleBefore=process.env.HELIX_GOOGLE_CLIENT_ID, googleSecret=process.env.HELIX_GOOGLE_CLIENT_SECRET;
  process.env.HELIX_GITHUB_CLIENT_ID='test-client';process.env.HELIX_GITHUB_CLIENT_SECRET='test-secret';
  delete process.env.HELIX_GOOGLE_CLIENT_ID;delete process.env.HELIX_GOOGLE_CLIENT_SECRET;
  assert.ok(!accountOAuth(db,'https://helix.example').methods().some(method=>method.id==='google'));
  process.env.HELIX_GOOGLE_CLIENT_ID='test-google-client';
  assert.ok(!accountOAuth(db,'https://helix.example').methods().some(method=>method.id==='google'),'Both Google credentials are required');
  process.env.HELIX_GOOGLE_CLIENT_SECRET='test-google-secret';
  const oauth=accountOAuth(db,'https://helix.example');const start=oauth.begin('github',a.id);const target=new URL(start.url);
  assert.equal(target.origin,'https://github.com');assert.equal(target.searchParams.get('code_challenge_method'),'S256');
  assert.equal(target.searchParams.get('redirect_uri'),'https://helix.example/account/oauth/github/callback');assert.match(start.cookie,/HttpOnly.*SameSite=Lax.*Secure/);
  const realFetch=globalThis.fetch;
  let googleSubject='google-user-456', googleName='Alice';
  globalThis.fetch=async (url,options)=>{
    if(url==='https://oauth2.googleapis.com/token') {
      assert.equal(options.body.get('client_id'),'test-google-client');
      assert.equal(options.body.get('client_secret'),'test-google-secret');
      assert.equal(options.body.get('redirect_uri'),'https://helix.example/account/oauth/google/callback');
      assert.equal(options.body.get('grant_type'),'authorization_code');assert.ok(options.body.get('code_verifier'));
      return Response.json({access_token:'temporary-google-token'});
    }
    if(url==='https://openidconnect.googleapis.com/v1/userinfo') {
      assert.equal(options.headers.Authorization,'Bearer temporary-google-token');
      return Response.json({sub:googleSubject,name:googleName});
    }
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
    const signInGoogle=async userId=>{
      const flow=oauth.begin('google',userId), url=new URL(flow.url), browser=flow.cookie.match(/helix_oauth=([^;]+)/)[1];
      assert.equal(url.origin,'https://accounts.google.com');assert.equal(url.searchParams.get('scope'),'openid profile');
      assert.equal(url.searchParams.get('code_challenge_method'),'S256');
      assert.equal(url.searchParams.get('redirect_uri'),'https://helix.example/account/oauth/google/callback');
      url.searchParams.set('code','google-code');
      const user=await oauth.finish('google',url.searchParams,browser,userId);
      await assert.rejects(oauth.finish('google',url.searchParams,browser,userId),/expired/);
      return user;
    };
    const first=await signInGoogle();assert.notEqual(first.id,a.id,'Matching names must not merge accounts');
    db.saveWorkspace(first.id,{checkpoint:'google-workspace'});
    googleName='Changed Display Name';
    const returning=await signInGoogle();assert.equal(returning.id,first.id);
    assert.equal(db.workspace(returning.id).checkpoint,'google-workspace');
    await assert.rejects(signInGoogle(b.id),/another Helix account/);
    googleSubject='google-link-789';
    db.saveWorkspace(b.id,{checkpoint:'existing-workspace'});
    assert.equal((await signInGoogle(b.id)).id,b.id);
    assert.equal((await signInGoogle()).id,b.id);
    assert.equal(db.workspace(b.id).checkpoint,'existing-workspace');
    assert.equal(oauth.methods(b.id).find(method=>method.id==='google').linked,true);
    db.saveWorkspace(b.id,null);
  } finally {
    globalThis.fetch=realFetch;
    if(before===undefined)delete process.env.HELIX_GITHUB_CLIENT_ID;else process.env.HELIX_GITHUB_CLIENT_ID=before;
    if(secret===undefined)delete process.env.HELIX_GITHUB_CLIENT_SECRET;else process.env.HELIX_GITHUB_CLIENT_SECRET=secret;
    if(googleBefore===undefined)delete process.env.HELIX_GOOGLE_CLIENT_ID;else process.env.HELIX_GOOGLE_CLIENT_ID=googleBefore;
    if(googleSecret===undefined)delete process.env.HELIX_GOOGLE_CLIENT_SECRET;else process.env.HELIX_GOOGLE_CLIENT_SECRET=googleSecret;
  }
  const token=db.session(a.id);db.saveWorkspace(a.id,{checkpoint:'alice'});db.close();
  db=accounts(root);assert.equal(db.authenticate(token.token).id,a.id);assert.equal(db.workspace(a.id).checkpoint,'alice');assert.equal(db.workspace(b.id),null);assert.deepEqual(db.identities(a.id),['github']);assert.equal(db.identify('google','123','alice').id,google.id);
  assert.equal(db.adminOverview('legacy').users[0].createdAt,null,'Do not invent signup dates for migrated accounts');
  const seen=db.adminOverview('alice').users.find(user=>user.id===a.id).lastSeenAt;
  db.authenticate(token.token);
  assert.equal(db.adminOverview('alice').users.find(user=>user.id===a.id).lastSeenAt,seen,'Activity writes must be throttled');
  for(let i=0;i<53;i++)db.identify('google',`page-${i}`,`pageuser${i}`);
  assert.equal(db.adminOverview('PAGEUSER').matching,53);
  assert.equal(db.adminOverview('pageuser').users.length,50);
  assert.equal(db.adminOverview('pageuser',1).users.length,3);
  assert.equal(db.adminOverview('%').matching,0,'Search is literal, not a SQL wildcard');
  assert.equal(db.adminOverview().summary.newUsers,db.users().length-1,'Only new accounts have signup timestamps');
  db.logout(token.token);assert.equal(db.authenticate(token.token),null);db.close();
  assert.ok(!(await readFile(join(root,'accounts.sqlite'))).includes(Buffer.from('a long password')));
  const origin='http://127.0.0.1:4328';
  let workspaceRequests=0;
  app=hostedServer({root,origin,adminGithubId:'123',makeWorkspaces:()=>({ensure:async id=>{workspaceRequests++;return {url:`http://127.0.0.1:${worker.address().port}`,token:id};},retain:()=>()=>{},invalidate:()=>{},close:()=>{}})});
  await new Promise(resolve=>app.server.listen(4328,'127.0.0.1',resolve));
  const login=async username=>{
    const r=await fetch(origin+'/account/login',{method:'POST',headers:{Origin:origin},body:JSON.stringify({username,password:`a long password for ${username}`})});
    assert.equal(r.status,200); const cookie=r.headers.get('set-cookie');assert.match(cookie,/HttpOnly/);assert.match(cookie,/SameSite=Lax/);
    return {cookie,Authorization:`Bearer ${(await r.json()).csrf}`};
  };
  const alice=await login('alice'),bob=await login('bob');
  const adminPath='/account/admin/users';
  assert.equal((await fetch(origin+'/admin')).status,200,'A bookmarked admin page must load the sign-in shell');
  assert.equal((await fetch(origin+adminPath)).status,401);
  assert.equal((await fetch(origin+adminPath,{headers:{cookie:alice.cookie}})).status,401);
  assert.equal((await fetch(origin+adminPath,{headers:bob})).status,403);
  assert.equal((await fetch(origin+'/account/session',{headers:{cookie:alice.cookie}}).then(r=>r.json())).isAdmin,true);
  assert.equal((await fetch(origin+'/account/session',{headers:{cookie:bob.cookie}}).then(r=>r.json())).isAdmin,false);
  const response=await fetch(origin+adminPath,{headers:alice});
  assert.equal(response.status,200);assert.equal(response.headers.get('cache-control'),'no-store');
  const overview=await response.json();
  assert.ok(overview.summary.users>50 && overview.summary.seenUsers>=2 && overview.summary.workspaces>=1);
  assert.ok(overview.users.every(user=>Object.keys(user).sort().join(',')==='createdAt,hasWorkspace,id,lastSeenAt,signIns,username'));
  assert.equal((await fetch(origin+adminPath+'?q=pageuser&page=1',{headers:alice}).then(r=>r.json())).users.length,3);
  for(const query of ['page=-1','page=1.5','page=Infinity','q='+ 'a'.repeat(65)])assert.equal((await fetch(origin+adminPath+'?'+query,{headers:alice})).status,400);
  assert.equal((await fetch(origin+adminPath,{method:'POST',headers:{...alice,Origin:origin}})).status,405);
  assert.equal(workspaceRequests,0,'Admin requests must not boot, wake, or query any ML workspace');
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
  assert.equal((await fetch(origin+adminPath,{headers:alice})).status,401);
  console.log('Hosted checks passed: owner-only admin, migration, pagination, literal search, activity tracking, no worker startup, Google/GitHub sign-in, account separation, CSRF, OAuth PKCE/state/replay protection and worker proxy.');
} finally {if(app)await app.close();await new Promise(resolve=>worker.close(resolve));await rm(root,{recursive:true,force:true});}
