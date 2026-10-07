import { createServer } from 'node:http';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { createReadStream } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join, extname } from 'node:path';
import { accounts } from './accounts.mjs';
import { accountOAuth } from './oauth.mjs';
import { workspaces } from './workspaces.mjs';
import { inside } from '../local/validate.mjs';

const project=fileURLToPath(new URL('..',import.meta.url));
const mime={'.html':'text/html; charset=utf-8','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml','.woff2':'font/woff2'};
async function body(req) {
  const chunks=[];let length=0;
  for await (const chunk of req) { if ((length+=chunk.length)>8000) throw new Error('Request too large.'); chunks.push(chunk); }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}
export function hostedServer({root=process.env.HELIX_DATA_DIR || '/data',origin=process.env.HELIX_PUBLIC_URL || (process.env.RAILWAY_PUBLIC_DOMAIN ? `https://${process.env.RAILWAY_PUBLIC_DOMAIN}` : `http://127.0.0.1:${process.env.PORT||3000}`),makeWorkspaces=workspaces}={}) {
  const store=accounts(root), workers=makeWorkspaces(store), oauth=accountOAuth(store,origin), attempts=new Map();
  let loginWindow=Date.now(), loginCount=0;
  const cookie=(token,age=30*86400)=>`helix_session=${token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${age}${origin.startsWith('https:')?'; Secure':''}`;
  const server=createServer(async(req,res)=>{
    const respond=(status,value)=>{res.writeHead(status,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify(value));};
    res.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'");
    res.setHeader('X-Content-Type-Options','nosniff'); res.setHeader('Referrer-Policy','no-referrer');
    try {
      const url=new URL(req.url,origin);
      if (url.pathname==='/health' && req.method==='GET') return respond(200,{name:'helix-ml',mode:'hosted'});
      if (req.headers.host!==new URL(origin).host) return respond(403,{error:'Invalid host.'});
      if (!['GET','HEAD'].includes(req.method) && req.headers.origin!==origin) return respond(403,{error:'Open Helix from its website to continue.'});
      const token=req.headers.cookie?.match(/(?:^|;\s*)helix_session=([A-Za-z0-9_-]+)/)?.[1];
      const account=store.authenticate(token);
      const oauthRoute=url.pathname.match(/^\/account\/oauth\/(google|github)(\/callback)?$/);
      if (req.method==='POST' && (url.pathname==='/account/login' || oauthRoute)) {
        const now=Date.now();
        if(now-loginWindow>60_000){loginWindow=now;loginCount=0;}
        if(++loginCount>60)return respond(429,{error:'Too many attempts. Try again in a minute.'});
      }
      if (url.pathname==='/account/methods' && req.method==='GET') return respond(200,oauth.methods(account?.id));
      if (oauthRoute && !oauthRoute[2] && req.method==='POST') {
        if (account && req.headers.authorization!==`Bearer ${account.csrf}`) return respond(401,{error:'Sign in to link an account.'});
        const flow=oauth.begin(oauthRoute[1],account?.id);
        res.setHeader('Set-Cookie',flow.cookie);return respond(200,{url:flow.url});
      }
      if (oauthRoute?.[2] && req.method==='GET') {
        try {
          const browser=req.headers.cookie?.match(/(?:^|;\s*)helix_oauth=([A-Za-z0-9_-]+)/)?.[1];
          const user=await oauth.finish(oauthRoute[1],url.searchParams,browser,account?.id);
          const session=store.session(user.id);store.logout(token);
          res.writeHead(303,{'Set-Cookie':[cookie(session.token),oauth.clearCookie()],Location:account?'/?signin=linked':'/','Cache-Control':'no-store'});res.end();
        } catch { res.writeHead(303,{'Set-Cookie':oauth.clearCookie(),Location:'/?signin=failed','Cache-Control':'no-store'});res.end(); }
        return;
      }
      if (url.pathname==='/account/login' && req.method==='POST') {
        const now=Date.now();
        const value=await body(req);
        const key=String(value.username || '').trim().toLowerCase().slice(0,32);
        for (const [ip,value] of attempts) if (now-value.since>60_000) attempts.delete(ip);
        const count=attempts.get(key)||{since:now,count:0}; attempts.set(key,count);
        if (++count.count>10) return respond(429,{error:'Too many attempts. Try again in a minute.'});
        const user=await store.signIn(value.username,value.password,value.register===true);
        const session=store.session(user.id);
        res.setHeader('Set-Cookie',cookie(session.token));
        return respond(200,{username:user.username,csrf:session.csrf});
      }
      if (url.pathname==='/account/session' && req.method==='GET') return respond(200,account ? {username:account.username,csrf:account.csrf} : null);
      if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/account/')) {
        if (!account || req.headers.authorization!==`Bearer ${account.csrf}`) return respond(401,{error:'Sign in to Helix to continue.'});
        if (url.pathname==='/account/logout' && req.method==='POST') {
          store.logout(token);res.setHeader('Set-Cookie',cookie('',0));return respond(200,{signedOut:true});
        }
        if (url.pathname==='/account/workspace' && req.method==='POST') {
          await workers.ensure(account.id);return respond(200,{ready:true});
        }
        let path=req.url;
        if (url.pathname==='/account/provider' && ['GET','POST','DELETE'].includes(req.method)) path='/native-login';
        else if (!url.pathname.startsWith('/api/') || url.pathname.startsWith('/api/session') || url.pathname.startsWith('/api/quit')) return respond(404,{error:'Not found'});
        const workspace=await workers.ensure(account.id);
        const release=workers.retain(account.id);
        try {
        const controller=new AbortController();
        req.on('aborted',()=>controller.abort());res.on('close',()=>{if(!res.writableFinished)controller.abort();});
        const upstream=await fetch(workspace.url+path,{method:req.method,headers:{Authorization:`Bearer ${workspace.token}`,'Content-Type':req.headers['content-type']||'application/json'},...(!['GET','HEAD'].includes(req.method)?{body:Readable.toWeb(req),duplex:'half'}:{}),redirect:'manual',signal:AbortSignal.any([controller.signal,AbortSignal.timeout(360_000)])});
        if (!upstream.ok) workers.invalidate(account.id);
        res.writeHead(upstream.status,{'Content-Type':upstream.headers.get('content-type')||'application/octet-stream','Cache-Control':'no-store',...(upstream.headers.get('content-disposition')?{'Content-Disposition':upstream.headers.get('content-disposition')}:{})});
        if (upstream.body) await pipeline(Readable.fromWeb(upstream.body),res); else res.end(); return;
        } catch (error) { workers.invalidate(account.id); throw error; } finally { release(); }
      }
      if (req.method!=='GET') return respond(405,{error:'Method not allowed.'});
      const name=url.pathname==='/'?'index.html':decodeURIComponent(url.pathname.slice(1));
      const file=await inside(join(project,'dist'),name);
      res.writeHead(200,{'Content-Type':mime[extname(file)]||'application/octet-stream'});createReadStream(file).pipe(res);
    } catch(error) { if(!res.headersSent)respond(400,{error:error.message});else res.destroy(); }
  });
  return {server,close:async()=>{server.closeAllConnections();await new Promise(resolve=>server.close(resolve));await workers.close();store.close();}};
}
if (process.argv[1]===fileURLToPath(import.meta.url)) {
  const app=hostedServer();app.server.listen(Number(process.env.PORT||3000),'0.0.0.0');
  for(const signal of ['SIGINT','SIGTERM'])process.once(signal,async()=>{await app.close();process.exit(0);});
}
