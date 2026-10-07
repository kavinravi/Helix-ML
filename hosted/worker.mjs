// Runs inside one user's VM. Native CLIs own their credentials; Helix never reads them.
import { createServer, request as httpRequest } from 'node:http';
import { spawn } from 'node:child_process';
import { timingSafeEqual } from 'node:crypto';
import { stripVTControlCharacters } from 'node:util';
import { createService } from '../local/server.mjs';

const token = process.env.HELIX_WORKER_TOKEN;
if (!/^[A-Za-z0-9_-]{43}$/.test(token || '')) throw new Error('Worker authentication is required.');
const service = await createService({root:'/var/lib/helix',token,port:4319});
await new Promise(resolve=>service.server.listen(4319,'127.0.0.1',resolve));
let login;
const reply = (res,status,value) => { res.writeHead(status,{'Content-Type':'application/json','Cache-Control':'no-store'}); res.end(JSON.stringify(value)); };
const stopLogin = () => { if (login?.child?.pid) { try { process.kill(-login.child.pid,'SIGKILL'); } catch {} } };
const server = createServer(async (req,res) => {
  const actual = Buffer.from(req.headers.authorization || ''), expected = Buffer.from(`Bearer ${token}`);
  if (actual.length !== expected.length || !timingSafeEqual(actual,expected)) return reply(res,401,{error:'Sign in to your Helix workspace.'});
  try {
    if (req.url === '/native-login' && req.method === 'GET') return reply(res,200,login?.public || null);
    if (req.url === '/native-login' && req.method === 'DELETE') { stopLogin(); return reply(res,200,{cancelled:true}); }
    if (req.url === '/native-login' && req.method === 'POST') {
      const chunks=[]; let size=0;
      for await (const chunk of req) { if ((size+=chunk.length)>8000) throw new Error('Request too large'); chunks.push(chunk); }
      const body=JSON.parse(Buffer.concat(chunks).toString());
      if (body.code !== undefined) {
        if (!login?.child || login.public.status!=='waiting' || login.public.agent!=='claude' || typeof body.code!=='string' || !body.code.trim() || body.code.length>4096 || /[\r\n\0]/.test(body.code)) throw new Error('Start Claude Code sign-in first.');
        login.child.stdin.write(body.code.trim()+'\n');
        return reply(res,200,login.public);
      }
      if (!['codex','claude'].includes(body.agent)) throw new Error('Choose an agent.');
      stopLogin();
      const state = {public:{agent:body.agent,status:'waiting',url:null,code:null},buffer:''};
      login=state;
      const child=spawn(body.agent,body.agent==='codex' ? ['login','--device-auth'] : ['auth','login'],{stdio:['pipe','pipe','pipe'],detached:true});
      state.child=child;
      const output=chunk=>{
        state.buffer=(state.buffer+stripVTControlCharacters(chunk.toString())).slice(-16000);
        const urls=state.buffer.match(/https:\/\/[^\s<>\x00-\x1f]+/g)||[];
        for (const value of urls) {
          try {
            const url=new URL(value);
            const allowed=body.agent==='codex' ? ['auth.openai.com','chatgpt.com'] : ['claude.com','claude.ai','platform.claude.com','console.anthropic.com'];
            if (allowed.includes(url.hostname)) state.public.url=url.href;
          } catch {}
        }
        if (body.agent==='codex') state.public.code=state.buffer.match(/\b[A-Z0-9]{4}-[A-Z0-9]{4,5}\b/)?.[0] || null;
      };
      child.stdout.on('data',output); child.stderr.on('data',output);
      const timeout=setTimeout(()=>{ if (login===state) stopLogin(); },300_000);
      const finish=ok=>{ clearTimeout(timeout); state.buffer=''; state.child=null; state.public.status=ok?'connected':'failed'; state.public.url=null; state.public.code=null; };
      child.once('error',()=>finish(false)); child.once('close',code=>finish(code===0));
      return reply(res,200,state.public);
    }
    if (req.url==='/sleep' && req.method==='POST') {
      stopLogin(); await service.close(); reply(res,200,{saved:true});
      setTimeout(()=>server.close(()=>process.exit(0)),100).unref(); return;
    }
    if (!req.url?.startsWith('/api/') || req.url.startsWith('/api/session') || req.url.startsWith('/api/quit')) return reply(res,404,{error:'Not found'});
    const upstream=httpRequest({host:'127.0.0.1',port:4319,path:req.url,method:req.method,headers:{'Content-Type':req.headers['content-type']||'application/json',Authorization:`Bearer ${token}`}},remote=>{
      res.writeHead(remote.statusCode,remote.headers); remote.pipe(res);
    });
    upstream.on('error',()=>{ if (!res.headersSent) reply(res,502,{error:'Workspace unavailable.'}); else res.destroy(); });
    req.on('aborted',()=>upstream.destroy()); req.pipe(upstream);
  } catch (error) { if (!res.headersSent) reply(res,400,{error:error.message}); else res.destroy(); }
});
server.listen(8080,'0.0.0.0');
const close=async()=>{stopLogin();await service.close();server.close(()=>process.exit(0));};
process.once('SIGTERM',close);process.once('SIGINT',close);
