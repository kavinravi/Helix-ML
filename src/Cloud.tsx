import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import App from "./App";
import { api } from "./api";
import type { AgentId, Connection } from "./types";

type SignInMethod = { id: "github" | "google"; name: string; linked: boolean };
type Session = { username: string; csrf: string };
type Login = { agent: AgentId; status: "waiting" | "connected" | "failed"; url: string | null; code: string | null };
export default function Cloud() {
  const [methods, setMethods] = useState<SignInMethod[]>([]);
  const [notice, setNotice] = useState(() => new URLSearchParams(location.search).get("signin") === "failed" ? "Sign-in did not finish. Please try again. If you already have a Helix account, sign in first and link it in Connections." : new URLSearchParams(location.search).get("signin") === "linked" ? "Sign-in method linked to your workspace." : "");
  const [hosted, setHosted] = useState<boolean | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [register, setRegister] = useState(false);
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const [linking, setLinking] = useState<AgentId | null>(null);
  const [login, setLogin] = useState<Login | null>(null);
  const [code, setCode] = useState("");
  const [retry, setRetry] = useState(0);
  const [loginAttempt, setLoginAttempt] = useState(0);
  const dialog = useRef<HTMLDialogElement>(null);
  const connection = useMemo<Connection | null>(() => session ? {url: location.origin, token: session.csrf} : null, [session]);
  useEffect(() => {
    let stopped = false;
    void (async () => {
      const health = await fetch("/health").then(r => r.json()).catch(() => null);
      if (health?.mode === "hosted") {
        const [value, available] = await Promise.all([
          fetch("/account/session").then(r => r.ok ? r.json() : null).catch(() => null),
          fetch("/account/methods").then(r => r.ok ? r.json() : []).catch(() => []),
        ]);
        if (!stopped) { setSession(value); setMethods(available); setHosted(true); }
        if (new URLSearchParams(location.search).has("signin")) history.replaceState(null, "", location.pathname);
      } else if (!stopped) setHosted(false);
    })();
    return () => { stopped = true; };
  }, []);
  useEffect(() => {
    setReady(false);
    if (!connection) return;
    const controller = new AbortController();
    setError("");
    void api(connection, "/account/workspace", {method: "POST", signal: AbortSignal.any([controller.signal, AbortSignal.timeout(360_000)])})
      .then(() => { if (!controller.signal.aborted) setReady(true); })
      .catch(e => { if (!controller.signal.aborted) { if (e.status === 401) setSession(null); else setError(e.message); } });
    return () => controller.abort();
  }, [connection, retry]);
  useEffect(() => {
    if (linking) dialog.current?.showModal(); else dialog.current?.close();
  }, [linking]);
  useEffect(() => {
    if (!linking || !connection) return;
    let stopped = false, timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const value = await api<Login | null>(connection, "/account/provider");
        if (stopped) return;
        if (value?.agent === linking) setLogin(value);
        if (value?.status === "connected") { setLinking(null); return; }
        if (value?.status === "failed") { setError("Sign-in did not finish. Please try again."); return; }
      } catch (e) { if (!stopped) setError((e as Error).message); }
      if (!stopped) timer = setTimeout(poll, 2000);
    };
    timer = setTimeout(poll, 1000);
    return () => { stopped = true; clearTimeout(timer); };
  }, [linking, connection, loginAttempt]);
  const signIn = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault(); setBusy(true); setError("");
    const form = new FormData(event.currentTarget);
    try {
      const response = await fetch("/account/login", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({username: form.get("username"), password: form.get("password"), register})});
      const value = await response.json();
      if (!response.ok) throw new Error(value.error);
      setSession(value); setRegister(false);
      setMethods(await fetch("/account/methods").then(r => r.json()));
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };
  const socialSignIn = async (method: SignInMethod) => {
    setBusy(true); setError(""); setNotice("");
    try {
      const response = await fetch(`/account/oauth/${method.id}`, {method: "POST", headers: connection ? {Authorization: `Bearer ${connection.token}`} : {}});
      const value = await response.json();
      if (!response.ok) throw new Error(value.error);
      location.assign(value.url);
    } catch (e) { setError((e as Error).message); setBusy(false); }
  };
  const signOut = async () => {
    if (!connection) return;
    try { await api(connection, "/account/logout", {method: "POST"}); setSession(null); setError(""); }
    catch (e) { setError((e as Error).message); }
  };
  const linkAgent = async (agent: AgentId) => {
    if (!connection) return;
    setLogin(null); setCode(""); setError(""); setLinking(agent); setLoginAttempt(v => v + 1);
    try { setLogin(await api<Login>(connection, "/account/provider", {method: "POST", body: JSON.stringify({agent})})); }
    catch (e) { setError((e as Error).message); }
  };
  const closeLogin = () => {
    setLinking(null); setError("");
    if (connection) void api(connection, "/account/provider", {method: "DELETE"}).catch(() => {});
  };
  const confirmCode = async (event: FormEvent) => {
    event.preventDefault(); if (!connection) return;
    setBusy(true); setError("");
    try { await api(connection, "/account/provider", {method: "POST", body: JSON.stringify({code})}); setCode(""); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };
  if (hosted === false) return <App />;
  if (hosted === null) return <main className="account-page"><span className="account-wordmark">helix.</span><p role="status">Opening workspace…</p></main>;
  if (!session) return <main className="account-page"><section className="account-card">
    <a className="account-wordmark" href="/">helix<span>.</span></a>
    <h1>{register ? "Create your workspace" : "Welcome back"}</h1>
    <p>ML experiments with your Codex or Claude Code subscription.</p>
    {!!methods.length && <><div className="social-signin">{methods.map(method => <button className={`social-button ${method.id}`} key={method.id} disabled={busy} onClick={() => void socialSignIn(method)}><img src={`/brands/${method.id}.svg`} alt="" width="20" height="20" />Continue with {method.name}</button>)}</div><div className="signin-divider">or</div></>}
    {notice && <p className="inline-error" role="status">{notice}</p>}
    <form onSubmit={signIn}>
      <label>Username<input name="username" autoComplete="username" required minLength={3} maxLength={32} pattern="[A-Za-z0-9]([A-Za-z0-9_]|-){2,31}" autoCapitalize="none" spellCheck={false} /></label>
      <label>Password<input name="password" type="password" autoComplete={register ? "new-password" : "current-password"} required minLength={12} maxLength={200} placeholder={register ? "At least 12 characters" : undefined} /></label>
      {error && <p className="inline-error" role="alert">{error}</p>}
      <button className="primary" disabled={busy}>{busy ? "Signing in…" : register ? "Create account" : "Sign in"}</button>
    </form>
    <button className="text-button account-switch" onClick={() => { setRegister(!register); setError(""); }}>{register ? "Already have an account? Sign in" : "New here? Create an account"}</button>
    <div className="account-agents"><img src="/brands/openai.svg" alt="OpenAI Codex" /><span>Codex</span><img src="/brands/claude.svg" alt="Claude Code" /><span>Claude Code</span></div>
  </section></main>;
  if (!ready || !connection) return <main className="account-page"><section className="account-card"><span className="account-wordmark">helix.</span><h1>Opening your workspace</h1><p role="status">Starting the training runtime. Your saved experiments will appear here.</p>{error && <><p role="alert" className="inline-error">{error}</p><button className="primary" onClick={() => setRetry(v => v + 1)}>Try again</button></>}<button className="text-button account-switch" onClick={() => void signOut()}>Sign out</button></section></main>;
  return <>
    <App cloud={{connection, username: session.username, linkAgent: agent => void linkAgent(agent), signOut: () => void signOut(), expired: () => setSession(null), signInMethods: methods.map(method => ({...method, connect: () => void socialSignIn(method)}))}} />
    {(error || notice) && !linking && <div className="cloud-error" role="status">{error || notice}<button onClick={() => { setError(""); setNotice(""); }}>Dismiss</button></div>}
    <dialog ref={dialog} className="modal provider-modal" onCancel={closeLogin} aria-labelledby="provider-title">
      <div className="modal-content"><header className="modal-header"><h2 id="provider-title">Connect {linking === "codex" ? "Codex" : "Claude Code"}</h2><button className="icon-button" aria-label="Close sign-in" onClick={closeLogin}>×</button></header>
        <div className="provider-login">
          {login?.url ? <>
            <p>Sign in with your {linking === "codex" ? "ChatGPT" : "Claude"} account.</p>
            {login.code && <div className="device-code"><span>Enter this code on the sign-in page</span><code>{login.code}</code></div>}
            <a className="primary provider-link" href={login.url} target="_blank" rel="noopener noreferrer">Open {linking === "codex" ? "OpenAI" : "Claude"} sign-in ↗</a>
            {linking === "claude" && <form onSubmit={confirmCode}><label>Confirmation code<input value={code} onChange={e => setCode(e.target.value)} autoComplete="off" placeholder="Paste the code from Claude" required maxLength={4096} /></label><button className="primary" disabled={busy || !code.trim()}>{busy ? "Connecting…" : "Finish connecting"}</button></form>}
            {linking === "codex" && <p className="login-wait" role="status">Waiting for sign-in…</p>}
          </> : !error && <p role="status">Preparing sign-in…</p>}
          {error && <><p className="inline-error" role="alert">{error}</p><button className="text-button" onClick={() => { if (linking) { void linkAgent(linking); } }}>Try again</button></>}
        </div>
      </div>
    </dialog>
  </>;
}
