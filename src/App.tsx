import { useEffect, useRef, useState, type FormEvent, type ComponentProps } from "react";
import { api, localConnection, validateConnection } from "./api";
import type { AgentId, Connection, DatasetInfo, ExportFormat, Metric, Policy, Provider, Run, Task, Tool } from "./types";

const defaults: Omit<Task, "agent"> = {
  dataset: "", target: "", objective: "", metric: "accuracy", minutes: 30, trials: 12,
  searchModels: true, model: "", output: "py", exportModel: true, exportFormat: "native",
  validation: "cv", folds: 3, seeds: [42], testFraction: .2, holdoutFraction: .2,
  splitStrategy: "independent", groupColumn: "", timeColumn: "", assetColumns: [],
  policy: { augmentation: false, regularization: true, features: true, tuning: true, pretrained: true, ensemble: true },
};
const strategies: [keyof Policy, string][] = [["augmentation", "Augmentation"], ["regularization", "Regularization"], ["features", "Feature engineering"], ["tuning", "Parameter tuning"], ["pretrained", "Pretrained models"], ["ensemble", "Ensembling"]];
const phases: Record<string, string> = { research: "Researching", baseline: "Training candidates", merge: "Comparing models", ablation: "Testing components", refinement: "Refining", ensemble: "Building ensemble", finalizing: "Finalizing", complete: "Complete" };
const duration = (seconds: number) => seconds >= 60 ? `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s` : `${Math.floor(seconds)}s`;
const score = (value: number | null | undefined) => value == null ? "—" : value.toFixed(4);
const datasetName = (path: string) => { const name = path.split(/[\\/]/).filter(Boolean).at(-1) || "Dataset"; return /^[a-f0-9-]{36}$/.test(name) ? "Attached dataset" : name; };
function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const paths: Record<string, string> = {
    plus: "M12 5v14M5 12h14", arrow: "M12 19V5m-5 5 5-5 5 5", chevron: "m8 10 4 4 4-4",
    folder: "M3 7h6l2 2h10v11H3zm0 0V4h6l2 3h10v2", file: "M6 3h8l4 4v14H6zM14 3v5h4M9 12h6M9 16h6",
    settings: "M4 7h9m4 0h3M4 17h3m4 0h9M13 4v6M7 14v6", pause: "M8 5v14M16 5v14", play: "m8 5 11 7-11 7z",
    stop: "M6 6h12v12H6z", close: "m6 6 12 12M18 6 6 18", check: "m5 12 4 4L19 6",
    download: "M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5", sidebar: "M3 4h18v16H3zM9 4v16", link: "M9 7H6a5 5 0 0 0 0 10h3m6-10h3a5 5 0 0 1 0 10h-3M8 12h8",
    trash: "M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7",
    attach: "m8 13 7-7a3 3 0 0 1 4 4l-9 9a5 5 0 0 1-7-7L13 2m-8 12 9-9",
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name] || paths.file} /></svg>;
}
function HelixMark({ size = 25 }: { size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 30" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><path d="M5 2c0 10 14 16 14 26M19 2c0 10-14 16-14 26M6 6h12M8 11h8M8 19h8M6 24h12" /></svg>;
}
function AgentMark({ agent }: { agent: AgentId }) {
  return <img className={`agent-logo ${agent}`} src={`/brands/${agent === "codex" ? "openai" : "claude"}.svg`} width="22" height="22" alt="" />;
}
function NumberField({ value, onChange, ...props }: Omit<ComponentProps<"input">, "value" | "onChange" | "type"> & { value: number; onChange: (value: number) => void }) {
  return <input {...props} type="number" required value={Number.isFinite(value) ? value : ""} onChange={event => onChange(event.currentTarget.valueAsNumber)} />;
}
const numberLabel = (value: number) => Number.isFinite(value) ? value : "—";
type Modal = "settings" | "connection" | "connections" | "run-settings" | null;
type Artifact = { path: string; size: number };

type CloudAccount = { connection: Connection; username: string; agentRevision: number; linkAgent: (agent: AgentId) => void; signOut: () => void; expired: () => void; signInMethods: {id: string; name: string; linked: boolean; connect: () => void}[] };
export default function App({ cloud }: { cloud?: CloudAccount }) {
  const [connection, setConnection] = useState<Connection | null>(() => {
    if (cloud) return cloud.connection;
    try { const value = JSON.parse(sessionStorage.getItem("helix-connection") || "null"); return value ? validateConnection(value.url, value.token) : null; } catch { return null; }
  });
  const [providers, setProviders] = useState<Provider[]>([]);
  const [tools, setTools] = useState<Tool[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<AgentId | "">("");
  const [runId, setRunId] = useState<string | null>(null);
  const [task, setTask] = useState(defaults);
  const [seedText, setSeedText] = useState("42");
  const [assetText, setAssetText] = useState("");
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [inspection, setInspection] = useState<{ key: string; data: DatasetInfo } | null>(null);
  const [busy, setBusy] = useState("");
  const [submission, setSubmission] = useState<{ task: Task; error?: string } | null>(null);
  const [connecting, setConnecting] = useState(!cloud);
  const [closed, setClosed] = useState(false);
  const [error, setError] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [modal, setModal] = useState<Modal>(() => window.matchMedia("(min-width: 1100px)").matches ? "settings" : null);
  const [sidebar, setSidebar] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [runnerUrl, setRunnerUrl] = useState("http://127.0.0.1:4319");
  const [pairCode, setPairCode] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const attachMenu = useRef<HTMLDetailsElement>(null);
  const agentMenu = useRef<HTMLDetailsElement>(null);
  const composer = useRef<HTMLFormElement>(null);
  const prompt = useRef<HTMLTextAreaElement>(null);
  const historyRevision = useRef(0);
  const run = runs.find((value) => value.id === runId);
  const pending = runId === "pending" ? submission : null;
  const sentTask = pending?.task || run?.task;
  const active = runs.find((value) => ["running", "queued"].includes(value.status));
  const enabled = !!connection && !!providers.find((value) => value.id === selected)?.authenticated;
  const selectedProvider = providers.find((value) => value.id === selected);
  const capability = selectedProvider?.capability;
  const checkingAgent = providers.some((value) => value.capability?.status === "checking");
  const runtime = tools.find((value) => value.id === "runtime");
  const running = !!run && ["running", "queued"].includes(run.status);
  const inspectionKey = JSON.stringify([task, seedText, selected]);
  const datasetInfo = inspection?.key === inspectionKey ? inspection.data : null;
  const readOnly = modal === "run-settings";
  const settings = readOnly && run ? { ...defaults, ...run.task } : task;
  const settingsVisible = modal === "settings" || readOnly;
  const seedCount = (readOnly ? settings.seeds.join(",") : seedText).split(",").filter((value) => /^\d+$/.test(value.trim())).length;
  const validationFits = settings.trials * (settings.validation === "cv" ? settings.folds : 1) * seedCount;
  const developmentPercent = 100 - Math.round(settings.testFraction * 100);
  const validationPercent = settings.validation === "holdout" ? Math.round(developmentPercent * settings.holdoutFraction * 10) / 10 : 0;
  const trainPercent = Math.round((developmentPercent - validationPercent) * 10) / 10;
  const bundle = artifacts.find((file) => file.path === "final/helix-solution.zip");
  const baselineGain = run?.baseline && run.score != null ? (run.score - run.baseline.score) * (["accuracy", "auroc"].includes(run.task.metric) ? 1 : -1) : null;
  const update = <K extends keyof Omit<Task, "agent">>(key: K, value: Omit<Task, "agent">[K]) => setTask((prev) => ({ ...prev, [key]: value }));

  const connectLocal = async () => {
    setConnecting(true);
    try {
      const next = await localConnection();
      if (next) {
        setConnection(next); setConnectionError("");
        try { sessionStorage.setItem("helix-connection", JSON.stringify(next)); } catch { /* Optional. */ }
      }
    } catch (e) { setConnectionError((e as Error).message); }
    finally { setConnecting(false); }
  };
  useEffect(() => { if (!cloud) void connectLocal(); }, []);
  useEffect(() => {
    if (!prompt.current || sentTask) return;
    prompt.current.style.height = "auto";
    prompt.current.style.height = `${prompt.current.scrollHeight}px`;
  }, [task.objective, !!sentTask]);
  useEffect(() => {
    if (!selected) setSelected(providers.find((value) => value.authenticated)?.id || "");
  }, [providers, selected]);
  useEffect(() => {
    if (!connection) return;
    let stopped = false, timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    const poll = async () => {
      const revision = historyRevision.current;
      const results = await Promise.allSettled([
        api<Provider[]>(connection, "/api/providers", { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(15_000)]) }),
        api<Run[]>(connection, "/api/runs", { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(15_000)]) }),
        api<Tool[]>(connection, "/api/tools", { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(15_000)]) }),
      ]);
      if (stopped) return;
      if (results[0].status === "fulfilled") { setProviders(results[0].value); setConnectionError(""); }
      else {
        setProviders([]); setConnectionError(cloud ? "Reconnecting to your workspace…" : "Helix disconnected. Reopen the app to reconnect.");
        if (connection.url === location.origin && results[0].reason?.status === 401) { if (cloud) cloud.expired(); else void connectLocal(); }
      }
      if (results[1].status === "fulfilled" && revision === historyRevision.current) setRuns(results[1].value);
      if (results[2].status === "fulfilled") setTools(results[2].value);
      timer = setTimeout(poll, 2000);
    };
    void poll();
    return () => { stopped = true; controller.abort(); clearTimeout(timer); };
  }, [connection, cloud?.agentRevision]);
  useEffect(() => {
    const desktop = window.matchMedia("(min-width: 1100px)");
    const show = () => {
      dialog.current?.close();
      if (!modal) return;
      if (settingsVisible && desktop.matches) dialog.current?.show();
      else dialog.current?.showModal();
    };
    show(); desktop.addEventListener("change", show);
    return () => desktop.removeEventListener("change", show);
  }, [modal, settingsVisible]);
  useEffect(() => {
    const closeMenus = (event: PointerEvent) => {
      for (const menu of [attachMenu, agentMenu]) if (!menu.current?.contains(event.target as Node)) menu.current?.removeAttribute("open");
    };
    document.addEventListener("pointerdown", closeMenus);
    return () => document.removeEventListener("pointerdown", closeMenus);
  }, []);
  useEffect(() => { folderInput.current?.setAttribute("webkitdirectory", ""); }, []);
  useEffect(() => {
    setArtifacts([]);
    if (!connection || !run) return;
    const controller = new AbortController();
    api<Artifact[]>(connection, `/api/runs/${runId}/artifacts`, { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(15_000)]) })
      .then(value => { if (!controller.signal.aborted) setArtifacts(value); }).catch((e) => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [connection, runId, run?.trials.length, run?.status]);

  const pair = async (event: FormEvent) => {
    event.preventDefault(); setBusy("pair"); setError("");
    try {
      const next = validateConnection(runnerUrl, pairCode.trim());
      const values = await api<Provider[]>(next, "/api/providers");
      setProviders(values); setConnection(next); setConnectionError("");
      if (!selected) setSelected(values.find((value) => value.authenticated)?.id || "");
      try { sessionStorage.setItem("helix-connection", JSON.stringify(next)); } catch { /* Connection works without storage. */ }
      setModal(window.matchMedia("(min-width: 1100px)").matches ? "settings" : null);
    } catch (e) { setError(e instanceof TypeError ? "Cannot reach the runner. Start it with npm start." : (e as Error).message); }
    finally { setBusy(""); }
  };
  const configuredTask = (): Task => {
    if (!selected) throw new Error("Choose Codex or Claude Code.");
    if (!task.dataset) throw new Error("Attach a dataset first.");
    if (!task.target.trim()) throw new Error("Enter the target column in Settings.");
    if (!task.objective.trim()) throw new Error("Describe your experiment.");
    if (!Number.isInteger(task.minutes) || task.minutes < 1 || task.minutes > 1440) throw new Error("Enter a time budget between 1 and 1440 minutes in Settings.");
    if (!Number.isInteger(task.trials) || task.trials < 3 || task.trials > 100) throw new Error("Enter a trial budget between 3 and 100 in Settings.");
    if (!Number.isFinite(task.testFraction) || (task.validation === "holdout" && !Number.isFinite(task.holdoutFraction))) throw new Error("Enter the split percentages in Settings.");
    const seeds = seedText.split(",").map((value) => Number(value.trim()));
    if (seedText.split(",").some((value) => !/^\d+$/.test(value.trim())) || seeds.length > 5 || new Set(seeds).size !== seeds.length || seeds.some((value) => !Number.isSafeInteger(value) || value < 0 || value > 2 ** 32 - 1)) throw new Error("Use up to five unique 32-bit seeds in Settings.");
    return { ...task, agent: selected, seeds };
  };
  const checkDataset = async () => {
    if (!connection || !enabled) return;
    setBusy("inspect"); setError("");
    try { const data = await api<DatasetInfo>(connection, "/api/datasets/inspect", { method: "POST", body: JSON.stringify(configuredTask()), signal: AbortSignal.timeout(125_000) }); setInspection({ key: inspectionKey, data }); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(""); }
  };
  const start = async (event: FormEvent) => {
    event.preventDefault(); setError("");
    if (!connection) { setModal("connection"); return; }
    if (active || busy || checkingAgent) return;
    if (runtime?.status !== "Ready") { setError(runtime?.description || "Start Docker and run npm run setup."); return; }
    setBusy("start");
    let submitted = false;
    try {
      const nextTask = configuredTask();
      if (!enabled) {
        const current = await api<Provider[]>(connection, "/api/providers?refresh=1");
        setProviders(current);
        if (!current.find(provider => provider.id === selected)?.authenticated) {
          if (cloud) { setModal(null); cloud.linkAgent(nextTask.agent); return; }
          throw new Error(current.find(provider => provider.id === selected)?.detail || "Connect your agent first.");
        }
      }
      setSubmission({ task: nextTask }); setRunId("pending"); update("objective", "");
      submitted = true;
      if (!window.matchMedia("(min-width: 1100px)").matches) setModal(null);
      const value = await api<Run>(connection, "/api/runs", { method: "POST", body: JSON.stringify(nextTask), signal: AbortSignal.timeout(360_000) });
      historyRevision.current++;
      setRuns((prev) => [value, ...prev.filter((saved) => saved.id !== value.id)]);
      setRunId(current => current === "pending" ? value.id : current); setSubmission(null);
    } catch (e) {
      if (submitted) setSubmission((prev) => prev ? { ...prev, error: (e as Error).message } : null);
      else setError((e as Error).message);
    }
    finally { setBusy(""); }
  };
  const action = async (name: string) => {
    if (!connection || !run) return;
    setBusy("action"); setError("");
    try { const result = await api<Run>(connection, `/api/runs/${run.id}/action`, { method: "POST", body: JSON.stringify({ action: name }), signal: AbortSignal.timeout(240_000) }); setRuns((prev) => prev.map((value) => value.id === result.id ? result : value)); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(""); }
  };
  const upload = async (files: FileList | File[] | null) => {
    if (!files?.length) return;
    if (!connection) { setModal("connection"); return; }
    if (busy || active) return;
    setBusy("upload"); setError(""); attachMenu.current?.removeAttribute("open");
    try {
      const dataset = await api<{ id: string; path: string }>(connection, "/api/datasets", { method: "POST", body: "{}" });
      for (const file of Array.from(files)) {
        const name = file.webkitRelativePath ? file.webkitRelativePath.split("/").slice(1).join("/") : file.name;
        const response = await fetch(`${connection.url}/api/datasets/${dataset.id}/file?name=${encodeURIComponent(name)}`, { method: "PUT", headers: { Authorization: `Bearer ${connection.token}` }, body: file, signal: AbortSignal.timeout(300_000) });
        if (!response.ok) throw new Error((await response.json()).error || "Upload failed.");
      }
      update("dataset", files.length === 1 && !files[0].webkitRelativePath ? `${dataset.path}/${files[0].name}` : dataset.path);
      requestAnimationFrame(() => prompt.current?.focus());
      return true;
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(""); if (fileInput.current) fileInput.current.value = ""; if (folderInput.current) folderInput.current.value = ""; }
  };
  const useExample = async () => {
    try {
      const response = await fetch("/examples/measurements.csv");
      if (!response.ok) throw new Error("Could not load the example dataset.");
      if (await upload([new File([await response.blob()], "measurements.csv", {type: "text/csv"})])) {
        setTask(prev => ({...defaults, dataset: prev.dataset, target: "label", objective: "Predict short or long from length and width", metric: "accuracy", minutes: 15, trials: 3, searchModels: false, model: "sklearn.tree.DecisionTreeClassifier", validation: "holdout", seeds: [42], policy: {augmentation: false, regularization: false, features: false, tuning: false, pretrained: false, ensemble: false}}));
        setSeedText("42"); setAssetText(""); setInspection(null); setModal(null);
      }
    } catch (e) { setError((e as Error).message); }
  };
  const download = async (path: string) => {
    if (!connection || !run) return;
    try {
      const response = await fetch(`${connection.url}/api/runs/${run.id}/file?path=${encodeURIComponent(path)}`, { headers: { Authorization: `Bearer ${connection.token}` }, signal: AbortSignal.timeout(60_000) });
      if (!response.ok) throw new Error((await response.json()).error);
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a"); link.href = url; link.download = path.split("/").pop()!; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) { setError((e as Error).message); }
  };
  const newRun = () => { if (busy === "start" || busy === "delete") return; setSubmission(null); if (readOnly) setModal("settings"); setRunId(null); setTask((prev) => ({ ...prev, objective: "", dataset: "", target: "" })); setError(""); setSidebar(false); requestAnimationFrame(() => prompt.current?.focus()); };
  const reuseRun = () => { if (!run) return; setSubmission(null); setTask({ ...defaults, ...run.task }); setSeedText(run.task.seeds.join(", ")); setAssetText((run.task.assetColumns || []).join(", ")); setSelected(run.task.agent); setRunId(null); setError(""); setModal("settings"); requestAnimationFrame(() => prompt.current?.focus()); };
  const openRun = (value: Run) => { if (busy === "delete") return; setRunId(value.id); setTask({ ...defaults, ...value.task, objective: "" }); setSeedText(value.task.seeds.join(", ")); setAssetText((value.task.assetColumns || []).join(", ")); setSelected(value.task.agent); setError(""); setSidebar(false); };
  const openSubmission = () => { if (!submission) return; setRunId("pending"); setTask({ ...submission.task, objective: "" }); setSeedText(submission.task.seeds.join(", ")); setAssetText(submission.task.assetColumns.join(", ")); setSelected(submission.task.agent); setError(""); setSidebar(false); };
  const deleteRun = async (value: Run) => {
    if (!connection || busy || !window.confirm(`Delete "${value.task.objective}"?\n\nThis permanently removes this experiment and its results. Uploaded datasets are kept.`)) return;
    setBusy("delete"); setError("");
    try {
      await api(connection, `/api/runs/${value.id}`, { method: "DELETE", signal: AbortSignal.timeout(60_000) });
      historyRevision.current++;
      setRuns(previous => previous.filter(item => item.id !== value.id));
      if (runId === value.id) { if (submission) openSubmission(); else newRun(); }
    } catch (e) { setError((e as Error).message); setSidebar(false); setModal(null); }
    finally { setBusy(""); }
  };
  const chooseUpload = (folder = false) => { attachMenu.current?.removeAttribute("open"); if (!connection) { setModal("connection"); return; } (folder ? folderInput : fileInput).current?.click(); };
  const editSubmission = () => {
    if (!submission) return;
    setTask(submission.task); setSelected(submission.task.agent); setSeedText(submission.task.seeds.join(", "));
    setAssetText(submission.task.assetColumns.join(", ")); setRunId(null); setSubmission(null); setError(""); requestAnimationFrame(() => prompt.current?.focus());
  };
  const quit = async () => {
    if (!connection?.desktop) return;
    setBusy("quit");
    try {
      await api(connection, "/api/quit", { method: "POST" });
      setConnection(null); setModal(null); setClosed(true);
      try { sessionStorage.removeItem("helix-connection"); } catch { /* Optional. */ }
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(""); }
  };

  if (closed) return <main className="closed-screen"><HelixMark size={40} /><h1>Helix is closed.</h1><p>You can close this tab. Open the Helix app to return.</p></main>;

  return <div className={`app-shell ${sidebar ? "sidebar-open" : ""} ${settingsVisible ? "setup-open" : ""}`}>
    {sidebar && <button className="sidebar-scrim" aria-label="Close sidebar" onClick={() => setSidebar(false)} />}
    <aside className="sidebar">
      <button className="brand" onClick={newRun} aria-label="Helix home"><HelixMark /><span>helix<span className="brand-dot">.</span></span></button>
      <button className="new-button" disabled={busy === "start" || busy === "delete"} onClick={newRun}><Icon name="plus" />New experiment</button>
      <div className="history-label">Experiments</div>
      <nav className="history" aria-label="Saved experiments">
      {submission && <div className={`history-row ${pending ? "selected" : ""}`}><button className="history-item" aria-current={pending ? "page" : undefined} onClick={openSubmission} title={submission.task.objective}><span className={`run-dot ${submission.error ? "failed" : "queued"}`} /><span className="history-title">{submission.task.objective}</span><span className="history-state">{submission.error ? "Failed" : "Starting"}</span></button></div>}
      {runs.map((value) => <div key={value.id} className={`history-row ${runId === value.id ? "selected" : ""}`}>
        <button className="history-item" disabled={busy === "delete"} aria-current={runId === value.id ? "page" : undefined} onClick={() => openRun(value)} title={value.task.objective}><span className={`run-dot ${value.status}`} /><span className="history-title">{value.task.objective}</span></button>
        {!["running", "queued"].includes(value.status) && <button className="history-delete" disabled={!!busy} aria-label={`Delete experiment: ${value.task.objective}`} title="Delete experiment" onClick={() => void deleteRun(value)}><Icon name="trash" size={15} /></button>}
      </div>)}</nav>
      <button className="connections-button" onClick={() => setModal("connections")}><Icon name="link" />Connections</button>
    </aside>
    <main className="workspace">
      <header className="topbar">
        <button className="icon-button mobile-menu" aria-label="Open sidebar" onClick={() => setSidebar(true)}><Icon name="sidebar" /></button>
        <span className="workspace-label">WORKSPACE <span>/</span></span><span className="conversation-title">{sentTask?.objective || "New experiment"}</span>
        <button className="connection-status" onClick={() => { setError(""); setModal("connections"); }}><span className={`connection-dot ${connection && !connectionError ? "online" : ""}`} />{connecting ? "Connecting…" : connection && !connectionError ? cloud ? "Cloud" : "Local" : "Connect"}</button>
        <button className={`icon-button setup-toggle ${settingsVisible ? "selected" : ""}`} aria-label="Toggle experiment setup" aria-expanded={settingsVisible} onClick={() => setModal(settingsVisible ? null : "settings")}><Icon name="settings" /></button>
        {cloud && <button className="account-avatar" aria-label={`Signed in as ${cloud.username}`} title={`Signed in as ${cloud.username}`} aria-haspopup="dialog" onClick={() => { setError(""); setModal("connections"); }}>{cloud.username.charAt(0).toUpperCase()}</button>}
      </header>
      <section className={`conversation ${sentTask ? "has-run" : "is-empty"}`} aria-label="Experiment conversation">
        {pending ? <div className="thread">
          <div className="experiment-heading"><span>NEW EXPERIMENT</span><span>{pending.error ? "Needs attention" : "Starting"}</span></div>
          <article className="user-message"><span className="brief-label">SENT <Icon name="check" size={12} /></span><p>{pending.task.objective}</p><span className="message-file"><Icon name="file" size={15} />{datasetName(pending.task.dataset)}</span></article>
          <article className="assistant-message"><div className="assistant-avatar"><HelixMark size={22} /></div><div className="assistant-content" role="status" aria-live="polite">
            <div className="response-heading">{pending.error ? "Startup needs attention" : "Preparing your experiment"}{!pending.error && <span className="working-dot" />}</div>
            {pending.error ? <><p className="inline-error">{pending.error}</p><button className="text-button" disabled={!!active || !!busy} onClick={editSubmission}>Edit message</button></> : <p className="submission-progress">{capability?.status === "checking" ? capability.detail : capability?.status === "verified" ? "Checking data and starting the run…" : "Checking data and getting your agent ready…"}</p>}
          </div></article>
        </div> : run ? <div className="thread" key={run.id}>
          <div className="experiment-heading"><span>EXPERIMENT / {run.id.slice(0, 8)}</span><span>{run.status}</span></div>
          <article className="user-message"><span className="brief-label">OBJECTIVE</span><p>{run.task.objective}</p><span className="message-file"><Icon name="file" size={15} />{datasetName(run.task.dataset)}</span></article>
          <article className="assistant-message">
            <div className="assistant-avatar"><HelixMark size={22} /></div>
            <div className="assistant-content">
              <div className="response-heading"><span>{run.status === "completed" ? "Experiment complete" : running ? run.progress ? `Trial ${run.progress.trial} · ${run.progress.stage}` : phases[run.phase] || run.phase : run.status === "paused" ? "Experiment paused" : run.status === "stopped" ? "Experiment stopped" : "Experiment failed"}</span>{running && <span className="working-dot" />}</div>
              <div className="run-meta">{duration(run.elapsed)}<span>·</span>{run.trials.length} completed {run.trials.length === 1 ? "trial" : "trials"}{running && run.progress?.stage === "Training" && <><span>·</span>{run.progress.completedFits}/{run.progress.totalFits} fits</>}<button onClick={() => setModal("run-settings")} aria-label="View experiment settings"><Icon name="settings" size={14} /></button></div>
              {run.score != null && <div className="result-metrics"><div><span>Validation {run.task.metric}</span><strong>{score(run.score)}</strong></div>{run.testScore != null && <div><span>Test {run.task.metric}</span><strong>{score(run.testScore)}</strong></div>}</div>}
              {run.baseline && <details className="baseline-comparison"><summary><span>{baselineGain != null && baselineGain > 0 ? "Beats baseline" : "Baseline not beaten"}</span><span>Baseline {score(run.baseline.score)}</span></summary><p>{run.baseline.name}, fitted on training labels for each split. Same validation rows and {run.task.metric} metric as the candidates.{run.testBaseline && ` Test baseline: ${score(run.testBaseline.score)}.`}</p></details>}
              {run.error && <div className="inline-error" role="alert">{run.error}</div>}
              <div className="run-actions">
                {bundle && <button className="download-button" onClick={() => void download(bundle.path)}><Icon name="download" size={16} />Download solution</button>}
                {running && <><button disabled={!!busy || run.phase === "finalizing"} onClick={() => void action("pause")}><Icon name="pause" size={15} />Pause</button><button disabled={!!busy} onClick={() => void action("stop")}><Icon name="stop" size={15} />Stop</button></>}
                {["paused", "failed"].includes(run.status) && <><button disabled={!enabled || !!busy || !!active || checkingAgent} onClick={() => void action("resume")}><Icon name="play" size={15} />Resume</button><button disabled={!!busy} onClick={() => void action("stop")}>Stop</button></>}
                {!running && <button disabled={!!busy || !!active} onClick={reuseRun}>Use these settings</button>}
              </div>
              {!!run.trials.length && <details className="response-details" open><summary>Trials <span>{run.trials.length}</span></summary><div className="trial-list"><div className="trial-columns"><span>Candidate</span><span>{run.task.metric}</span></div>{run.trials.map((trial, index) => <details className="trial" key={trial.id}><summary><span><i>{String(index + 1).padStart(2, "0")}</i>{trial.name}</span><span className={trial.status === "accepted" ? "accepted" : ""}>{score(trial.score)}</span></summary><div><p>{trial.detail}</p>{trial.deviation != null && <p className="muted">Fold standard deviation: {trial.deviation.toFixed(4)}</p>}<button className="text-button" onClick={() => void download(`trials/${trial.artifact}`)}><Icon name="download" size={14} />Source</button></div></details>)}</div></details>}
              <details className="response-details"><summary>Activity <span>{run.logs.length}</span></summary><div className="activity-log">{run.logs.map((entry, index) => <div className={`log-entry ${entry.type}`} key={`${entry.time}-${index}`}><time>{new Date(entry.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time><p>{entry.message}</p></div>)}</div></details>
              {!!artifacts.length && <details className="response-details"><summary>Files <span>{artifacts.length}</span></summary><div className="artifact-list">{artifacts.map((file) => <button key={file.path} onClick={() => void download(file.path)}><Icon name="file" size={15} /><span>{file.path}</span><Icon name="download" size={14} /></button>)}</div></details>}
            </div>
          </article>
        </div> : <div className="welcome"><div className="specimen-mark"><span className="specimen-axis">Y</span><HelixMark size={52} /><span className="specimen-axis">X</span></div><div className="welcome-label">NEW EXPERIMENT</div><h1>What do you want<br /> to predict?</h1><button className="attach-start" onClick={() => chooseUpload()}><Icon name="plus" size={16} />Attach a dataset</button>{connection && <button className="text-button example-start" disabled={!!busy || !!active} onClick={() => void useExample()}>Try an example</button>}</div>}
      </section>
      <div className={`composer-dock ${sentTask ? "is-collapsed" : ""}`}>
        <div className="composer-width">
          {submission && !pending ? <button className="active-notice" onClick={openSubmission}>{!submission.error && <span className="working-dot" />}{submission.error ? "An experiment needs attention" : "An experiment is starting"}<Icon name="chevron" size={14} /></button> : active && active.id !== runId && <button className="active-notice" onClick={() => openRun(active)}><span className="working-dot" />An experiment is running<Icon name="chevron" size={14} /></button>}
          {(error || connectionError) && !modal && <div className="composer-error" role="alert">{error || connectionError}<button aria-label="Dismiss error" onClick={() => { setError(""); setConnectionError(""); }}><Icon name="close" size={14} /></button></div>}
          {sentTask && <div className="run-dock"><span><AgentMark agent={sentTask.agent} />{sentTask.agent === "claude" ? "Claude Code" : "Codex"}<span className="run-dock-status">{pending ? pending.error ? "Needs attention" : "Starting" : run?.status}</span></span><button onClick={() => setModal(run ? "run-settings" : "settings")}><Icon name="settings" size={16} />View setup</button></div>}
          <div hidden={!!sentTask}>
          <form ref={composer} className={`composer ${dragging ? "dragging" : ""}`} onSubmit={start}
            onDragOver={(e) => { if (e.dataTransfer.types.includes("Files")) { e.preventDefault(); setDragging(true); } }}
            onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragging(false); }}
            onDrop={(e) => { e.preventDefault(); setDragging(false); void upload(e.dataTransfer.files); }}>
            {task.dataset && <div className="attachment-row"><span className="dataset-chip" title={task.dataset}><Icon name="file" size={16} /><span>{datasetName(task.dataset)}</span><button type="button" aria-label="Remove dataset" disabled={!!active || !!busy} onClick={() => update("dataset", "")}><Icon name="close" size={13} /></button></span></div>}
            <textarea ref={prompt} aria-label="Experiment message" rows={1} value={task.objective} disabled={!!active || !!busy || !!submission} placeholder={busy === "upload" ? "Uploading data…" : "Describe your experiment…"} maxLength={4000} onChange={(e) => update("objective", e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); composer.current?.requestSubmit(); } }} />
            <div className="composer-toolbar">
              <details className="attach-menu" ref={attachMenu}><summary title="Attach data" aria-label="Attach data"><Icon name="plus" size={21} /></summary><div className="attach-options"><button type="button" disabled={!!active || !!busy} onClick={() => chooseUpload()}><Icon name="file" />Upload CSV</button><button type="button" disabled={!!active || !!busy} onClick={() => chooseUpload(true)}><Icon name="folder" />Upload folder</button>{!cloud && <button type="button" onClick={() => { attachMenu.current?.removeAttribute("open"); setModal("settings"); }}><Icon name="link" />Local path</button>}</div></details>
              <button className="settings-button" type="button" aria-label="Experiment settings" onClick={() => setModal("settings")}><Icon name="settings" size={17} /><span>Settings</span></button>
              <div className="composer-spacer" />
              <details className="agent-picker" ref={agentMenu} onKeyDown={(e) => { if (e.key === "Escape") { e.preventDefault(); agentMenu.current?.removeAttribute("open"); agentMenu.current?.querySelector("summary")?.focus(); } }}>
                <summary aria-label="Choose coding agent">{selected && <AgentMark agent={selected} />}<span>{selected === "claude" ? "Claude Code" : selected === "codex" ? "Codex" : "Choose agent"}</span><Icon name="chevron" size={13} /></summary>
                <div className="agent-options" role="radiogroup" aria-label="Coding agent">{(["claude", "codex"] as const).map((agent) => <label key={agent}><input type="radio" name="agent" value={agent} checked={selected === agent} disabled={!!active || !!busy} onChange={() => { setSelected(agent); setError(""); agentMenu.current?.removeAttribute("open"); agentMenu.current?.querySelector("summary")?.focus(); if (!connection) setModal("connection"); else if (cloud && !providers.find(p => p.id === agent)?.authenticated) { setModal(null); cloud.linkAgent(agent); } }} /><AgentMark agent={agent} /><span>{agent === "claude" ? "Claude Code" : "OpenAI Codex"}</span>{selected === agent && <Icon name="check" size={15} />}</label>)}</div>
              </details>
              <button className="send-button" type="submit" aria-label="Run experiment" title="Run experiment" disabled={!!active || !!busy || checkingAgent || !task.objective.trim()}><Icon name="arrow" size={20} /></button>
            </div>
          </form>
          {selected && connection && !enabled && !connectionError && <button className="signin-notice" onClick={() => setModal("connections")}>{cloud ? `Connect ${selected === "codex" ? "Codex" : "Claude Code"}` : providers.find((value) => value.id === selected)?.detail || "Checking agent…"}</button>}
          {enabled && !submission && <button className={`agent-readiness ${capability?.status || "unchecked"}`} onClick={() => setModal("connections")}><span className="connection-dot" />{busy === "verify" || capability?.status === "checking" ? capability?.status === "checking" ? capability.detail : "Checking agent…" : capability?.status === "verified" ? "Agent verified" : capability?.status === "failed" ? "Agent check failed · View details" : "Agent check runs before your first experiment"}</button>}
          </div>
          <input ref={fileInput} type="file" accept=".csv" hidden onChange={(e) => void upload(e.target.files)} />
          <input ref={folderInput} type="file" multiple hidden onChange={(e) => void upload(e.target.files)} />
        </div>
      </div>
    </main>
    <dialog ref={dialog} className={`modal ${settingsVisible ? "setup-panel" : ""} ${modal === "connection" ? "connection-modal" : ""}`} onCancel={() => setModal(null)} onClick={(e) => { if (e.target === e.currentTarget) setModal(null); }} aria-labelledby="modal-title">
      {modal && <div className="modal-content">
        <header className="modal-header"><h2 id="modal-title">{modal === "connection" ? "Advanced connection" : modal === "connections" ? "Connections" : readOnly ? "Saved configuration" : "Experiment setup"}</h2>{settingsVisible && <span className="setup-badge">{readOnly ? "SAVED" : "DRAFT"}</span>}<button className="icon-button" aria-label="Close dialog" onClick={() => setModal(null)}><Icon name="close" /></button></header>
        {modal === "connection" ? <form onSubmit={pair} className="connection-form">
          <code>npm start</code><label>Runner address<input required value={runnerUrl} onChange={(e) => setRunnerUrl(e.target.value)} /></label><label>Pairing code<input required type="password" autoComplete="off" placeholder="From your terminal" value={pairCode} onChange={(e) => setPairCode(e.target.value)} /></label>
          {error && <div className="inline-error" role="alert">{error}</div>}
          <button className="primary" disabled={!!busy}>{busy === "pair" ? "Connecting…" : "Connect"}</button>
          {connection && <button className="text-button" type="button" onClick={() => { setConnection(null); setProviders([]); setTools([]); setRuns([]); setRunId(null); setSelected(""); setConnectionError(""); setModal(null); sessionStorage.removeItem("helix-connection"); }}>Disconnect</button>}
        </form> : modal === "connections" ? <div className="connection-list">
          <div className="runner-row"><span>{cloud ? `Cloud · ${cloud.username}` : "On this computer"}</span><span>{connection && !connectionError ? <><Icon name="check" size={14} />Connected</> : <button disabled={connecting} onClick={() => cloud ? location.reload() : void connectLocal()}>{connecting ? "Connecting…" : "Reconnect"}</button>}</span></div>
          {providers.map(provider => <div className="provider-check" key={provider.id}>
            <div className="tool-row"><AgentMark agent={provider.id} /><div><strong>{provider.name}</strong><p>{cloud && !provider.authenticated ? "Connect your subscription" : provider.detail}</p></div>
              {cloud && !provider.authenticated ? <button className="verify-button" disabled={!!active || !!busy || checkingAgent} onClick={() => { setModal(null); cloud.linkAgent(provider.id); }}>Connect</button> : provider.authenticated && <span className="provider-connected"><Icon name="check" size={14} />Connected</span>}
            </div>
            {provider.authenticated && provider.capability?.status === "checking" && <p className="verification-detail" role="status">{provider.capability.detail}</p>}
            {provider.authenticated && provider.capability?.status === "failed" && <div className="provider-failure"><p role="alert">{provider.name} couldn't start the experiment. Try sending it again.</p><details><summary>Technical details</summary><pre>{provider.capability.detail}</pre></details></div>}
          </div>)}
          {tools.map((tool) => <details className="tool-detail" key={tool.id}><summary>{tool.name}<span>{tool.status}</span></summary><p>{tool.description}</p></details>)}
          {cloud?.signInMethods.map(method => <div className="runner-row" key={method.id}><span>{method.name} sign-in</span>{method.linked ? <span><Icon name="check" size={14} />Linked</span> : <button onClick={method.connect}>Link account</button>}</div>)}
          {cloud ? <button className="text-button" onClick={cloud.signOut}>Sign out</button> : <button className="text-button advanced-connection" onClick={() => { setError(""); setModal("connection"); }}>Advanced connection</button>}
          {connection?.desktop && <button className="text-button" disabled={!!busy} onClick={() => void quit()}>{active ? "Pause experiment & quit Helix" : "Quit Helix"}</button>}
        </div> : <form onSubmit={(e) => { e.preventDefault(); setModal(null); }}>
          <div className="settings-body"><fieldset disabled={readOnly || !!active || !!busy}>
            <section className="settings-section">
              <h3><span>01</span> Dataset</h3>
              {settings.dataset && <div className="setup-dataset"><Icon name="file" size={17} /><span title={settings.dataset}>{datasetName(settings.dataset)}</span>{datasetInfo && <span>{datasetInfo.rows.toLocaleString()} rows</span>}</div>}
              {!readOnly && <div className="dataset-actions"><button type="button" onClick={() => chooseUpload()}><Icon name="plus" size={14} />{settings.dataset ? "Replace CSV" : "Choose CSV"}</button><button type="button" onClick={() => chooseUpload(true)}><Icon name="folder" size={14} />Choose folder</button></div>}
              {!cloud && <details className="settings-detail dataset-path"><summary>Use a local path</summary><label>CSV or folder path<input value={settings.dataset} onChange={(e) => update("dataset", e.target.value)} placeholder="/path/to/data.csv" /></label></details>}
              <div className="field-grid"><label>Target column<input value={settings.target} onChange={(e) => update("target", e.target.value)} placeholder="label" /></label><label>Metric<select value={settings.metric} onChange={(e) => update("metric", e.target.value as Metric)}><option value="accuracy">Accuracy</option><option value="auroc">AUROC</option><option value="log_loss">Log loss</option><option value="rmse">RMSE</option><option value="mae">MAE</option></select></label></div>
              <details className="settings-detail"><summary>Asset columns</summary><label>Columns containing file paths<input value={readOnly ? settings.assetColumns.join(", ") : assetText} onChange={(e) => { setAssetText(e.target.value); update("assetColumns", e.target.value.split(",").map((s) => s.trim()).filter(Boolean)); }} placeholder="image, audio, text_file" /></label></details>
            </section>
            <details className="settings-section settings-group">
              <summary><h3><span>02</span> Models & strategies</h3><span className="group-value">{settings.searchModels ? "Auto" : "Custom"}</span></summary>
              <div className="model-choice" role="radiogroup" aria-label="Model selection"><label><input type="radio" name="model-search" checked={settings.searchModels} onChange={() => update("searchModels", true)} /><span>Auto search</span></label><label><input type="radio" name="model-search" checked={!settings.searchModels} onChange={() => update("searchModels", false)} /><span>Choose model</span></label></div>
              <label>{settings.searchModels ? "Model constraints" : "Model or family"}<input value={settings.model} onChange={(e) => update("model", e.target.value)} placeholder={settings.searchModels ? "Any model, or add a constraint" : "e.g. tree-based models only"} /></label>
              <div className="strategy-grid">{strategies.map(([key, label]) => <label key={key} className="strategy-option"><input type="checkbox" aria-label={label} checked={settings.policy[key]} onChange={(e) => update("policy", { ...task.policy, [key]: e.target.checked })} /><span>{label}</span></label>)}</div>
            </details>
            <section className="settings-section">
              <h3><span>03</span> Evaluation</h3>
              <div className="field-grid"><label>Validation<select value={settings.validation} onChange={(e) => update("validation", e.target.value as Task["validation"])}><option value="cv">Cross-validation</option><option value="holdout">Fixed split</option></select></label><label>Test split (%)<NumberField min="0" max="99" step="1" placeholder="e.g. 20" value={Math.round(settings.testFraction * 100)} onChange={value => update("testFraction", value / 100)} /></label></div>
              <div className="split-preview" role="img" aria-label={settings.validation === "cv" ? `${numberLabel(developmentPercent)}% training, ${numberLabel(100 - developmentPercent)}% test` : `${numberLabel(trainPercent)}% training, ${numberLabel(validationPercent)}% validation, ${numberLabel(100 - developmentPercent)}% test`}><span className="split-train" style={{ flexGrow: Number.isFinite(trainPercent) ? Math.max(0, trainPercent) : 0 }} /><span className="split-validation" style={{ flexGrow: Number.isFinite(validationPercent) ? Math.max(0, validationPercent) : 0 }} /><span className="split-test" style={{ flexGrow: Number.isFinite(developmentPercent) ? Math.max(0, 100 - developmentPercent) : 0 }} /></div>
              <div className="split-legend"><span><i className="split-train" />Train <b>{numberLabel(trainPercent)}%</b></span>{settings.validation === "holdout" && <span><i className="split-validation" />Validate <b>{numberLabel(validationPercent)}%</b></span>}<span><i className="split-test" />Test <b>{numberLabel(100 - developmentPercent)}%</b></span></div>
              <div className="field-grid">{settings.validation === "cv" ? <label>Folds<select value={settings.folds} onChange={(e) => update("folds", Number(e.target.value))}>{[3, 5, 10].map((fold) => <option key={fold} value={fold}>{fold} folds</option>)}</select></label> : <label>Validation (% of train)<NumberField min="1" max="99" step="1" placeholder="e.g. 20" value={Math.round(settings.holdoutFraction * 100)} onChange={value => update("holdoutFraction", value / 100)} /></label>}<label>Seed<input value={readOnly ? settings.seeds.join(", ") : seedText} onChange={(e) => setSeedText(e.target.value)} placeholder="42" title="Actual seed values. Separate multiple values with commas." /></label></div>
              <label>Split by<select value={settings.splitStrategy} onChange={(e) => update("splitStrategy", e.target.value as Task["splitStrategy"])}><option value="independent">Independent rows</option><option value="group">Groups</option><option value="time">Time</option></select></label>
              {settings.splitStrategy === "group" && <label>Group column<input value={settings.groupColumn} onChange={(e) => update("groupColumn", e.target.value)} /></label>}{settings.splitStrategy === "time" && <label>Time column<input value={settings.timeColumn} onChange={(e) => update("timeColumn", e.target.value)} /></label>}
            </section>
            <section className="settings-section">
              <h3><span>04</span> Run budget</h3>
              <div className="field-grid"><label>Minutes<NumberField min="1" max="1440" placeholder="e.g. 30" value={settings.minutes} onChange={value => update("minutes", value)} /></label><label>Max trials<NumberField min="3" max="100" placeholder="e.g. 12" value={settings.trials} onChange={value => update("trials", value)} /></label></div>
              <div className="workload"><span>Validation fits</span><strong>up to {numberLabel(validationFits)}</strong></div>
            </section>
            <details className="settings-section settings-group">
              <summary><h3><span>05</span> Deliverables</h3><span className="group-value">.{settings.output}</span></summary>
              <label>Source<select value={settings.output} onChange={(e) => update("output", e.target.value as Task["output"])}><option value="py">Python (.py)</option><option value="ipynb">Notebook (.ipynb)</option></select></label>
              <label className="check-label"><input type="checkbox" checked={settings.exportModel} onChange={(e) => update("exportModel", e.target.checked)} /><span>Export trained model</span></label>
              {settings.exportModel && <label>Model format<select value={settings.exportFormat} onChange={(e) => update("exportFormat", e.target.value as ExportFormat)}><option value="native">Native</option><option value="joblib">Joblib</option><option value="pickle">Pickle</option><option value="pytorch">PyTorch state_dict</option><option value="torchscript">TorchScript</option><option value="keras">Keras</option><option value="savedmodel">SavedModel</option><option value="onnx">ONNX</option></select></label>}
            </details>
          </fieldset>
          {(error || connectionError) && <div className="inline-error" role="alert">{error || connectionError}</div>}
          {!readOnly && datasetInfo && <div className="inspection" role="status"><Icon name="check" size={16} />{datasetInfo.rows.toLocaleString()} rows · {datasetInfo.developmentRows.toLocaleString()} train · {datasetInfo.testRows.toLocaleString()} test</div>}
          </div><footer className="modal-footer">{!readOnly && <button type="button" className="text-button" disabled={!enabled || !!busy || !!active} onClick={() => void checkDataset()}>{busy === "inspect" ? "Checking…" : "Check dataset"}</button>}<button className="primary">{readOnly ? "Close" : "Done"}</button></footer>
        </form>}
      </div>}
    </dialog>
  </div>;
}
