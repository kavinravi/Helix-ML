import { createServer } from "node:http";
import { randomBytes, randomUUID, timingSafeEqual } from "node:crypto";
import { mkdir, readdir, lstat, rm, rmdir, unlink, open, writeFile, readFile, rename } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { pipeline } from "node:stream/promises";
import { Transform } from "node:stream";
import { setTimeout as delay } from "node:timers/promises";
import { resolve, join, dirname, basename, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { providers } from "./agents.mjs";
import { loadRuns, snapshot, createRun, execute, saveRun, log, elapsed, discussRun, continueRun } from "./engine.mjs";
import { inside, redact, validateTask } from "./validate.mjs";
import { runProcess } from "./process.mjs";
import { runtimeStatus, cleanupContainers } from "./runtime.mjs";
import { checkAgentCapability, readinessKey } from "./readiness.mjs";
import { inspectDataset } from "./evaluation.mjs";

const project = fileURLToPath(new URL("..", import.meta.url));
const mime = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css",
  ".svg": "image/svg+xml",
  ".woff2": "font/woff2",
};

async function jsonBody(request) {
  const chunks = [];
  let length = 0;
  for await (const chunk of request) {
    length += chunk.length;
    if (length > 64_000) throw new Error("Request body must be smaller than 64 KB.");
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

async function lockRunner(root) {
  await mkdir(root, { recursive: true, mode: 0o700 });
  const folder = join(root, ".runner-lock"), owner = `${process.pid}-${randomUUID()}`;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      await mkdir(folder, { mode: 0o700 });
      await writeFile(join(folder, owner), "", { flag: "wx", mode: 0o600 });
      return async () => {
        await unlink(join(folder, owner));
        await rmdir(folder).catch(error => { if (!["ENOENT", "ENOTEMPTY"].includes(error.code)) throw error; });
      };
    } catch (error) { if (error.code !== "EEXIST" && error.code !== "ENOENT") throw error; }
    const entries = await readdir(folder).catch(error => { if (error.code === "ENOENT") return []; throw error; });
    for (const entry of entries) {
      if (!/^\d+-[a-f0-9-]{36}$/.test(entry)) throw new Error(`Unrecognized runner lock: ${folder}`);
      try { process.kill(Number(entry.split("-")[0]), 0); }
      catch (error) {
        if (error.code !== "ESRCH") throw new Error("Another Helix runner already owns this data folder.");
        await unlink(join(folder, entry)).catch(error => { if (error.code !== "ENOENT") throw error; });
        continue;
      }
      throw new Error("Another Helix runner already owns this data folder. Use its existing browser tab or stop it first.");
    }
    // Remove only an empty lock directory. A concurrent owner with a new file stays protected.
    await rmdir(folder).catch(error => { if (!["ENOENT", "ENOTEMPTY"].includes(error.code)) throw error; });
  }
  throw new Error("Another runner is starting. Retry in a moment.");
}

export async function createService({
  root = resolve(project, ".helix"),
  token = randomBytes(32).toString("base64url"),
  origins = [],
  port = 4319,
  desktop = false,
} = {}) {
  const release = await lockRunner(root);
  const runRoot = join(root, "runs"),
    uploads = join(root, "datasets");
  let runs;
  try {
    runs = await loadRuns(runRoot);
    await mkdir(uploads, { recursive: true, mode: 0o700 });
  } catch (error) { await release(); throw error; }
  const allowed = new Set([
    `http://127.0.0.1:${port}`,
    `http://localhost:${port}`,
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    ...origins,
  ]);
  const auth = (request) => {
    const supplied = String(request.headers.authorization || "").replace(
      /^Bearer /,
      "",
    );
    return (
      Buffer.byteLength(supplied) === Buffer.byteLength(token) &&
      timingSafeEqual(Buffer.from(supplied), Buffer.from(token))
    );
  };
  let stopping = false;
  const shutdown = new AbortController();
  // ponytail: one CPU experiment at a time; use a queue if concurrent runs become necessary.
  let active = null, starting = false, verification = null, chatting = null;
  const mutating = new Set();
  const capabilities = new Map();
  const launch = (run) => {
    const controller = new AbortController();
    const entry = { run, controller, promise: null };
    active = entry;
    entry.promise = execute(runRoot, run, controller.signal).catch(async (error) => {
      run.status = "failed";
      run.error = redact(error.message);
      await saveRun(runRoot, run);
    }).finally(() => { if (active === entry) active = null; });
  };
  let providerCache = null,
    providerPending = null,
    toolCache = null;
  const getProviders = async (refresh = false) => {
    if (refresh) await providerPending;
    if (refresh || !providerCache || Date.now() - providerCache.time > 10_000) {
      providerPending ||= providers().then(value => { providerCache = { time: Date.now(), value }; }).finally(() => { providerPending = null; });
      await providerPending;
    }
    return providerCache.value.map((provider) => {
      const previous = capabilities.get(provider.id);
      if (previous && previous.status !== "checking" && (!provider.authenticated || previous.version !== provider.version)) capabilities.delete(provider.id);
      const capability = capabilities.get(provider.id) || { status: "unchecked", detail: "Code writing and training have not been checked." };
      return { ...provider, capability };
    });
  };
  const verify = async (agent, { force = false, forRun = false } = {}) => {
    const provider = (await getProviders(true)).find((p) => p.id === agent);
    if (!provider?.installed || !provider.authenticated) throw new Error(process.env.HELIX_HOSTED_WORKER === "1" ? "Connect your agent in Connections, then send the experiment again." : "Sign in to the selected CLI before starting an experiment.");
    const runtime = await runtimeStatus();
    if (!runtime.ready) throw new Error(runtime.detail);
    if (active || (!forRun && (starting || chatting))) throw new Error("An experiment is starting or running. Wait before checking an agent.");
    if (verification) {
      if (verification.agent === agent) return verification.promise;
      throw new Error("Another agent check is running. Wait for it to finish.");
    }
    if (stopping) throw new Error("The runner is shutting down.");
    const cache = join(root, `verified-${agent}.json`), previousSession = capabilities.get(agent);
    const state = { status: "checking", detail: "Preparing the agent check…", version: provider.version, image: runtime.imageId };
    capabilities.set(agent, state);
    const entry = { agent, controller: new AbortController(), promise: null };
    verification = entry;
    entry.promise = (async () => {
      const key = await readinessKey(provider.version, runtime.imageId);
      const previous = previousSession || await readFile(cache, "utf8").then(JSON.parse).catch(() => null);
      const age = Date.now() - Date.parse(previous?.verifiedAt);
      if (!force && previous?.status === "verified" && previous.key === key && age >= 0 && age < 7 * 86_400_000 &&
          Array.isArray(previous.checks) && ["mcp", "write", "train", "predict", "reload"].every(check => previous.checks.includes(check)) &&
          Number.isFinite(previous.score) && previous.score >= .8 && /^[a-f0-9]{64}$/.test(previous.sourceHash)) {
        capabilities.set(agent, previous);
        return previous;
      }
      // Invalidate before rechecking so a failed forced check stays invalid after restart.
      await rm(cache, { force: true });
      if (entry.controller.signal.aborted) throw new Error("Interrupted");
      const result = await checkAgentCapability(join(root, "checks"), agent, {
        image: runtime.imageId, signal: entry.controller.signal,
        onProgress: (detail) => { state.detail = detail; },
      });
      Object.assign(state, { status: "verified", key, detail: "Code writing, CPU training and model reload passed.", verifiedAt: new Date().toISOString(), ...result });
      await writeFile(cache + ".tmp", JSON.stringify(state), { mode: 0o600 });
      await rename(cache + ".tmp", cache);
      return state;
    })().catch((error) => {
      state.status = "failed";
      state.detail = redact(error.message);
      throw new Error(`Agent verification failed: ${state.detail}`);
    }).finally(() => { if (verification === entry) verification = null; });
    return entry.promise;
  };
  const ready = async (agent) => {
    await verify(agent, { forRun: true });
    await cleanupContainers(runRoot);
  };
  const toolStatus = async () => {
    if (toolCache && Date.now() - toolCache.time < 30_000)
      return toolCache.value;
    const [runtime, github, kaggle] = await Promise.all([
      runtimeStatus(),
      runProcess("gh", ["auth", "status"], { timeout: 8000 })
        .then(() => true)
        .catch(() => false),
      runProcess("kaggle", ["--version"], { timeout: 8000 })
        .then(() => true)
        .catch(() => false),
    ]);
    const value = [
      {
        id: "search",
        name: "Web search",
        description:
          "Research models and implementation examples through the selected agent.",
        status: "Built in",
        kind: "Agent tool",
      },
      {
        id: "runtime",
        name: "Training runtime",
        description: runtime.detail,
        status: runtime.ready ? "Ready" : "Setup required",
        kind: "Local tool",
      },
      {
        id: "evaluation",
        name: "Validation",
        description:
          "Recorded folds and seeds, independent scoring, and a reserved final test set.",
        status: "Ready",
        kind: "Harness tool",
      },
      {
        id: "huggingface",
        name: "Hugging Face",
        description:
          "Model search, pretrained weight caching, and model metadata.",
        status: "Public access",
        kind: "Connector",
      },
      {
        id: "arxiv",
        name: "arXiv",
        description:
          "Search primary research papers for task-specific methods.",
        status: "Public access",
        kind: "Connector",
      },
      {
        id: "github",
        name: "GitHub",
        description:
          "Read repository source files. Your gh sign-in also enables private repositories.",
        status: github ? "Signed in" : "Public access",
        kind: "Connector",
      },
      {
        id: "kaggle",
        name: "Kaggle",
        description:
          "List competition data through your local Kaggle CLI. Downloads remain local.",
        status: kaggle ? "CLI installed" : "Install Kaggle CLI",
        kind: "Connector",
      },
      {
        id: "mcp",
        name: "Helix MCP",
        description:
          "Helix tools in an isolated session configuration. Model discovery and pretrained downloads follow the experiment permissions.",
        status: "Built in",
        kind: "MCP server",
      },
    ];
    toolCache = { time: Date.now(), value };
    return value;
  };
  const server = createServer(async (request, response) => {
    const respond = (status, value) => {
      response.writeHead(status, {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
      });
      response.end(JSON.stringify(value));
    };
    try {
      const host = request.headers.host || "";
      if (!/^((127\.0\.0\.1)|(localhost))(?::\d+)?$/.test(host))
        return respond(403, { error: "Invalid runner host." });
      const origin = request.headers.origin;
      const url = new URL(request.url || "/", "http://127.0.0.1");
      // Only the UI served by this runner may bootstrap a session. Even explicitly
      // allowed remote UIs must pair; cross-origin preflights get no CORS access.
      if (url.pathname === "/api/session") {
        if (request.method !== "POST") return respond(405, { error: "Method not allowed." });
        if (origin !== new URL(`http://${host}`).origin ||
            Number(new URL(`http://${host}`).port || 80) !== request.socket.localPort ||
            request.headers["x-helix-local"] !== "1" ||
            (request.headers["sec-fetch-site"] && request.headers["sec-fetch-site"] !== "same-origin"))
          return respond(403, { error: "Open Helix from its local app address." });
        return respond(200, { token, desktop });
      }
      if (origin && !allowed.has(origin))
        return respond(403, {
          error:
            "This site is not paired with the runner. Restart with --origin followed by the site URL.",
        });
      if (origin) {
        response.setHeader("Access-Control-Allow-Origin", origin);
        response.setHeader("Vary", "Origin");
      }
      response.setHeader("X-Content-Type-Options", "nosniff");
      if (request.method === "OPTIONS") {
        response.setHeader(
          "Access-Control-Allow-Methods",
          "GET,POST,PUT,DELETE,OPTIONS",
        );
        response.setHeader(
          "Access-Control-Allow-Headers",
          "Content-Type,Authorization",
        );
        response.setHeader("Access-Control-Allow-Private-Network", "true");
        response.writeHead(204);
        return response.end();
      }
      if (url.pathname === "/health" && request.method === "GET")
        return respond(200, { name: "helix-ml", version: "0.1.0" });
      if (!url.pathname.startsWith("/api/")) {
        if (request.method !== "GET")
          return respond(405, { error: "Method not allowed." });
        const name =
          url.pathname === "/"
            ? "index.html"
            : decodeURIComponent(url.pathname.slice(1));
        const file = await inside(join(project, "dist"), name);
        response.writeHead(200, {
          "Content-Type": mime[extname(file)] || "application/octet-stream",
          "Referrer-Policy": "no-referrer",
        });
        return createReadStream(file).pipe(response);
      }
      if (!auth(request))
        return respond(401, { error: "Pair your local runner to continue." });
      if (stopping)
        return respond(503, { error: "The runner is shutting down." });
      const path = url.pathname.slice(4);
      if (path === "/quit" && request.method === "POST" && desktop) {
        respond(200, { closing: true });
        setImmediate(() => server.emit("quit"));
        return;
      }
      if (path === "/providers" && request.method === "GET")
        return respond(200, await getProviders(url.searchParams.get("refresh") === "1"));
      const providerCheck = path.match(/^\/providers\/(codex|claude)\/verify$/);
      if (providerCheck && request.method === "POST") {
        if (active || starting || chatting) return respond(409, { error: "Pause or stop the current experiment before checking an agent." });
        const body = await jsonBody(request);
        if (!body || (body.force !== undefined && typeof body.force !== "boolean")) throw new Error("force must be a boolean.");
        return respond(200, await verify(providerCheck[1], { force: body.force === true }));
      }
      if (path === "/tools" && request.method === "GET")
        return respond(200, await toolStatus());
      if (path === "/datasets/inspect" && request.method === "POST") {
        const task = validateTask(await jsonBody(request));
        const controller = new AbortController();
        const cancel = () => { if (!response.writableFinished) controller.abort(); };
        response.once("close", cancel);
        try {
          return respond(200, await inspectDataset(task, { signal: controller.signal }));
        } finally {
          response.removeListener("close", cancel);
        }
      }
      if (path === "/runs" && request.method === "GET")
        return respond(
          200,
          [...runs.values()]
            .map(snapshot)
            .sort((a, b) => b.createdAt.localeCompare(a.createdAt)),
        );
      if (path === "/runs" && request.method === "POST") {
        if (verification) return respond(409, { error: "An agent check is running. Wait for it to finish." });
        if (active || starting || chatting) return respond(409, { error: "Another experiment is running. Pause or stop it first." });
        starting = true;
        try {
          const task = validateTask(await jsonBody(request));
          await inspectDataset(task, { signal: shutdown.signal });
          await ready(task.agent);
          if (stopping) throw new Error("The runner is shutting down.");
          const run = await createRun(runRoot, task);
          runs.set(run.id, run);
          if (stopping) {
            run.status = "paused";
            await saveRun(runRoot, run);
            throw new Error("The runner shut down while creating the experiment. Resume it after restarting.");
          }
          launch(run);
          return respond(201, snapshot(run));
        } finally { starting = false; }
      }
      const match = path.match(
        /^\/runs\/([a-f0-9-]{36})(?:\/(action|artifacts|file|messages))?$/,
      );
      if (match) {
        const run = runs.get(match[1]);
        if (!run) return respond(404, { error: "Run not found." });
        if (!match[2] && request.method === "GET")
          return respond(200, snapshot(run));
        if (!match[2] && request.method === "DELETE") {
          if (starting || chatting?.run.id === run.id || active?.run.id === run.id || !["completed", "stopped", "failed", "paused"].includes(run.status))
            return respond(409, { error: "Wait for the experiment to finish, pause, or stop before deleting it." });
          if (mutating.has(run.id)) return respond(409, { error: "This experiment is being updated. Try again shortly." });
          mutating.add(run.id);
          try {
            await rm(join(runRoot, match[1]), { recursive: true, force: true });
            runs.delete(match[1]);
            return respond(200, { deleted: true });
          } finally { mutating.delete(run.id); }
        }
        if (match[2] === "messages" && request.method === "POST") {
          if (active || starting || chatting || verification || mutating.has(run.id))
            return respond(409, { error: "Wait for the current experiment or reply to finish." });
          if (run.status !== "completed" || !run.best) throw new Error("Follow-ups are available after the experiment completes.");
          mutating.add(run.id);
          starting = true;
          try {
            const body = await jsonBody(request);
            if (typeof body.message !== "string" || !body.message.trim() || body.message.length > 4000)
              throw new Error("Write a message of 1 to 4000 characters.");
            if (!["chat", "trials"].includes(body.mode)) throw new Error("Choose chat or more trials.");
            if ((run.messages?.length || 0) >= 100) throw new Error("This conversation has reached 100 messages. Start a new experiment.");
            if (body.mode === "trials") validateTask({ ...run.task, minutes: body.minutes, trials: body.trials });
            const message = { id: randomUUID(), message: body.message.trim(), mode: body.mode, status: "pending", createdAt: new Date().toISOString() };
            (run.messages ||= []).push(message);
            await saveRun(runRoot, run);
            const entry = { run, controller: new AbortController(), promise: null };
            chatting = entry;
            entry.promise = (async () => {
              if (body.mode === "trials") {
                await ready(run.task.agent);
                if (stopping || entry.controller.signal.aborted) throw new Error("Interrupted before starting more trials.");
                const child = await continueRun(runRoot, run, message.message, body.minutes, body.trials);
                runs.set(child.id, child);
                message.childRunId = child.id;
                message.reply = `Started up to ${body.trials} additional trials from the selected model, with a ${body.minutes}-minute budget. The original result is saved.`;
                launch(child);
              } else {
                const provider = (await getProviders(true)).find(value => value.id === run.task.agent);
                if (!provider?.authenticated) throw new Error("Reconnect your agent in Connections, then resend your message.");
                message.reply = redact(await discussRun(runRoot, run, message, entry.controller.signal)).slice(0, 24000);
                if (!message.reply.trim()) throw new Error("The agent returned no reply. Please try again.");
              }
              message.status = "completed";
            })().catch(error => {
              message.status = "failed";
              message.error = redact(error.message);
            }).finally(async () => {
              try { await saveRun(runRoot, run); }
              finally { if (chatting === entry) chatting = null; }
            });
            return respond(202, snapshot(run));
          } finally { starting = false; mutating.delete(run.id); }
        }
        if (match[2] === "action" && request.method === "POST") {
          if (mutating.has(run.id)) return respond(409, { error: "This experiment is being updated. Try again shortly." });
          mutating.add(run.id);
          try {
            const { action } = await jsonBody(request);
            if (action === "resume") {
              if (verification) return respond(409, { error: "An agent check is running. Wait for it to finish." });
              if (active || starting || chatting) return respond(409, { error: "Another experiment is running or stopping." });
              if (!["paused", "failed"].includes(run.status)) throw new Error("Only paused or failed runs can resume.");
              if (run.selectionFrozen && !Number.isFinite(run.testScore)) {
                const final = JSON.parse(await readFile(join(runRoot, run.id, "final-test.json"), "utf8").catch(() => "{}"));
                if (final.status !== "completed") throw new Error("Final evaluation was interrupted or failed. Start a new experiment; the final fit cannot be retried.");
              }
              if (elapsed(run) >= run.task.minutes * 60) throw new Error("This run used its time budget. Start a new experiment with a larger budget.");
              starting = true;
              try {
                await ready(run.task.agent);
                if (stopping) throw new Error("The runner is shutting down.");
                if (!["paused", "failed"].includes(run.status)) throw new Error("The experiment was stopped while its agent was being checked.");
                launch(run);
              } finally { starting = false; }
            } else if (["pause", "stop"].includes(action)) {
              if (action === "pause" && run.phase === "finalizing") throw new Error("Final evaluation runs once and cannot be paused. Allow it to finish, or stop the experiment.");
              if (!["running", "queued", "paused", "failed"].includes(run.status)) throw new Error("This experiment has already ended.");
              run.status = action === "pause" ? "paused" : "stopped";
              log(run, "system", action === "pause" ? "Paused. Resume restarts the unfinished trial within the remaining budget." : "Stopped. Completed trial artifacts are preserved.");
              const entry = active?.run.id === run.id ? active : null;
              entry?.controller.abort();
              if (entry) await entry.promise;
              await saveRun(runRoot, run);
            } else throw new Error("Choose pause, resume, or stop.");
            return respond(200, snapshot(run));
          } finally { mutating.delete(run.id); }
        }
        if (match[2] === "artifacts" && request.method === "GET") {
          const files = [];
          const visit = async (folder, depth = 0) => {
            if (depth > 3) return;
            for (const entry of await readdir(join(runRoot, run.id, folder), {
              withFileTypes: true,
            }).catch(() => [])) {
              if (
                entry.name.startsWith(".") ||
                ["context.json", "AGENTS.md", "CLAUDE.md"].includes(
                  entry.name,
                ) ||
                entry.isSymbolicLink()
              )
                continue;
              const name = folder + "/" + entry.name;
              if (entry.isDirectory()) await visit(name, depth + 1);
              else if (entry.isFile())
                files.push({
                  path: name,
                  size: (await lstat(join(runRoot, run.id, name))).size,
                });
            }
          };
          for (const folder of ["final", "research", "trials"])
            await visit(folder);
          return respond(200, files);
        }
        if (match[2] === "file" && request.method === "GET") {
          const name = url.searchParams.get("path") || "";
          if (
            !/^(trials|final|research)\//.test(name) ||
            ["context.json", "AGENTS.md", "CLAUDE.md"].includes(basename(name))
          )
            throw new Error("Artifact not available.");
          const [folder, ...parts] = name.split("/");
          const file = await inside(join(runRoot, run.id, folder), parts.join("/"));
          response.writeHead(200, {
            "Content-Type": "application/octet-stream",
            "Content-Disposition": `attachment; filename="${basename(file).replace(/[^a-zA-Z0-9_.-]/g, "_")}"`,
          });
          return createReadStream(file).pipe(response);
        }
      }
      if (path === "/datasets" && request.method === "POST") {
        const id = randomUUID();
        await mkdir(join(uploads, id), { mode: 0o700 });
        return respond(201, { id, path: join(uploads, id) });
      }
      const upload = path.match(/^\/datasets\/([a-f0-9-]{36})\/file$/);
      if (upload && request.method === "PUT") {
        const name = url.searchParams.get("name");
        if (
          !name ||
          name.length > 1000 ||
          name.startsWith("/") ||
          name.includes("\\") ||
          name
            .split("/")
            .some((part) => part === ".." || part.startsWith(".") || !part)
        )
          throw new Error("Invalid upload filename.");
        await lstat(join(uploads, upload[1]));
        const destination = join(uploads, upload[1], name);
        await mkdir(dirname(destination), { recursive: true });
        let size = 0;
        const limit = new Transform({
          transform(chunk, _encoding, callback) {
            size += chunk.length;
            callback(
              size > 2_000_000_000
                ? new Error("Each file must be smaller than 2 GB.")
                : null,
              chunk,
            );
          },
        });
        // Open first so a duplicate upload cannot delete the original file in the error handler.
        const handle = await open(destination, "wx", 0o600);
        try {
          await pipeline(request, limit, handle.createWriteStream());
        } catch (error) {
          await rm(destination, { force: true });
          throw error;
        }
        return respond(201, { name, size });
      }
      return respond(404, { error: "Endpoint not found." });
    } catch (error) {
      if (!response.headersSent)
        respond(error.code === "ENOENT" ? 404 : 400, {
          error: redact(error.message),
        });
      else response.destroy();
    }
  });
  let closing;
  const close = () => closing ||= (async () => {
    stopping = true;
    shutdown.abort();
    if (verification) {
      verification.controller.abort();
      await verification.promise.catch(() => {});
    }
    while (starting) await delay(20);
    if (chatting) {
      chatting.controller.abort();
      await chatting.promise;
    }
    if (active) {
      active.run.status = "paused";
      active.controller.abort();
      await active.promise;
    }
    server.closeAllConnections();
    await new Promise((resolveClose) => server.close(resolveClose));
    await release();
  })();
  return { server, close, token, runs, refreshProviders: () => getProviders(true) };
}

if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  const port = Number(process.env.HELIX_PORT || 4319);
  const index = process.argv.indexOf("--origin");
  const origin =
    index >= 0 ? process.argv[index + 1] : process.env.HELIX_UI_ORIGIN;
  if (origin && !/^https?:\/\/[^/]+$/.test(origin))
    throw new Error(
      "Use --origin with the site origin, without a trailing slash.",
    );
  const service = await createService({
    origins: origin ? [origin] : [],
    port,
  });
  await writeFile(resolve(project, ".helix", "pairing-code"), service.token, {
    mode: 0o600,
  });
  service.server.listen(port, "127.0.0.1", () => {
    console.log(
      `Helix runner: http://127.0.0.1:${port}\nPairing code: ${service.token}\nOpen your Helix site, select an agent, and pair this runner.`,
    );
  });
  for (const signal of ["SIGINT", "SIGTERM"])
    process.once(signal, async () => {
      await service.close();
      process.exit(0);
    });
}
