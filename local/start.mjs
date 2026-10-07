// First-run setup uses only Node's standard library, before npm dependencies exist.
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { runtimeStatus } from "./runtime.mjs";
import { providers } from "./agents.mjs";

const project = fileURLToPath(new URL("..", import.meta.url));
const npm = process.platform === "win32" ? "npm.cmd" : "npm";
let stopping = false;

async function run(command, args) {
  await new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd: project, stdio: "inherit", detached: process.platform !== "win32" });
    const stop = (signal) => {
      stopping = true;
      if (!child.pid) return;
      try { process.platform === "win32" ? child.kill(signal) : process.kill(-child.pid, signal); }
      catch (error) { if (error.code !== "ESRCH") console.error(error.message); }
    };
    const interrupt = () => stop("SIGINT");
    const terminate = () => stop("SIGTERM");
    process.once("SIGINT", interrupt);
    process.once("SIGTERM", terminate);
    const cleanup = () => { process.removeListener("SIGINT", interrupt); process.removeListener("SIGTERM", terminate); };
    child.once("error", (error) => { cleanup(); reject(error); });
    child.once("exit", (code, signal) => { cleanup(); code === 0 && !stopping ? resolve() : reject(new Error(stopping ? "Helix stopped." : `${command} ${signal ? "interrupted" : `exited with ${code}`}.`)); });
  });
}

try {
  const [major, minor] = process.versions.node.split(".").map(Number);
  if (major < 22 || (major === 22 && minor < 13)) throw new Error("Install Node.js 22.13 or newer, then retry npm run local.");
  console.log("Checking Python, Docker and agent sign-in…");
  const [runtime, agents] = await Promise.all([runtimeStatus(), providers()]);
  if (!runtime.ready && runtime.missing !== "image") throw new Error(runtime.detail);
  if (!agents.some((agent) => agent.installed && agent.authenticated))
    throw new Error(agents.map((agent) => `${agent.name}: ${agent.detail}`).join("\n"));
  if (!runtime.ready) await run(process.execPath, ["local/setup.mjs"]);
  await run(npm, ["ci"]);
  await run(npm, ["run", "build"]);
  console.log("Setup complete. Keep this terminal open. Next time, use npm start.");
  await run(process.execPath, ["local/server.mjs"]);
} catch (error) {
  console.error(error.message);
  process.exitCode = stopping ? 130 : 1;
}
