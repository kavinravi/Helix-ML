import { createHash, randomUUID } from "node:crypto";
import { readdir, lstat, mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { runProcess, expired, remainingTime } from "./process.mjs";

export const IMAGE = "helix-ml:local";
export const owner = (root) => createHash("sha256").update(resolve(root)).digest("hex").slice(0, 16);

export function runtimeMemoryMb() {
  const value = Number(process.env.HELIX_TRAIN_MEMORY_MB || (process.env.HELIX_HOSTED_WORKER ? 1200 : 3072));
  if (!Number.isInteger(value) || value < 128 || value > 65536) throw new Error("HELIX_TRAIN_MEMORY_MB must be an integer from 128 to 65536.");
  return value;
}

export async function runtimeStatus() {
  try {
    await runProcess(process.env.HELIX_PYTHON || "python3", ["-c", "import sys; assert sys.version_info >= (3,11), 'Python 3.11 or newer is required'"], { timeout: 5000 });
  } catch { return { ready: false, missing: "python", detail: "Install Python 3.11 or newer (python3), or set HELIX_PYTHON to its executable, then restart the runner." }; }
  try {
    await runProcess("docker", ["info", "--format", "{{.ServerVersion}}"], { timeout: 8000 });
  } catch { return { ready: false, missing: "docker", detail: "Docker is unavailable. Start Docker Desktop or the Docker daemon." + (process.env.WSL_DISTRO_NAME ? " Enable Docker Desktop's WSL integration for this distribution." : "") }; }
  try {
    const image = await runProcess("docker", ["image", "inspect", IMAGE, "--format", '{{index .Config.Labels "helix.runtime"}} {{.Id}}'], { timeout: 8000 });
    const [version, imageId] = image.output.trim().split(/\s+/);
    if (version !== "2" || !/^sha256:[a-f0-9]{64}$/.test(imageId)) throw new Error("Runtime needs rebuilding");
    return { ready: true, imageId, detail: process.env.HELIX_HOSTED_WORKER ? "Cloud CPU training ready" : `CPU training ready · 2 cores and ${runtimeMemoryMb()} MiB RAM per fit. Docker Desktop must remain running.` };
  } catch { return { ready: false, missing: "image", detail: "Run npm run setup to build or update the CPU training runtime." }; }
}

export async function cleanupContainers(root, runId) {
  const filters = ["ps", "-aq", "--filter", `label=helix.owner=${owner(root)}`];
  if (runId) filters.push("--filter", `label=helix.run=${runId}`);
  const { output } = await runProcess("docker", filters, { timeout: 10_000 });
  const ids = output.trim().split(/\s+/).filter((id) => /^[a-f0-9]{12,64}$/.test(id));
  if (ids.length) await runProcess("docker", ["rm", "-f", ...ids], { timeout: 15_000 });
}

export async function directoryBytes(root) {
  let size = 0;
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const path = join(root, entry.name);
    if (entry.isSymbolicLink()) throw new Error("Generated artifacts cannot contain symlinks.");
    if (entry.isDirectory()) size += await directoryBytes(path);
    else if (entry.isFile()) size += (await lstat(path)).size;
    else throw new Error("Generated artifacts must be regular files.");
  }
  return size;
}

export async function container(context, command, { mounts = [], network = false, signal, onLine, writable, limit = 1_000_000_000 } = {}) {
  if (mounts.some(([path]) => path.includes(",") || path.includes("\n"))) throw new Error("Docker workspaces and datasets must use paths without commas or newlines.");
  if (expired(context.deadline)) throw new Error("The run budget has expired.");
  const name = `helix-${context.runId}-${randomUUID().slice(0, 8)}`;
  const remaining = remainingTime(context.deadline);
  const seconds = remaining === null ? null : Math.max(1, Math.floor(remaining / 1000));
  const controller = new AbortController();
  const combined = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
  let diskError, checking = false;
  if (writable) await mkdir(writable, { recursive: true });
  // ponytail: aggregate disk quota is polled each second; use a quota-backed volume for a hard aggregate ceiling.
  const timer = writable && setInterval(async () => {
    if (checking) return;
    checking = true;
    try {
      if (await directoryBytes(writable) > limit) throw new Error("The runtime's 1 GB output limit was exceeded.");
    } catch (error) {
      diskError = error;
      controller.abort();
    } finally { checking = false; }
  }, 1000);
  try {
    const result = await runProcess("docker", [
      "run", "--name", name,
      "--label", `helix.owner=${owner(context.root)}`, "--label", `helix.run=${context.runId}`,
      ...(network ? [] : ["--network=none"]), "--read-only", "--cap-drop=ALL",
      "--security-opt=no-new-privileges", "--pids-limit=128", `--memory=${runtimeMemoryMb()}m`, "--cpus=2",
      "--ulimit", "fsize=268435456:268435456", "--tmpfs=/tmp:rw,size=512m",
      "--user", context.user,
      "-e", `HELIX_TRAIN_MEMORY_MB=${runtimeMemoryMb()}`,
      "-e", "PYTHONPATH=/code:/packages", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "NUMBA_CACHE_DIR=/tmp/numba",
      ...mounts.flatMap(([source, target, mode = "ro"]) => ["--mount", `type=bind,source=${source},target=${target}${mode === "ro" ? ",readonly" : ""}`]),
      context.image || IMAGE, ...(seconds === null ? [] : ["timeout", "--kill-after=5", String(seconds)]), ...command,
    ], { timeout: seconds === null ? null : (seconds + 5) * 1000, signal: combined, onLine });
    if (writable && await directoryBytes(writable) > limit) throw new Error("Runtime output exceeds the disk limit.");
    return result;
  } catch (error) {
    if (!combined.aborted && !diskError) {
      const state = await runProcess("docker", ["inspect", "--format", "{{.State.OOMKilled}}", name], { timeout: 10_000 }).catch(() => null);
      if (state?.output.trim() === "true") throw new Error(`Execution exceeded this worker's ${runtimeMemoryMb()} MiB RAM limit. Use chunked loading, smaller batches, fewer features, or a model with a smaller working set. The completed trials are preserved.`);
    }
    throw diskError || error;
  } finally {
    if (timer) clearInterval(timer);
    await runProcess("docker", ["rm", "-f", name], { timeout: 10_000 }).catch(() => {});
  }
}
