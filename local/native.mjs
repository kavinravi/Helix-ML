import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, dirname, join } from "node:path";
import { runProcess, expired, remainingTime } from "./process.mjs";
import { directoryBytes } from "./runtime.mjs";

// Native mode is trusted local execution, not an OS sandbox.
export function nativeEnvironment(runtime, temporary, mounts) {
  const env = {};
  for (const key of ["PATH", "Path", "SystemRoot", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR"])
    if (process.env[key]) env[key] = process.env[key];
  delete env.Path;
  env.PATH = dirname(runtime.python) + delimiter + (process.env.PATH || process.env.Path || "");
  Object.assign(env, {
    HOME: temporary, USERPROFILE: temporary, TMP: temporary, TEMP: temporary, TMPDIR: temporary,
    PYTHONNOUSERSITE: "1", PYTHONDONTWRITEBYTECODE: "1", PYTHONUNBUFFERED: "1",
    HELIX_TRAIN_MEMORY_MB: String(runtime.memoryMiB),
    OMP_NUM_THREADS: String(runtime.cpus), OPENBLAS_NUM_THREADS: String(runtime.cpus),
    MKL_NUM_THREADS: String(runtime.cpus), NUMBA_NUM_THREADS: String(runtime.cpus),
    NUMBA_CACHE_DIR: join(temporary, "numba"), JUPYTER_RUNTIME_DIR: join(temporary, "jupyter"),
    HELIX_DEVICE: runtime.device || "cpu", PYTORCH_ENABLE_MPS_FALLBACK: "0",
    CUDA_VISIBLE_DEVICES: runtime.device === "cuda" ? (process.env.CUDA_VISIBLE_DEVICES ?? "0") : "-1",
    CUBLAS_WORKSPACE_CONFIG: ":4096:8", TORCH_HOME: join(temporary, "torch"), CUDA_CACHE_PATH: join(temporary, "cuda"),
    ...(runtime.compute?.torchVersion ? { HELIX_TORCH_VERSION: runtime.compute.torchVersion } : {}),
  });
  const names = { "/code": "HELIX_SOURCE", "/data": "HELIX_DATA", "/work": "HELIX_OUTPUT", "/models": "HELIX_MODELS", "/packages": "HELIX_PACKAGES" };
  for (const [source, target] of mounts) if (names[target]) env[names[target]] = source;
  env.PYTHONPATH = mounts.filter(([, target]) => ["/code", "/packages", "/harness"].includes(target)).map(([source]) => source).join(delimiter);
  return env;
}

export function nativeArguments(command, mounts) {
  if (command[0] !== "python") throw new Error("The native runtime only executes Python.");
  return ["-B", "-s", "-P", ...command.slice(1).map(arg => {
    const mount = mounts.find(([, target]) => arg === target || arg.startsWith(target + "/"));
    return mount ? join(mount[0], arg.slice(mount[1].length)) : arg;
  })];
}

export async function nativeProcess(context, command, { mounts = [], signal, onLine, writable, limit = 1_000_000_000 } = {}) {
  if (expired(context.deadline)) throw new Error("The run budget has expired.");
  const temporary = await mkdtemp(join(tmpdir(), "helix-fit-"));
  const controller = new AbortController();
  const combined = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
  let diskError, checking = false;
  if (writable) await mkdir(writable, { recursive: true });
  // shortcut: poll output size; native mode cannot enforce a hard disk or RAM quota.
  const timer = writable && setInterval(async () => {
    if (checking) return;
    checking = true;
    try { if (await directoryBytes(writable) > limit) throw new Error("Runtime output exceeds the disk limit."); }
    catch (error) { diskError = error; controller.abort(); }
    finally { checking = false; }
  }, 1000);
  try {
    const result = await runProcess(context.runtime.python, nativeArguments(command, mounts), {
      cwd: writable || temporary, env: nativeEnvironment(context.runtime, temporary, mounts),
      signal: combined, timeout: remainingTime(context.deadline), onLine,
    });
    if (writable && await directoryBytes(writable) > limit) throw new Error("Runtime output exceeds the disk limit.");
    return result;
  } catch (error) { throw diskError || error; }
  finally { if (timer) clearInterval(timer); await rm(temporary, { recursive: true, force: true }); }
}
