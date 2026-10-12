import { mkdtemp, mkdir, rm, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
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
    PYTHONNOUSERSITE: "1", PYTHONDONTWRITEBYTECODE: "1", PYTHONUNBUFFERED: "1", PYTHONUTF8: "1",
    HELIX_TRAIN_MEMORY_MB: String(runtime.memoryMiB),
    OMP_NUM_THREADS: String(runtime.cpus), OPENBLAS_NUM_THREADS: String(runtime.cpus),
    MKL_NUM_THREADS: String(runtime.cpus), NUMBA_NUM_THREADS: String(runtime.cpus),
    NUMBA_CACHE_DIR: join(temporary, "numba"), JUPYTER_RUNTIME_DIR: join(temporary, "jupyter"),
    PIP_CACHE_DIR: join(dirname(dirname(dirname(runtime.python))), "cache", "pip"),
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

export async function nativeLibraryPaths(roots) {
  const paths = [];
  const entries = root => readdir(root, {withFileTypes:true}).catch(error => { if (error.code === "ENOENT") return []; throw error; });
  // Wheel RPATHs assume one site-packages directory; candidates also reuse base wheels.
  for (const root of roots) {
    for (const entry of await entries(root)) {
      if (!entry.isDirectory()) continue;
      const folder = join(root, entry.name);
      if (entry.name.endsWith(".libs")) paths.push(folder);
      for (const child of await entries(folder)) if (child.isDirectory() && ["lib", "lib64"].includes(child.name)) paths.push(join(folder, child.name));
    }
    for (const vendor of await entries(join(root, "nvidia"))) if (vendor.isDirectory())
      for (const child of await entries(join(root, "nvidia", vendor.name))) if (child.isDirectory() && ["lib", "lib64"].includes(child.name)) paths.push(join(root, "nvidia", vendor.name, child.name));
  }
  return paths;
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
    const env = nativeEnvironment(context.runtime, temporary, mounts);
    let args = nativeArguments(command, mounts);
    if (env.HELIX_PACKAGES) {
      args = [...args.slice(0, 3), fileURLToPath(new URL("native_entry.py", import.meta.url)), ...args.slice(3)];
      if (process.platform === "linux") {
        const base = join(dirname(dirname(context.runtime.python)), "lib", "python" + context.runtime.pythonVersion.split(".").slice(0, 2).join("."), "site-packages");
        env.LD_LIBRARY_PATH = (await nativeLibraryPaths([env.HELIX_PACKAGES, base])).join(delimiter);
      }
    }
    const result = await runProcess(context.runtime.python, args, {
      cwd: writable || temporary, env,
      signal: combined, timeout: remainingTime(context.deadline), onLine,
    });
    if (writable && await directoryBytes(writable) > limit) throw new Error("Runtime output exceeds the disk limit.");
    return result;
  } catch (error) { throw diskError || error; }
  finally { if (timer) clearInterval(timer); await rm(temporary, { recursive: true, force: true }); }
}
