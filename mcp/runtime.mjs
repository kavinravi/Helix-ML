import { access, mkdir, readFile, writeFile, rm } from "node:fs/promises";
import { availableParallelism, totalmem } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { runProcess } from "../local/process.mjs";
import { lockRunner } from "../local/lock.mjs";

const requirements = fileURLToPath(new URL("requirements.txt", import.meta.url));
export const pythonPath = home => join(home, "python", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
const check = "import sys, numpy, pandas, sklearn, scipy, PIL, nbformat, nbclient, ipykernel, huggingface_hub; assert sys.version_info >= (3,11); print(sys.version.split()[0])";

export async function runtimeConfig(home) {
  const config = JSON.parse(await readFile(join(home, "runtime.json"), "utf8"));
  const expected = createHash("sha256").update(await readFile(requirements)).digest("hex");
  if (config.requirementsHash !== expected || config.kind !== "native" || config.localCodeConsent !== true)
    throw new Error("Run helix setup --allow-local-code to update the local runtime.");
  await access(pythonPath(home));
  return { ...config, python: pythonPath(home) };
}

export async function setupRuntime(home, { allowLocalCode, report = console.error, uv } = {}) {
  if (!allowLocalCode) throw new Error("Local training runs generated Python with your user permissions, including file and network access. To use this trusted local mode, run again with --allow-local-code. No Docker is required.");
  const release = await lockRunner(join(home, "setup"));
  try {
    try {
      const config = await runtimeConfig(home);
      await runProcess(config.python, ["-I", "-c", check], { timeout: 30_000 });
      report("Helix Python runtime is ready.");
      return config;
    } catch { /* Missing, changed or interrupted setup is repaired below. */ }
    await rm(join(home, "runtime.json"), { force: true });
    if (uv) {
      const env = { ...process.env, UV_PYTHON_INSTALL_DIR: join(home, "interpreters"), UV_CACHE_DIR: join(home, "cache", "uv"), UV_PYTHON_INSTALL_BIN: "0", UV_PYTHON_INSTALL_REGISTRY: "0" };
      report("Downloading Helix's private Python and preparing its environment…");
      const progress = setInterval(() => report("Helix setup is still running. First-time downloads can take a few minutes…"), 15_000);
      try {
        await runProcess(uv, ["venv", "--no-config", "--managed-python", "--python", "3.12", "--seed", "--allow-existing", join(home, "python")], { env, timeout: 900_000, onLine: report });
        report("Installing ML dependencies…");
        await runProcess(uv, ["pip", "install", "--no-config", "--python", pythonPath(home), "-r", requirements], { env, timeout: 900_000, onLine: report });
      } finally { clearInterval(progress); }
    } else {
      let found;
      const candidates = process.env.HELIX_PYTHON ? [[process.env.HELIX_PYTHON, []]] : process.platform === "win32" ? [["py", ["-3"]], ["python", []], ["python3", []]] : [["python3", []], ["python", []]];
      for (const [command, prefix] of candidates) {
        try {
          await runProcess(command, [...prefix, "-c", "import sys; assert sys.version_info >= (3,11)"], { timeout: 5000 });
          found = [command, prefix]; break;
        } catch { /* Try the next platform convention. */ }
      }
      if (!found) throw new Error("Install Python 3.11 or newer from python.org (on Windows enable the Python launcher), then rerun this command. You can also set HELIX_PYTHON to the Python executable.");
      report("Creating Helix's Python environment…");
      await runProcess(found[0], [...found[1], "-m", "venv", join(home, "python")], { timeout: 120_000 });
      report("Installing ML dependencies. The first setup can take several minutes…");
      await runProcess(pythonPath(home), ["-m", "pip", "install", "--disable-pip-version-check", "--no-input", "-r", requirements], { timeout: 900_000, onLine: report });
    }
    const version = (await runProcess(pythonPath(home), ["-I", "-c", check], { timeout: 60_000 })).output.trim();
    const config = {
      kind: "native", python: pythonPath(home), pythonVersion: version, localCodeConsent: true,
      cpus: Math.max(1, availableParallelism() - 1), memoryMiB: Math.min(65536, Math.max(128, Math.floor(totalmem() / 2 / 1024 ** 2))),
      requirementsHash: createHash("sha256").update(await readFile(requirements)).digest("hex"),
    };
    await mkdir(home, { recursive: true, mode: 0o700 });
    await writeFile(join(home, "runtime.json"), JSON.stringify(config, null, 2), { mode: 0o600 });
    report(`Ready: Python ${version}, ${config.cpus} CPU threads, ${config.memoryMiB} MiB estimated fit budget.`);
    return config;
  } finally { await release(); }
}
