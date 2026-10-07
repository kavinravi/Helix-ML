import { spawn } from "node:child_process";

export function runProcess(
  command,
  args,
  { cwd, env = process.env, signal, timeout = 30_000, onLine = () => {} } = {},
) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new Error("Interrupted"));
    const child = spawn(command, args, {
      cwd,
      env,
      stdio: ["pipe", "pipe", "pipe"],
      detached: process.platform !== "win32",
    });
    let output = "",
      error = "",
      buffer = "",
      limitError = "",
      killed = false;
    const stop = () => {
      killed = true;
      if (process.platform === "win32")
        spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"]);
      else {
        try {
          process.kill(-child.pid, "SIGKILL");
        } catch {
          /* Already exited. */
        }
      }
    };
    const timer = setTimeout(stop, timeout);
    signal?.addEventListener("abort", stop, { once: true });
    child.stdout.on("data", (chunk) => {
      output = (output + chunk.toString()).slice(-2_000_000);
      buffer += chunk.toString();
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      if (buffer.length > 2_000_000) {
        limitError = "A process emitted a line larger than 2 MB.";
        stop();
        buffer = "";
      }
      for (const line of lines) onLine(line);
    });
    child.stderr.on("data", (chunk) => {
      error = (error + chunk.toString()).slice(-20_000);
    });
    const cleanup = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", stop);
    };
    child.on("error", (err) => {
      cleanup();
      reject(err);
    });
    child.on("close", (code) => {
      cleanup();
      if (buffer) onLine(buffer);
      if (killed)
        reject(
          new Error(limitError || (signal?.aborted ? "Interrupted" : "Time limit reached")),
        );
      else if (code !== 0)
        reject(
          new Error(
            (error || output || `${command} exited with ${code}`).slice(
              -12_000,
            ),
          ),
        );
      else resolve({ output, error });
    });
    child.stdin.end();
  });
}
