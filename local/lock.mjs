import { mkdir, writeFile, readdir, unlink, rmdir } from "node:fs/promises";
import { join } from "node:path";
import { randomUUID } from "node:crypto";

export async function lockRunner(root) {
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
