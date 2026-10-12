import { createInterface } from "node:readline";
import { createMcpService } from "./service.mjs";
import { tools } from "./schema.mjs";

export async function serve(home, runtime) {
  const service = await createMcpService(home, runtime);
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  const send = value => process.stdout.write(JSON.stringify({ jsonrpc: "2.0", ...value }) + "\n");
  const stop = () => { lines.close(); process.stdin.destroy(); void service.close(); };
  process.once("SIGINT", stop); process.once("SIGTERM", stop);
  process.stdout.on("error", stop);
  // JSON-RPC framing is one JSON value per line. Bound incomplete lines too.
  let pendingBytes = 0;
  const bound = chunk => {
    for (const part of chunk.toString().split(/(?<=\n)/)) {
      pendingBytes += Buffer.byteLength(part);
      if (pendingBytes > 2_000_000) { console.error("Helix request exceeds 2 MB."); stop(); return; }
      if (part.endsWith("\n")) pendingBytes = 0;
    }
  };
  process.stdin.on("data", bound);
  try {
    for await (const line of lines) {
      let request;
      try { request = JSON.parse(line); }
      catch { send({ id: null, error: { code: -32700, message: "Invalid JSON" } }); continue; }
      if (!request || request.jsonrpc !== "2.0" || typeof request.method !== "string" || (request.id !== undefined && typeof request.id !== "string" && typeof request.id !== "number")) {
        send({ id: null, error: { code: -32600, message: "Invalid JSON-RPC request" } }); continue;
      }
      if (request.id === undefined) continue;
      let result;
      if (request.method === "initialize") result = {
        protocolVersion: ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"].includes(request.params?.protocolVersion) ? request.params.protocolVersion : "2025-06-18",
        capabilities: { tools: {} }, serverInfo: { name: "helix-ml-local", version: "0.2.1" },
        instructions: "Helix runs ML experiments on this computer. Start with helix_info. Propose task, target, metrics and budget before training. Create an experiment, read its status for the code contract, write candidate source using your editor, submit trials and compare measured validation scores, then finalize once. Long jobs return immediately: poll status every few seconds. Local execution has normal user permissions. Never access held-out labels or write into Helix's private state folder.",
      };
      else if (request.method === "ping") result = {};
      else if (request.method === "tools/list") result = { tools };
      else if (request.method === "tools/call") {
        try {
          const value = await service.call(request.params?.name, request.params?.arguments ?? {});
          result = { content: [{ type: "text", text: JSON.stringify(value) }] };
        } catch (error) { result = { content: [{ type: "text", text: error.message }], isError: true }; }
      } else { send({ id: request.id, error: { code: -32601, message: "Method not found" } }); continue; }
      send({ id: request.id, result });
    }
  } finally {
    process.removeListener("SIGINT", stop); process.removeListener("SIGTERM", stop); process.stdin.removeListener("data", bound); process.stdout.removeListener("error", stop);
    await service.close();
  }
}
