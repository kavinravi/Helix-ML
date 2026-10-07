import { readFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import { callTool, availableTools } from "./tools.mjs";

const context = JSON.parse(await readFile(process.argv[2], "utf8"));
const run = JSON.parse(await readFile(context.runFile, "utf8"));
const tools = availableTools(run.task, context.mode);
const lines = createInterface({ input: process.stdin });
for await (const line of lines) {
  let request;
  try {
    request = JSON.parse(line);
    if (request.id === undefined) continue;
    let result;
    if (request.method === "initialize")
      result = {
        protocolVersion: ["2024-11-05", "2025-03-26", "2025-06-18"].includes(request.params?.protocolVersion) ? request.params.protocolVersion : "2025-06-18",
        capabilities: { tools: {} },
        serverInfo: { name: "helix-ml", version: "0.1.0" },
      };
    else if (request.method === "ping") result = {};
    else if (request.method === "tools/list") result = { tools };
    else if (request.method === "tools/call") {
      try {
        const value = await callTool(
          request.params.name,
          request.params.arguments ?? {},
          context,
        );
        result = {
          content: [
            {
              type: "text",
              text: typeof value === "string" ? value : JSON.stringify(value),
            },
          ],
        };
      } catch (error) {
        result = {
          content: [{ type: "text", text: error.message }],
          isError: true,
        };
      }
    } else {
      process.stdout.write(
        JSON.stringify({
          jsonrpc: "2.0",
          id: request.id,
          error: { code: -32601, message: "Method not found" },
        }) + "\n",
      );
      continue;
    }
    process.stdout.write(
      JSON.stringify({ jsonrpc: "2.0", id: request.id, result }) + "\n",
    );
  } catch {
    process.stdout.write(
      JSON.stringify({
        jsonrpc: "2.0",
        id: request?.id ?? null,
        error: { code: -32700, message: "Invalid JSON-RPC message" },
      }) + "\n",
    );
  }
}
