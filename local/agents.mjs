import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { runProcess } from "./process.mjs";
import { tools as helixTools } from "./tools.mjs";

const mcpScript = fileURLToPath(new URL("./mcp.mjs", import.meta.url));

export function subscriptionEnvironment() {
  const env = { ...process.env };
  for (const key of [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "CLAUDE_CONFIG_DIR",
    "CLAUDE_CODE_API_KEY_HELPER_TTL_MS",
    "HELIX_WORKER_TOKEN",
    "RAILWAY_TOKEN",
    "RAILWAY_API_TOKEN",
  ])
    delete env[key];
  return env;
}

export async function providers() {
  return Promise.all(
    ["codex", "claude"].map(async (id) => {
      const provider = {
        id,
        name: id === "codex" ? "Codex" : "Claude Code",
        installed: false,
        authenticated: false,
        detail: "Not installed",
      };
      try {
        const version = await runProcess(id, ["--version"], { timeout: 8_000 });
        provider.version = version.output.trim().slice(0, 200);
        provider.installed = true;
        const { output, error } = await runProcess(
          id,
          id === "codex" ? ["login", "status"] : ["auth", "status", "--json"],
          { env: subscriptionEnvironment(), timeout: 10_000 },
        );
        if (id === "codex")
          provider.authenticated = /ChatGPT/i.test(output + error);
        else {
          const auth = JSON.parse(output);
          provider.authenticated =
            auth.loggedIn === true && auth.authMethod === "claude.ai";
        }
        provider.detail = provider.authenticated
          ? "Signed in with your subscription"
          : `Sign in using ${id === "codex" ? "codex login" : "claude auth login"}`;
      } catch {
        if (provider.installed)
          provider.detail = "Subscription sign-in needs attention";
      }
      return provider;
    }),
  );
}

export function agentArguments(id, workspace, context, prompt) {
  const args = [mcpScript, resolve(context)];
  if (id === "codex")
    return [
      "exec",
      "--json",
      "--skip-git-repo-check",
      "--ignore-user-config",
      "--ignore-rules",
      "--ephemeral",
      // Every invocation includes Helix's contract. Avoid scanning unrelated AGENTS.md
      // files through a nested namespace that hosted runtimes may not support.
      "-c",
      "project_doc_max_bytes=0",
      "-c",
      'approval_policy="never"',
      "-c",
      'default_permissions="helix"',
      "-c",
      'permissions.helix.filesystem={":minimal"="read",":workspace_roots"="write"}',
      "-c",
      'permissions.helix.network.enabled=false',
      "-c",
      'features.plugins=false',
      "-c",
      'features.hooks=false',
      "-c",
      'features.multi_agent=false',
      "-c",
      'features.shell_tool=false',
      "-c",
      'features.view_image=false',
      "-c",
      'features.image_generation=false',
      "-c",
      'web_search="live"',
      "-c",
      `mcp_servers.helix.command=${JSON.stringify(process.execPath)}`,
      "-c",
      `mcp_servers.helix.args=${JSON.stringify(args)}`,
      "-c",
      "mcp_servers.helix.required=true",
      ...helixTools.flatMap((tool) => ["-c", `mcp_servers.helix.tools.${tool.name}.approval_mode="approve"`]),
      "--cd",
      workspace,
      prompt,
    ];
  if (id === "claude")
    return [
      "-p",
      prompt,
      "--output-format",
      "stream-json",
      "--verbose",
      "--restricted",
      "--strict-mcp-config",
      "--no-session-persistence",
      "--disable-slash-commands",
      "--tools",
      "Read,Edit,Write,Glob,Grep,WebSearch",
      "--permission-mode",
      "dontAsk",
      "--allowedTools",
      "Read,Edit,Write,Glob,Grep,WebSearch,mcp__helix__*",
      "--mcp-config",
      JSON.stringify({
        mcpServers: { helix: { command: process.execPath, args } },
      }),
    ];
  throw new Error("Select Codex or Claude Code.");
}

export async function invokeAgent(id, workspace, context, prompt, options = {}) {
  let reply = "";
  const stream = agentStream(id, (type, text) => {
    if (type === "agent") reply = text;
    options.onEvent?.(type, text);
  });
  await runProcess(
    id,
    agentArguments(id, workspace, context, prompt),
    {
      ...options,
      cwd: workspace,
      env: subscriptionEnvironment(),
      onLine: stream.line,
    },
  );
  stream.finish();
  return reply;
}

// Terminal events determine success, including zero-exit provider failures.
export function agentStream(id, onEvent = () => {}) {
  let completed = false, failure = "", malformed = false;
  return {
    line(line) {
      if (!line.trim()) return;
      let event;
      try { event = JSON.parse(line); } catch { malformed = true; return; }
      if (!event || typeof event !== "object") { malformed = true; return; }
      if ((id === "codex" && event.type === "turn.completed") || (id === "claude" && event.type === "result" && event.subtype === "success" && !event.is_error)) completed = true;
      if (event.type === "error" || event.type === "turn.failed" || (event.type === "result" && (event.is_error || event.subtype !== "success"))) {
        failure = String(event.message || event.error?.message || event.result || event.errors?.join("; ") || "The agent could not complete this step.");
      }
        try {
          if (
            event.type === "error" ||
            event.type === "turn.failed" ||
            (event.type === "result" && event.is_error)
          )
            onEvent(
              "error",
              event.message ||
                event.error?.message ||
                event.result ||
                "The agent could not complete this step.",
            );
          if (
            event.type === "item.completed" &&
            event.item?.type === "agent_message"
          )
            onEvent("agent", event.item.text);
          if (event.type === "assistant")
            for (const part of event.message?.content ?? []) {
              if (part.type === "text") onEvent("agent", part.text);
              if (part.type === "tool_use")
                onEvent(
                  "tool",
                  part.name.replace("mcp__helix__", ""),
                );
            }
          if (
            event.type === "item.completed" &&
            ["web_search", "mcp_tool_call"].includes(event.item?.type)
          )
            onEvent(
              "tool",
              event.item.tool || event.item.type.replaceAll("_", " "),
            );
        } catch {
          malformed = true;
        }
      },
    finish() {
      if (failure) throw new Error(failure);
      if (malformed) throw new Error("The agent returned a malformed event stream.");
      if (!completed) throw new Error("The agent stream ended without a successful completion event.");
    },
  };
}
