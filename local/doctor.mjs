import { runtimeStatus } from "./runtime.mjs";
import { providers } from "./agents.mjs";

const [runtime, agents] = await Promise.all([runtimeStatus(), providers()]);
console.log(`Node ${process.version} · ${process.platform}/${process.arch}`);
console.log(`${runtime.ready ? "OK" : "SETUP"} ${runtime.detail}`);
for (const agent of agents) console.log(`${agent.installed && agent.authenticated ? "OK" : "SETUP"} ${agent.name}: ${agent.detail}`);
if (!runtime.ready || !agents.some((agent) => agent.installed && agent.authenticated)) process.exitCode = 1;

console.log("Agent execution is verified before the first experiment, or through Connections → Verify agent.");
