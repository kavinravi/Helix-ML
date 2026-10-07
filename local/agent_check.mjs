// Opt-in check: each installed CLI writes a real experiment through Helix MCP.
// Helix then trains, scores, and verifies a model reload in the CPU runtime.
import assert from "node:assert/strict";
import { mkdtemp, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createService } from "./server.mjs";

const root = await mkdtemp(join(tmpdir(), "helix-agent-check-"));
let service;
try {
  service = await createService({ root, port: 0 });
  await new Promise(resolve => service.server.listen(0, "127.0.0.1", resolve));
  const url = `http://127.0.0.1:${service.server.address().port}/api`;
  const request = async (path, body) => {
    const response = await fetch(url + path, { method: body ? "POST" : "GET", headers: { Authorization: `Bearer ${service.token}`, "Content-Type": "application/json" }, ...(body ? { body: JSON.stringify(body) } : {}) });
    const result = await response.json();
    assert.equal(response.status, 200, JSON.stringify(result));
    return result;
  };
  for (const provider of await request("/providers")) {
    if (process.env.HELIX_TEST_AGENT && provider.id !== process.env.HELIX_TEST_AGENT) continue;
    assert.ok(provider.authenticated, `${provider.name}: ${provider.detail}`);
    assert.equal(provider.capability.status, "unchecked");
    console.log(`${provider.name}: checking native agent → MCP source write → CPU training → prediction → model reload…`);
    const result = await request(`/providers/${provider.id}/verify`, {});
    assert.equal(result.status, "verified");
    assert.deepEqual(result.checks, ["mcp", "write", "train", "predict", "reload"]);
    assert.ok(Number.isFinite(result.score) && result.score >= .8);
    assert.equal(result.sourceHash.length, 64);
    assert.equal((await request(`/providers/${provider.id}/verify`, {})).verifiedAt, result.verifiedAt, "Repeated checks should reuse current evidence");
    const refreshed = (await request("/providers")).find(p => p.id === provider.id);
    assert.equal(refreshed.capability.status, "verified");
    assert.equal(service.runs.size, 0, "Readiness checks must not enter experiment history");
    assert.deepEqual(await readdir(join(root, "checks")), [], "Temporary sample and model files should be removed");
    console.log(`${provider.name}: PASS. All five checks verified; accuracy ${result.score.toFixed(4)}. Cached for this CLI/runtime session.`);
  }
} finally {
  await service?.close();
  await rm(root, { recursive: true, force: true });
}
