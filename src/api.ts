import type { Connection } from "./types";

export async function api<T>(
  connection: Connection,
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${connection.url}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${connection.token}`,
      ...init.headers,
    },
    signal: init.signal ?? AbortSignal.timeout(15_000),
  });
  const body = await response.json();
  if (!response.ok)
    throw Object.assign(new Error(body.error ?? `The runner returned ${response.status}.`), { status: response.status });
  return body as T;
}

export async function localConnection(): Promise<Connection | null> {
  if (location.protocol !== "http:" || !["127.0.0.1", "localhost"].includes(location.hostname)) return null;
  const response = await fetch("/api/session", {
    method: "POST", headers: { "X-Helix-Local": "1" }, signal: AbortSignal.timeout(10_000),
  });
  if (!response.ok) throw new Error("Could not connect to Helix. Reopen the app and try again.");
  const { token, desktop } = await response.json();
  return { ...validateConnection(location.origin, token), desktop: desktop === true };
}

export function validateConnection(url: string, token: string): Connection {
  const parsed = new URL(url);
  if (
    parsed.protocol !== "http:" ||
    !["127.0.0.1", "localhost"].includes(parsed.hostname) ||
    parsed.username ||
    parsed.password ||
    parsed.pathname !== "/" ||
    parsed.search ||
    parsed.hash
  ) {
    throw new Error(
      "Use the local runner address, such as http://127.0.0.1:4319.",
    );
  }
  if (!/^[A-Za-z0-9_-]{24,128}$/.test(token))
    throw new Error("Enter the pairing code printed by your local runner.");
  return { url: parsed.origin, token };
}
