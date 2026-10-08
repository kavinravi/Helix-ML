import { useEffect, useState } from "react";
import { api } from "./api";
import type { Connection } from "./types";

type Overview = {
  summary: { users: number; newUsers: number; seenUsers: number; workspaces: number };
  users: { id: string; username: string; createdAt: number | null; lastSeenAt: number | null; signIns: string[]; hasWorkspace: boolean }[];
  matching: number; page: number; pageSize: number; updatedAt: number;
};
const date = (value: number | null) => value === null ? "Not recorded" : new Date(value).toLocaleString(undefined, { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" });
const methods: Record<string, string> = { github: "GitHub", google: "Google", password: "Password" };

export default function Admin({ connection, username, accountError, signOut, expired }: { connection: Connection; username: string; accountError: string; signOut: () => void; expired: () => void }) {
  const [data, setData] = useState<Overview | null>(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("");
  const [page, setPage] = useState(0);
  const [revision, setRevision] = useState(0);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    setBusy(true); setError("");
    void api<Overview>(connection, `/account/admin/users?${new URLSearchParams({ q: filter, page: String(page) })}`, { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(15_000)]) })
      .then(value => { if (!controller.signal.aborted) setData(value); })
      .catch(e => { if (!controller.signal.aborted) { setData(null); if (e.status === 401) expired(); else setError(e.message); } })
      .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    return () => controller.abort();
  }, [connection, filter, page, revision, expired]);
  return <main className="admin-page">
    <header className="admin-header"><a className="account-wordmark" href="/">helix<span>.</span></a><span className="admin-owner">Owner</span><div className="admin-account"><span title={username}>{username}</span><button className="text-button" onClick={signOut}>Sign out</button></div></header>
    <section className="admin-content" aria-labelledby="admin-title">
      <a className="admin-back" href="/">← Experiments</a>
      <div className="admin-heading"><h1 id="admin-title">Users & activity</h1><button className="admin-action" onClick={() => setRevision(value => value + 1)} disabled={busy}>{busy ? "Updating…" : "Refresh"}</button></div>
      {(error || accountError) && <p className="inline-error" role="alert">{error || accountError}</p>}
      {data && <dl className="admin-stats">
        <div><dt>Total users</dt><dd>{data.summary.users.toLocaleString()}</dd></div>
        <div><dt>Seen in 7 days</dt><dd>{data.summary.seenUsers.toLocaleString()}</dd></div>
        <div><dt>New in 7 days</dt><dd>{data.summary.newUsers.toLocaleString()}</dd></div>
        <div><dt>Saved workspaces</dt><dd>{data.summary.workspaces.toLocaleString()}</dd></div>
      </dl>}
      <form className="admin-search" onSubmit={event => { event.preventDefault(); setFilter(query.trim()); setPage(0); setRevision(value => value + 1); }}><label htmlFor="admin-query">Find a user</label><input id="admin-query" type="search" placeholder="Search username" maxLength={64} value={query} onChange={event => setQuery(event.target.value)} /><button className="admin-action" disabled={busy}>Search</button>{filter && <button className="text-button" type="button" disabled={busy} onClick={() => { setQuery(""); setFilter(""); setPage(0); }}>Clear</button>}</form>
      <div className="admin-table-wrap" aria-busy={busy} tabIndex={0} role="region" aria-label="Users">
        <table className="admin-table"><caption className="sr-only">Helix users and account activity</caption><thead><tr><th scope="col">User</th><th scope="col">Helix sign-in</th><th scope="col">Joined</th><th scope="col">Last seen</th><th scope="col">Workspace</th></tr></thead><tbody>
          {data?.users.map(user => <tr key={user.id}><th scope="row"><span className="admin-user"><span className="account-avatar" aria-hidden="true">{user.username.charAt(0).toUpperCase()}</span><span>{user.username}</span></span></th><td><div className="admin-methods">{user.signIns.map(method => <span key={method}>{method !== "password" && <img src={`/brands/${method}.svg`} alt="" width="14" height="14" />}{methods[method] || method}</span>)}</div></td><td><time dateTime={user.createdAt === null ? undefined : new Date(user.createdAt).toISOString()}>{date(user.createdAt)}</time></td><td><time dateTime={user.lastSeenAt === null ? undefined : new Date(user.lastSeenAt).toISOString()}>{date(user.lastSeenAt)}</time></td><td><span className={user.hasWorkspace ? "admin-saved" : ""}>{user.hasWorkspace ? "Saved" : "Not created"}</span></td></tr>)}
          {!data?.users.length && <tr><td colSpan={5} className="admin-empty" role="status">{busy ? "Loading users…" : error ? "User data is unavailable." : filter ? "No users match this search." : "No users yet."}</td></tr>}
        </tbody></table>
      </div>
      {data && <div className="admin-pagination"><span>{data.matching ? `${Math.min(data.page * data.pageSize + 1, data.matching)}–${Math.min((data.page + 1) * data.pageSize, data.matching)} of ${data.matching.toLocaleString()} users` : "0 users"}</span><nav aria-label="User pages"><button disabled={busy || page === 0} onClick={() => setPage(value => value - 1)}>Previous</button><button disabled={busy || (page + 1) * data.pageSize >= data.matching} onClick={() => setPage(value => value + 1)}>Next</button></nav></div>}
      <footer className="admin-footnote"><p>Activity tracking starts with this update. Earlier signup dates are not recorded. Last seen reflects authenticated requests, updated at most every 5 minutes.</p>{data && <span role="status">Updated {new Date(data.updatedAt).toLocaleTimeString()}</span>}</footer>
    </section>
  </main>;
}
