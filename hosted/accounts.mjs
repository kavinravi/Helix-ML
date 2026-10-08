import { DatabaseSync } from 'node:sqlite';
import { randomBytes, randomUUID, createHash, scrypt as derive, timingSafeEqual } from 'node:crypto';
import { promisify } from 'node:util';
import { mkdirSync, chmodSync } from 'node:fs';
import { join } from 'node:path';

const scrypt = promisify(derive);
const digest = value => createHash('sha256').update(value).digest('hex');
export function accounts(root) {
  mkdirSync(root, {recursive:true, mode:0o700});
  const file = join(root, 'accounts.sqlite');
  const db = new DatabaseSync(file);
  chmodSync(file, 0o600);
  db.exec(`PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, salt TEXT NOT NULL, password TEXT NOT NULL, workspace TEXT);
    CREATE TABLE IF NOT EXISTS sessions (hash TEXT PRIMARY KEY, userId TEXT NOT NULL, csrf TEXT NOT NULL, expires INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS identities (provider TEXT NOT NULL, subject TEXT NOT NULL, userId TEXT NOT NULL REFERENCES users(id), PRIMARY KEY(provider,subject), UNIQUE(provider,userId));
    CREATE TABLE IF NOT EXISTS oauth_states (hash TEXT PRIMARY KEY, browser TEXT NOT NULL, provider TEXT NOT NULL, verifier TEXT NOT NULL, userId TEXT, expires INTEGER NOT NULL);`);
  const columns = db.prepare('PRAGMA table_info(users)').all().map(column => column.name);
  for (const column of ['createdAt', 'lastSeenAt']) {
    if (!columns.includes(column)) db.exec(`ALTER TABLE users ADD COLUMN ${column} INTEGER`);
  }
  const user = id => db.prepare('SELECT id, username, workspace FROM users WHERE id=?').get(id);
  const workspace = value => value?.workspace ? JSON.parse(value.workspace) : null;
  return {
    async signIn(username, password, register) {
      username = typeof username === 'string' ? username.trim().toLowerCase() : '';
      if (!/^[a-z0-9][a-z0-9_-]{2,31}$/.test(username) || typeof password !== 'string' || password.length < 12 || password.length > 200)
        throw new Error('Use a 3–32 character username and a password of at least 12 characters.');
      const existing = db.prepare('SELECT * FROM users WHERE username=?').get(username);
      const salt = existing?.salt || randomBytes(32).toString('hex');
      const key = await scrypt(password, salt, 64);
      if (register) {
        if (existing) throw new Error('That username is already taken.');
        const id = randomUUID();
        try { db.prepare('INSERT INTO users(id,username,salt,password,createdAt) VALUES(?,?,?,?,?)').run(id,username,salt,key.toString('hex'),Date.now()); }
        catch { throw new Error('That username is already taken.'); }
        return user(id);
      }
      if (!existing || existing.password.length !== 128 || !timingSafeEqual(key, Buffer.from(existing.password, 'hex'))) throw new Error('Incorrect username or password.');
      return user(existing.id);
    },
    identities(id) { return db.prepare('SELECT provider FROM identities WHERE userId=?').all(id).map(row=>row.provider); },
    startOAuth(provider, browser, userId = null) {
      const state = randomBytes(32).toString('base64url'), verifier = randomBytes(32).toString('base64url');
      db.prepare('DELETE FROM oauth_states WHERE expires<?').run(Date.now());
      db.prepare('INSERT INTO oauth_states VALUES(?,?,?,?,?,?)').run(digest(state),digest(browser),provider,verifier,userId,Date.now()+600_000);
      return {state, challenge:createHash('sha256').update(verifier).digest('base64url')};
    },
    takeOAuth(provider, state, browser, currentUser = null) {
      if (!/^[A-Za-z0-9_-]{43}$/.test(state || '') || !/^[A-Za-z0-9_-]{43}$/.test(browser || '')) throw new Error('Sign-in expired. Please try again.');
      const value = db.prepare('DELETE FROM oauth_states WHERE hash=? AND browser=? AND provider=? AND expires>? RETURNING verifier,userId').get(digest(state),digest(browser),provider,Date.now());
      if (!value || value.userId !== currentUser) throw new Error('Sign-in expired. Please try again.');
      return value;
    },
    identify(provider, subject, label, linkUser = null) {
      if (!['google','github'].includes(provider) || typeof subject !== 'string' || !subject || subject.length > 200) throw new Error('Invalid sign-in identity.');
      const existing = db.prepare('SELECT userId FROM identities WHERE provider=? AND subject=?').get(provider,subject);
      if (existing) {
        if (linkUser && existing.userId !== linkUser) throw new Error('That sign-in belongs to another Helix account.');
        return user(existing.userId);
      }
      if (linkUser && db.prepare('SELECT 1 FROM identities WHERE provider=? AND userId=?').get(provider,linkUser)) throw new Error('This account already has that sign-in method.');
      const id = linkUser || randomUUID();
      db.exec('BEGIN IMMEDIATE');
      try {
        if (!linkUser) {
          const name = String(label || provider).toLowerCase().replace(/[^a-z0-9_-]/g,'').replace(/^[^a-z0-9]+/,'').slice(0,20) || provider;
          db.prepare('INSERT INTO users(id,username,salt,password,createdAt) VALUES(?,?,?,?,?)').run(id,`${name}-${id.slice(0,8)}`,'','',Date.now());
        } else if (!user(linkUser)) throw new Error('Sign in to your existing account before linking.');
        db.prepare('INSERT INTO identities VALUES(?,?,?)').run(provider,subject,id);
        db.exec('COMMIT');
      } catch (error) { db.exec('ROLLBACK'); throw error; }
      return user(id);
    },
    session(id) {
      const token = randomBytes(32).toString('base64url'), csrf = randomBytes(24).toString('base64url');
      db.prepare('DELETE FROM sessions WHERE expires<?').run(Date.now());
      db.prepare('INSERT INTO sessions VALUES(?,?,?,?)').run(digest(token),id,csrf,Date.now()+30*86400_000);
      db.prepare('UPDATE users SET lastSeenAt=? WHERE id=?').run(Date.now(),id);
      return {token, csrf};
    },
    authenticate(token) {
      if (!/^[A-Za-z0-9_-]{43}$/.test(token || '')) return null;
      const session = db.prepare('SELECT userId, csrf FROM sessions WHERE hash=? AND expires>?').get(digest(token),Date.now());
      // Record authenticated activity at most once every five minutes per user.
      if (session) db.prepare('UPDATE users SET lastSeenAt=? WHERE id=? AND (lastSeenAt IS NULL OR lastSeenAt<?)').run(Date.now(),session.userId,Date.now()-300_000);
      return session ? {...user(session.userId),csrf:session.csrf} : null;
    },
    isAdmin(id, githubSubject) {
      return !!githubSubject && !!db.prepare("SELECT 1 FROM identities WHERE userId=? AND provider='github' AND subject=?").get(id,githubSubject);
    },
    adminOverview(query = '', page = 0) {
      const week = Date.now()-7*86400_000, pageSize = 50;
      const summary = db.prepare(`SELECT count(*) AS users, count(CASE WHEN createdAt>=? THEN 1 END) AS newUsers,
        count(CASE WHEN lastSeenAt>=? THEN 1 END) AS seenUsers,
        count(CASE WHEN workspace IS NOT NULL AND workspace!='null' THEN 1 END) AS workspaces FROM users`).get(week,week);
      const matching = db.prepare('SELECT count(*) AS count FROM users WHERE instr(username,?)>0').get(query.toLowerCase()).count;
      const rows = db.prepare(`SELECT id, username, createdAt, lastSeenAt, (password!='') AS passwordSignIn,
        (workspace IS NOT NULL AND workspace!='null') AS hasWorkspace FROM users WHERE instr(username,?)>0
        ORDER BY lastSeenAt DESC, username, id LIMIT ? OFFSET ?`).all(query.toLowerCase(),pageSize,page*pageSize);
      const users = rows.map(({passwordSignIn,hasWorkspace,...row}) => ({...row,hasWorkspace:!!hasWorkspace,
        signIns:[...(passwordSignIn ? ['password'] : []),...db.prepare('SELECT provider FROM identities WHERE userId=? ORDER BY provider').all(row.id).map(identity => identity.provider)]}));
      return {summary,users,matching,page,pageSize,updatedAt:Date.now()};
    },
    logout(token) { if (token) db.prepare('DELETE FROM sessions WHERE hash=?').run(digest(token)); },
    saveWorkspace(id, value) { db.prepare('UPDATE users SET workspace=? WHERE id=?').run(JSON.stringify(value),id); },
    workspace(id) { return workspace(user(id)); },
    users() { return db.prepare('SELECT id, username, workspace FROM users').all().map(value=>({...value,workspace:workspace(value)})); },
    close() { db.close(); },
  };
}
