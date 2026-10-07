import {randomBytes} from 'node:crypto';

const providers = {
  github: {name:'GitHub', authorize:'https://github.com/login/oauth/authorize', token:'https://github.com/login/oauth/access_token', profile:'https://api.github.com/user', scope:''},
  google: {name:'Google', authorize:'https://accounts.google.com/o/oauth2/v2/auth', token:'https://oauth2.googleapis.com/token', profile:'https://openidconnect.googleapis.com/v1/userinfo', scope:'openid profile'},
};
export function accountOAuth(store, origin) {
  const configured = Object.fromEntries(Object.entries(providers).flatMap(([id,value]) => {
    const client = process.env[`HELIX_${id.toUpperCase()}_CLIENT_ID`], secret = process.env[`HELIX_${id.toUpperCase()}_CLIENT_SECRET`];
    return client && secret ? [[id,{...value, client, secret, callback:`${origin}/account/oauth/${id}/callback`}]] : [];
  }));
  const cookie = (value, age=600) => `helix_oauth=${value}; Path=/account/oauth; HttpOnly; SameSite=Lax; Max-Age=${age}${origin.startsWith('https:')?'; Secure':''}`;
  return {
    methods(userId) {
      const linked = userId ? store.identities(userId) : [];
      return Object.entries(configured).map(([id,provider]) => ({id,name:provider.name,linked:linked.includes(id)}));
    },
    begin(id, userId) {
      const provider = configured[id];
      if (!provider) throw new Error('This sign-in method is not available.');
      const browser = randomBytes(32).toString('base64url');
      const {state,challenge} = store.startOAuth(id,browser,userId);
      const url = new URL(provider.authorize);
      url.search = new URLSearchParams({client_id:provider.client,redirect_uri:provider.callback,response_type:'code',scope:provider.scope,state,code_challenge:challenge,code_challenge_method:'S256'}).toString();
      return {url:url.href,cookie:cookie(browser)};
    },
    clearCookie: () => cookie('',0),
    async finish(id, params, browser, currentUser) {
      const provider = configured[id];
      if (!provider) throw new Error('This sign-in method is not available.');
      const flow = store.takeOAuth(id,params.get('state'),browser,currentUser);
      const code = params.get('code');
      if (params.has('error') || !code || code.length>4096) throw new Error('Sign-in did not finish.');
      const response = await fetch(provider.token,{method:'POST',headers:{Accept:'application/json','Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({client_id:provider.client,client_secret:provider.secret,code,code_verifier:flow.verifier,redirect_uri:provider.callback,grant_type:'authorization_code'}),signal:AbortSignal.timeout(15_000)});
      const token = await response.json();
      if (!response.ok || typeof token.access_token!=='string' || token.access_token.length>8192) throw new Error('Provider sign-in failed.');
      const profileResponse = await fetch(provider.profile,{headers:{Authorization:`Bearer ${token.access_token}`,Accept:'application/json','User-Agent':'Helix-ML'},signal:AbortSignal.timeout(15_000)});
      const profile = await profileResponse.json();
      const subject = id==='google' ? profile.sub : Number.isSafeInteger(profile.id) ? String(profile.id) : null;
      if (!profileResponse.ok || typeof subject!=='string') throw new Error('Could not verify your account.');
      // Provider tokens are used once for identity and discarded; they never authorize ML runs.
      return store.identify(id,subject,id==='google'?profile.name:profile.login,flow.userId);
    },
  };
}
