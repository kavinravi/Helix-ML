// Run against npm run dev with playwright-cli run-code --filename local/ui_check.js.
async page => {
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  const origin = new URL(page.url()).origin;
  let signedIn = false, refreshed = false, submitted = null;
  let savedRuns = [], failDelete = false, deleteRequests = 0;
  const providers = () => [
    {id:'codex',name:'Codex',installed:true,authenticated:refreshed,detail:refreshed?'Signed in with your subscription':'Subscription sign-in needs attention',capability:{status:'failed',detail:'Codex-only test failure'}},
    {id:'claude',name:'Claude Code',installed:true,authenticated:false,detail:'Sign in using claude auth login'},
  ];
  await page.route(origin+'/**', async route => {
    const url = new URL(route.request().url()), path = url.pathname;
    const json = (value, status=200) => route.fulfill({status,contentType:'application/json',body:JSON.stringify(value)});
    if(path==='/health') return json({mode:'hosted'});
    if(path==='/account/session') return json({username:'kavinravi',csrf:'browser-test-session'});
    if(path==='/account/methods') return json([]);
    if(path==='/account/logout') return json({signedOut:true});
    if(path==='/account/workspace') return json({ready:true});
    if(path==='/api/providers') { if(url.searchParams.has('refresh')) refreshed=signedIn; return json(providers()); }
    if(path==='/api/tools') return json([{id:'runtime',name:'Training',status:'Ready',description:'CPU ready'}]);
    if(path==='/api/runs') {
      if(route.request().method()==='POST') { submitted=route.request().postDataJSON(); return json({error:'Intentional test stop after send'},400); }
      return json(savedRuns);
    }
    if (/^\/api\/runs\/[^/]+\/artifacts$/.test(path)) return json([]);
    if (/^\/api\/runs\/[^/]+$/.test(path) && route.request().method()==='DELETE') {
      deleteRequests++;
      if (failDelete) return json({error:'Deletion test failure'},500);
      savedRuns=savedRuns.filter(run=>run.id!==path.split('/').at(-1));
      return json({deleted:true});
    }
    if(path==='/account/provider') { signedIn=true; return json({agent:'codex',status:'connected',url:null,code:null}); }
    if(path==='/api/datasets') return json({id:'example',path:'/test-data'},201);
    if(path.startsWith('/api/datasets/')) return json({size:100},201);
    return route.continue();
  });
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.reload();await page.getByRole('button',{name:'Cloud',exact:true}).waitFor();
  const target=page.getByRole('textbox',{name:'Target column',exact:true});
  check(await target.inputValue()==='','A new experiment must not default to a target column');
  check(await target.getAttribute('placeholder')==='Enter column name','The target hint must not look like a filled column name');
  check(await target.evaluate(el=>el.matches(':placeholder-shown') && getComputedStyle(el,'::placeholder').color!==getComputedStyle(el).color),'An empty target must show a grey placeholder');
  await page.locator('summary').filter({hasText:'Models & strategies'}).click();
  const model=page.getByRole('textbox',{name:'Model or family',exact:true});
  check(await model.count()===0 && await page.getByRole('textbox',{name:'Model constraints',exact:true}).count()===0,'Auto search must not show a model input');
  await page.getByRole('radio',{name:'Choose model',exact:true}).check();
  await model.fill('sklearn.tree.DecisionTreeClassifier');
  await page.getByRole('radio',{name:'Auto search',exact:true}).check();
  check(await model.count()===0,'Switching to auto search must remove the model input');
  await page.getByRole('radio',{name:'Choose model',exact:true}).check();
  check(await model.inputValue()==='','Auto search must clear the previous model constraint');
  await page.getByRole('radio',{name:'Auto search',exact:true}).check();
  const minutes=page.getByRole('spinbutton',{name:'Minutes',exact:true});
  check(await minutes.inputValue()==='30','Expected the default 30 minutes');
  await minutes.focus();await minutes.press('End');await minutes.press('Backspace');await minutes.press('Backspace');
  check(await minutes.inputValue()==='','Minutes must remain empty after deleting both digits');
  check(await minutes.getAttribute('placeholder')==='e.g. 30','Empty minutes need an example placeholder');
  check(await minutes.evaluate(el=>el.matches(':placeholder-shown')),'The placeholder must be visual, not an input value');
  check(await minutes.evaluate(el=>getComputedStyle(el,'::placeholder').color!==getComputedStyle(el).color),'The placeholder must be greyed out');
  await page.getByRole('button',{name:'Done',exact:true}).click();
  check(await minutes.isVisible(),'An empty required budget must not silently close settings');
  await minutes.fill('5');check(await minutes.inputValue()==='5','Replacing 30 with 5 must not insert a zero');
  const trials=page.getByRole('spinbutton',{name:'Max trials',exact:true});await trials.fill('');check(await trials.inputValue()==='','Trials must be clearable');await trials.fill('3');
  const split=page.getByRole('spinbutton',{name:'Test split (%)',exact:true});await split.fill('');check(await split.inputValue()==='','Split must be clearable');await split.fill('0');check(await split.inputValue()==='0','Zero is a valid explicit test split');
  await page.getByRole('combobox',{name:'Validation',exact:true}).selectOption('holdout');
  const holdout=page.getByRole('spinbutton',{name:'Validation (% of train)',exact:true});await holdout.fill('');check(await holdout.inputValue()==='','Holdout must be clearable');await holdout.fill('20');
  await page.getByRole('button',{name:'Done',exact:true}).click();
  await page.getByRole('button',{name:'Try an example',exact:true}).click();
  await page.getByRole('textbox',{name:'Experiment message',exact:true}).waitFor();
  await page.getByLabel('Choose coding agent',{exact:true}).click();await page.getByRole('radio',{name:'OpenAI Codex',exact:true}).check();
  await page.getByRole('dialog',{name:'Connect Codex',exact:true}).waitFor({state:'hidden'});
  check(await page.getByRole('textbox',{name:'Experiment message',exact:true}).inputValue()==='Predict short or long from length and width','Login must preserve the draft');
  check(await page.locator('.composer input[aria-label="Target column"]').count()===0,'The composer must not duplicate the target setting');
  await page.getByRole('button',{name:'Experiment settings',exact:true}).click();await minutes.fill('5');
  check(await target.count()===1,'Settings must contain the only target-column input');
  check(await target.inputValue()==='label','Try an example must set a real target value');
  await page.locator('summary').filter({hasText:'Models & strategies'}).click();
  await page.getByRole('radio',{name:'Auto search',exact:true}).check();
  await target.fill('outcome');await page.getByRole('button',{name:'Done',exact:true}).click();
  await page.getByRole('button',{name:'Run experiment',exact:true}).click();
  await page.getByText('Intentional test stop after send',{exact:true}).waitFor();
  check(submitted?.target==='outcome','Send must use the target column entered in Settings');
  check(submitted?.searchModels===true && submitted.model==='','Auto search must not submit the example model as a hidden constraint');
  check(submitted?.minutes===5 && submitted.agent==='codex','Send must refresh stale sign-in state and submit the exact edited budget');
  await page.getByRole('button',{name:'Cloud',exact:true}).click();
  check(await page.getByRole('button',{name:/Verify agent|Recheck/}).count()===0,'Signing in must not require a separate verification button');
  const rows=page.locator('.provider-check');check(await rows.count()===2,'Expected both agent connections');
  check((await rows.nth(0).innerText()).includes("Codex couldn't start"),'Codex failure needs a provider-specific summary');
  check(!(await rows.nth(1).innerText()).includes('Codex'),'A Codex error must not appear in the Claude row');
  check(await page.getByText('Codex-only test failure',{exact:true}).count()===1,'Technical errors must not be duplicated');
  check(!(await page.getByText('Codex-only test failure',{exact:true}).isVisible()),'Technical errors should be collapsed initially');
  await page.getByRole('button',{name:'Close dialog',exact:true}).click();
  check(await page.getByText('PROTOCOL',{exact:true}).count()===0,'The redundant protocol strip must be removed');
  check(await page.getByRole('button',{name:'Edit experiment protocol',exact:true}).count()===0,'The composer must not have a duplicate protocol button');
  savedRuns=['completed','running','queued'].map((status,index)=>({id:`00000000-0000-4000-8000-00000000000${index}`,task:{...submitted,objective:`${status} experiment`},status,phase:status==='completed'?'complete':'baseline',createdAt:new Date().toISOString(),startedAt:null,elapsed:10,trials:[],logs:[],best:null,score:null,next:0}));
  const past=page.getByRole('button',{name:'completed experiment',exact:true});
  const remove=page.getByRole('button',{name:'Delete experiment: completed experiment',exact:true});
  await past.waitFor();await page.mouse.move(800,20);
  check(await remove.evaluate(el=>getComputedStyle(el).opacity)==='0','Trash must be hidden until hover or keyboard focus');
  await past.hover();check(await remove.evaluate(el=>getComputedStyle(el).opacity)==='1','Hover must reveal the trash button');
  check(await page.getByRole('button',{name:/Delete experiment: (running|queued)/}).count()===0,'Active experiments must not offer deletion');
  await past.click();await page.mouse.move(800,20);
  check(await remove.evaluate(el=>getComputedStyle(el).opacity)==='0','Mouse selection must not leave the trash button visible');
  await past.press('Tab');
  check(await remove.evaluate(el=>getComputedStyle(el).opacity)==='1','Keyboard users must be able to reach deletion');
  page.once('dialog',dialog=>dialog.dismiss());await remove.press('Enter');
  check(deleteRequests===0,'Cancelling must not send a deletion request');
  failDelete=true;page.once('dialog',dialog=>dialog.accept());await remove.press('Enter');
  await page.getByText('Deletion test failure',{exact:true}).waitFor();
  check(await past.count()===1,'A failed deletion must keep the experiment');
  failDelete=false;await remove.focus();page.once('dialog',dialog=>dialog.accept());await remove.press('Enter');
  await past.waitFor({state:'detached'});
  await page.getByText('Intentional test stop after send',{exact:true}).waitFor();
  check(await page.getByRole('button',{name:'Edit message',exact:true}).isVisible(),'Deleting an older experiment must preserve a separate failed submission');
  check(await page.getByRole('button',{name:'running experiment',exact:true}).count()===1,'Deleting history must preserve other experiments');
  await page.reload();await page.getByRole('button',{name:'running experiment',exact:true}).waitFor();
  check(await past.count()===0,'Deleted history must stay gone after refresh');
  await page.getByRole('button',{name:'Close dialog',exact:true}).click();
  const avatar=page.getByRole('button',{name:'Signed in as kavinravi',exact:true});
  check(await avatar.innerText()==='K','The account badge must show only the first initial');
  await page.setViewportSize({width:390,height:844});
  const bounds=await avatar.boundingBox();
  check(bounds && bounds.x>=0 && bounds.x+bounds.width<=390,'The account badge must stay visible on mobile');
  await avatar.focus();await avatar.press('Enter');
  await page.getByRole('heading',{name:'Connections',exact:true}).waitFor();
  check(await page.getByText('Cloud · kavinravi',{exact:true}).isVisible(),'The badge must open the signed-in account');
  await page.getByRole('button',{name:'Sign out',exact:true}).click();
  await page.getByRole('heading',{name:'Welcome back',exact:true}).waitFor();
  check(await avatar.count()===0,'The badge must disappear after signing out');
  await page.setViewportSize({width:1440,height:900});
  check(errors.length===0,errors.join('\n'));
  await page.unroute(origin+'/**');
  await page.evaluate(()=>{document.body.dataset.uiCheck='passed';});
}
