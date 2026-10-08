// Run against npm run dev with playwright-cli run-code --filename local/startup_ui_check.js.
async page => {
  const check=(condition,message)=>{if(!condition)throw new Error(message);};
  const origin=new URL(page.url()).origin;
  const objective='Predict short or long from length and width using a DecisionTreeClassifier';
  const task={agent:'codex',dataset:'/example/measurements.csv',target:'label',objective,metric:'accuracy',minutes:5,trials:3,searchModels:false,model:'DecisionTreeClassifier',output:'py',exportModel:true,exportFormat:'native',validation:'holdout',folds:3,seeds:[42],testFraction:.2,holdoutFraction:.2,splitStrategy:'independent',groupColumn:'',timeColumn:'',assetColumns:[],policy:{augmentation:false,regularization:false,features:false,tuning:false,pretrained:false,ensemble:false}};
  const previous={id:'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',task,status:'completed',phase:'complete',createdAt:'2026-01-01T00:00:00Z',startedAt:null,elapsed:120,trials:[],logs:[],best:null,score:.73,next:0};
  const next={...previous,id:'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb',status:'running',phase:'baseline',score:.89};
  let runs=[previous],finishStart=null,failStart=false,holdRead=false,releaseRead=null,requests=0,submittedTask=null;
  const artifactRequests=[],errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.route(origin+'/**',async route=>{
    const path=new URL(route.request().url()).pathname;
    const json=(value,status=200)=>route.fulfill({status,contentType:'application/json',body:JSON.stringify(value)});
    if(path==='/health')return json({mode:'hosted'});
    if(path==='/account/session')return json({username:'startup-check',csrf:'test-session'});
    if(path==='/account/methods')return json([]);
    if(path==='/account/workspace')return json({ready:true});
    if(path==='/api/providers')return json([{id:'codex',name:'Codex',installed:true,authenticated:true,capability:{status:'verified'}}]);
    if(path==='/api/tools')return json([{id:'runtime',name:'Training',status:'Ready'}]);
    if(path==='/api/tasks/plan') return json({task:route.request().postDataJSON(),reason:'Use classification.'});
    if(path==='/api/runs'){
      if(route.request().method()==='POST'){
        requests++;
        submittedTask=route.request().postDataJSON();
        await new Promise(resolve=>{finishStart=resolve;});
        if(failStart)return json({error:'Startup test failure'},400);
        runs=[next,previous];return json(next,201);
      }
      const snapshot=structuredClone(runs);
      if(holdRead){holdRead=false;await new Promise(resolve=>{releaseRead=resolve;});}
      return json(snapshot);
    }
    if(path.endsWith('/artifacts')){artifactRequests.push(path);return json([]);}
    return route.continue();
  });
  const history=page.getByRole('navigation',{name:'Saved experiments'});
  const past=()=>history.getByRole('button',{name:objective,exact:true}).last();
  const pending=()=>history.getByRole('button',{name:objective+' Starting',exact:true});
  const oldHeading=page.getByText('EXPERIMENT / aaaaaaaa',{exact:true});
  const newHeading=page.getByText('EXPERIMENT / bbbbbbbb',{exact:true});
  const frames=()=>page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  const send=async()=>{
    const requested=page.waitForRequest(request=>request.url()===origin+'/api/runs'&&request.method()==='POST');
    await page.getByRole('button',{name:'Run experiment',exact:true}).click();
    await page.getByRole('button',{name:'Start experiment',exact:true}).click();
    await pending().waitFor();await requested;await frames();
    check(Object.keys(task).every(key=>JSON.stringify(submittedTask[key])===JSON.stringify(task[key])),'The repeated experiment must keep the exact previous settings');
  };
  const startAgain=async()=>{
    await past().click();await page.getByRole('button',{name:'Use these settings',exact:true}).click();
    await page.getByRole('button',{name:'Done',exact:true}).click();
    await send();
  };
  await page.setViewportSize({width:1440,height:900});
  await page.reload();await past().waitFor();
  await startAgain();
  check(await pending().getAttribute('aria-current')==='page','A repeated experiment must immediately select its own Starting entry');
  check(await history.locator('.history-row').count()===2,'Both identical experiments must be visible during startup');
  check(await page.getByText('Starting your experiment',{exact:true}).isVisible(),'The new task needs its own startup view');
  check(await past().isEnabled(),'Startup must not lock finished experiments');
  holdRead=true;
  await past().click();await oldHeading.waitFor();
  check(await page.getByText('0.7300',{exact:true}).isVisible(),'Opening history must show the previous result');
  await pending().click();await page.getByText('Starting your experiment',{exact:true}).waitFor();
  await past().click();await oldHeading.waitFor();
  // Hold an old poll until after POST succeeds to reproduce stale history responses.
  for(let attempt=0;!releaseRead&&attempt<300;attempt++)await frames();
  check(!!releaseRead && !!finishStart,'Expected a pending startup request and history poll');
  finishStart();await pending().waitFor({state:'detached'});
  check(await oldHeading.isVisible(),'Startup completion must not switch away from the result being read');
  const staleResponse=page.waitForResponse(response=>response.url()===origin+'/api/runs'&&response.request().method()==='GET');
  releaseRead();await staleResponse;await frames();
  check(await history.getByRole('button',{name:objective,exact:true}).count()===2,'A late history response must not remove the new experiment');
  await history.getByRole('button',{name:objective,exact:true}).first().click();await newHeading.waitFor();
  check(await page.getByText('0.8900',{exact:true}).isVisible(),'The new entry must open its own result');
  check(requests===1,'Browsing history must not submit a second experiment');
  runs=[previous];finishStart=null;failStart=true;
  await page.reload();await past().waitFor();await startAgain();await past().click();
  finishStart();
  const failed=history.getByRole('button',{name:objective+' Failed',exact:true});await failed.waitFor();
  check(await oldHeading.isVisible(),'A startup failure must not replace the previous experiment');
  await failed.click();await page.getByText('Startup test failure',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Edit message & settings',exact:true}).click();
  check(await page.getByRole('textbox',{name:'Experiment message',exact:true}).inputValue()===objective,'A failed start must keep the message for retry');
  finishStart=null;failStart=false;
  await send();
  finishStart();await newHeading.waitFor();
  check(await history.getByRole('button',{name:objective,exact:true}).first().getAttribute('aria-current')==='page','The pending view must become the actual run when it starts');
  check(artifactRequests.every(path=>!path.includes('/pending/')),'Pending experiments must never fetch a previous run or placeholder artifacts');
  check(errors.length===0,errors.join('\n'));
  await page.unroute(origin+'/**');
  return 'Startup checks passed: identical experiments, history navigation, late responses, failure and retry.';
}
