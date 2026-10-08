// Browser check for inferred tasks, confirmation, dimensionality controls, and real result rendering.
async page => {
  const check=(condition,message)=>{if(!condition)throw new Error(message);};
  const origin=new URL(page.url()).origin;let runs=[],posts=0,proposalBody,runBody;
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route(origin+'/**',async route=>{
    const path=new URL(route.request().url()).pathname;
    const json=(value,status=200)=>route.fulfill({status,contentType:'application/json',body:JSON.stringify(value)});
    if(path==='/health')return json({mode:'hosted'});
    if(path==='/account/session')return json({username:'kavinravi',csrf:'test'});
    if(path==='/account/methods')return json([]);
    if(path==='/account/workspace')return json({ready:true});
    if(path==='/api/providers')return json([{id:'codex',name:'Codex',installed:true,authenticated:true,capability:{status:'verified'}}]);
    if(path==='/api/tools')return json([{id:'runtime',status:'Ready'}]);
    if(path==='/api/datasets')return json({id:'data',path:'/data'},201);
    if(path.startsWith('/api/datasets/'))return json({},201);
    if(path==='/api/tasks/plan') {
      proposalBody=route.request().postDataJSON();const reduction=proposalBody.objective.includes('PCA');
      return json({task:{...proposalBody,learning:reduction?'reduction':'clustering',metric:reduction?'trustworthiness':'silhouette',target:'',searchModels:!reduction,model:reduction?'sklearn.decomposition.PCA':'',excludedColumns:['label'],policy:{...proposalBody.policy,ensemble:false}},reason:reduction?'Fit PCA to retain the requested variance.':'Cluster numeric measurements and maximize silhouette.'});
    }
    if(path==='/api/runs'){
      if(route.request().method()==='POST'){
        posts++;runBody=route.request().postDataJSON();
        const run={id:'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',task:runBody,status:'completed',phase:'complete',createdAt:new Date().toISOString(),startedAt:null,elapsed:20,trials:[],logs:[],score:.92,best:'pca',next:1,projection:{kind:'reduction',axes:['Dimension 1','Dimension 2'],rows:60,scope:'Full dataset',dimensions:2,cumulativeVariance:.963,points:[{row:0,x:-1,y:2},{row:1,x:1,y:-2}]}};
        runs=[run];return json(run,201);
      }
      return json(runs);
    }
    if(path.endsWith('/artifacts'))return json([{path:'final/representation.csv',size:120}]);
    return route.continue();
  });
  await page.setViewportSize({width:1440,height:900});await page.reload();await page.getByRole('button',{name:'Cloud',exact:true}).waitFor();
  check(await page.getByRole('combobox',{name:'Metric',exact:true}).count()===0,'Metric must be inferred from chat');
  check(await page.getByRole('radio',{name:'Choose model',exact:true}).count()===0,'Model selector must be removed');
  await page.getByRole('button',{name:'Done',exact:true}).click();
  await page.getByRole('button',{name:'Try an example',exact:true}).click();
  const message=page.getByRole('textbox',{name:'Experiment message',exact:true});await message.fill('Cluster and maximize silhouette score');
  await page.getByRole('button',{name:'Run experiment',exact:true}).click();
  await page.getByRole('button',{name:'Start experiment',exact:true}).waitFor();
  check(posts===0,'Send must not start training before confirmation');
  check(await page.locator('.experiment-preview').innerText().then(t=>t.includes('Clustering')&&t.includes('silhouette')&&t.includes('Agent chooses')),'Show inferred clustering setup');
  await page.getByRole('button',{name:'Edit message & settings',exact:true}).click();
  await message.fill('Use PCA to preserve 95% cumulative variance');
  await page.getByRole('button',{name:'Experiment settings',exact:true}).click();
  await page.locator('summary').filter({hasText:'Strategies & dimensions'}).click();
  await page.locator('summary').filter({hasText:'Dimensionality reduction'}).click();
  const dims=page.getByRole('spinbutton',{name:'Output dimensions',exact:true});await dims.fill('');check(await dims.inputValue()==='','Dimension count must be clearable');await dims.fill('3');
  const mode=page.getByRole('combobox',{name:'Output constraint',exact:true});await mode.selectOption('variance');
  check(await dims.count()===0,'Only one output constraint can be active');
  const variance=page.getByRole('spinbutton',{name:'Cumulative variance (%)',exact:true});await variance.fill('');check(await variance.inputValue()==='','Variance target must be clearable');await variance.fill('95');
  await page.getByRole('button',{name:'Done',exact:true}).click();await page.getByRole('button',{name:'Run experiment',exact:true}).click();
  await page.getByRole('button',{name:'Start experiment',exact:true}).waitFor();
  check(proposalBody.reductionMode==='variance'&&proposalBody.varianceTarget===.95,'The percentage must reach the server as a fraction');
  check(posts===0,'Editing a proposal must still require confirmation');
  check((await page.locator('.experiment-preview').innerText()).includes('95.0% cumulative variance'),'Show variance constraint before training');
  await page.screenshot({path:'output/unsupervised-preview.png',fullPage:true});
  await page.getByRole('button',{name:'Start experiment',exact:true}).click();
  await page.getByRole('button',{name:'Download coordinates',exact:true}).waitFor();
  check(posts===1&&runBody.learning==='reduction'&&runBody.target==='','Confirm submits exactly one unlabeled experiment');
  check((await page.locator('.projection-note').innerText()).includes('96.3% variance retained'),'Show measured retained variance');
  check(await page.getByRole('img',{name:/First two output dimensions/}).count()===1,'Show a coordinate plot');
  await page.screenshot({path:'output/unsupervised-result.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'The result must fit mobile width');
  check(errors.length===0,errors.join('\n'));await page.unroute(origin+'/**');await page.setViewportSize({width:1440,height:900});
  return 'Unsupervised browser checks passed';
}
