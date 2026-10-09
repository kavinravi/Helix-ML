// Browser fixture for proposal confirmation and individually measured objectives.
async page => {
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  const origin = new URL(page.url()).origin, errors = [];
  let runs = [], posts = 0;
  page.on('pageerror', error => errors.push(error.message));
  await page.route(origin + '/**', async route => {
    const path = new URL(route.request().url()).pathname;
    const json = (value, status = 200) => route.fulfill({status, contentType:'application/json', body:JSON.stringify(value)});
    if (path === '/health') return json({mode:'hosted'});
    if (path === '/account/session') return json({username:'kavinravi', csrf:'test'});
    if (path === '/account/methods') return json([]);
    if (path === '/account/workspace') return json({ready:true});
    if (path === '/api/providers') return json([{id:'codex', name:'Codex', installed:true, authenticated:true, capability:{status:'verified'}}]);
    if (path === '/api/tools') return json([{id:'runtime', status:'Ready'}]);
    if (path === '/api/datasets') return json({id:'data', path:'/data'}, 201);
    if (path.startsWith('/api/datasets/')) return json({}, 201);
    if (path === '/api/tasks/plan') return json({task:{...route.request().postDataJSON(), learning:'supervised', metric:'auroc', metrics:['auroc','f1'], positiveClass:'short', target:'label', searchModels:false, model:'sklearn.linear_model.LogisticRegression'}, reason:'Maximize the equal-weight mean of ROC-AUC and F1; the positive class is short.'});
    if (path === '/api/runs') {
      if (route.request().method() === 'POST') {
        posts++;
        const task = route.request().postDataJSON();
        check(task.objective === 'Run a logistic regression model and maximize roc-auc & f1 score', 'Preserve the submitted message');
        const trials = [
          {id:'first',name:'Logistic regression',phase:'baseline',status:'accepted',score:.75,metricScores:{auroc:.9,f1:.6}},
          {id:'second',name:'Refined logistic regression',phase:'refine',status:'accepted',score:.84,metricScores:{auroc:.88,f1:.8}}
        ];
        const run = {id:'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', task, status:'completed', phase:'complete', createdAt:new Date().toISOString(), elapsed:20, trials, logs:[], score:.84, testScore:.82, metricScores:{auroc:.88,f1:.8}, testMetricScores:{auroc:.86,f1:.78}, best:'second', next:2};
        runs = [run]; return json(run, 201);
      }
      return json(runs);
    }
    if (path.endsWith('/artifacts')) return json([]);
    return route.continue();
  });
  await page.setViewportSize({width:1440,height:1000});
  await page.reload();
  await page.getByRole('button',{name:'Cloud',exact:true}).waitFor();
  await page.getByRole('button',{name:'Done',exact:true}).click();
  await page.getByRole('button',{name:'Try an example',exact:true}).click();
  await page.waitForFunction(() => document.querySelector('textarea[aria-label="Experiment message"]')?.value.includes('DecisionTreeClassifier'));
  await page.getByRole('textbox',{name:'Experiment message',exact:true}).fill('Run a logistic regression model and maximize roc-auc & f1 score');
  await page.getByRole('button',{name:'Run experiment',exact:true}).click();
  await page.getByRole('button',{name:'Start experiment',exact:true}).waitFor();
  check(posts === 0, 'Training must wait for confirmation');
  const preview = await page.locator('.experiment-preview').innerText();
  check(preview.includes('Maximize Mean (ROC-AUC + F1)') && preview.includes('positive class is short'), 'Show the joint objective before training');
  await page.screenshot({path:'output/playwright/metrics-preview.png',fullPage:true});
  await page.getByRole('button',{name:'Start experiment',exact:true}).click();
  await page.locator('.metric-scores').waitFor();
  check(posts === 1, 'Confirm exactly one experiment');
  const values = await page.locator('.metric-scores').innerText();
  check(values.includes('ROC-AUC') && values.includes('F1') && values.includes('0.8800') && values.includes('0.7800') && values.includes('Equal-weight mean'), 'Show every individual validation/test score and mean');
  check(await page.getByRole('img',{name:/Validation (ROC-AUC|F1)\. 2 measured/}).count() === 2, 'Separate metric curves');
  await page.screenshot({path:'output/playwright/metrics-results.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Results must fit a mobile screen');
  await page.screenshot({path:'output/playwright/metrics-mobile.png',fullPage:true});
  check(errors.length === 0, errors.join('\n'));
  await page.unroute(origin+'/**');
  return 'Joint-metric preview, confirmation, results, curves, and mobile layout passed.';
}
